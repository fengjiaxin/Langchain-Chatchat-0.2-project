import streamlit as st
from webui_pages.utils import *
from streamlit_chatbox import *
from datetime import datetime
import os

from configs import (TEMPERATURE, HISTORY_LEN, LLM_MODELS, MAX_TOKENS, PRESENCE_PENALTY, SEARCH_TOP_K)
from typing import List, Dict

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "chatchat_icon_blue_square_v2.png"
    )
)


def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)


@st.cache_data
def upload_temp_doc(files, _api: ApiRequest) -> Tuple[str, str]:
    '''
    将文件上传到临时目录，用于文件总结对话
    返回文件名称
    '''
    response = _api.upload_temp_doc(files)
    print(response)
    data = response.get("data", {})
    return data.get("id"), data.get("msg")


def dialogue_page(api: ApiRequest):
    if not chat_box.chat_inited:
        st.toast(
            f"当前运行的模型`{LLM_MODELS[0]}`\n您可以开始提问了."
        )
        chat_box.init_session()

    with st.sidebar:
        # 多会话
        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"已切换到 {mode} 模式。"
            st.toast(text)

        dialogue_modes = ["问答助手",
                          "文件助手",
                          ]
        dialogue_mode = st.selectbox("请选择对话模式：",
                                     dialogue_modes,
                                     index=0,
                                     on_change=on_mode_change,
                                     key="dialogue_mode",
                                     )

        temperature = st.slider("temperature:[数值越大,随机性越高]", 0.0, 2.0, TEMPERATURE, 0.1)
        max_tokens = st.slider("max_token:  [控制模型生成文本长度]", 256, 2048, MAX_TOKENS, 128)
        presence_penalty = st.slider("presence_penalty:[惩罚重复文本]", 0.0, 2.0, PRESENCE_PENALTY, 0.1)
        if dialogue_mode == "问答助手":
            history_len = st.number_input("历史对话轮数：", 0, 10, HISTORY_LEN)
        else:  #"文件助手"
            with st.expander("文件配置", True):
                file = st.file_uploader("上传知识文件：",
                                        [".pdf"],
                                        accept_multiple_files=False)
                files = [file]
                top_k = st.number_input("匹配知识条数：", 1, 20, SEARCH_TOP_K)
                if st.button("开始上传", disabled=len(files) == 0):
                    file_id, msg = upload_temp_doc(files, api)
                    st.toast(msg)
                    st.session_state["file_id"] = file_id

    # Display chat messages from history on app rerun
    chat_box.output_messages()

    chat_input_placeholder = "请输入对话内容，换行请使用Shift+Enter。"

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        chat_box.user_say(prompt)
        if dialogue_mode == "问答助手":
            history = get_messages_history(history_len)
            chat_box.ai_say("正在思考...")
            text = ""
            message_id = ""
            r = api.chat_chat(prompt,
                              history=history,
                              temperature=temperature,
                              max_tokens=max_tokens,
                              presence_penalty=presence_penalty)
            for t in r:
                if error_msg := check_error_msg(t):  # check whether error occured
                    st.error(error_msg)
                    break
                text += t.get("text", "")
                chat_box.update_msg(text)
                message_id = t.get("message_id", "")

            metadata = {
                "message_id": message_id,
            }
            chat_box.update_msg(text, streaming=False, metadata=metadata)  # 更新最终的字符串，去除光标

        else:  #"文件助手"
            if st.session_state["file_id"] is None:
                st.error("请先上传文件再进行对话")
                st.stop()
            chat_box.ai_say([
                f"正在阅读 `{st.session_state['file_id']}`，并思考问题 ...",
                Markdown("...", in_expander=True, title="文件匹配结果", state="complete"),
            ])
            text = ""
            for d in api.long_file_chat(prompt,
                                        filename=st.session_state["file_id"],
                                        top_k=top_k,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        presence_penalty=presence_penalty
                                        ):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, element_index=0)
            chat_box.update_msg(text, element_index=0, streaming=False)
            chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

    now = datetime.now()
    with st.sidebar:

        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )
