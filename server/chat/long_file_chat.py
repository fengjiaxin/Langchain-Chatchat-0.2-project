from fastapi import Body, File, UploadFile
from sse_starlette.sse import EventSourceResponse
from configs import (LLM_MODELS, TEMPERATURE, DEFAULT_MAX_REF_SIZE)
from server.knowledge_base.doc_parser import Chunk
from server.knowledge_base.split_query import SplitQuery
from server.utils import (wrap_done, get_ChatOpenAI,
                          BaseResponse, get_prompt_template, get_temp_dir)
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from server.knowledge_base.lru_cache import longPdfCachePool
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.history import History
from server.knowledge_base.long_file import LongPdfFile
import json
import os
from server.knowledge_base.keyword_search import KeywordSearch


# 上传文件到指定目录下, 目前只支持pdf
def parse_file(
        dir: str,
        file: UploadFile) -> tuple[bool, str, str, List[Chunk]]:
    filename = file.filename
    file_path = os.path.join(dir, filename)
    file_content = file.file.read()  # 读取上传文件的内容

    if not os.path.isdir(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "wb") as f:
        f.write(file_content)

    try:
        # 读取file_path的内容
        longPdfFile = LongPdfFile(file_path=file_path)
        chunks = longPdfFile.file2chunks()
        return True, filename, f"成功上传文件 {filename}", chunks
    except Exception as e:
        msg = f"{filename} 文件上传失败，报错信息为: {e}"
        return False, filename, msg, []


def upload_temp_doc(
        files: List[UploadFile] = File(..., description="上传文件，单个文件")) -> BaseResponse:
    '''
    将文件保存到临时目录
    返回文件名称
    '''
    temp_dir, _ = get_temp_dir()
    success, filename, msg, chunks = parse_file(temp_dir, files[0])

    if success:
        longPdfCachePool.put(filename, chunks)

    return BaseResponse(data={"id": filename, "msg": msg})


async def long_file_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                         filename: str = Body(..., description="文件名称"),
                         stream: bool = Body(True, description="流式输出"),
                         temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                         max_tokens: Optional[int] = Body(None,
                                                          description="限制LLM生成Token数量，默认None代表模型最大值"),
                         ):
    if not longPdfCachePool.contain(filename):
        return BaseResponse(code=404, msg=f"未找到临时文件 {filename}，请先上传文件")

    async def file_base_chat_iterator() -> AsyncIterable[str]:
        nonlocal max_tokens
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=LLM_MODELS[0],
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        # 1.判断query是问题还是总结类，判断比较简单
        summary_flag = '总结' in query
        query_dic = {"text": query, "information": ''}

        # 2.利用大模型抽取关键信息
        if not summary_flag:  # 非总结类
            splitQuery = SplitQuery()
            query_dic["information"] = splitQuery.run(query=query)

        # 3.获取chunks
        chunks = longPdfCachePool.get(filename)

        # 3.keyword 搜索
        content_list = get_content(query_dic["information"], chunks)
        context = "\n\n".join(content_list)

        # 4.送到大模型里
        prompt_template = get_prompt_template("knowledge_base_chat", "default")
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages([input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(file_base_chat_iterator())


def get_content(query: str, chunks: List[Chunk]) -> list:
    if chunks:
        keyWordSearch = KeywordSearch()
        txt_list = keyWordSearch.call(query=query, max_ref_size=DEFAULT_MAX_REF_SIZE,
                                      chunks=chunks)
        return txt_list
    else:
        return []
