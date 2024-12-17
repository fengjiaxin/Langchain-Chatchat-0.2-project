from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
import json
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from server.chat.history import History
from server.utils import get_prompt_template


async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               history: Union[int, List[History]] = Body([],
                                                         description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                         examples=[[
                                                             {"role": "user",
                                                              "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                             {"role": "assistant", "content": "虎头虎脑"}]]
                                                         ),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               ):
    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history, max_tokens
        callback = AsyncIteratorCallbackHandler()
        callbacks = [callback]

        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=LLM_MODELS[0],
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

        if history:  # 优先使用前端传入的历史消息
            history = [History.from_data(h) for h in history]
            prompt_template = get_prompt_template("llm_chat", "default")
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
        else:
            prompt_template = get_prompt_template("llm_chat", "default")
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        async for token in callback.aiter():
            # Use server-sent-events to stream the response
            yield json.dumps(
                {"text": token},
                ensure_ascii=False)

        await task

    return EventSourceResponse(chat_iterator())
