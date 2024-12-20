import openai
from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE, PRESENCE_PENALTY
from server.utils import fschat_openai_api_address
import json
from typing import List, Optional


async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               history: List[dict] = Body([],
                                          description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                          examples=[[
                                              {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                              {"role": "assistant", "content": "虎头虎脑"}]]
                                          ),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               presence_penalty: float = Body(PRESENCE_PENALTY, description="重复惩罚", ge=-2.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               ):
    async def chat_iterator():
        nonlocal max_tokens
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        client = openai.AsyncOpenAI(
            api_key="EMPTY",
            base_url=fschat_openai_api_address()
        )
        messages = history
        messages.append({"role": "user", "content": query})
        completion = await client.chat.completions.create(
            model=LLM_MODELS[0],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            stream=True,
        )

        async for event in completion:
            text = event.choices[0].delta.content
            if text:
                yield json.dumps(
                    {"text": text},
                    ensure_ascii=False)

    return EventSourceResponse(chat_iterator())  # SSE streaming
