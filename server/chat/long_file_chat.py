from string import Template
import openai
from fastapi import Body, File, UploadFile
from sse_starlette.sse import EventSourceResponse
from configs import (LLM_MODELS, TEMPERATURE, DEFAULT_MAX_REF_SIZE, KNOWLEDGE_CHAT_TEMPLATE, PRESENCE_PENALTY,
                     USE_LLM_EXTRACT_INFO, SEARCH_TOP_K)
from server.knowledge_base.doc_parser import Chunk
from server.knowledge_base.split_query import SplitQuery
from server.utils import (BaseResponse, get_temp_dir, fschat_openai_api_address)
from typing import List, Optional

from server.knowledge_base.lru_cache import longPdfCachePool
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
                         temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                         top_k: int = Body(SEARCH_TOP_K, description="匹配条数", ge=1, le=20),
                         presence_penalty: float = Body(PRESENCE_PENALTY, description="重复惩罚", ge=-2.0, le=2.0),
                         max_tokens: Optional[int] = Body(None,
                                                          description="限制LLM生成Token数量，默认None代表模型最大值"),
                         ):
    if not longPdfCachePool.contain(filename):
        return BaseResponse(code=404, msg=f"未找到临时文件 {filename}，请先上传文件")

    async def file_base_chat_iterator():
        nonlocal max_tokens
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        client = openai.AsyncOpenAI(
            api_key="EMPTY",
            base_url=fschat_openai_api_address()
        )

        # 1.判断query是问题还是总结类，判断比较简单
        summary_flag = '总结' in query
        query_dic = {"text": query, "information": ''}

        # 2.利用大模型抽取关键信息
        if not summary_flag:  # 非总结类
            if USE_LLM_EXTRACT_INFO:
                splitQuery = SplitQuery()
                query_dic["information"] = splitQuery.run(query=query)
            else:
                query_dic["information"] = query

        # 3.获取chunks
        chunks = longPdfCachePool.get(filename)

        # 3.keyword 搜索
        chunk_list = get_content(query_dic["information"],top_k, chunks)
        context = "\n".join([chunk.content for chunk in chunk_list])

        # 4.送到大模型里
        template = Template(KNOWLEDGE_CHAT_TEMPLATE)
        prompt = template.substitute(context=context, question=query)

        messages = [{"role": "user", "content": prompt}]
        completion = await client.chat.completions.create(
            model=LLM_MODELS[0],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            stream=True,
        )

        source_documents = []
        for idx, chunk in enumerate(chunk_list):
            text = f"""出处 [{idx + 1}] [{chunk.page_num}页] \n\n{chunk.content}\n\n"""
            source_documents.append(text)

        async for event in completion:
            text = event.choices[0].delta.content
            if text:
                yield json.dumps(
                    {"answer": text},
                    ensure_ascii=False)
        yield json.dumps({"docs": source_documents}, ensure_ascii=False)

    return EventSourceResponse(file_base_chat_iterator())


def get_content(query: str, top_k:int, chunks: List[Chunk]) -> List[Chunk]:
    if chunks:
        keyWordSearch = KeywordSearch()
        txt_list = keyWordSearch.call(query=query, top_k = top_k,max_ref_size=DEFAULT_MAX_REF_SIZE,
                                      chunks=chunks)
        return txt_list
    else:
        return []
