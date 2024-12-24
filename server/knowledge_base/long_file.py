import os
import time

from langchain.text_splitter import MarkdownTextSplitter

from configs import (
    CHUNK_SIZE,
    logger,
)
from server.knowledge_base.doc_parser import parse_pdf, Chunk, PageDoc
from typing import List

from text_splitter import ChineseRecursiveTextSplitter


# 长文本问题回答，实现方式采用RAG 只支持pdf
# 1.抽取出信息 2.抽取出关键词 3.bm25搜索 4.送到大模型里
class LongPdfFile:
    def __init__(
            self,
            file_path: str,
    ):
        self.filename = os.path.basename(file_path)
        self.ext = os.path.splitext(self.filename)[-1].lower()
        if self.ext != ".pdf":
            raise ValueError(f"文件格式不正确 {self.filename}")
        self.filepath = file_path
        self.pages = None
        self.chunks = None
        self.bm25 = None
        self.doc = None

    def file2pages(self, refresh: bool = False) -> List[PageDoc]:
        if self.pages is None or refresh:
            logger.info(f'Start parsing {self.filename}...')
            time1 = time.time()
            self.pages = parse_pdf(self.filepath)
            time2 = time.time()
            logger.info(f'Finished parsing {self.filename}. Time spent: {time2 - time1} seconds.')
        return self.pages

    def pages2chunks(self, pages: List[PageDoc] = None,
                     refresh: bool = False,
                     chunk_size: int = CHUNK_SIZE) -> List[Chunk]:
        pages = pages or self.file2pages(refresh)
        if not pages:
            return []

        text_splitter = ChineseRecursiveTextSplitter(
            keep_separator=True,
            is_separator_regex=True,
            chunk_size=chunk_size,
            chunk_overlap=25
        )

        chunk_list = []
        chunk_id = 0
        for page in pages:
            for txt in text_splitter.split_text(page.content):
                chunk_list.append(Chunk(content=txt, chunk_id=chunk_id, page_num=page.page_num))
                chunk_id += 1
        self.chunks = chunk_list
        return self.chunks

    def file2chunks(self, refresh: bool = False, chunk_size: int = CHUNK_SIZE) -> List[Chunk]:
        if self.chunks is None or refresh:
            pages = self.file2pages()
            # self.chunks = self.pages2chunks(pages=pages, chunk_size=chunk_size, refresh=refresh)
            self.chunks = self.pages2chunks(pages=pages, refresh=refresh)
        return self.chunks

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)
