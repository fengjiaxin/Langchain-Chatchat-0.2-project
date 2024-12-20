from pydantic import BaseModel
import re
from typing import List
from langchain.document_loaders import PyMuPDFLoader


class PageDoc(BaseModel):
    content: str
    page_num: int = 0

    def __init__(self, content: str, page_num: int):
        super().__init__(content=content, page_num=page_num)


class Chunk(BaseModel):
    content: str
    chunk_id: int
    page_num: int

    def __init__(self, content: str, chunk_id: int, page_num: int):
        super().__init__(content=content, chunk_id=chunk_id, page_num=page_num)


class Doc(BaseModel):
    path: str
    title: str
    pages: int
    chunks: List[Chunk]

    def __init__(self, path: str = '', title: str = '', pages: int = 0, chunks: List[Chunk] = None):
        super().__init__(path=path, title=title, pages=pages, chunks=chunks)


def rm_cid(text):
    text = re.sub(r'\(cid:\d+\)', '', text)
    return text


def rm_continuous_placeholders(text):
    text = re.sub(r'[.\- —。_*]{7,}', '\t', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def rm_hexadecimal(text):
    text = re.sub(r'[0-9A-Fa-f]{21,}', '', text)
    return text


def clean_paragraph(text):
    text = rm_cid(text)
    text = rm_hexadecimal(text)
    text = rm_continuous_placeholders(text)
    return text


PARAGRAPH_SPLIT_SYMBOL = '\n'


def parse_pdf(pdf_path: str) -> List[PageDoc]:
    loader = PyMuPDFLoader(pdf_path)
    docs = []
    pages = loader.load_and_split()
    for page in pages:
        content = page.page_content
        content = clean_paragraph(content)
        metadata = page.metadata
        page_num = metadata['page']
        docs.append(PageDoc(content=content, page_num=page_num))
    return docs


def get_plain_doc(docs: list[PageDoc]):
    content_list = []
    for page in docs:
        content_list.append(page.content)
    return PARAGRAPH_SPLIT_SYMBOL.join(content_list)