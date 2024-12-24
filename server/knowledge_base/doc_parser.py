from pydantic import BaseModel
from typing import List
from langchain.document_loaders import PyMuPDFLoader
from server.knowledge_base.word_helper import clean_paragraph


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
    title: str
    pages: int
    chunks: List[Chunk]

    def __init__(self, title: str = '', pages: int = 0, chunks: List[Chunk] = None):
        super().__init__(title=title, pages=pages, chunks=chunks)


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


if __name__ == "__main__":
    pdf_file = r"D:\git\Qwen-finetune\test\files\大模型功能设计.pdf"
    page_list = parse_pdf(pdf_file)
    for page in page_list:
        print(page)
