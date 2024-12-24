from typing import List, Tuple
from configs import DEFAULT_MAX_REF_SIZE, SEARCH_TOP_K
from configs import logger
from server.knowledge_base.doc_parser import Chunk
from server.knowledge_base.word_helper import split_text_into_keywords


def get_the_front_part(chunks: List[Chunk], max_ref_size: int = DEFAULT_MAX_REF_SIZE) -> List[Chunk]:
    res = []
    available_token = max_ref_size
    for chunk in chunks:
        if available_token <= 0:
            break
        if len(chunk.content) <= available_token:
            res.append(chunk)
            available_token -= len(chunk.content)
        else:
            break
    return res


class KeywordSearch:
    def __init__(self):
        pass

    def call(self,
             query: str = None,
             top_k : int = SEARCH_TOP_K,
             max_ref_size: int = DEFAULT_MAX_REF_SIZE,
             chunks: List[Chunk] = None) -> List[Chunk]:

        if not chunks:
            return []
        if not query:
            return get_the_front_part(chunks, max_ref_size)
        all_tokens = count_all_tokens(chunks)
        logger.info(f'all tokens: {all_tokens}')
        if all_tokens <= max_ref_size:
            logger.info('use full ref')
            return chunks

        return self.search(query=query, top_k=top_k,chunks=chunks, max_ref_size=max_ref_size)

    def search(self, query: str, top_k:int, chunks: List[Chunk], max_ref_size: int = DEFAULT_MAX_REF_SIZE) -> List[Chunk]:
        chunk_and_score = self.sort_by_scores(query=query, chunks=chunks)
        if not chunk_and_score:
            return get_the_front_part(chunks, max_ref_size)

        max_sims = chunk_and_score[0][1]

        if max_sims != 0:
            return self.get_topk(chunk_and_score=chunk_and_score,
                                 top_k=top_k,
                                 chunks=chunks)
        else:
            return get_the_front_part(chunks, max_ref_size)

    def sort_by_scores(self, query: str, chunks: List[Chunk]) -> List[Tuple[int, float]]:
        wordlist = split_text_into_keywords(query)
        logger.info('wordlist: ' + ','.join(wordlist))
        if not wordlist:
            # This represents the queries that do not use retrieval: summarize, etc.
            return []

        # Using bm25 retrieval
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([split_text_into_keywords(x.content) for x in chunks])
        doc_scores = bm25.get_scores(wordlist)
        chunk_and_score = [
            (chk.chunk_id, score) for chk, score in zip(chunks, doc_scores)
        ]
        chunk_and_score.sort(key=lambda item: item[1], reverse=True)
        assert len(chunk_and_score) > 0

        return chunk_and_score

    def get_topk(self,
                 chunk_and_score: List[Tuple[int, float]],
                 top_k : int,
                 chunks: List[Chunk]) -> List[Chunk]:

        res_list = []
        for chunk_id, _ in chunk_and_score:
            if len(res_list) >= top_k:
                break
            else:
                chunk = chunks[chunk_id]
                res_list.append(chunk)
        return res_list


def count_all_tokens(chunks: List[Chunk]) -> int:
    res = 0
    for chunk in chunks:
        res += len(chunk.content)
    return res

