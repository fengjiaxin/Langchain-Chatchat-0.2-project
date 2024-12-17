import re
import string
from typing import List, Tuple, Any
from configs import DEFAULT_MAX_REF_SIZE
from configs import logger
from server.knowledge_base.doc_parser import Chunk


def get_the_front_part(chunks: List[Chunk], max_ref_size: int = DEFAULT_MAX_REF_SIZE) -> List[str]:
    res = []
    available_token = max_ref_size
    for chunk in chunks:
        if available_token <= 0:
            break
        if len(chunk.content) <= available_token:
            res.append(chunk.content)
            available_token -= len(chunk.content)
        else:
            break
    return res


class KeywordSearch:
    def __init__(self):
        pass

    def call(self,
             query: str = None,
             max_ref_size: int = DEFAULT_MAX_REF_SIZE,
             chunks: List[Chunk] = None) -> List[str]:

        if not chunks:
            return []
        if not query:
            return get_the_front_part(chunks, max_ref_size)
        all_tokens = count_all_tokens(chunks)
        logger.info(f'all tokens: {all_tokens}')
        if all_tokens <= max_ref_size:
            logger.info('use full ref')
            return [chunk.content for chunk in chunks]

        return self.search(query=query, chunks=chunks, max_ref_size=max_ref_size)

    def search(self, query: str, chunks: List[Chunk], max_ref_size: int = DEFAULT_MAX_REF_SIZE) -> list:
        chunk_and_score = self.sort_by_scores(query=query, chunks=chunks)
        if not chunk_and_score:
            return get_the_front_part(chunks, max_ref_size)

        max_sims = chunk_and_score[0][1]

        if max_sims != 0:
            return self.get_topk(chunk_and_score=chunk_and_score,
                                 chunks=chunks,max_ref_size=max_ref_size)
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
                 chunks: List[Chunk],
                 max_ref_size: int = DEFAULT_MAX_REF_SIZE) -> List[str]:
        available_token = max_ref_size
        context_list = []
        for chunk_id, _ in chunk_and_score:
            if available_token <= 0:
                break
            chunk = chunks[chunk_id]
            if len(chunk.content) > available_token:
                break
            else:
                context_list.append(chunk.content)
                available_token -= len(chunk.content)
        return context_list


def count_all_tokens(chunks: List[Chunk]) -> int:
    res = 0
    for chunk in chunks:
        res += len(chunk.content)
    return res


WORDS_TO_IGNORE = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
    'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
    "wouldn't", '', '\\t', '\\n', '\\\\', '\n', '\t', '\\', ' ', ',', '，', ';', '；', '/', '.', '。', '-', '_', '——',
    '的',
    '吗', '是', '了', '啊', '呢', '怎么', '如何', '什么', '(', ')', '（', '）', '【', '】', '[', ']', '{', '}', '？', '?',
    '！', '!',
    '“', '”', '‘', '’', "'", '"', ':', '：', '讲了', '描述', '讲', '总结', 'summarize', '总结下', '总结一下', '文档',
    '文章', 'article',
    'paper', '文稿', '稿子', '论文', 'PDF', 'pdf', '这个', '这篇', '这', '我', '帮我', '那个', '下', '翻译', '说说',
    '讲讲', '介绍', 'summary'
]

ENGLISH_PUNCTUATIONS = string.punctuation.replace('%', '').replace('.', '').replace(
    '@', '')  # English punctuations to remove. We're separately handling %, ., and @
CHINESE_PUNCTUATIONS = '。？！，、；：“”‘’（）《》【】……—『』「」_'
PUNCTUATIONS = ENGLISH_PUNCTUATIONS + CHINESE_PUNCTUATIONS


def clean_en_token(token: str) -> str:
    punctuations_to_strip = PUNCTUATIONS

    # Detect if the token is a special case like U.S.A., E-mail, percentage, etc.
    # and skip further processing if that is the case.
    special_cases_pattern = re.compile(r'^(?:[A-Za-z]\.)+|\w+[@]\w+\.\w+|\d+%$|^(?:[\u4e00-\u9fff]+)$')
    if special_cases_pattern.match(token):
        return token

    # Strip unwanted punctuations from front and end
    token = token.strip(punctuations_to_strip)

    return token


def tokenize_and_filter(input_text: str) -> str:
    patterns = r"""(?x)                    # Enable verbose mode, allowing regex to be on multiple lines and ignore whitespace
                (?:[A-Za-z]\.)+          # Match abbreviations, e.g., U.S.A.
                |\d+(?:\.\d+)?%?         # Match numbers, including percentages
                |\w+(?:[-']\w+)*         # Match words, allowing for hyphens and apostrophes
                |(?:[\w\-\']@)+\w+       # Match email addresses
                """

    tokens = re.findall(patterns, input_text)

    stop_words = WORDS_TO_IGNORE

    filtered_tokens = []
    for token in tokens:
        token_lower = clean_en_token(token).lower()
        if token_lower not in stop_words and not all(char in PUNCTUATIONS for char in token_lower):
            filtered_tokens.append(token_lower)

    return filtered_tokens


def string_tokenizer(text: str) -> List[str]:
    text = text.lower().strip()
    if has_chinese_chars(text):
        import jieba
        _wordlist_tmp = list(jieba.lcut(text))
        _wordlist = []
        for word in _wordlist_tmp:
            if not all(char in PUNCTUATIONS for char in word):
                _wordlist.append(word)
    else:
        try:
            _wordlist = tokenize_and_filter(text)
        except Exception:
            _wordlist = text.split()
    _wordlist_res = []
    for word in _wordlist:
        if word in WORDS_TO_IGNORE:
            continue
        else:
            _wordlist_res.append(word)

    import snowballstemmer
    stemmer = snowballstemmer.stemmer('english')
    return stemmer.stemWords(_wordlist_res)


def split_text_into_keywords(text: str) -> List[str]:
    _wordlist = string_tokenizer(text)
    wordlist = []
    for x in _wordlist:
        if x in WORDS_TO_IGNORE:
            continue
        wordlist.append(x)
    return wordlist


CHINESE_CHAR_RE = re.compile(r'[\u4e00-\u9fff]')


def has_chinese_chars(data: Any) -> bool:
    text = f'{data}'
    return bool(CHINESE_CHAR_RE.search(text))
