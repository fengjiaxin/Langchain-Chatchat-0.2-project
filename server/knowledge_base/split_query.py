from string import Template
import json5
import time
from openai import OpenAI
from server.utils import get_completion, get_openai_client
from configs import logger, EXTRACT_INFO_TEMPLATE


# 利用大模型提取关键信息

def extract_information(query: str = '', extracted_content: str = '') -> str:
    information = extracted_content.strip()
    if information.startswith('```json'):
        information = information[len('```json'):]
    if information.endswith('```'):
        information = information[:-3]
    try:
        information = '\n'.join(json5.loads(information)['information']).strip()
        if 0 < len(information) <= len(query):
            return information
        else:
            return query
    except Exception:
        return query


class SplitQuery:
    def __init__(self,
                 client: OpenAI = get_openai_client()):
        self.client = client
        self.template = Template(EXTRACT_INFO_TEMPLATE)

    def run(self,
            query: str = '',
            temperature: float = 0.7,
            max_tokens: int = None, ) -> str:

        prompt = self.template.substitute(user_request=query)
        messages = [{"role": "user", "content": prompt}]
        answer = get_completion(messages, self.client, temperature, max_tokens)
        extracted_content = answer.strip()
        logger.info(f'Extracted info from query: {extracted_content}')
        if extracted_content.endswith('}') or extracted_content.endswith('```'):
            return extract_information(query, extracted_content)
        else:
            extracted_content += '"]}'
            return extract_information(query, extracted_content)

    # 提取关键词


if __name__ == "__main__":
    splitQuery = SplitQuery()
    query = "帮我定位风险问题"
    start = time.time()
    response = splitQuery.run(query)
    end = time.time()
    print(response)
    print(end - start)  # 3s

    # x = '{"information": ["风险问题"], "instruction": ["帮我定位"]}'
    # dic = json5.loads(x)
    # print(dic)

    # sent = '帮我定位风险问题,并逐条列举'
    # start = time.time()
    # res = jieba.analyse.extract_tags(sent)
    # end = time.time()
    # print(end - start)  # 0.48s
    # for i in res:
    #     print(i)
