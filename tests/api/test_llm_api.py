import time
import requests
from openai import OpenAI

test_content = """
请提取问题中的可以帮助检索的重点信息片段和任务描述，以JSON的格式给出：{{"information": ["重点信息片段1", "重点信息片段2"], "instruction": ["任务描述片段1", "任务描述片段2"]}}。
如果是提问，则默认任务描述为：回答问题

Question: MMDET.UTILS是什么
Result: {{"information": ["MMDET.UTILS是什么"], "instruction": ["回答问题"]}}
Observation: ...

Question: 总结
Result: {{"information": [], "instruction": ["总结"]}}
Observation: ...

Question: 要非常详细描述2.1 DATA，2.2 TOKENIZATION，2.3 ARCHITECTURE。另外你能把这篇论文的方法融合进去吗
Result: {{"information": ["2.1 DATA，2.2 TOKENIZATION，2.3 ARCHITECTURE"], "instruction": ["要非常详细描述", "另外你能把这篇论文的方法融合进去吗"]}}
Observation: ...

Question: 帮我统计不同会员等级的业绩
Result: {{"information": ["会员等级的业绩"], "instruction": ["帮我统计"]}}
Observation: ...

Question: 什么是边际效应
Result:
"""


# 测试api相应是否无误
def request_test():
    # 发送POST请求

    url = "http://127.0.0.1:20000/v1/chat/completions"  # API的地址
    data = {

        "model": "Qwen2-0.5B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": test_content
            }
        ]
    }

    response = requests.post(url, json=data)

    # 解析响应
    if response.status_code == 200:
        answer = response.json()
        choice = answer["choices"][0]
        message = choice["message"]
        content = message["content"]
        print(content)
    else:
        print("Request failed:", response.text)


def open_api_test():
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:20000/v1"
    )

    def get_completion(messages, model="Qwen2-0.5B-Instruct"):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7

        )
        return response.choices[0].message.content

    # messages = [{"role": "user", "content": prompt}]
    # prompt = "如何绕过CDN找到真实IP，请列举五种方法"
    messages = [{'role': 'user', 'content':test_content}]

    # prompt = test_content
    print(get_completion(messages, model="Qwen2-0.5B-Instruct"))


if __name__ == '__main__':
    time_start = time.time()
    open_api_test()
    # request_test()
    time_end = time.time()
    print("测试时间 - {:.2f} s".format(time_end - time_start))
