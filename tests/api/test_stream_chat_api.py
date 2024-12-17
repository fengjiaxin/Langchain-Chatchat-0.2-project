import requests
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from server.utils import api_address

from pprint import pprint


api_base_url = api_address()


def dump_input(d, title):
    print("\n")
    print("=" * 30 + title + "  input " + "="*30)
    pprint(d)


def dump_output(r, title):
    print("\n")
    print("=" * 30 + title + "  output" + "="*30)
    for line in r.iter_content(None, decode_unicode=True):
        print(line, end="", flush=True)


headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}

data = {
    "query": "请用100字左右的文字介绍自己",
    "history": [
        {
            "role": "user",
            "content": "你好"
        },
        {
            "role": "assistant",
            "content": "你好，我是人工智能大模型"
        }
    ],
    "stream": True,
    "temperature": 0.7,
}


def test_chat_chat(api="/chat/chat"):
    url = f"{api_base_url}{api}"
    dump_input(data, api)
    response = requests.post(url, headers=headers, json=data, stream=True)
    dump_output(response, api)
    assert response.status_code == 200
