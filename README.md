
## 介绍

🤖️ 该项目从langchain-chatchat项目简化而来,
1. 首先本地部署大模型服务，利用fastchat框架
2. 利用大模型服务接口，开发后端服务
3. 开发web服务

### 功能：
1. llm对话功能：直接利用大模型能力。
2. 长文本对话功能, 上传文件后，根据用户搜索出相关文档，和问题一起输入到大模型中，得到回答。



## 快速上手

### 1. 环境配置

+ 首先，确保你的机器安装了 Python 3.8 - 3.11 (我们强烈推荐使用 Python3.11)。

```
$ python --version
Python 3.11.7
```

接着，创建一个虚拟环境，并在虚拟环境内安装项目的依赖

```shell

# 安装全部依赖
$ pip install -r requirements.txt 
```


### 2. 模型下载

模型下载到本地（翻墙下载）
下载到本地后，在configs.model.config.py配置模型名称和路径


### 3. 一键启动
模型下载后，修改configs里面的配置信息
按照以下命令启动项目

```shell
# 启动fastchat模型服务，类似openAI接口服务
$ python startup.py --llm-api 

# 1.启动fastchat模型服务 2.启动项目后台服务
$ python startup.py --all-api  

# 1.启动fastchat模型服务 2.启动项目后台服务 3.启动web服务
$ python startup.py -a  

```



## 代码概览

* chains: llm访问大模型的测试方法
* configs: 包内模型的配置信息
* server: 后端目录
  * chat: 对话目录
    * chat.py: llm对话的方法
    * long_file_chat.py: 长文本对话的方法
    * history.py: 历史信息类
  * knowledge_base: 处理文件的目录
    * doc_parser.py : 读取pdf文件
    * keyword_search.py: 根据关键字从文档中搜索相关文档
    * lru_cache.py: 线程安全缓存类
    * split_query.py: 利用大模型，从query中抽取出关键信息
    * long_file.py: LongPdfFile类
  * api.py: api相关方法
  * minx_chat_openai.py : langchain调用openai的客户端方法
  * utils.py: 各种api，地址的辅助方法类
* tests:测试类
* text_splitter: 切分长文本
* webui_pages: 前端目录
  * dialogue 目录
    * dialogue.py : 前端对话
  * utils.py : 前端使用的方法  

