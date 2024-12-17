import nltk
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs import VERSION
from configs.server_config import OPEN_CROSS_DOMAIN
import argparse
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from server.chat.chat import chat
from server.chat.long_file_chat import upload_temp_doc, long_file_chat
from server.utils import (BaseResponse, FastAPI, MakeFastAPIOffline )


async def document():
    return RedirectResponse(url="/docs")


def create_app(run_mode: str = None):
    app = FastAPI(
        title="Langchain-Chatchat API Server",
        version=VERSION
    )
    MakeFastAPIOffline(app)
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app, run_mode=run_mode)
    return app


def mount_app_routes(app: FastAPI, run_mode: str = None):
    app.get("/",
            response_model=BaseResponse,
            summary="swagger 文档")(document)

    # Tag: Chat
    app.post("/chat/chat",
             tags=["Chat"],
             summary="与llm模型对话(通过LLMChain)",
             )(chat)

    # TAG: long file chat
    app.post("/chat/long_file_chat",
             tags=["Long File Chat"],
             summary="长文本问答"
             )(long_file_chat)

    app.post("/file_base/upload_temp_doc",
             tags=["Long File Chat"],
             summary="上传文件到临时目录，用于长文件问答总结。"
             )(upload_temp_doc)


def run_api(host, port, **kwargs):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='langchain-ChatGLM',
                                     description='About langchain-ChatGLM, local knowledge based ChatGLM with langchain'
                                                 ' ｜ 基于本地知识库的 ChatGLM 问答')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    app = create_app()

    run_api(host=args.host,
            port=args.port,
            )
