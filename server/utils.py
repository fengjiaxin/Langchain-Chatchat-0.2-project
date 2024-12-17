import pydantic
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from pathlib import Path
import asyncio
from configs import (LLM_MODELS, LLM_DEVICE,
                     MODEL_PATH, logger, log_verbose,
                     FSCHAT_MODEL_WORKERS, HTTPX_DEFAULT_TIMEOUT)
import os
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import httpx
from typing import (
    Optional,
    Callable,
    Dict,
    Any,
    Awaitable,
    Union,
    Tuple
)
import logging
from server.minx_chat_openai import MinxChatOpenAI


async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn
    except Exception as e:
        logging.exception(e)
        msg = f"Caught exception: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
    finally:
        # Signal the aiter to stop.
        event.set()


def api_address() -> str:
    from configs.server_config import API_SERVER

    host = API_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = API_SERVER["port"]
    return f"http://{host}:{port}"


def get_model_worker_config(model_name: str = None) -> dict:
    '''
    加载model worker的配置项。
    优先级:FSCHAT_MODEL_WORKERS[model_name] > FSCHAT_MODEL_WORKERS["default"]
    '''
    config = FSCHAT_MODEL_WORKERS.get("default", {}).copy()
    config.update(FSCHAT_MODEL_WORKERS.get(model_name, {}).copy())

    # 本地模型
    llm_model_path = MODEL_PATH[model_name]
    config["model_path"] = llm_model_path
    if llm_model_path and os.path.isdir(llm_model_path):
        config["model_path_exists"] = True
    config["device"] = LLM_DEVICE
    return config


def fschat_openai_api_address() -> str:
    from configs.server_config import FSCHAT_OPENAI_API

    host = FSCHAT_OPENAI_API["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_OPENAI_API["port"]
    return f"http://{host}:{port}/v1"


def get_ChatOpenAI(
        model_name: str,
        temperature: float,
        max_tokens: int = None,
        streaming: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        **kwargs: Any,
) -> ChatOpenAI:
    config = get_model_worker_config(model_name)
    ChatOpenAI._get_encoding_model = MinxChatOpenAI.get_encoding_model
    model = ChatOpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_proxy=config.get("openai_proxy"),
        **kwargs
    )
    return model


class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


def MakeFastAPIOffline(
        app: FastAPI,
        static_dir=Path(__file__).parent / "static",
        static_url="/static-offline-docs",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import HTMLResponse

    openapi_url = app.openapi_url
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        '''
        remove original route from app
        '''
        index = None
        for i, r in enumerate(app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        if isinstance(index, int):
            app.routes.pop(index)

    # Set up static file mount
    app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    if docs_url is not None:
        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()

    if redoc_url is not None:
        remove_route(redoc_url)

        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"

            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False,
                redoc_favicon_url=favicon,
            )


def webui_address() -> str:
    from configs.server_config import WEBUI_SERVER

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]
    return f"http://{host}:{port}"


def fschat_controller_address() -> str:
    from configs.server_config import FSCHAT_CONTROLLER

    host = FSCHAT_CONTROLLER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_CONTROLLER["port"]
    return f"http://{host}:{port}"


def fschat_model_worker_address(model_name: str = LLM_MODELS[0]) -> str:
    if model := get_model_worker_config(model_name):
        host = model["host"]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = model["port"]
        return f"http://{host}:{port}"
    return ""


def get_prompt_template(type: str, name: str) -> Optional[str]:
    '''
    从prompt_config中加载模板内容
    type: "llm_chat","agent_chat","knowledge_base_chat","search_engine_chat"的其中一种，如果有新功能，应该进行加入。
    '''

    from configs import prompt_config
    import importlib
    importlib.reload(prompt_config)
    return prompt_config.PROMPT_TEMPLATES[type].get(name)


def set_httpx_config(
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        proxy: Union[str, Dict] = None,
):
    '''
    设置httpx默认timeout。httpx默认timeout是5秒，在请求LLM回答时不够用。
    将本项目相关服务加入无代理列表，避免fastchat的服务器请求错误。(windows下无效)
    对于chatgpt等在线API，如要使用代理需要手动配置。搜索引擎的代理如何处置还需考虑。
    '''

    import httpx
    import os

    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    # 在进程范围内设置系统级代理
    proxies = {}
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    for k, v in proxies.items():
        os.environ[k] = v

    # set host to bypass proxy
    no_proxy = [x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()]
    no_proxy += [
        # do not use proxy for locahost
        "http://127.0.0.1",
        "http://localhost",
    ]
    # do not use proxy for user deployed fastchat servers
    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        host = ":".join(x.split(":")[:2])
        if host not in no_proxy:
            no_proxy.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    def _get_proxies():
        return proxies

    import urllib.request
    urllib.request.getproxies = _get_proxies


def get_httpx_client(
        use_async: bool = False,
        proxies: Union[str, Dict] = None,
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    '''
    helper to get httpx client with default proxies that bypass local addesses.
    '''
    default_proxies = {
        # do not use proxy for locahost
        "all://127.0.0.1": None,
        "all://localhost": None,
    }
    # do not use proxy for user deployed fastchat servers
    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        host = ":".join(x.split(":")[:2])
        default_proxies.update({host: None})

    # get proxies from system envionrent
    # proxy not str empty string, None, False, 0, [] or {}
    default_proxies.update({
        "http://": (os.environ.get("http_proxy")
                    if os.environ.get("http_proxy") and len(os.environ.get("http_proxy").strip())
                    else None),
        "https://": (os.environ.get("https_proxy")
                     if os.environ.get("https_proxy") and len(os.environ.get("https_proxy").strip())
                     else None),
        "all://": (os.environ.get("all_proxy")
                   if os.environ.get("all_proxy") and len(os.environ.get("all_proxy").strip())
                   else None),
    })
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            # default_proxies.update({host: None}) # Origin code
            default_proxies.update({'all://' + host: None})  # PR 1838 fix, if not add 'all://', httpx will raise error

    # merge default proxies with user provided proxies
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # construct Client
    kwargs.update(timeout=timeout, proxies=default_proxies)

    if log_verbose:
        logger.info(f'{get_httpx_client.__class__.__name__}:kwargs: {kwargs}')

    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)


def get_temp_dir(id: str = None) -> Tuple[str, str]:
    '''
    创建一个临时目录，返回（路径，文件夹名称）
    '''
    from configs.basic_config import BASE_TEMP_DIR
    import tempfile

    if id is not None:  # 如果指定的临时目录已存在，直接返回
        path = os.path.join(BASE_TEMP_DIR, id)
        if os.path.isdir(path):
            return path, id

    path = tempfile.mkdtemp(dir=BASE_TEMP_DIR)
    return path, os.path.basename(path)


def get_openai_client() -> OpenAI:
    client = OpenAI(
        api_key="EMPTY",
        base_url=fschat_openai_api_address()
    )
    return client


# messages = [{"role": "user", "content": prompt}]
# 获取大模型回复
def get_completion(messages: List[dict] = None,
                   client: OpenAI = None,
                   temperature: float = 0.7,
                   max_tokens: int = None,
                   ) -> str:
    response = client.chat.completions.create(
        model=LLM_MODELS[0],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
