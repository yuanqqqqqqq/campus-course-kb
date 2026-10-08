"""FastAPI 依赖：把应用启动时建好的单例取给路由函数用。

**为什么要有这一层**

向量库连接、Embedding 模型、LLM 客户端、FAQ 缓存、意图分类器都是"有状态且构建昂贵"
的东西：Embedding 模型加载要几秒，Chroma 打开要连 sqlite，LLM 客户端持有 HTTP 连接池。
每个请求新建一份，延迟和内存都会立刻失控。

所以组装只做一次：``src/main.py`` 的 lifespan 里调用
:func:`~src.services.pipeline.build_pipeline` 建好 :class:`~src.services.pipeline.ChatPipeline`，
挂在 ``app.state`` 上；这里只负责把它取出来。测试则用
``app.dependency_overrides[get_pipeline]`` 换成注入假实现的实例——这也是本模块存在的
第二个理由：路由函数不直接摸 ``app.state``，替换点就只有一个。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.routing.faq_cache import FAQCache
from src.services.pipeline import ChatPipeline
from src.utils.exceptions import ServiceNotReadyError
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_pipeline(request: Request) -> ChatPipeline:
    """取出应用级的问答链路。

    :raises ServiceNotReadyError: 应用没走 lifespan 就被调用（例如直接实例化 ASGI
        app 而没进入上下文），映射为 503。这种情况报"服务未就绪"比抛 ``AttributeError``
        再变成 500 更容易定位，且响应体与其它错误同构。
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        logger.error("app.state.pipeline 不存在：应用可能绕过了 lifespan 启动流程")
        raise ServiceNotReadyError("问答服务尚未就绪，请稍后重试。")
    return pipeline


def get_faq_cache(request: Request) -> FAQCache:
    """取出应用级的 FAQ 缓存。

    ``POST /api/faq`` 新增的条目必须写进**业务链路正在用的那一份**缓存，否则会出现
    "接口说加成功了，但提问时依然不命中"——所以这里返回的是同一个实例，而不是新建一个。
    """
    cache = getattr(request.app.state, "faq_cache", None)
    if cache is None:
        logger.error("app.state.faq_cache 不存在：应用可能绕过了 lifespan 启动流程")
        raise ServiceNotReadyError("FAQ 服务尚未就绪，请稍后重试。")
    return cache


#: 路由函数里用的类型别名：``pipeline: PipelineDep`` 即可，不用重复写 Depends。
PipelineDep = Annotated[ChatPipeline, Depends(get_pipeline)]
FAQCacheDep = Annotated[FAQCache, Depends(get_faq_cache)]
