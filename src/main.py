"""FastAPI 应用入口。

启动方式：

.. code-block:: bash

    uvicorn src.main:app --reload

职责有三块：

1. **组装**：lifespan 里构建一次问答链路（Chroma 连接、Embedding 模型、LLM 客户端、
   FAQ 缓存），挂在 ``app.state`` 上供所有请求复用；
2. **挂路由**：``/api/health``、``/api/chat``、``/api/chat/stream``、``/api/faq``；
3. **错误 → HTTP 状态码**：把项目自有的异常体系映射成合适的状态码，并保持响应体
   永远是同一种结构（:class:`~src.schemas.common.ErrorResponse`）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api import chat, faq, health
from src.config import settings
from src.schemas.common import ErrorResponse
from src.services.pipeline import ChatPipeline, build_pipeline
from src.utils.exceptions import (
    AppError,
    ConfigurationError,
    DocumentLoadError,
    FAQDataError,
    GenerationError,
    IndexNotReadyError,
    LLMTimeoutError,
    NoRelevantContextError,
    RetrievalError,
    RoutingError,
    ServiceNotReadyError,
    UnauthorizedError,
)
from src.utils.logging import get_logger, new_request_id, set_request_id, setup_logging

logger = get_logger(__name__)

#: 所有 API 路由的统一前缀。
API_PREFIX = "/api"

#: 异常类型 → HTTP 状态码。**顺序有意义**：从上往下第一个 ``isinstance`` 命中即用，
#: 所以子类必须排在父类前面（``LLMTimeoutError`` 在 ``GenerationError`` 之前）。
#:
#: 语义是照着"调用方该怎么办"分的，不是照着"哪个模块抛的"：
#:
#: - 4xx 只有请求校验失败（由 FastAPI 的 422 处理），业务分支（拒答、闲聊）都是 200
#: - 500 = 服务端配置/数据错了，重试没用，得改配置或改数据
#: - 502 / 504 = 上游（DeepSeek）出错或超时，是"对端的问题"，可以稍后重试
#: - 503 = 自身暂时不可用（索引没建），修好索引即可
_STATUS_BY_ERROR: tuple[tuple[type[AppError], int], ...] = (
    (LLMTimeoutError, status.HTTP_504_GATEWAY_TIMEOUT),
    (GenerationError, status.HTTP_502_BAD_GATEWAY),
    (IndexNotReadyError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (RetrievalError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (ServiceNotReadyError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (UnauthorizedError, status.HTTP_401_UNAUTHORIZED),
    (ConfigurationError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    (FAQDataError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    (RoutingError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    (DocumentLoadError, status.HTTP_500_INTERNAL_SERVER_ERROR),
    # 拒答现在是一条正常分支（Pipeline 直接返回话术），不会走到这里；保留映射是
    # 为了万一有别的调用方按"抛异常"的方式表达同一件事时不至于变成 500。
    (NoRelevantContextError, status.HTTP_404_NOT_FOUND),
)


def status_code_for(error: AppError) -> int:
    """查出某个业务异常应当映射成的 HTTP 状态码。"""
    for error_type, code in _STATUS_BY_ERROR:
        if isinstance(error, error_type):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期钩子。

    启动时初始化日志、打印关键配置（**不打印密钥**）、构建问答链路；关闭时记录退出日志。

    **构建是惰性的**：:func:`~src.services.pipeline.build_pipeline` 不在此时连接 Chroma、
    也不加载 Embedding 模型或建 LLM 客户端（都在首次使用时才发生），所以启动很快，
    缺 API Key 也不会导致服务起不来——部署时可以先探活、再补配置。

    唯一会真正读盘的是 FAQ：文件损坏就在这里失败。这是刻意的：宁可起不来，也不要
    带着一个"永远不命中"的 FAQ 快路径跑起来。
    """
    setup_logging()
    logger.info(
        "应用启动 | name=%s version=%s env=%s log_level=%s faq_admin_token=%s",
        settings.app_name,
        settings.app_version,
        settings.app_env,
        settings.log_level,
        "已设置" if settings.has_faq_admin_token else "未设置",
    )
    logger.info(
        "关键配置 | llm_model=%s base_url=%s embedding_provider=%s embedding_model=%s",
        settings.deepseek_model,
        settings.deepseek_base_url,
        settings.embedding_provider.value,
        settings.embedding_model,
    )
    logger.info(
        "路径配置 | chroma_persist_dir=%s faq_path=%s",
        settings.chroma_persist_dir,
        settings.faq_path,
    )
    if not settings.has_deepseek_api_key:
        # 只警告不阻断：健康检查与 FAQ 快路径不依赖 LLM，应当仍可访问。
        logger.warning(
            "未检测到 DEEPSEEK_API_KEY，涉及 LLM 生成的接口将返回 500（CONFIGURATION_ERROR）。"
            "请在项目根目录的 .env 中配置（可参考 .env.example）。"
        )

    if not settings.has_faq_admin_token:
        # 写入接口默认不校验令牌（演示取舍），但风险必须说出来，不能让它悄悄存在。
        logger.warning(
            "FAQ_ADMIN_TOKEN 未设置：POST /api/faq 对任何能访问本服务的人开放，"
            "写入的内容会被原样返回给之后所有问同类问题的用户。"
            "对公网/多用户环境请在 .env 中设置该令牌。"
        )

    pipeline: ChatPipeline | None = getattr(app.state, "pipeline", None)
    if pipeline is None:
        pipeline = build_pipeline()
        app.state.pipeline = pipeline
        app.state.faq_cache = pipeline.faq_cache
    logger.info(
        "问答链路就绪 | faq_entries=%d collection=%s top_k=%d relevance_threshold=%.2f",
        len(pipeline.faq_cache),
        settings.chroma_collection_name,
        settings.top_k,
        settings.relevance_threshold,
    )

    yield

    logger.info("应用关闭 | name=%s", settings.app_name)


def _build_cors_settings() -> dict[str, object]:
    """根据配置推导 CORS 参数。

    ``allow_origins=["*"]`` 与 ``allow_credentials=True`` 同时出现会被浏览器
    拒绝（CORS 规范不允许），因此在通配来源下自动关闭凭证支持。
    """
    origins = settings.cors_origins
    allow_all = "*" in origins
    return {
        "allow_origins": origins,
        "allow_credentials": not allow_all,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


def create_app() -> FastAPI:
    """构造并返回 FastAPI 应用实例。

    抽成工厂函数便于测试中用不同配置构造独立实例。
    """
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="校园课程知识库 RAG 问答系统",
        lifespan=lifespan,
    )

    application.add_middleware(CORSMiddleware, **_build_cors_settings())
    application.middleware("http")(_request_context_middleware)

    @application.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        """把项目自有异常统一转换成结构化 JSON + 合适的状态码。

        未在 :class:`~src.utils.exceptions.AppError` 体系内的异常（真正的代码 bug）
        仍然走 FastAPI 默认处理，返回 500，以免被静默吞掉。
        """
        status_code = status_code_for(exc)
        logger.error(
            "业务异常 | path=%s status=%d error=%s", request.url.path, status_code, exc
        )
        payload = ErrorResponse(**exc.to_dict())
        # 用 jsonable_encoder 兜底：details 里可能塞进 Path / datetime 等
        # 不可直接 json.dumps 的对象，直接序列化会在异常处理路径上二次抛错。
        return JSONResponse(status_code=status_code, content=jsonable_encoder(payload))

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """请求参数不合法也返回 :class:`~src.schemas.common.ErrorResponse`。

        FastAPI 默认给的是 ``{"detail": [...]}``，和本项目的错误结构不一致——客户端
        就得写两套解析逻辑。这里统一成同一种结构（状态码仍是标准的 422），并把字段
        路径压成一句话，方便直接贴给用户看。
        """
        problems = "; ".join(_format_validation_problem(error) for error in exc.errors())
        logger.warning("请求参数校验失败 | path=%s problems=%s", request.url.path, problems)
        payload = ErrorResponse(
            code="VALIDATION_ERROR",
            message=f"请求参数不合法：{problems}",
            details={"errors": jsonable_encoder(exc.errors())},
        )
        return JSONResponse(
            # 用新名字：starlette 1.7 起 HTTP_422_UNPROCESSABLE_ENTITY 已废弃，
            # 访问它会发 DeprecationWarning（15 条参数校验用例刷 15 条警告）。
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=jsonable_encoder(payload),
        )

    application.include_router(health.router, prefix=API_PREFIX)
    application.include_router(chat.router, prefix=API_PREFIX)
    application.include_router(faq.router, prefix=API_PREFIX)

    return application


async def _request_context_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """为每个请求绑定 request_id，并记录一条访问日志。

    **request_id 从哪来**：优先沿用调用方传的 ``X-Request-ID``（网关、前端、压测工具
    往往已经有自己的 id，链路追踪要靠它对齐），没有就新生成一个，并通过响应头回传。

    **为什么不在 finally 里 reset**：响应体（尤其是 SSE 流）是在中间件返回**之后**
    才被消费的。提前 reset 会让"生成流式答案"那一段日志丢掉 request_id——而那正是
    最需要它的地方。每个请求跑在独立的 asyncio Task 里，contextvar 随 Task 结束失效，
    下一个请求进来时也一定会被重新赋值，因此不存在跨请求污染。

    **延迟口径**：这里记的是"到响应头发出为止"的耗时。流式接口的完整耗时由 SSE 的
    ``done`` 事件给出（见 ``src/api/chat.py``），两者刻意分开，混在一起会让人误以为
    SSE 只要几毫秒。
    """
    request_id = request.headers.get("X-Request-ID") or new_request_id()
    set_request_id(request_id)
    started_at = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        # 未预期的异常交给 Starlette 的 ServerErrorMiddleware 转成 500；
        # 这里只负责留下一条带 request_id 的记录，方便顺着 id 找上下文。
        logger.exception(
            "请求处理失败 | method=%s path=%s", request.method, request.url.path
        )
        raise

    response.headers["X-Request-ID"] = request_id
    logger.info(
        "请求完成 | method=%s path=%s status=%d latency_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        (perf_counter() - started_at) * 1000,
    )
    return response


def _format_validation_problem(error: dict[str, object]) -> str:
    """把一条 FastAPI 校验错误压成 ``字段: 原因`` 的形式。"""
    location = error.get("loc") or ()
    field = ".".join(str(item) for item in location if item != "body") or "body"
    return f"{field}: {error.get('msg', '校验失败')}"


app = create_app()
