"""问答接口：一次性返回与 SSE 流式返回。

本模块只做两件事：**把请求交给 Pipeline、把结果翻译成 HTTP（或 SSE 帧）**。
它不含任何业务判断——没有意图分支、没有阈值、没有提示词。

**同步还是异步**

两个接口都用同步 ``def``。整条链路（Chroma 查询、OpenAI SDK 调用）都是阻塞式
同步代码，写成 ``async def`` 会在事件循环里阻塞整个服务；交给 FastAPI 丢进线程池
反而是正确且简单的做法。

**错误怎么变成状态码**

路由函数里只跑"流开始之前"的部分，所有异常都交给 ``src/main.py`` 的异常处理器
按类型映射（``GenerationError`` → 502、``LLMTimeoutError`` → 504、
``IndexNotReadyError`` → 503 …）。

流一旦开始，HTTP 状态码就再也改不了了（响应头已经发出），此时中途失败只能发一条
``event: error``。这就是 :class:`~src.services.pipeline.PreparedAnswer` 与
:meth:`~src.services.pipeline.ChatPipeline.stream_answer` 分开的原因。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from time import perf_counter

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.api.dependencies import PipelineDep
from src.schemas.chat import ChatRequest, ChatResponse
from src.services.pipeline import ChatPipeline, PreparedAnswer
from src.utils.exceptions import AppError
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])

#: SSE 媒体类型。SSE 的格式是纯文本协议，这里所有帧都是 UTF-8。
SSE_MEDIA_TYPE = "text/event-stream"


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="课程问答",
    description=(
        "完整问答链路：意图分类 → FAQ 快路径 / 向量检索 → 相关性校验 → DeepSeek 生成。\n\n"
        "- `cached=true` 表示由 FAQ 直接命中，未调用 Embedding / Chroma / LLM\n"
        "- 证据不足时返回固定拒答话术（HTTP 仍为 200，这是业务分支而非错误）\n"
        "- `route` 是命中的意图，决定走了哪条链路"
    ),
)
def chat(request: ChatRequest, pipeline: PipelineDep) -> ChatResponse:
    """跑完整条链路并一次性返回。"""
    return pipeline.run(request)


@router.post(
    "/chat/stream",
    summary="课程问答（SSE 流式）",
    description=(
        "与 `POST /api/chat` 完全相同的链路，区别是答案以 SSE 逐片段下发。\n\n"
        "事件序列：\n"
        "- `event: meta`  —— route / cached / sources（先发元信息，前端可以先渲染引用）\n"
        "- `event: delta` —— `{\"text\": \"...\"}`，若干个，按顺序拼接即完整回答\n"
        "- `event: done`  —— `{\"latency_ms\": ...}`，正常结束\n"
        "- `event: error` —— `{\"code\": ..., \"message\": ...}`，流中途失败\n\n"
        "闲聊、FAQ 命中与拒答这三条分支不调用 LLM，会把整段答案放在**一个** delta 里，"
        "客户端不需要为它们写第二套逻辑。"
    ),
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
            "description": "SSE 事件流",
        }
    },
)
def chat_stream(request: ChatRequest, pipeline: PipelineDep) -> StreamingResponse:
    """准备阶段同步执行（错误能正常变成 HTTP 状态码），生成阶段流式下发。"""
    started_at = perf_counter()
    # 这一步会把"分类 → FAQ → 检索 → 相关性"跑完。失败就抛，交给异常处理器产出
    # 正常的 JSON 错误响应——此时还没有任何 SSE 帧被写出去，状态码还能改。
    prepared = pipeline.prepare(request)

    return StreamingResponse(
        _sse_frames(pipeline, prepared, started_at),
        media_type=SSE_MEDIA_TYPE,
        headers={
            # 关掉中间层的缓冲：Nginx 默认会把响应攒够一个缓冲区再转发，
            # 那样"流式"就退化成了"等很久然后一次性出现"。
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_frames(
    pipeline: ChatPipeline,
    prepared: PreparedAnswer,
    started_at: float,
) -> Iterator[str]:
    """产出 SSE 帧。

    :param prepared: :meth:`~src.services.pipeline.ChatPipeline.prepare` 的结果。
    :param started_at: 请求开始时刻（``perf_counter``），用于最后的耗时统计。

    每帧形如 ``event: delta\\ndata: {...}\\n\\n``。**payload 里的换行由 json 转义**，
    所以每个 data 永远只占一行——SSE 协议里换行是分隔符，一旦数据里出现裸换行，
    客户端会把一行 JSON 拆成两行而解析失败。
    """
    yield _frame(
        "meta",
        {
            "route": prepared.route.value,
            "cached": prepared.cached,
            "sources": [source.model_dump() for source in prepared.sources],
        },
    )

    try:
        for piece in pipeline.stream_answer(prepared):
            yield _frame("delta", {"text": piece})
    except AppError as exc:
        # 流已经开始，HTTP 状态码改不了了。发一条错误事件，让客户端知道
        # "这段话是不完整的"，而不是把半截回答当成完整答案展示。
        logger.error("流式生成中途失败 | code=%s error=%s", exc.code, exc)
        yield _frame("error", {"code": exc.code, "message": exc.message})
        return
    except Exception:
        # 未预期的异常（代码 bug）同样必须转成错误帧：响应头与部分 delta 已经发出，
        # 抛出去只会让连接被掐断，而客户端无法区分"生成中断"与"回答到此结束"。
        # 堆栈进日志，不返回给调用方。
        logger.exception("流式生成中途出现未预期错误")
        yield _frame(
            "error",
            {"code": "INTERNAL_ERROR", "message": "生成过程中出现内部错误，本次回答不完整。"},
        )
        return

    yield _frame("done", {"latency_ms": round((perf_counter() - started_at) * 1000, 3)})


def _frame(event: str, payload: dict[str, object]) -> str:
    """把事件名与数据序列化成一帧 SSE。

    ``ensure_ascii=False``：中文答案直接以 UTF-8 下发，比 ``\\uXXXX`` 转义省一半体积；
    SSE 本身就有 charset 约定，浏览器与 ``EventSource`` 都能正确处理。
    """
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"
