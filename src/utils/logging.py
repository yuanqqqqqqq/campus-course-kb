"""统一日志配置：格式、request_id、脱敏。

全项目禁止使用 ``print`` 输出运行时信息，一律使用本模块提供的 logger：

.. code-block:: python

    from src.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("检索完成 | hits=%d", len(docs))

``setup_logging()`` 只需在进程入口调用一次（``src/main.py`` 的 lifespan 与
``scripts/`` 下的脚本各调用一次即可），重复调用是幂等的。

**request_id：把一次请求的日志串起来**

一次问答会经过 API → Pipeline → 分类 / FAQ / 检索 / 生成 五六个模块，每个模块各自
打日志。没有关联标识时，并发下这些行会交错在一起，排查问题只能靠时间戳猜。
所以：

1. HTTP 中间件为每个请求生成（或沿用调用方传来的）一个 id；
2. 它存在 :class:`~contextvars.ContextVar` 里，日志格式自动带上，业务代码**不需要**
   在自己每一行日志里手写 request_id。

.. code-block:: text

    2026-10-08 10:30:21 | INFO     | request_id=8f3c2a1b | src.services.pipeline | 问答完成 | route=faq ...

**为什么不在中间件里 reset**

响应体（尤其是 SSE 流）是在中间件返回**之后**才被消费的，提前 reset 会让流式输出
阶段的日志丢掉 request_id。每个 HTTP 请求跑在独立的 asyncio Task 里，contextvar
随 Task 结束自然失效，不存在跨请求污染；下一个请求进来时也一定会被重新赋值。

**脱敏**

用户问题默认**不写进日志**（``LOG_REQUEST_CONTENT=false``），只记长度。日志会被
采集、转发、长期留存，"谁问了什么"属于用户内容，不该默认进日志。需要排查时把开关
打开即可。
"""

from __future__ import annotations

import contextlib
import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Final

from src.config import settings

#: 日志格式：时间 | 级别 | request_id | logger 名 | 消息
LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | request_id=%(request_id)s | %(name)s | %(message)s"
)

#: 时间格式：2026-10-07 23:59:59
DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"

#: 没有请求上下文时 request_id 的占位（脚本、启动日志）。
NO_REQUEST_ID: Final[str] = "-"

#: uvicorn 自己的 logger，清空其 handler 后统一由根 logger 输出，
#: 避免同一行日志被打印两次、格式还前后不一致。
_UVICORN_LOGGERS: Final[tuple[str, ...]] = ("uvicorn", "uvicorn.error", "uvicorn.access")

#: 这些第三方库在 INFO 级别过于聒噪（每次 HTTP 调用都打一行），
#: 统一压到 WARNING，需要排查时把级别调成 DEBUG 即可。
_NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "httpx",
    # openai 3.x 把 HTTP 客户端换成了自带的 httpx2：不一起压掉的话，
    # 每次 LLM 调用都会在 INFO 级别打一行请求日志（含完整 URL）。
    "httpx2",
    "httpcore",
    "chromadb",
    "urllib3",
    "sentence_transformers",
    "filelock",
)

#: 标记"这个 handler / filter 是本模块装的"。
#: 目标：``setup_logging`` 幂等（重复调用不会出现双份日志），但又**不去清空
#: 别人挂在 root 上的 handler**——pytest 的日志捕获、gunicorn 的配置都靠它们，
#: 一并清掉会让测试和部署行为变得难以预料。
_HANDLER_TAG: Final[str] = "_campus_kb_handler"

#: 当前请求的 id。业务代码不直接读它，交给日志格式自动带出。
_REQUEST_ID_VAR: ContextVar[str] = ContextVar("request_id", default=NO_REQUEST_ID)

#: 日志里允许出现的用户问题长度上限。
_QUESTION_LOG_LIMIT: Final[int] = 200


# ---------------------------------------------------------------------------
# request_id
# ---------------------------------------------------------------------------
def new_request_id() -> str:
    """生成一个新的短 request_id（8 位十六进制，够用且不占日志宽度）。"""
    return uuid.uuid4().hex[:8]


def get_request_id() -> str:
    """取当前上下文的 request_id；不在请求里时返回 :data:`NO_REQUEST_ID`。"""
    return _REQUEST_ID_VAR.get()


def set_request_id(request_id: str) -> Token[str]:
    """把 request_id 绑到当前上下文，返回可用于恢复的 token。

    调用方通常是 HTTP 中间件；脚本里请用 :func:`bind_request_id`。
    """
    return _REQUEST_ID_VAR.set(request_id or NO_REQUEST_ID)


def reset_request_id(token: Token[str]) -> None:
    """恢复 :func:`set_request_id` 之前的取值（主要用于测试与脚本）。"""
    _REQUEST_ID_VAR.reset(token)


@contextmanager
def bind_request_id(request_id: str | None = None) -> Iterator[str]:
    """在 ``with`` 块内绑定一个 request_id，退出时恢复。

    .. code-block:: python

        with bind_request_id() as rid:
            logger.info("开始建库 | 本次 request_id=%s", rid)

    与 HTTP 中间件不同，这里**会**恢复原值：脚本与测试不是"一个请求一个 Task"，
    不恢复会把 id 泄漏到后续代码里。
    """
    resolved = request_id or new_request_id()
    token = set_request_id(resolved)
    try:
        yield resolved
    finally:
        reset_request_id(token)


def _install_record_factory() -> None:
    """让每条日志记录在**创建时**就带上 request_id。

    为什么不用 ``logger.addFilter``：父 logger 上的 filter 对"子 logger 产生、
    向上冒泡"的记录**不生效**（``callHandlers`` 只遍历 handler，不调用祖先的
    filter），所以挂在 root 上的 filter 会漏掉 ``src.*`` 打的所有日志，格式化时
    直接抛 KeyError。换成包装 :func:`logging.getLogRecordFactory`，任何 handler
    （自带的、pytest 的 caplog、gunicorn 配的）拿到的记录都已经有值。

    幂等：重复调用不会再包一层（否则每包一层都多一次函数调用，且看不出问题）。
    """
    current = logging.getLogRecordFactory()
    if getattr(current, _HANDLER_TAG, False):
        return

    previous_factory = current

    def _record_factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = previous_factory(*args, **kwargs)
        record.request_id = get_request_id()
        return record

    setattr(_record_factory, _HANDLER_TAG, True)
    logging.setLogRecordFactory(_record_factory)


# ---------------------------------------------------------------------------
# 脱敏
# ---------------------------------------------------------------------------
def question_for_log(question: str, *, limit: int = _QUESTION_LOG_LIMIT) -> str:
    """返回可写进日志的问题文本。

    ``LOG_REQUEST_CONTENT=false``（默认）时只输出长度，不输出内容：日志会被采集、
    转发、长期留存，"谁问了什么"是用户内容，不该默认进日志。

    :param question: 原始问题。
    :param limit: 开启内容记录时的截断长度（防超长输入把日志刷爆）。
    """
    if not settings.log_request_content:
        return f"<已脱敏 len={len(question)}>"
    flattened = " ".join(question.split())
    if len(flattened) <= limit:
        return flattened
    return f"{flattened[:limit]}…"


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------
def _is_owned_by_us(marker: object) -> bool:
    """判断 handler / filter 是否为 :func:`setup_logging` 安装的。"""
    return getattr(marker, _HANDLER_TAG, False)


def _build_stream_handler() -> logging.StreamHandler:
    """构造输出到 stdout 的 handler。"""
    stream = sys.stdout
    # Windows 的 stdout 默认是 GBK，日志里出现中文以外的字符（↑ emoji 等）
    # 会抛 UnicodeEncodeError。这里尽力切到 UTF-8；在 pytest 等已经把
    # stdout 换成非 TextIOWrapper 的场景下 reconfigure 会失败，忽略即可。
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        with contextlib.suppress(ValueError, OSError):  # pragma: no cover - 取决于宿主环境
            reconfigure(encoding="utf-8", errors="replace")

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
    setattr(handler, _HANDLER_TAG, True)
    return handler


def setup_logging(level: str | int | None = None) -> None:
    """初始化根 logger。

    :param level: 日志级别。传 ``None`` 时使用 ``settings.log_level``。
                  接受标准级别名（``"INFO"``）或 :mod:`logging` 的整数常量。

    本函数幂等：重复调用时只保留最后一次安装的 handler，因此 uvicorn
    ``--reload`` 重新导入模块不会出现日志重复输出。注意它只清理本模块
    自己安装的 handler，不会动 pytest / gunicorn 等挂在 root 上的 handler。
    """
    resolved_level: int | str = level if level is not None else settings.log_level
    if isinstance(resolved_level, str):
        # 允许传小写，同时对拼错的级别给出明确报错而不是静默降级
        normalized = resolved_level.strip().upper()
        if normalized not in logging.getLevelNamesMapping():
            raise ValueError(f"无效的日志级别：{level!r}")
        resolved_level = normalized

    root = logging.getLogger()
    root.setLevel(resolved_level)

    # 只移除本模块之前装的 handler，保证幂等；别人的 handler 原样保留
    for handler in list(root.handlers):
        if _is_owned_by_us(handler):
            root.removeHandler(handler)
            handler.close()

    _install_record_factory()
    root.addHandler(_build_stream_handler())

    # uvicorn 的日志交给根 logger 统一输出
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        for handler in list(uvicorn_logger.handlers):
            uvicorn_logger.removeHandler(handler)
            handler.close()
        uvicorn_logger.propagate = True

    # 第三方库降噪
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取一个具名 logger。

    :param name: 通常直接传 ``__name__``。
    """
    return logging.getLogger(name)
