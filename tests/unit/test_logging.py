"""``src.utils.logging`` 的单元测试。

重点验证三件事：日志级别能正确设置、重复调用是幂等的、不会误伤别人挂在
root 上的 handler。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from src.config import settings
from src.utils.logging import (
    _HANDLER_TAG,
    _UVICORN_LOGGERS,
    LOG_FORMAT,
    NO_REQUEST_ID,
    bind_request_id,
    get_logger,
    get_request_id,
    new_request_id,
    question_for_log,
    reset_request_id,
    set_request_id,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _restore_root_logger() -> Iterator[None]:
    """测试结束后把 root logger 恢复原状，避免污染其他测试。

    只移除本模块安装的 handler（靠 ``_HANDLER_TAG`` 识别），其余原样保留。
    """
    root = logging.getLogger()
    original_level = root.level
    yield
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
            handler.close()
    root.setLevel(original_level)


def _owned_handlers() -> list[logging.Handler]:
    """返回 root 上属于本模块的 handler。"""
    return [h for h in logging.getLogger().handlers if getattr(h, _HANDLER_TAG, False)]


def test_setup_logging_installs_single_owned_handler() -> None:
    """一次调用应当且只应当安装一个本模块的 handler。"""
    setup_logging("INFO")
    assert len(_owned_handlers()) == 1


def test_setup_logging_is_idempotent() -> None:
    """重复调用不应产生多份 handler（否则日志会重复输出）。"""
    setup_logging("INFO")
    setup_logging("DEBUG")
    setup_logging("WARNING")

    owned = _owned_handlers()
    assert len(owned) == 1
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_applies_level_case_insensitively() -> None:
    """级别名大小写不敏感。"""
    setup_logging("debug")
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_accepts_integer_level() -> None:
    """也允许直接传 logging 的整数常量。"""
    setup_logging(logging.ERROR)
    assert logging.getLogger().level == logging.ERROR


def test_setup_logging_rejects_unknown_level() -> None:
    """拼错的级别必须报错，而不是静默降级成默认级别。"""
    with pytest.raises(ValueError, match="无效的日志级别"):
        setup_logging("VERBOSE")


def test_setup_logging_preserves_foreign_handlers() -> None:
    """不应当清空别人（pytest / gunicorn）挂在 root 上的 handler。"""
    foreign = logging.NullHandler()
    root = logging.getLogger()
    root.addHandler(foreign)
    try:
        setup_logging("INFO")
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


def test_handler_format_matches_declared_format() -> None:
    """handler 的格式串应与模块常量一致，避免出现两套日志格式。"""
    setup_logging("INFO")
    handler = _owned_handlers()[0]
    assert handler.formatter is not None
    assert handler.formatter._fmt == LOG_FORMAT


def test_uvicorn_loggers_delegate_to_root() -> None:
    """uvicorn 的 logger 应当清空自有 handler 并向上冒泡，避免一行打两遍。"""
    setup_logging("INFO")
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        assert uvicorn_logger.handlers == []
        assert uvicorn_logger.propagate is True


def test_noisy_third_party_loggers_are_downgraded() -> None:
    """第三方库的 INFO 噪音应当被压到 WARNING。"""
    setup_logging("INFO")
    assert logging.getLogger("httpx").level == logging.WARNING


def test_get_logger_returns_named_logger() -> None:
    """get_logger 应返回同名的标准 logger。"""
    logger = get_logger("src.some.module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "src.some.module"
    assert logger is logging.getLogger("src.some.module")


# ---------------------------------------------------------------------------
# request_id
# ---------------------------------------------------------------------------
def test_log_format_contains_request_id() -> None:
    """格式串里必须真的带上 request_id 字段。"""
    assert "%(request_id)s" in LOG_FORMAT


def test_new_request_id_is_short_and_unique() -> None:
    """id 要短（不把日志挤宽）且互不相同。"""
    ids = {new_request_id() for _ in range(50)}

    assert len(ids) == 50
    assert all(len(value) == 8 for value in ids)


def test_request_id_defaults_to_placeholder_outside_a_request() -> None:
    """没有请求上下文时（脚本、启动日志）用占位符，而不是空字段。"""
    token = set_request_id("")
    try:
        assert get_request_id() == NO_REQUEST_ID
    finally:
        reset_request_id(token)


def test_bind_request_id_restores_previous_value() -> None:
    """``with`` 块结束后要恢复原值，别把 id 泄漏给后续代码。"""
    assert get_request_id() == NO_REQUEST_ID

    with bind_request_id("abc12345") as rid:
        assert rid == "abc12345"
        assert get_request_id() == "abc12345"

    assert get_request_id() == NO_REQUEST_ID


def test_bind_request_id_generates_one_when_not_given() -> None:
    """不传参数时自动生成。"""
    with bind_request_id() as rid:
        assert len(rid) == 8


def test_request_id_is_attached_to_every_record(caplog: pytest.LogCaptureFixture) -> None:
    """每条日志记录都要带上 request_id —— 业务代码不需要自己写。

    注意这里用的是**子 logger**：父 logger 上的 filter 对冒泡上来的记录不生效，
    所以实现换成了包装 record factory。这条测试正是钉住这个坑。
    """
    setup_logging("INFO")

    with bind_request_id("8f3c2a1b"), caplog.at_level(logging.INFO):
        get_logger("src.demo.child").info("测试消息")

    record = next(r for r in caplog.records if r.message == "测试消息")
    assert record.request_id == "8f3c2a1b"


def test_record_factory_is_installed_once() -> None:
    """重复 setup_logging 不该把 record factory 一层层包下去。"""
    setup_logging("INFO")
    first = logging.getLogRecordFactory()
    setup_logging("INFO")

    assert logging.getLogRecordFactory() is first
    assert getattr(first, _HANDLER_TAG, False) is True


def test_records_outside_a_request_still_format(caplog: pytest.LogCaptureFixture) -> None:
    """没有请求上下文时也要能正常格式化（用占位符），不能抛 KeyError。"""
    setup_logging("INFO")

    with caplog.at_level(logging.INFO):
        get_logger("src.demo.outside").info("脚本里的日志")

    record = next(r for r in caplog.records if r.message == "脚本里的日志")
    assert record.request_id == NO_REQUEST_ID
    assert "request_id=-" in logging.Formatter(LOG_FORMAT).format(record)


# ---------------------------------------------------------------------------
# 脱敏
# ---------------------------------------------------------------------------
def test_question_is_masked_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认只记长度，不记内容。"""
    monkeypatch.setattr(settings, "log_request_content", False)

    rendered = question_for_log("数据结构的先修课是什么")

    assert "数据结构" not in rendered
    assert "len=11" in rendered


def test_question_is_recorded_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """开关打开后记录原文，并把换行压平（否则一条日志会被拆成多行）。"""
    monkeypatch.setattr(settings, "log_request_content", True)

    assert question_for_log("第一行\n第二行") == "第一行 第二行"


def test_long_question_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """超长输入要截断，否则一次请求就能把日志刷爆。"""
    monkeypatch.setattr(settings, "log_request_content", True)

    rendered = question_for_log("问" * 500, limit=20)

    assert len(rendered) == 21  # 20 个字 + 省略号
    assert rendered.endswith("…")
