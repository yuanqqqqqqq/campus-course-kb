"""文本与校验错误的公共处理。

**为什么单独成模块**：这两个能力原先在各处各写了一份（``as_text`` 有 4 份副本，
Pydantic 错误处理有 3 份），而它们的口径必须完全一致：

- :func:`as_text` 是 metadata → 字符串的**唯一口径**。``Course.to_metadata`` 用它渲染
  抬头，``splitter`` 用它还原前缀——两处算法一旦不一致，抬头就会对不上（而且不会报错）。
- :func:`format_validation_error` / :func:`simplify_validation_errors` 负责把 Pydantic
  错误变成"人话"与"可序列化结构"。后者尤其讲究：``exc.errors()`` 的 ``ctx`` 里可能带
  异常对象，直接塞进 ``details`` 会在序列化响应时二次抛错——漏改任何一份副本，
  那份就会重新踩这个坑。

放在 ``utils`` 而不是 ``schemas``：它是纯函数工具，且 ``utils`` 处在依赖图最底层
（只依赖 ``config``），任何层都能安全引用而不产生环。
"""

from __future__ import annotations

from pydantic import ValidationError


def as_text(value: object) -> str:
    """把任意取值安全地转成去空白的字符串。

    :param value: 可能是 ``str`` / 数字 / ``None``（metadata 里各种标量）。
    :return: ``None`` 与空串都返回 ``""``；其余先 ``str()`` 再去首尾空白。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def format_validation_error(exc: ValidationError) -> str:
    """把 Pydantic 校验错误压成一行可读文本（``字段: 原因`` 用 ``；`` 连接）。"""
    parts: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        parts.append(f"{location}: {error.get('msg', '校验失败')}")
    return "；".join(parts)


def simplify_validation_errors(exc: ValidationError) -> list[dict[str, str]]:
    """把校验错误转成可 JSON 序列化的精简结构，用于写进异常的 ``details``。

    直接放 ``exc.errors()`` 是不安全的：其中的 ``ctx`` 可能携带异常对象，一旦这些
    ``details`` 需要被序列化（例如经 API 层返回）就会二次报错。
    """
    return [
        {
            "loc": ".".join(str(item) for item in error.get("loc", ())) or "<root>",
            "msg": str(error.get("msg", "")),
            "type": str(error.get("type", "")),
        }
        for error in exc.errors(include_url=False)
    ]
