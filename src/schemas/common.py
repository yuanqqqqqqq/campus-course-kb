"""跨模块复用的通用 Pydantic 模型。

本模块只放"到处都要用"的结构，例如健康检查响应、统一错误响应。
业务模型分别放在 ``chat.py`` / ``course.py``。
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

#: 文档 metadata 允许的取值类型。
#:
#: ChromaDB 的 metadata 只接受标量（str / int / float / bool），不接受 list、
#: dict 或 None。全项目的 Document.metadata 都必须遵守这条约束，因此把允许的
#: 类型固化成别名，任何"想把 list 塞进 metadata"的写法都会在类型检查阶段被拦下。
MetadataValue: TypeAlias = str | int | float | bool


class HealthResponse(BaseModel):
    """``GET /api/health`` 的响应体。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"] = Field(default="ok", description="服务状态，正常时恒为 ok。")


class ErrorResponse(BaseModel):
    """统一错误响应体。

    由 ``src/main.py`` 中的异常处理器构造，保证任何异常都返回同一种结构，
    调用方不需要为每种错误类型写不同的解析逻辑。
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="机器可读的错误码，例如 CONFIGURATION_ERROR。")
    message: str = Field(description="给开发者看的错误描述。")
    details: dict[str, object] = Field(default_factory=dict, description="附加上下文，便于排查。")
