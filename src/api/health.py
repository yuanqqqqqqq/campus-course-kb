"""健康检查路由。

这是当前阶段唯一对外暴露的接口，用于验证"项目能装起来、能跑起来"，
不触碰任何 RAG / Chroma / LLM / FAQ 逻辑。
"""

from __future__ import annotations

from fastapi import APIRouter

from src.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="返回服务存活状态，用于部署探针与本地自检。",
)
async def health() -> HealthResponse:
    """返回 ``{"status": "ok"}``。"""
    return HealthResponse()
