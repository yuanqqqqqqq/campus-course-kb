"""FAQ 管理接口：动态新增问答对。

新增的条目写进**应用正在使用的那一份缓存**（由 :func:`~src.api.dependencies.get_faq_cache`
注入），所以下一个请求立刻就能命中，不需要重启服务；同时落盘到 ``FAQ_PATH``，
重启后仍然在。

**写入保护**

`FAQ_ADMIN_TOKEN` 配置了就校验请求头 ``X-Admin-Token``（用 ``compare_digest``
做定长比较，避免时序侧信道）。**默认留空 = 不校验**：这是刻意的取舍——本项目是
演示/作品集项目，`/api/faq` 是演示里"加一条 FAQ 立刻生效"的关键一步，默认加锁
会让本地示例多一道配置。但它的风险必须说清楚：**不设令牌时，任何能访问该服务的人
都可以把任意文本写进 FAQ，并被原样返回给之后所有问同类问题的用户**。对公网或多用户
环境，请在 `.env` 里设置该令牌（启动日志也会在未设置时告警）。

**为什么默认就落盘**

"动态添加"若只改内存，服务一重启答案就没了——那更像一个调试后门而不是接口。
代价是每次新增都写一次文件（数据量小，可以接受），写入是原子的（先写 ``*.tmp``
再 ``os.replace``），进程挂掉不会留下半截 JSON。
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Header, status

from src.api.dependencies import FAQCacheDep
from src.config import settings
from src.schemas.chat import FAQCreateRequest, FAQCreateResponse
from src.utils.exceptions import UnauthorizedError
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["faq"])


@router.post(
    "/faq",
    response_model=FAQCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="新增一条 FAQ",
    description=(
        "把一组问法与答案写入 FAQ 快路径，立即生效并落盘。\n\n"
        "问法会被归一化（小写、去空白、去标点）后建索引，所以「CS101 几学分」与"
        "「cs101，几学分？」命中同一条。**建议把课程编号写进问法**：模糊匹配是对整串"
        "算相似度的，只写「几学分」这种短词命中率很差。"
    ),
)
def create_faq(
    request: FAQCreateRequest,
    cache: FAQCacheDep,
    x_admin_token: Annotated[str | None, Header()] = None,
) -> FAQCreateResponse:
    """新增一条 FAQ 并落盘。

    :raises UnauthorizedError: 配置了 FAQ_ADMIN_TOKEN 但请求头缺失或不匹配（401）。
    :raises FAQDataError: 条目不合规（正常会被请求模型挡在 422，这里是兜底）。
    :raises ConfigurationError: 缓存未加载（应用启动流程出错）。
    """
    _verify_admin_token(x_admin_token)
    entry = cache.add_faq(
        patterns=request.patterns,
        answer=request.answer,
        course_id=request.course_id,
    )
    cache.save()
    logger.info("FAQ 新增接口 | id=%s patterns=%d", entry.id, len(entry.patterns))

    return FAQCreateResponse(
        id=entry.id,
        patterns=list(entry.patterns),
        answer=entry.answer,
        course_id=entry.course_id,
        saved=True,
    )


def _verify_admin_token(provided: str | None) -> None:
    """校验写入令牌。

    - 未配置 ``FAQ_ADMIN_TOKEN`` → 直接放行（演示/本地默认，见模块文档的取舍说明）；
    - 已配置 → 请求头必须完全匹配，否则 401。

    用 :func:`secrets.compare_digest` 而不是 ``==``：后者在第一个不同字节处就会
    返回，比较耗时会随匹配前缀长度变化，理论上可被用来逐字节猜令牌。
    """
    expected = settings.faq_admin_token.get_secret_value().strip()
    if not expected:
        return
    if provided is None or not secrets.compare_digest(provided, expected):
        logger.warning("FAQ 写入被拒绝：令牌缺失或不匹配")
        raise UnauthorizedError(
            "该接口需要 X-Admin-Token 请求头（服务端已配置 FAQ_ADMIN_TOKEN）。",
            details={"header": "X-Admin-Token"},
        )
