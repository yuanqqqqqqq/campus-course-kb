"""``/api/health`` 的集成测试。

走真实的 FastAPI 应用（含 lifespan 与 CORS 中间件），验证阶段 1 的骨架
确实是可启动、可访问的。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.main import API_PREFIX, app

HEALTH_URL = f"{API_PREFIX}/health"


@pytest.fixture
def client() -> Iterator[TestClient]:
    """带 lifespan 的测试客户端。

    使用 ``with`` 进入上下文才会真正执行 startup/shutdown 钩子，
    这样 ``setup_logging`` 的启动路径也被覆盖到。
    """
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    """健康检查应当返回 200 且响应体恰好是 {"status": "ok"}。"""
    response = client.get(HEALTH_URL)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_response_schema(client: TestClient) -> None:
    """响应内容类型应当是 JSON。"""
    response = client.get(HEALTH_URL)

    assert response.headers["content-type"].startswith("application/json")


def test_unknown_route_returns_404(client: TestClient) -> None:
    """未实现的路由应当是 404，而不是被兜底处理成 200。"""
    assert client.get(f"{API_PREFIX}/not-implemented").status_code == 404


def test_cors_header_is_sent_for_allowed_origin(client: TestClient) -> None:
    """来自白名单来源的请求应当带上 Access-Control-Allow-Origin。"""
    response = client.get(HEALTH_URL, headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_preflight_is_handled(client: TestClient) -> None:
    """预检请求应当被 CORS 中间件处理。"""
    response = client.options(
        HEALTH_URL,
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


def test_openapi_schema_is_served(client: TestClient) -> None:
    """OpenAPI 文档可用，且健康检查已被注册。"""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert HEALTH_URL in response.json()["paths"]
