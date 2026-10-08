"""``/api/chat``、``/api/chat/stream``、``/api/faq`` 的集成测试。

**真的走 FastAPI**（路由、依赖注入、异常处理器、序列化、SSE 分帧都是真实代码），
但把三个慢依赖换成假实现：LLM（不联网）、向量库（不开 Chroma）、FAQ（写临时文件而不是
仓库里的 ``data/faq.json``）。

这样就覆盖到了"单元测试测不到的那一层"：状态码映射、响应字段、错误结构、SSE 帧格式。
按阶段 6 的要求逐条覆盖：FAQ 命中 / 未命中、course_query、process_query、chitchat、
空检索结果、拒答、LLM 错误、参数错误、SSE。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.api.dependencies import get_faq_cache, get_pipeline
from src.config import settings
from src.main import API_PREFIX, app
from src.retrieval.relevance import RelevanceChecker
from src.routing.faq_cache import FAQCache
from src.schemas.chat import IntentType
from src.services.pipeline import CHITCHAT_REPLY, REFUSAL_REPLY, ChatPipeline
from src.utils.exceptions import (
    ConfigurationError,
    GenerationError,
    LLMTimeoutError,
)
from tests.fakes import FakeIntentClassifier, FakeLLMService, FakeVectorStore, make_document

CHAT_URL = f"{API_PREFIX}/chat"
STREAM_URL = f"{API_PREFIX}/chat/stream"
FAQ_URL = f"{API_PREFIX}/faq"

#: 会被判成 faq 意图、且能命中 conftest 里那条 FAQ 的问题。
FAQ_HIT_QUESTION = "CS101 几学分"

#: 会被判成 faq 意图、但 FAQ 里没有的问题 —— 必须继续走 RAG，而不是拒答。
FAQ_MISS_QUESTION = "CS999 的教材是什么"

#: 会被判成 course_query 的问题。
COURSE_QUESTION = "数据结构主要讲什么"

GOOD_HITS = [(make_document(), 0.82)]
WEAK_HITS = [(make_document(), 0.18)]


@contextmanager
def app_client(
    pipeline: ChatPipeline,
    faq_cache: FAQCache,
    *,
    raise_server_exceptions: bool = True,
) -> Iterator[TestClient]:
    """把给定链路与缓存挂进依赖覆盖，返回进入过 lifespan 的客户端。

    用 ``dependency_overrides`` 而不是改 ``app.state``：请求走的是完整的 FastAPI 栈
    （含 lifespan、CORS、异常处理器），只有那三个慢依赖是假的；替换点也只有一个。

    出口处清空覆盖——``app`` 是模块级单例，覆盖残留会让别的测试文件莫名其妙地用上
    假依赖。

    :param raise_server_exceptions: ``TestClient`` 默认会把服务端未捕获异常直接抛给
        测试，这对"验证 500 响应体"是帮倒忙（异常先炸在测试里，根本拿不到响应）。
        需要看真实 500 时传 ``False``，那才是 uvicorn 下的行为。
    """
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_faq_cache] = lambda: faq_cache
    try:
        with TestClient(app, raise_server_exceptions=raise_server_exceptions) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def _default_classifier() -> FakeIntentClassifier:
    """一份自洽的意图映射：FAQ 文件里的问法 → faq，寒暄 → chitchat，其余 course_query。"""
    return FakeIntentClassifier(
        {
            FAQ_HIT_QUESTION: IntentType.FAQ,
            FAQ_MISS_QUESTION: IntentType.FAQ,
            "你好": IntentType.CHITCHAT,
            "怎么选课": IntentType.PROCESS_QUERY,
        }
    )


@pytest.fixture
def client(faq_cache: FAQCache, make_pipeline) -> Iterator[TestClient]:
    """默认场景：检索命中（0.82）、生成成功、FAQ 文件可用。"""
    pipeline = make_pipeline(
        classifier=_default_classifier(),
        faq_cache=faq_cache,
        vector_store=FakeVectorStore(hits=GOOD_HITS),
        llm=FakeLLMService(),
    )
    with app_client(pipeline, faq_cache) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# 基本契约
# ---------------------------------------------------------------------------
def test_chat_returns_the_documented_response_shape(client: TestClient) -> None:
    """响应字段与文档一致：answer / sources / route / cached / latency_ms。"""
    response = client.post(CHAT_URL, json={"question": COURSE_QUESTION})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"answer", "sources", "route", "cached", "latency_ms"}
    assert body["route"] == "course_query"
    assert body["cached"] is False
    assert body["latency_ms"] >= 0


def test_source_fields_are_serialized(client: TestClient) -> None:
    """来源里的课程 / 类型 / 文件 / 分数都要出现在响应里。"""
    body = client.post(CHAT_URL, json={"question": COURSE_QUESTION}).json()

    source = body["sources"][0]
    assert source["course_id"] == "CS201"
    assert source["type"] == "syllabus"
    assert source["source"] == "course_syllabus.md"
    assert source["score"] == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# FAQ 命中 / 未命中
# ---------------------------------------------------------------------------
def test_faq_hit_returns_cached_answer(client: TestClient) -> None:
    """FAQ 命中：cached=true，route=faq，来源指向 FAQ 条目。"""
    body = client.post(CHAT_URL, json={"question": FAQ_HIT_QUESTION}).json()

    assert body["cached"] is True
    assert body["route"] == "faq"
    assert body["answer"] == "CS101《程序设计基础》为 4 学分。"
    assert body["sources"][0]["type"] == "faq"


def test_faq_miss_falls_through_to_rag(client: TestClient) -> None:
    """FAQ 未命中：继续检索 + 生成，绝不拒答。"""
    body = client.post(CHAT_URL, json={"question": FAQ_MISS_QUESTION}).json()

    assert body["cached"] is False
    assert body["route"] == "faq"
    assert body["answer"] != REFUSAL_REPLY
    assert body["sources"][0]["score"] == pytest.approx(0.82)


def test_use_faq_false_bypasses_the_cache(client: TestClient) -> None:
    """``use_faq=false`` 时命中也不走 FAQ，用于对比"FAQ 是不是答错了"。"""
    body = client.post(
        CHAT_URL, json={"question": FAQ_HIT_QUESTION, "use_faq": False}
    ).json()

    assert body["cached"] is False
    assert body["answer"] != "CS101《程序设计基础》为 4 学分。"


# ---------------------------------------------------------------------------
# 各条路由
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "route"),
    [
        (COURSE_QUESTION, "course_query"),
        ("怎么选课", "process_query"),
    ],
)
def test_retrieval_routes(client: TestClient, question: str, route: str) -> None:
    """course_query 与 process_query 都走检索 + 生成。"""
    body = client.post(CHAT_URL, json={"question": question}).json()

    assert body["route"] == route
    assert body["cached"] is False
    assert body["sources"]


def test_chitchat_returns_guidance(client: TestClient) -> None:
    """闲聊直接返回引导话术，且没有来源。"""
    body = client.post(CHAT_URL, json={"question": "你好"}).json()

    assert body["route"] == "chitchat"
    assert body["answer"] == CHITCHAT_REPLY
    assert body["cached"] is False
    assert body["sources"] == []


# ---------------------------------------------------------------------------
# 拒答
# ---------------------------------------------------------------------------
def test_empty_hits_are_refused_with_http_200(faq_cache: FAQCache, make_pipeline) -> None:
    """空检索结果 → 拒答，状态码仍是 200（业务分支，不是错误）。"""
    pipeline = make_pipeline(
        classifier=_default_classifier(),
        faq_cache=faq_cache,
        vector_store=FakeVectorStore(hits=[], total=17),
    )

    with app_client(pipeline, faq_cache) as test_client:
        response = test_client.post(CHAT_URL, json={"question": COURSE_QUESTION})

    assert response.status_code == 200
    assert response.json()["answer"] == REFUSAL_REPLY
    assert response.json()["sources"] == []


def test_weak_hits_are_refused_without_generation(faq_cache: FAQCache, make_pipeline) -> None:
    """分数不达标 → 拒答，并且一次 LLM 都没调用。"""
    llm = FakeLLMService()
    pipeline = make_pipeline(
        classifier=_default_classifier(),
        faq_cache=faq_cache,
        vector_store=FakeVectorStore(hits=WEAK_HITS),
        relevance=RelevanceChecker(threshold=0.6),
        llm=llm,
    )

    with app_client(pipeline, faq_cache) as test_client:
        body = test_client.post(CHAT_URL, json={"question": COURSE_QUESTION}).json()

    assert body["answer"] == REFUSAL_REPLY
    assert llm.call_count == 0


# ---------------------------------------------------------------------------
# 错误 → 状态码
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (GenerationError("上游炸了"), 502, "GENERATION_ERROR"),
        (LLMTimeoutError("超时"), 504, "LLM_TIMEOUT"),
        (ConfigurationError("没配 Key"), 500, "CONFIGURATION_ERROR"),
    ],
)
def test_llm_errors_map_to_status_codes(
    faq_cache: FAQCache,
    make_pipeline,
    error: Exception,
    expected_status: int,
    expected_code: str,
) -> None:
    """LLM 侧的问题按"谁的问题"映射：上游错误 502、上游超时 504、本地没配好 500。"""
    pipeline = make_pipeline(
        classifier=_default_classifier(),
        faq_cache=faq_cache,
        vector_store=FakeVectorStore(hits=GOOD_HITS),
        llm=FakeLLMService(error=error),
    )

    with app_client(pipeline, faq_cache) as test_client:
        response = test_client.post(CHAT_URL, json={"question": COURSE_QUESTION})

    assert response.status_code == expected_status
    assert response.json()["code"] == expected_code
    assert response.json()["message"]


def test_empty_index_maps_to_503(faq_cache: FAQCache, make_pipeline) -> None:
    """索引为空是"自身暂时不可用"：503，且错误码能直接告诉运维该去建库。"""
    pipeline = make_pipeline(
        classifier=_default_classifier(),
        faq_cache=faq_cache,
        vector_store=FakeVectorStore(hits=[], total=0),
    )

    with app_client(pipeline, faq_cache) as test_client:
        response = test_client.post(CHAT_URL, json={"question": COURSE_QUESTION})

    assert response.status_code == 503
    assert response.json()["code"] == "INDEX_NOT_READY"


def test_unexpected_error_returns_500_without_internal_details(
    faq_cache: FAQCache, make_pipeline
) -> None:
    """代码 bug 走 FastAPI 默认处理：500，且不把内部细节泄给调用方。"""
    pipeline = make_pipeline(
        classifier=_default_classifier(),
        faq_cache=faq_cache,
        vector_store=FakeVectorStore(hits=GOOD_HITS),
        llm=FakeLLMService(error=TypeError("内部实现细节不该外泄")),
    )

    with app_client(pipeline, faq_cache, raise_server_exceptions=False) as test_client:
        response = test_client.post(CHAT_URL, json={"question": COURSE_QUESTION})

    assert response.status_code == 500
    assert "内部实现细节不该外泄" not in response.text


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "问题", "top_k": 0},
        {"question": "问题", "top_k": 999},
        {"question": "问题", "use_faq": "yes please"},
        {"question": "问题", "unknown_field": 1},
        {"question": "x" * 1001},
    ],
)
def test_invalid_request_is_422_with_unified_body(client: TestClient, payload: dict) -> None:
    """参数不合法返回 422，且响应体与业务错误同构（客户端只需一套解析逻辑）。"""
    response = client.post(CHAT_URL, json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert body["details"]["errors"]


def test_unknown_route_is_still_404(client: TestClient) -> None:
    """新加的路由不影响 404 语义。"""
    assert client.post(f"{API_PREFIX}/nope", json={"question": "x"}).status_code == 404


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------
def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 响应体解析成 ``[(事件名, 数据)]``。

    只按协议拆帧（空行分隔、``event:`` / ``data:`` 前缀），**不依赖实现细节**——
    这样帧格式一旦写错，测试会真的失败。
    """
    events: list[tuple[str, dict]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: ") :])
        events.append((event_name, json.loads("\n".join(data_lines))))
    return events


def test_stream_emits_meta_deltas_then_done(client: TestClient) -> None:
    """SSE 事件序列：meta → delta… → done，且 delta 拼起来就是完整回答。"""
    response = client.post(STREAM_URL, json={"question": COURSE_QUESTION})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert "delta" in names

    meta = events[0][1]
    assert meta["route"] == "course_query"
    assert meta["cached"] is False
    assert meta["sources"][0]["course_id"] == "CS201"

    answer = "".join(data["text"] for name, data in events if name == "delta")
    assert answer
    assert events[-1][1]["latency_ms"] >= 0


def test_stream_of_faq_hit_uses_a_single_delta(client: TestClient) -> None:
    """FAQ 命中不调 LLM：整段答案放在一个 delta 里，客户端不必写第二套逻辑。"""
    response = client.post(STREAM_URL, json={"question": FAQ_HIT_QUESTION})

    events = _parse_sse(response.text)
    deltas = [data["text"] for name, data in events if name == "delta"]

    assert len(deltas) == 1
    assert deltas[0] == "CS101《程序设计基础》为 4 学分。"
    assert events[0][1]["cached"] is True


def test_stream_of_chitchat_only_emits_guidance(client: TestClient) -> None:
    """闲聊同样走 SSE，一个 delta 里是引导话术。"""
    events = _parse_sse(client.post(STREAM_URL, json={"question": "你好"}).text)

    deltas = [data["text"] for name, data in events if name == "delta"]
    assert deltas == [CHITCHAT_REPLY]


def test_stream_reports_mid_stream_failure_as_an_error_event(
    faq_cache: FAQCache, make_pipeline
) -> None:
    """流已开始就没法改状态码了，只能发一条 error 事件告诉客户端"这段不完整"。"""
    pipeline = make_pipeline(
        classifier=_default_classifier(),
        faq_cache=faq_cache,
        vector_store=FakeVectorStore(hits=GOOD_HITS),
        llm=FakeLLMService(error=GenerationError("生成到一半炸了")),
    )

    with app_client(pipeline, faq_cache) as test_client:
        response = test_client.post(STREAM_URL, json={"question": COURSE_QUESTION})

    assert response.status_code == 200  # 响应头已经发出去了

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert "error" in names
    assert "done" not in names

    error_payload = next(data for name, data in events if name == "error")
    assert error_payload["code"] == "GENERATION_ERROR"
    assert error_payload["message"]


def test_stream_preflight_errors_keep_http_status(faq_cache: FAQCache, make_pipeline) -> None:
    """准备阶段的失败发生在任何 SSE 帧之前，仍然返回正常的状态码与 JSON 错误体。"""
    pipeline = make_pipeline(
        classifier=_default_classifier(),
        faq_cache=faq_cache,
        vector_store=FakeVectorStore(hits=[], total=0),
    )

    with app_client(pipeline, faq_cache) as test_client:
        response = test_client.post(STREAM_URL, json={"question": COURSE_QUESTION})

    assert response.status_code == 503
    assert response.json()["code"] == "INDEX_NOT_READY"


def test_stream_reports_unexpected_errors_as_an_error_event(
    faq_cache: FAQCache, make_pipeline
) -> None:
    """代码 bug（非 AppError）同样必须转成 error 帧。

    否则连接被直接掐断：客户端只收到半截 delta，既没有 ``error`` 也没有 ``done``，
    无法区分"生成中断"与"回答到此结束"。
    """
    pipeline = make_pipeline(
        classifier=_default_classifier(),
        faq_cache=faq_cache,
        vector_store=FakeVectorStore(hits=GOOD_HITS),
        llm=FakeLLMService(error=TypeError("代码 bug")),
    )

    with app_client(pipeline, faq_cache) as test_client:
        response = test_client.post(STREAM_URL, json={"question": COURSE_QUESTION})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert "error" in names
    assert "done" not in names

    payload = next(data for name, data in events if name == "error")
    assert payload["code"] == "INTERNAL_ERROR"
    assert "代码 bug" not in payload["message"]  # 不把内部细节回给调用方


def test_stream_validates_the_request(client: TestClient) -> None:
    """流式接口同样做参数校验。"""
    assert client.post(STREAM_URL, json={"question": ""}).status_code == 422


# ---------------------------------------------------------------------------
# POST /api/faq
# ---------------------------------------------------------------------------
def test_create_faq_persists_and_takes_effect_immediately(client: TestClient, faq_cache: FAQCache) -> None:
    """新增 FAQ：201、拿到 id、落盘，并且**下一个请求立刻命中**。"""
    response = client.post(
        FAQ_URL,
        json={
            "patterns": ["CS101 多少学分", "程序设计基础学分是多少"],
            "answer": "CS101 为 4 学分。",
            "course_id": "CS101",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"].startswith("faq_")
    assert body["saved"] is True
    assert body["patterns"] == ["CS101 多少学分", "程序设计基础学分是多少"]

    # 落盘了：重新加载同一份文件能看到这条
    reloaded = FAQCache(path=faq_cache.path)
    reloaded.load()
    assert any(entry.id == body["id"] for entry in reloaded.entries)

    # 立即生效：缓存对象就是业务链路正在用的那一个
    assert faq_cache.lookup("CS101 多少学分") is not None


@pytest.mark.parametrize(
    "payload",
    [
        {"patterns": [], "answer": "有答案"},
        {"patterns": ["   "], "answer": "有答案"},
        {"patterns": ["问法"], "answer": "   "},
        {"patterns": ["问法"]},
        {"answer": "有答案"},
        {"patterns": "不是数组", "answer": "有答案"},
        {"patterns": ["问法"], "answer": "答案", "extra": 1},
    ],
)
def test_create_faq_rejects_invalid_payload(client: TestClient, payload: dict) -> None:
    """不合规的 FAQ 请求被 422 挡下，不会写进文件。"""
    response = client.post(FAQ_URL, json=payload)

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


def test_created_faq_is_answerable_through_chat(faq_cache: FAQCache, make_pipeline) -> None:
    """端到端：先 POST /api/faq，再问同一个问题，应当命中 FAQ 快路径。

    这条用例里的分类器把一切判成 faq 意图——真实链路上"新加的问法能不能被分类成
    faq"取决于第 4 阶段的规则与 LLM，这里只验证"加完之后快路径确实查得到"。
    """
    question = "MA101 的老师是谁"
    pipeline = make_pipeline(
        classifier=FakeIntentClassifier(default=IntentType.FAQ),
        faq_cache=faq_cache,
        vector_store=FakeVectorStore(hits=GOOD_HITS),
        llm=FakeLLMService(),
    )

    with app_client(pipeline, faq_cache) as test_client:
        created = test_client.post(
            FAQ_URL,
            json={
                "patterns": [question],
                "answer": "MA101《高等数学（上）》的授课教师是王芳。",
            },
        )
        body = test_client.post(CHAT_URL, json={"question": question}).json()

    assert created.status_code == 201
    assert body["cached"] is True
    assert body["answer"] == "MA101《高等数学（上）》的授课教师是王芳。"


def test_create_faq_requires_the_admin_token_when_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """配置了 FAQ_ADMIN_TOKEN 之后：缺令牌 401、令牌不对 401、对了才 201。"""
    monkeypatch.setattr(settings, "faq_admin_token", SecretStr("s3cret-token"))
    payload = {"patterns": ["CS999 几学分"], "answer": "CS999 为 2 学分。"}

    missing = client.post(FAQ_URL, json=payload)
    wrong = client.post(FAQ_URL, json=payload, headers={"X-Admin-Token": "nope"})
    accepted = client.post(FAQ_URL, json=payload, headers={"X-Admin-Token": "s3cret-token"})

    assert missing.status_code == 401
    assert missing.json()["code"] == "UNAUTHORIZED"
    assert wrong.status_code == 401
    assert accepted.status_code == 201


def test_create_faq_is_open_when_no_token_is_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未配置令牌时保持开放（演示取舍），但这条行为是被文档与启动告警明确标注的。"""
    monkeypatch.setattr(settings, "faq_admin_token", SecretStr(""))

    response = client.post(FAQ_URL, json={"patterns": ["CS998 几学分"], "answer": "2 学分。"})

    assert response.status_code == 201


def test_overlong_pattern_is_rejected(client: TestClient) -> None:
    """单条问法有长度上限：超长 pattern 会拖慢之后每一次 FAQ 匹配。"""
    response = client.post(
        FAQ_URL, json={"patterns": ["问" * 500], "answer": "答案"}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# 文档
# ---------------------------------------------------------------------------
def test_openapi_documents_all_routes_and_schemas(client: TestClient) -> None:
    """``/docs`` 依赖的 OpenAPI 里四个接口与关键 schema 都在。"""
    schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) >= {CHAT_URL, STREAM_URL, FAQ_URL, f"{API_PREFIX}/health"}
    assert {"ChatRequest", "ChatResponse", "Source"} <= set(schema["components"]["schemas"])
    assert client.get("/docs").status_code == 200


def test_chat_response_schema_exposes_the_intent_enum(client: TestClient) -> None:
    """``route`` 在文档里指向枚举 schema，Swagger 上能看到四个取值。"""
    schema = client.get("/openapi.json").json()

    route_schema = schema["components"]["schemas"]["ChatResponse"]["properties"]["route"]
    rendered = json.dumps(route_schema, ensure_ascii=False)
    assert "IntentType" in rendered
    assert "course_query" in json.dumps(
        schema["components"]["schemas"]["IntentType"], ensure_ascii=False
    )


# ---------------------------------------------------------------------------
# 未就绪
# ---------------------------------------------------------------------------
def test_endpoints_report_503_with_the_unified_body() -> None:
    """绕过 lifespan 直接请求时返回 503，且响应体与其它错误同构。

    早期这里抛的是 FastAPI 的 ``HTTPException``，返回 ``{"detail": ...}``——
    客户端按文档写一套 ``code/message/details`` 解析就会在这里崩掉。
    """
    bare_client = TestClient(app)  # 不进入上下文 → 不执行 lifespan
    app.state.pipeline = None
    app.state.faq_cache = None
    try:
        chat_response = bare_client.post(CHAT_URL, json={"question": "问题"})
        faq_response = bare_client.post(
            FAQ_URL, json={"patterns": ["问法"], "answer": "答案"}
        )
    finally:
        del app.state.pipeline
        del app.state.faq_cache

    for response in (chat_response, faq_response):
        assert response.status_code == 503
        body = response.json()
        assert body["code"] == "SERVICE_NOT_READY"
        assert body["message"]
        assert "detail" not in body
