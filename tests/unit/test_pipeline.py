"""``src.services.pipeline`` 的单元测试。

关注**分支与代价**，而不是 HTTP 细节（那些在
``tests/integration/test_chat_api.py``）：每条分支该不该提前返回、有没有白花掉一次
LLM 调用、失败时是抛还是吞、耗时是不是真的量了。

所有依赖都是假实现（``tests/fakes.py``），没有网络、没有模型、没有 Chroma。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pytest

from src.config import settings
from src.retrieval.relevance import RelevanceChecker
from src.schemas.chat import ChatRequest, IntentType
from src.services.pipeline import (
    CHITCHAT_REPLY,
    REFUSAL_REPLY,
    ChatPipeline,
    PreparedAnswer,
)
from src.utils.exceptions import GenerationError, IndexNotReadyError, RetrievalError
from tests.fakes import (
    FakeIntentClassifier,
    FakeLLMService,
    FakeVectorStore,
    make_document,
)

#: 命中 FAQ 的问题（conftest 里的 FAQ_PAYLOAD 有这条问法）。
FAQ_QUESTION = "CS101 几学分"

#: 会被分类成 course_query 的问题。
COURSE_QUESTION = "数据结构与算法主要讲什么"

#: 文档 + 高分，用来构造"检索命中且相关"的场景。
GOOD_HITS = [(make_document(), 0.82)]

#: 文档 + 低分，用来构造"检索到东西但不够相关"的场景。
WEAK_HITS = [(make_document(), 0.21)]


def _request(question: str = COURSE_QUESTION, **overrides: object) -> ChatRequest:
    """造一个请求。"""
    payload: dict[str, object] = {"question": question}
    payload.update(overrides)
    return ChatRequest.model_validate(payload)


# ---------------------------------------------------------------------------
# 闲聊：不碰任何外部组件
# ---------------------------------------------------------------------------
def test_chitchat_returns_guidance_without_touching_anything(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """闲聊直接给引导话术：FAQ / Embedding / Chroma / LLM 一个都不许碰。"""
    classifier = FakeIntentClassifier(default=IntentType.CHITCHAT)
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    llm = FakeLLMService()
    pipeline = make_pipeline(classifier=classifier, vector_store=vector_store, llm=llm)

    response = pipeline.run(_request("你好呀"))

    assert response.route is IntentType.CHITCHAT
    assert response.answer == CHITCHAT_REPLY
    assert response.cached is False
    assert response.sources == []
    assert vector_store.queries == []
    assert llm.call_count == 0


# ---------------------------------------------------------------------------
# FAQ
# ---------------------------------------------------------------------------
def test_faq_hit_short_circuits_before_retrieval(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """FAQ 命中：直接返回答案，不检索、不生成。"""
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    llm = FakeLLMService()
    pipeline = make_pipeline(vector_store=vector_store, llm=llm)

    response = pipeline.run(_request(FAQ_QUESTION))

    assert response.cached is True
    assert response.route is IntentType.FAQ
    assert "4 学分" in response.answer
    assert vector_store.queries == []
    assert llm.call_count == 0


def test_faq_hit_reports_the_faq_entry_as_source(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """FAQ 命中的来源指向 FAQ 条目本身，而不是编一个课程名。"""
    pipeline = make_pipeline()

    source = pipeline.run(_request(FAQ_QUESTION)).sources[0]

    assert source.course_id == "CS101"
    assert source.type == "faq"
    assert source.source == "faq.json"
    assert source.section == "faq_002"
    assert source.score is None


def test_faq_miss_falls_through_to_rag(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """FAQ 未命中不是拒答：继续检索 + 生成，cached 保持 false。"""
    classifier = FakeIntentClassifier(default=IntentType.FAQ)
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    llm = FakeLLMService(answer="（走 RAG 生成的答案）")
    pipeline = make_pipeline(classifier=classifier, vector_store=vector_store, llm=llm)

    response = pipeline.run(_request("CS999 的教材是什么"))

    assert response.cached is False
    assert response.answer == "（走 RAG 生成的答案）"
    assert len(vector_store.queries) == 1
    assert llm.call_count == 1


def test_use_faq_false_skips_the_faq_fast_path(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """``use_faq=false`` 时即使意图是 faq 也走 RAG —— 用于对比"FAQ 是否答错了"。"""
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    llm = FakeLLMService()
    pipeline = make_pipeline(vector_store=vector_store, llm=llm)

    response = pipeline.run(_request(FAQ_QUESTION, use_faq=False))

    assert response.cached is False
    assert len(vector_store.queries) == 1
    assert llm.call_count == 1


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------
def test_relevant_hits_go_to_generation(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """相关就交给生成层，回答与来源都来自这一条链路。"""
    llm = FakeLLMService(answer="根据资料，先修课程是 CS101。")
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=GOOD_HITS), llm=llm)

    response = pipeline.run(_request())

    assert response.answer == "根据资料，先修课程是 CS101。"
    assert response.cached is False
    assert llm.calls[0][0] == COURSE_QUESTION
    assert len(llm.calls[0][1]) == 1


def test_generation_sources_carry_relevance_scores(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """RAG 路径的来源带上相关性分数，并按课程 / 文件 / 章节去重。"""
    hits = [
        (make_document(content="切片一"), 0.82),
        (make_document(content="切片二"), 0.71),
        (make_document(content="切片三", course_id="CS101", name="程序设计基础"), 0.66),
    ]
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=hits))

    sources = pipeline.run(_request()).sources

    assert [source.course_id for source in sources] == ["CS201", "CS101"]
    assert [source.score for source in sources] == [0.82, 0.66]
    assert sources[0].type == "syllabus"


def test_top_k_from_request_reaches_the_vector_store(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """请求里的 ``top_k`` 必须真的传下去，否则它只是一个装饰品。"""
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    pipeline = make_pipeline(vector_store=vector_store)

    pipeline.run(_request(top_k=13))

    assert vector_store.queries == [(COURSE_QUESTION, 13)]


def test_top_k_defaults_to_the_server_setting(
    make_pipeline: Callable[..., ChatPipeline], monkeypatch: pytest.MonkeyPatch
) -> None:
    """请求不带 top_k 时用服务端的 TOP_K —— 否则运维改了配置却不生效。"""
    monkeypatch.setattr(settings, "top_k", 9)
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    pipeline = make_pipeline(vector_store=vector_store)

    response = pipeline.run(ChatRequest(question=COURSE_QUESTION))

    assert vector_store.queries == [(COURSE_QUESTION, 9)]
    assert response.route is IntentType.COURSE_QUERY


def test_explicit_top_k_still_wins_over_the_setting(
    make_pipeline: Callable[..., ChatPipeline], monkeypatch: pytest.MonkeyPatch
) -> None:
    """请求显式给了 top_k 就以请求为准。"""
    monkeypatch.setattr(settings, "top_k", 9)
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    pipeline = make_pipeline(vector_store=vector_store)

    pipeline.run(_request(top_k=2))

    assert vector_store.queries == [(COURSE_QUESTION, 2)]


def test_question_is_stripped_before_classification(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """分类与检索用的是去空白后的问题，避免空格影响 FAQ 精确匹配。"""
    classifier = FakeIntentClassifier()
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    pipeline = make_pipeline(classifier=classifier, vector_store=vector_store)

    pipeline.run(_request("  CS101 几学分  "))

    assert classifier.questions == ["CS101 几学分"]
    assert vector_store.queries[0][0] == "CS101 几学分"


# ---------------------------------------------------------------------------
# 拒答
# ---------------------------------------------------------------------------
def test_weak_hits_are_refused_without_calling_generation(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """分数不达标：返回拒答话术，**不调用生成 LLM**（这是省钱的关键一步）。"""
    llm = FakeLLMService()
    pipeline = make_pipeline(
        vector_store=FakeVectorStore(hits=WEAK_HITS),
        llm=llm,
        threshold=0.5,
    )

    response = pipeline.run(_request())

    assert response.answer == REFUSAL_REPLY
    assert response.sources == []
    assert llm.call_count == 0


def test_empty_hits_are_refused(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """检索没命中也是拒答，不是 500。"""
    llm = FakeLLMService()
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=[], total=17), llm=llm)

    response = pipeline.run(_request())

    assert response.answer == REFUSAL_REPLY
    assert llm.call_count == 0


def test_but_an_empty_index_is_an_operational_error(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """索引里一条文档都没有 = 忘记建库，报 503 比装成一次拒答有用得多。"""
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=[], total=0))

    with pytest.raises(IndexNotReadyError):
        pipeline.run(_request())


def test_threshold_boundary_is_inclusive(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """分数恰好等于阈值算相关（与 RelevanceChecker 的口径一致）。"""
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=[(make_document(), 0.5)]), threshold=0.5)

    assert pipeline.run(_request()).answer != REFUSAL_REPLY


# ---------------------------------------------------------------------------
# 延迟与日志
# ---------------------------------------------------------------------------
def test_latency_is_measured_and_non_negative(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """耗时用 perf_counter 测量，且是个合理的正数。"""
    pipeline = make_pipeline()

    response = pipeline.run(_request(FAQ_QUESTION))

    assert response.latency_ms >= 0.0
    assert response.latency_ms < 10_000  # 假实现应当毫秒级返回


def test_completion_is_logged_with_route_and_cache_flag(
    make_pipeline: Callable[..., ChatPipeline],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """每条请求都要留下可排查的日志：路由、是否命中缓存、耗时。"""
    pipeline = make_pipeline()

    with caplog.at_level(logging.INFO, logger="src.services.pipeline"):
        pipeline.run(_request(FAQ_QUESTION))

    messages = [record.getMessage() for record in caplog.records]
    assert any("route=faq" in message and "cached=true" in message for message in messages)
    assert any("latency_ms=" in message for message in messages)


def test_completion_log_carries_the_operational_fields(
    make_pipeline: Callable[..., ChatPipeline],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """完成日志必须能回答"这次到底做了什么"：检索了几条、证据够不够、有没有调模型。"""
    llm = FakeLLMService()
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=GOOD_HITS), llm=llm, threshold=0.5)

    with caplog.at_level(logging.INFO, logger="src.services.pipeline"):
        pipeline.run(_request(top_k=7))

    completion = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("问答完成")
    )
    assert "route=course_query" in completion
    assert "cached=false" in completion
    assert "top_k=7" in completion
    assert "retrieval_count=1" in completion
    assert "relevance=relevant" in completion
    assert "best_score=0.8200" in completion
    assert "llm_called=true" in completion
    assert "llm_latency_ms=" in completion


def test_completion_log_marks_rejection_and_skipped_llm(
    make_pipeline: Callable[..., ChatPipeline],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """拒答时日志要说明"证据不足、没有调用 LLM"，否则无法解释为什么没花钱。"""
    llm = FakeLLMService()
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=WEAK_HITS), llm=llm, threshold=0.6)

    with caplog.at_level(logging.INFO, logger="src.services.pipeline"):
        pipeline.run(_request())

    completion = next(
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("问答完成")
    )
    assert "relevance=insufficient" in completion
    assert "llm_called=false" in completion
    assert llm.call_count == 0


def test_question_is_masked_in_logs_by_default(
    make_pipeline: Callable[..., ChatPipeline],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认不把用户问题写进日志（只记长度）——日志会被长期留存，那是用户内容。"""
    monkeypatch.setattr(settings, "log_request_content", False)
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=GOOD_HITS))

    with caplog.at_level(logging.INFO, logger="src.services.pipeline"):
        pipeline.run(_request(COURSE_QUESTION))

    joined = " ".join(record.getMessage() for record in caplog.records)
    assert COURSE_QUESTION not in joined
    assert "question=<已脱敏 len=" in joined


def test_question_is_logged_when_explicitly_enabled(
    make_pipeline: Callable[..., ChatPipeline],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """排查具体问题时可以打开原文记录。"""
    monkeypatch.setattr(settings, "log_request_content", True)
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=GOOD_HITS))

    with caplog.at_level(logging.INFO, logger="src.services.pipeline"):
        pipeline.run(_request(COURSE_QUESTION))

    joined = " ".join(record.getMessage() for record in caplog.records)
    assert COURSE_QUESTION in joined


# ---------------------------------------------------------------------------
# 错误传播
# ---------------------------------------------------------------------------
def test_generation_error_propagates_untouched(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """生成失败要原样抛出（由 API 层映射成 502），不能吞成一个"系统错误"字符串。"""
    error = GenerationError("上游 500")
    pipeline = make_pipeline(
        vector_store=FakeVectorStore(hits=GOOD_HITS), llm=FakeLLMService(error=error)
    )

    with pytest.raises(GenerationError) as excinfo:
        pipeline.run(_request())

    assert excinfo.value is error


def test_retrieval_error_is_logged_with_stage(
    make_pipeline: Callable[..., ChatPipeline],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """检索失败要记下"是哪一段炸的"，否则线上只知道 503、不知道该看哪里。"""
    pipeline = make_pipeline(
        vector_store=FakeVectorStore(error=RetrievalError("Chroma 打不开"))
    )

    with caplog.at_level(logging.ERROR, logger="src.services.pipeline"), pytest.raises(RetrievalError):
        pipeline.run(_request())

    assert any("stage=retrieval" in record.getMessage() for record in caplog.records)


def test_unexpected_error_is_logged_as_unexpected(
    make_pipeline: Callable[..., ChatPipeline],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """代码 bug 与业务失败要能分辨：前者带堆栈且标记为"未预期"。"""
    pipeline = make_pipeline(vector_store=FakeVectorStore(error=TypeError("代码 bug")))

    with caplog.at_level(logging.ERROR, logger="src.services.pipeline"), pytest.raises(TypeError):
        pipeline.run(_request())

    assert any("未预期" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# prepare / stream_answer 的配合
# ---------------------------------------------------------------------------
def test_prepare_returns_canned_answer_for_faq_hit(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """准备阶段就能定论的请求，``needs_generation`` 必须为 False。"""
    pipeline = make_pipeline()

    prepared = pipeline.prepare(_request(FAQ_QUESTION))

    assert prepared.needs_generation is False
    assert prepared.cached is True
    assert "4 学分" in (prepared.answer or "")


def test_prepare_returns_documents_for_generation(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """需要生成的请求，准备阶段交出问题与资料，且不调用 LLM。"""
    llm = FakeLLMService()
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=GOOD_HITS), llm=llm)

    prepared = pipeline.prepare(_request())

    assert prepared.needs_generation is True
    assert prepared.question == COURSE_QUESTION
    assert len(prepared.documents) == 1
    assert llm.call_count == 0


def test_stream_answer_yields_one_piece_for_canned_answers(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """闲聊 / FAQ / 拒答也走同一条"片段流"：只产出一个片段。"""
    pipeline = make_pipeline()

    pieces = list(pipeline.stream_answer(pipeline.prepare(_request(FAQ_QUESTION))))

    assert len(pieces) == 1
    assert "4 学分" in pieces[0]


def test_stream_answer_uses_the_generation_stream(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """需要生成时逐片段产出，顺序不变。"""
    llm = FakeLLMService(pieces=["根据资料，", "先修课程是 ", "CS101。"])
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=GOOD_HITS), llm=llm)

    pieces = list(pipeline.stream_answer(pipeline.prepare(_request())))

    assert pieces == ["根据资料，", "先修课程是 ", "CS101。"]
    assert llm.stream_calls[0][0] == COURSE_QUESTION


def test_prepared_answer_flags_are_consistent(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """``needs_generation`` 与 ``answer`` 必须互为反义，别出现"有答案却还要生成"。"""
    pipeline = make_pipeline()

    prepared = pipeline.prepare(_request(FAQ_QUESTION))

    assert isinstance(prepared, PreparedAnswer)
    assert prepared.needs_generation == (prepared.answer is None)


def test_explicit_relevance_checker_is_used(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """注入的相关性判定器必须生效（阈值可被外部控制）。"""
    strict = RelevanceChecker(threshold=0.95)
    pipeline = make_pipeline(
        vector_store=FakeVectorStore(hits=[(make_document(), 0.9)]),
        relevance=strict,
        llm=FakeLLMService(),
    )

    assert pipeline.run(_request()).answer == REFUSAL_REPLY
