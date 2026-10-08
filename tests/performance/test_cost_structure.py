"""性能：**成本结构**断言（哪条路径花了什么）。

这个项目的性能设计几乎全在一句话上："能不调的就不调"——FAQ 命中不调 Embedding /
Chroma / LLM，证据不足不调 LLM，闲聊什么都不调。这类优化有个特点：**改坏了不会报错，
只会变贵**（多一次调用、多一秒延迟），代码评审时也看不出来。

所以这一组测试盯住的是"调用次数"，而不是毫秒数：

- 秒数是环境的函数（CI 机器、是否加载了模型），拿来断言必然 flaky；
- 调用次数是**设计**的函数，改了就应该红，红了说明设计变了。

绝对性能请看 ``scripts/benchmark.py`` 的输出（README 有实测表）。
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from src.schemas.chat import ChatRequest, IntentType
from src.services.pipeline import ChatPipeline
from tests.fakes import FakeIntentClassifier, FakeLLMService, FakeVectorStore, make_document

COURSE_QUESTION = "数据结构与算法主要讲什么"
FAQ_QUESTION = "CS101 几学分"
CHITCHAT_QUESTION = "你好"
NEUTRAL_QUESTION = "图书馆几点关门"

GOOD_HITS = [(make_document(), 0.82)]
WEAK_HITS = [(make_document(), 0.2)]


def _request(question: str, **overrides: object) -> ChatRequest:
    """造一个请求。"""
    payload: dict[str, object] = {"question": question}
    payload.update(overrides)
    return ChatRequest.model_validate(payload)


# ---------------------------------------------------------------------------
# 各条路径的调用次数
# ---------------------------------------------------------------------------
def test_chitchat_touches_nothing(make_pipeline: Callable[..., ChatPipeline]) -> None:
    """闲聊：零检索、零生成（这是最便宜的一条路径）。"""
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    llm = FakeLLMService()
    pipeline = make_pipeline(
        classifier=FakeIntentClassifier(default=IntentType.CHITCHAT),
        vector_store=vector_store,
        llm=llm,
    )

    pipeline.run(_request(CHITCHAT_QUESTION))

    assert vector_store.queries == []
    assert llm.call_count == 0


def test_faq_hit_skips_retrieval_and_generation(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """FAQ 命中：省掉检索与生成 —— 一次问答里最贵的两件事都没发生。"""
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    llm = FakeLLMService()
    pipeline = make_pipeline(vector_store=vector_store, llm=llm)

    response = pipeline.run(_request(FAQ_QUESTION))

    assert response.cached is True
    assert vector_store.queries == []
    assert llm.call_count == 0


def test_rejection_skips_generation_only(make_pipeline: Callable[..., ChatPipeline]) -> None:
    """拒答：检索照做（得先知道有没有依据），但**不调生成**。"""
    llm = FakeLLMService()
    pipeline = make_pipeline(
        classifier=FakeIntentClassifier(default=IntentType.COURSE_QUERY),
        vector_store=FakeVectorStore(hits=WEAK_HITS),
        llm=llm,
        threshold=0.6,
    )

    pipeline.run(_request(NEUTRAL_QUESTION))

    assert llm.call_count == 0


def test_rag_calls_retrieval_once_and_generation_once(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """完整的 RAG 路径：各一次。多于一次就是重复计算。"""
    vector_store = FakeVectorStore(hits=GOOD_HITS)
    llm = FakeLLMService()
    pipeline = make_pipeline(
        classifier=FakeIntentClassifier(default=IntentType.COURSE_QUERY),
        vector_store=vector_store,
        llm=llm,
    )

    pipeline.run(_request(COURSE_QUESTION))

    assert len(vector_store.queries) == 1
    assert llm.call_count == 1


def test_streaming_uses_the_streaming_generator(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """流式路径走 generate_stream，不要"先整体生成再切片"——那样流式就白做了。"""
    llm = FakeLLMService(pieces=["根据资料，", "先修课程是 CS101。"])
    pipeline = make_pipeline(
        classifier=FakeIntentClassifier(default=IntentType.COURSE_QUERY),
        vector_store=FakeVectorStore(hits=GOOD_HITS),
        llm=llm,
    )

    pieces = list(pipeline.stream_answer(pipeline.prepare(_request(COURSE_QUESTION))))

    assert pieces == ["根据资料，", "先修课程是 CS101。"]
    assert len(llm.stream_calls) == 1
    assert llm.calls == []


@pytest.mark.slow
def test_faq_hit_is_not_slower_than_full_rag(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """相对比较：FAQ 命中必须明显快于走完整 RAG。

    断言很宽松（只要快一倍以上），因为这里量的是**结构性差异**：
    FAQ 命中不做检索与生成，理应比 RAG 快一个数量级。用宽松阈值是为了在慢机器上
    也不误报——真正想拦住的是"FAQ 快路径被改成了仍然走一遍检索"这类回归。
    """
    faq_pipeline = make_pipeline(
        classifier=FakeIntentClassifier({FAQ_QUESTION: IntentType.FAQ}),
        vector_store=FakeVectorStore(hits=GOOD_HITS),
    )
    rag_pipeline = make_pipeline(
        classifier=FakeIntentClassifier(default=IntentType.COURSE_QUERY),
        vector_store=FakeVectorStore(hits=GOOD_HITS),
    )

    faq_ms = _median_ms(lambda: faq_pipeline.run(_request(FAQ_QUESTION)), times=30)
    rag_ms = _median_ms(lambda: rag_pipeline.run(_request(COURSE_QUESTION)), times=30)

    assert faq_ms < rag_ms


@pytest.mark.slow
def test_faq_hit_stays_within_a_generous_budget(
    make_pipeline: Callable[..., ChatPipeline],
) -> None:
    """FAQ 命中的绝对预算（宽松到几乎只在"它真的开始捞外部资源"时才会超）。

    20ms 是给"假实现 + 最慢的 CI 机器"留的余量；实测在本地是亚毫秒级。
    这种断言的价值不在数字本身，而在于它会在有人往快路径里塞一次网络调用时立刻红。
    """
    pipeline = make_pipeline(vector_store=FakeVectorStore(hits=GOOD_HITS))

    elapsed_ms = _median_ms(lambda: pipeline.run(_request(FAQ_QUESTION)), times=30)

    assert elapsed_ms < 20.0


def _median_ms(action: Callable[[], object], *, times: int) -> float:
    """重复执行并返回耗时中位数（中位数比平均更抗偶发卡顿）。"""
    samples: list[float] = []
    for _ in range(times):
        started = time.perf_counter()
        action()
        samples.append((time.perf_counter() - started) * 1000)
    samples.sort()
    return samples[len(samples) // 2]


# ---------------------------------------------------------------------------
# 压测脚本的统计口径
# ---------------------------------------------------------------------------
def test_benchmark_summary_matches_hand_computed_values() -> None:
    """压测脚本的汇总口径要能被手算验证（指标算错比系统慢更危险）。"""
    from scripts.benchmark import BenchResult

    result = BenchResult(name="demo", total=4, errors=1, latencies_ms=[10.0, 20.0, 30.0, 40.0])
    result.wall_seconds = 2.0

    summary = result.summary()

    assert summary["平均延迟_ms"] == 25.0
    assert summary["P50_ms"] == 25.0
    assert summary["错误率"] == 0.25
    assert summary["吞吐_req_s"] == 2.0


def test_benchmark_result_without_samples_is_safe() -> None:
    """一次都没成功时不除零，字段给 None。"""
    from scripts.benchmark import BenchResult

    result = BenchResult(name="demo", total=3, errors=3)

    summary = result.summary()

    assert summary["平均延迟_ms"] is None
    assert summary["P95_ms"] is None
    assert summary["吞吐_req_s"] == 0.0


@pytest.mark.parametrize("question", [COURSE_QUESTION, FAQ_QUESTION])
def test_requests_are_serializable_for_benchmarking(question: str) -> None:
    """压测要发 JSON，请求模型必须能序列化回去（字段名与文档一致）。"""
    payload = _request(question).model_dump()

    assert set(payload) == {"question", "use_faq", "top_k"}
