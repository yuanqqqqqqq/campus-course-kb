"""``src.retrieval.relevance`` 的单元测试。

重点验证三件事：距离到分数的换算方向正确、阈值比较用的是**最高分**而不是
随便哪一条、以及 LLM 复核只在向量判定失败时才被调用。
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from langchain_core.documents import Document

from src.retrieval.relevance import (
    MAX_RELEVANCE_SCORE,
    MIN_RELEVANCE_SCORE,
    RelevanceChecker,
    check_relevance,
    to_relevance_score,
)
from src.utils.exceptions import ConfigurationError, GenerationError, RetrievalError


def _document(course_id: str = "CS201") -> Document:
    """造一个最小文档。"""
    return Document(page_content=f"课程 {course_id} 的正文。", metadata={"course_id": course_id})


# ---------------------------------------------------------------------------
# 距离 → 分数
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0.0, 1.0),  # 完全相同的向量
        (0.2929, 0.7071),  # 余弦相似度 0.7071 的实测距离
        (1.0, 0.0),  # 正交
        (2.0, 0.0),  # 反方向 → 截断到 0，不产生负分数
        (-0.5, 1.0),  # 越界入参也截断到合法区间
    ],
)
def test_to_relevance_score_maps_distance_to_similarity(distance: float, expected: float) -> None:
    """余弦距离 d 应当换算成余弦相似度 1 - d，并截断到 [0, 1]。"""
    assert to_relevance_score(distance) == pytest.approx(expected, abs=1e-4)


def test_to_relevance_score_is_monotonic_decreasing() -> None:
    """距离越大，分数必须越低 —— 方向搞反的话整条阈值逻辑就废了。"""
    scores = [to_relevance_score(d) for d in (0.0, 0.2, 0.5, 0.8, 1.0)]

    assert scores == sorted(scores, reverse=True)


def test_to_relevance_score_clips_negative_similarity() -> None:
    """反方向向量（距离 > 1）不能产生负分数。

    负的余弦相似度在检索场景下没有意义，留着它只会让阈值比较变得难以解释。
    """
    assert to_relevance_score(2.0) == MIN_RELEVANCE_SCORE
    assert MIN_RELEVANCE_SCORE == 0.0
    assert MAX_RELEVANCE_SCORE == 1.0


# ---------------------------------------------------------------------------
# 空结果
# ---------------------------------------------------------------------------
def test_empty_results_are_not_relevant() -> None:
    """空结果判为不相关，且不抛异常 —— 拒答是上层 Pipeline 的职责。"""
    result = RelevanceChecker(threshold=0.6).check("数据结构的先修课是什么", [])

    assert result.is_relevant is False
    assert result.best_score is None
    assert result.candidate_count == 0
    assert result.checked_by_llm is False
    assert "未返回任何候选" in result.reason


def test_empty_results_do_not_trigger_llm_check() -> None:
    """没有候选文档时不该调用 LLM —— 没有材料可判。"""
    calls: list[str] = []

    def judge(query: str, documents: Sequence[Document]) -> bool:
        calls.append(query)
        return True

    checker = RelevanceChecker(threshold=0.6, enable_llm_check=True, judge=judge)
    result = checker.check("任意问题", [])

    assert result.is_relevant is False
    assert calls == []


def test_check_relevance_function_returns_false_for_empty() -> None:
    """便捷函数对空结果同样返回 False。"""
    assert check_relevance("问题", []) is False


# ---------------------------------------------------------------------------
# 阈值比较
# ---------------------------------------------------------------------------
def test_score_above_threshold_is_relevant() -> None:
    """最高分达到阈值即放行。"""
    results = [(_document(), 0.81), (_document("CS101"), 0.55)]

    result = RelevanceChecker(threshold=0.6).check("问题", results)

    assert result.is_relevant is True
    assert result.best_score == pytest.approx(0.81)


def test_score_exactly_at_threshold_is_relevant() -> None:
    """边界值取"大于等于"，恰好等于阈值时放行。"""
    result = RelevanceChecker(threshold=0.6).check("问题", [(_document(), 0.6)])

    assert result.is_relevant is True


def test_score_below_threshold_is_not_relevant() -> None:
    """全部分数低于阈值时判为不相关。"""
    results = [(_document(), 0.59), (_document("CS101"), 0.42)]

    result = RelevanceChecker(threshold=0.6).check("问题", results)

    assert result.is_relevant is False
    assert result.best_score == pytest.approx(0.59)
    assert result.checked_by_llm is False


def test_best_score_is_used_not_the_first_one() -> None:
    """判定用最高分，不是列表里第一条的分数。

    Chroma 一般按距离升序返回，也就是分数降序，但检索层不应当依赖这个顺序 ——
    排序逻辑一旦变化，这类假设会安静地失效。
    """
    results = [(_document(), 0.10), (_document("CS101"), 0.95)]

    assert RelevanceChecker(threshold=0.6).check("问题", results).is_relevant is True


def test_custom_threshold_is_honoured() -> None:
    """传入的阈值应当覆盖默认值。"""
    results = [(_document(), 0.7)]

    assert RelevanceChecker(threshold=0.75).check("问题", results).is_relevant is False
    assert RelevanceChecker(threshold=0.65).check("问题", results).is_relevant is True


def test_threshold_comes_from_settings_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不传阈值时取配置里的 RELEVANCE_THRESHOLD，而不是写死的常量。"""
    from src.retrieval import relevance as relevance_module

    monkeypatch.setattr(relevance_module.settings, "relevance_threshold", 0.95, raising=False)

    checker = RelevanceChecker()

    assert checker.threshold == pytest.approx(0.95)
    assert checker.check("问题", [(_document(), 0.9)]).is_relevant is False


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.5])
def test_invalid_threshold_rejected(bad_threshold: float) -> None:
    """阈值必须在 [0, 1] 内，越界直接报错而不是默默失效。"""
    with pytest.raises(ConfigurationError, match="RELEVANCE_THRESHOLD"):
        RelevanceChecker(threshold=bad_threshold)


# ---------------------------------------------------------------------------
# 分数区间校验
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_score", [1.5, 2.0, -0.3])
def test_out_of_range_score_rejected(bad_score: float) -> None:
    """分数超出 [0, 1] 时报错。

    这条校验专门拦"把原始距离当分数传进来"的错误：余弦距离的取值范围是
    [0, 2]，一旦误用，阈值比较不会报错，只会让检索结果安静地全错。
    """
    with pytest.raises(RetrievalError, match="不在"):
        RelevanceChecker(threshold=0.6).check("问题", [(_document(), bad_score)])


def test_raw_distance_error_message_points_to_the_right_method() -> None:
    """报错信息要直接告诉调用方该改用哪个方法。"""
    with pytest.raises(RetrievalError, match="similarity_search_with_relevance"):
        RelevanceChecker(threshold=0.6).check("问题", [(_document(), 1.8)])


# ---------------------------------------------------------------------------
# LLM 复核
# ---------------------------------------------------------------------------
class _RecordingJudge:
    """记录调用次数的假 LLM 判定器。"""

    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, documents: Sequence[Document]) -> bool:
        self.calls.append((query, len(documents)))
        return self.verdict


def test_llm_check_is_skipped_when_vector_score_passes() -> None:
    """向量分数达标时**不能**调用 LLM。

    这是本阶段最关心的成本问题：分类、相关性、生成三个 LLM 每次全跑一遍，
    延迟和费用都会失控。向量能判定的就别问 LLM。
    """
    judge = _RecordingJudge(verdict=True)
    checker = RelevanceChecker(threshold=0.6, enable_llm_check=True, judge=judge)

    result = checker.check("问题", [(_document(), 0.9)])

    assert result.is_relevant is True
    assert result.checked_by_llm is False
    assert judge.calls == []


def test_llm_check_rescues_borderline_results() -> None:
    """向量分数不达标且开启了复核时，交给 LLM 判定，可以翻案。"""
    judge = _RecordingJudge(verdict=True)
    checker = RelevanceChecker(threshold=0.6, enable_llm_check=True, judge=judge)

    result = checker.check("数据结构的先修课是什么", [(_document(), 0.45)])

    assert result.is_relevant is True
    assert result.checked_by_llm is True
    assert judge.calls == [("数据结构的先修课是什么", 1)]


def test_llm_check_can_also_reject() -> None:
    """LLM 复核也可以维持"不相关"的结论。"""
    judge = _RecordingJudge(verdict=False)
    checker = RelevanceChecker(threshold=0.6, enable_llm_check=True, judge=judge)

    result = checker.check("问题", [(_document(), 0.45)])

    assert result.is_relevant is False
    assert result.checked_by_llm is True
    assert len(judge.calls) == 1


def test_llm_check_passes_all_candidates_to_the_judge() -> None:
    """复核应当看到全部候选，而不是只看最高分那一条。"""
    judge = _RecordingJudge(verdict=True)
    checker = RelevanceChecker(threshold=0.6, enable_llm_check=True, judge=judge)
    results = [(_document(), 0.5), (_document("CS101"), 0.4), (_document("CS301"), 0.3)]

    checker.check("问题", results)

    assert judge.calls == [("问题", 3)]


def test_enabling_llm_check_without_judge_raises() -> None:
    """开启复核却没注入 judge 时必须明确报错，而不是静默跳过复核。

    静默跳过会让"已经开启了二次校验"这个配置变成一句空话，比直接报错危险。
    """
    checker = RelevanceChecker(threshold=0.6, enable_llm_check=True, judge=None)

    with pytest.raises(ConfigurationError, match="judge"):
        checker.check("问题", [(_document(), 0.4)])


def test_llm_check_disabled_by_default() -> None:
    """默认关闭 LLM 复核，不产生任何额外调用。"""
    assert RelevanceChecker().enable_llm_check is False


def test_judge_not_needed_when_llm_check_disabled() -> None:
    """关闭复核时，即使传入 judge 也不该被调用。"""
    judge = _RecordingJudge(verdict=True)
    checker = RelevanceChecker(threshold=0.6, enable_llm_check=False, judge=judge)

    result = checker.check("问题", [(_document(), 0.4)])

    assert result.is_relevant is False
    assert judge.calls == []


# ---------------------------------------------------------------------------
# 判定结果与便捷函数
# ---------------------------------------------------------------------------
def test_result_carries_explainable_details() -> None:
    """判定结果要能回答"为什么拒答"。"""
    result = RelevanceChecker(threshold=0.7).check("问题", [(_document(), 0.55)])

    assert result.threshold == pytest.approx(0.7)
    assert result.best_score == pytest.approx(0.55)
    assert result.candidate_count == 1
    assert "0.5500" in result.reason
    assert "0.7" in result.reason


def test_check_relevance_function_matches_checker() -> None:
    """便捷函数与显式构造 Checker 的结论必须一致。"""
    results = [(_document(), 0.8)]

    assert check_relevance("问题", results, threshold=0.6) is True
    assert check_relevance("问题", results, threshold=0.9) is False


@pytest.mark.parametrize("score", [0.0, 0.3, 0.6, 1.0])
def test_checker_accepts_full_valid_score_range(score: float) -> None:
    """[0, 1] 闭区间内的分数都应当是合法输入。"""
    RelevanceChecker(threshold=0.5).check("问题", [(_document(), score)])


def test_checker_is_reusable_across_calls() -> None:
    """同一个 Checker 可以被反复调用，不残留上一次的状态。"""
    checker = RelevanceChecker(threshold=0.6)

    assert checker.check("问题一", [(_document(), 0.9)]).is_relevant is True
    assert checker.check("问题二", [(_document(), 0.1)]).is_relevant is False
    assert checker.check("问题三", [(_document(), 0.7)]).is_relevant is True


def test_document_payload_is_not_modified() -> None:
    """判定过程不能改动文档本身。"""
    document = _document()
    before = dict(document.metadata)

    RelevanceChecker(threshold=0.6).check("问题", [(document, 0.9)])

    assert document.metadata == before


def test_callable_protocol_accepts_plain_function() -> None:
    """judge 只要是可调用对象即可，不要求是特定类。"""

    def judge(query: str, documents: Sequence[Document]) -> bool:
        return True

    checker: RelevanceChecker = RelevanceChecker(
        threshold=0.9, enable_llm_check=True, judge=judge
    )

    assert checker.check("问题", [(_document(), 0.1)]).is_relevant is True
    assert isinstance(checker.threshold, float)


def test_relevance_judge_type_alias_is_exported() -> None:
    """RelevanceJudge 类型别名应当可导入（供阶段 5 标注 judge 签名用）。"""
    from src.retrieval.relevance import RelevanceJudge

    assert callable(RelevanceJudge)


def test_judge_failure_degrades_to_rejection() -> None:
    """LLM 复核失败时按"不相关"处理（拒答），而不是让请求变成 502。

    复核只是"救回低分命中"的额外尝试：向量分数本来就没达标，它失败时拒答才是
    安全方向；把一次可选调用的失败升级成上游错误，等于让一个抖动变成一次 5xx。
    """

    def judge(query: str, documents: Sequence[Document]) -> bool:
        raise GenerationError("复核这条路挂了")

    checker = RelevanceChecker(threshold=0.6, enable_llm_check=True, judge=judge)

    result = checker.check("任意问题", [(_document(), 0.3)])

    assert result.is_relevant is False
    assert result.checked_by_llm is True
    assert "复核失败" in result.reason
