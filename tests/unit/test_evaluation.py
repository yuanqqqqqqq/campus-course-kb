"""评测体系自身的单元测试（``tests/evaluation/`` + ``tests/eval_set.json``）。

**为什么评测代码也要测**

指标算错比系统答错更危险：答错你看得见，指标算错你会拿一个错误的数字去做优化决策。
这里覆盖三块：

1. **评测集**：结构、分布、以及最关键的一条——评测集里的 FAQ 用例必须真的能在
   ``data/faq.json`` 里命中。改了 FAQ 数据却忘了改评测集，这条会立刻失败。
2. **指标**：Recall@K、百分位、关键词口径、分母为 0 时返回 ``None`` 而不是 0。
3. **报告结构**：``details`` 的字段与阶段 7 规格一致（下游要按字段名读）。

全部是纯函数测试，不加载模型、不开 Chroma、不联网。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from src.routing.faq_cache import FAQCache
from src.schemas.chat import IntentType
from src.utils.exceptions import AppError
from tests.evaluation.dataset import (
    DEFAULT_EVAL_SET_PATH,
    EvalCase,
    EvalSet,
    QuestionType,
    load_eval_set,
)
from tests.evaluation.metrics import (
    CaseOutcome,
    keyword_coverage,
    keywords_all_present,
    percentile,
    rate,
    recall_at_k,
    summarize,
    threshold_sweep,
)

#: 规格要求的 details 字段。
_REQUIRED_DETAIL_FIELDS = {
    "id",
    "question",
    "expected_route",
    "actual_route",
    "route_correct",
    "expected_course_id",
    "retrieved_course_ids",
    "retrieval_hit",
    "should_reject",
    "actual_rejected",
    "keyword_hit",
    "latency_ms",
}

#: 规格要求的 summary 字段。
_REQUIRED_SUMMARY_FIELDS = {
    "total",
    "route_accuracy",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "rejection_accuracy",
    "faq_hit_rate",
    "keyword_hit_rate",
    "average_latency_ms",
    "p50_latency_ms",
    "p95_latency_ms",
}


def _outcome(
    *,
    case_id: str = "eval_001",
    expected_route: str = "course_query",
    actual_route: str = "course_query",
    expected_ids: list[str] | None = None,
    retrieved_ids: list[str] | None = None,
    should_reject: bool = False,
    actual_rejected: bool = False,
    keyword_hit: bool | None = True,
    keyword_cov: float | None = 1.0,
    cached: bool = False,
    latency_ms: float = 100.0,
    case_type: str = "factual",
    error: str | None = None,
    recall_hits: dict[str, bool] | None = None,
) -> CaseOutcome:
    """造一条评测结果。"""
    return CaseOutcome(
        id=case_id,
        question="问题",
        type=case_type,
        expected_route=expected_route,
        actual_route=actual_route,
        route_correct=expected_route == actual_route,
        expected_course_ids=expected_ids or [],
        retrieved_course_ids=retrieved_ids or [],
        should_reject=should_reject,
        actual_rejected=actual_rejected,
        keyword_hit=keyword_hit,
        keyword_coverage=keyword_cov,
        cached=cached,
        latency_ms=latency_ms,
        recall_hits=recall_hits if recall_hits is not None else {},
        error=error,
    )


# ---------------------------------------------------------------------------
# 评测集
# ---------------------------------------------------------------------------
def test_shipped_eval_set_has_required_size_and_distribution() -> None:
    """30 条，且五类题型的数量与阶段 7 的要求一致（10/5/5/5/5）。"""
    eval_set = load_eval_set()

    assert len(eval_set.cases) == 30
    assert Counter(case.type.value for case in eval_set.cases) == {
        "factual": 10,
        "relation": 5,
        "process": 5,
        "faq": 5,
        "rejection": 5,
    }


def test_shipped_eval_set_has_unique_ids_and_valid_routes() -> None:
    """id 唯一（重复会让报告里同一条出现两次），路由取值合法。"""
    eval_set = load_eval_set()

    ids = [case.id for case in eval_set.cases]
    assert len(ids) == len(set(ids))
    for case in eval_set.cases:
        assert isinstance(case.expected_route, IntentType)


def test_every_answerable_case_has_keywords() -> None:
    """可回答的用例必须有关键词，否则关键词命中率无从计算。"""
    for case in load_eval_set().cases:
        if not case.should_reject:
            assert case.expected_answer_keywords, f"{case.id} 缺少 expected_answer_keywords"


def test_faq_cases_are_hittable_in_the_shipped_faq_file() -> None:
    """**最关键的一条**：评测集里的 FAQ 用例必须真的能命中 data/faq.json。

    改了 FAQ 数据却没同步评测集（或调坏了 FAQ_MATCH_THRESHOLD），faq_hit_rate 会
    无声下降，而报告里看不出原因。这条断言把"评测集与 FAQ 数据脱节"变成一次失败。
    """
    cache = FAQCache()
    cache.load()

    misses = [
        case.id
        for case in load_eval_set().faq_cases
        if cache.lookup(case.question) is None
    ]

    assert misses == [], f"这些 FAQ 用例在 data/faq.json 里命中不了：{misses}"


def test_process_and_rejection_cases_expect_no_evidence() -> None:
    """流程题与拒答题都应当拒答：语料里确实没有教务流程内容。"""
    eval_set = load_eval_set()

    for case in eval_set.cases:
        if case.type in {QuestionType.PROCESS, QuestionType.REJECTION}:
            assert case.should_reject is True
            assert case.expected_ids == []


def test_expected_ids_merges_single_and_multi() -> None:
    """``expected_course_id`` 与 ``expected_course_ids`` 合并去重。"""
    case = EvalCase(
        id="x",
        question="哪些课需要 CS101？",
        type=QuestionType.RELATION,
        expected_route=IntentType.COURSE_QUERY,
        expected_course_id="CS201",
        expected_course_ids=["CS202", "CS201"],
        expected_answer_keywords=["CS201"],
    )

    assert case.expected_ids == ["CS201", "CS202"]


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (
            {
                "id": "x",
                "question": "q",
                "type": "faq",
                "expected_route": "course_query",
                "expected_course_id": "CS101",
                "expected_answer_keywords": ["a"],
            },
            "type=faq 但 expected_route 不是 faq",
        ),
        (
            {
                "id": "x",
                "question": "q",
                "type": "rejection",
                "expected_route": "course_query",
                "should_reject": True,
                "expected_course_id": "CS101",
            },
            "应当拒答却有期望课程",
        ),
        (
            {
                "id": "x",
                "question": "q",
                "type": "factual",
                "expected_route": "course_query",
                "expected_course_id": "CS101",
            },
            "可回答却没有关键词",
        ),
    ],
)
def test_inconsistent_cases_are_rejected(payload: dict, reason: str) -> None:
    """自相矛盾的用例在加载阶段就该被拦住（构造时即报，不用等到跑评测）。"""
    with pytest.raises(Exception):
        EvalCase.model_validate(payload)


def test_duplicate_ids_are_rejected() -> None:
    """id 重复要报错。"""
    case = EvalCase(
        id="dup",
        question="q",
        type=QuestionType.FACTUAL,
        expected_route=IntentType.COURSE_QUERY,
        expected_course_id="CS201",
        expected_answer_keywords=["a"],
    )

    with pytest.raises(Exception):
        EvalSet(cases=[case, case])


def test_load_eval_set_accepts_bare_list(tmp_path: Path) -> None:
    """裸数组写法也接受（手写评测集时最自然的形式）。"""
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "eval_001",
                    "question": "数据结构的学分是多少？",
                    "type": "factual",
                    "expected_route": "course_query",
                    "expected_course_id": "CS201",
                    "expected_answer_keywords": ["4"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert len(load_eval_set(path).cases) == 1


@pytest.mark.parametrize(
    "content",
    [
        "{不是 JSON",
        "{}",
        '{"cases": {}}',
        '{"cases": [{"id": "x"}]}',
        '[{"id": "x", "question": "q", "type": "unknown", "expected_route": "faq",'
        ' "expected_answer_keywords": ["a"]}]',
        '[{"id": "x", "question": "q", "type": "factual", "expected_route": "course_query",'
        ' "expected_answer_keywords": ["a"], "typo_field": 1}]',
    ],
)
def test_invalid_eval_sets_raise(tmp_path: Path, content: str) -> None:
    """坏评测集要报 :class:`~tests.evaluation.dataset.EvalDataError`，而且信息可读。"""
    path = tmp_path / "eval.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(AppError) as excinfo:
        load_eval_set(path)

    assert excinfo.value.code == "EVAL_DATA_ERROR"


def test_missing_eval_set_raises(tmp_path: Path) -> None:
    """文件不存在时报错，而不是拿空评测集算出 0/0。"""
    with pytest.raises(AppError):
        load_eval_set(tmp_path / "nope.json")


def test_default_eval_set_path_points_at_the_shipped_file() -> None:
    """默认路径真的存在（打包/移动目录后这里会先失败）。"""
    assert DEFAULT_EVAL_SET_PATH.exists()
    assert DEFAULT_EVAL_SET_PATH.name == "eval_set.json"


# ---------------------------------------------------------------------------
# 指标：基础件
# ---------------------------------------------------------------------------
def test_rate_returns_none_for_empty_denominator() -> None:
    """"没有样本"和"0%"是两件事，前者必须是 None。"""
    assert rate(0, 0) is None
    assert rate(3, 4) == 0.75
    assert rate(0, 5) == 0.0


@pytest.mark.parametrize(
    ("values", "percent", "expected"),
    [
        ([1, 2, 3, 4], 50, 2.5),
        ([1, 2, 3, 4], 0, 1.0),
        ([1, 2, 3, 4], 100, 4.0),
        ([1, 2, 3, 4], 95, 3.85),
        ([10.0], 95, 10.0),
        ([3, 1, 2], 50, 2.0),
    ],
)
def test_percentile_interpolates(values: list[float], percent: float, expected: float) -> None:
    """线性插值百分位（与 numpy 默认口径一致）。"""
    assert percentile(values, percent) == pytest.approx(expected, abs=1e-6)


def test_percentile_of_empty_is_none() -> None:
    """没有样本时返回 None。"""
    assert percentile([], 50) is None


def test_percentile_rejects_invalid_percent() -> None:
    """越界百分位要报错，而不是悄悄给出一个错的数。"""
    with pytest.raises(ValueError):
        percentile([1, 2], 101)


# ---------------------------------------------------------------------------
# 指标：Recall / 关键词
# ---------------------------------------------------------------------------
def test_recall_counts_any_expected_id_in_top_k() -> None:
    """多答案问题：任意一个期望课程出现在前 K 条即算命中。"""
    outcomes = [
        _outcome(
            expected_ids=["CS201", "CS202"],
            retrieved_ids=["CS201", "CS301"],
            recall_hits={"3": True, "5": True, "10": True},
        ),
        _outcome(
            case_id="eval_002",
            expected_ids=["CS301"],
            retrieved_ids=["CS201", "CS302"],
            recall_hits={"3": False, "5": False, "10": True},
        ),
    ]

    assert recall_at_k(outcomes, 3) == 0.5
    assert recall_at_k(outcomes, 10) == 1.0


def test_recall_ignores_cases_without_expectations() -> None:
    """没有期望课程的用例（拒答题）不进 Recall 的分母。"""
    outcomes = [_outcome(expected_ids=[], recall_hits={}), _outcome(recall_hits={"3": True})]

    assert recall_at_k(outcomes, 3) == 1.0


def test_recall_is_none_when_nothing_to_score() -> None:
    """一条可评的用例都没有时返回 None。"""
    assert recall_at_k([_outcome()], 3) is None


def test_keywords_require_all_present() -> None:
    """一条用例的关键词要全部出现才算命中。"""
    assert keywords_all_present("该课程 4 学分。", ["4", "学分"]) is True
    assert keywords_all_present("该课程 4 学分。", ["4", "教师"]) is False
    assert keywords_all_present("CS101 为 4 学分", ["cs101"]) is True  # 大小写不敏感
    assert keywords_all_present("任何回答", []) is None


def test_keyword_coverage_reports_partial_hits() -> None:
    """覆盖率用于区分"完全答偏"与"只差一个词"。"""
    assert keyword_coverage("4 学分", ["4", "学分"]) == 1.0
    assert keyword_coverage("4 学分", ["4", "教师"]) == 0.5
    assert keyword_coverage("4 学分", []) is None


# ---------------------------------------------------------------------------
# 指标：汇总
# ---------------------------------------------------------------------------
def test_summarize_contains_all_required_metrics() -> None:
    """汇总里必须有阶段 7 要求的全部指标。"""
    summary = summarize([_outcome()])

    assert set(summary) >= _REQUIRED_SUMMARY_FIELDS


def test_summarize_computes_rates_correctly() -> None:
    """手算一遍：路由 5/6、拒答 1/2、FAQ 命中 1/2、关键词 4/4。"""
    outcomes = [
        _outcome(case_id="a", expected_route="course_query", actual_route="course_query"),
        _outcome(case_id="b", expected_route="faq", actual_route="course_query"),
        _outcome(
            case_id="c",
            case_type="faq",
            expected_route="faq",
            actual_route="faq",
            cached=True,
        ),
        _outcome(case_id="d", case_type="faq", expected_route="faq", actual_route="faq"),
        _outcome(case_id="e", should_reject=True, actual_rejected=True, keyword_hit=None),
        _outcome(case_id="f", should_reject=True, actual_rejected=False, keyword_hit=None),
    ]

    summary = summarize(outcomes)

    assert summary["total"] == 6
    assert summary["route_accuracy"] == round(5 / 6, 4)
    assert summary["rejection_accuracy"] == 0.5
    assert summary["faq_hit_rate"] == 0.5
    assert summary["keyword_hit_rate"] == round(4 / 4, 4)
    assert summary["keyword_coverage"] == 1.0


def test_over_rejection_is_reported_separately() -> None:
    """只看拒答准确率会漏掉"阈值太高、什么都拒"——反向指标必须单独给。"""
    outcomes = [
        _outcome(case_id="a", should_reject=True, actual_rejected=True, keyword_hit=None),
        _outcome(case_id="b", actual_rejected=True),  # 该答却拒
    ]

    summary = summarize(outcomes)

    assert summary["rejection_accuracy"] == 1.0
    assert summary["over_rejection_rate"] == 1.0


def test_summarize_treats_empty_groups_as_none() -> None:
    """没有拒答题 / 没有关键词题时，对应指标是 None 而不是 0 或 1。"""
    summary = summarize([_outcome(keyword_hit=None, keyword_cov=None)])

    assert summary["rejection_accuracy"] is None
    assert summary["faq_hit_rate"] is None
    assert summary["keyword_hit_rate"] is None
    assert summary["faithfulness"] is None


def test_latency_percentiles_in_summary() -> None:
    """延迟统计含平均值与 P50/P95。

    P95 用线性插值：``[10,20,30,40,1000]`` 的位置是 ``(5-1)*0.95 = 3.8``，
    落在 40 与 1000 之间 → ``40 + 960*0.8 = 808``。
    """
    outcomes = [
        _outcome(case_id=f"c{i}", latency_ms=float(value))
        for i, value in enumerate([10, 20, 30, 40, 1000])
    ]

    summary = summarize(outcomes)

    assert summary["average_latency_ms"] == 220.0
    assert summary["p50_latency_ms"] == 30.0
    assert summary["p95_latency_ms"] == 808.0


def test_threshold_sweep_finds_the_separating_threshold() -> None:
    """有干净间隔时，扫描应当找出中间那个阈值，且零误判。"""
    outcomes = [
        # 可回答：分数 0.60 / 0.70
        _outcome(case_id="ok1"),
        _outcome(case_id="ok2"),
        # 应拒：分数 0.20 / 0.30
        _outcome(case_id="rej1", should_reject=True, keyword_hit=None),
        _outcome(case_id="rej2", should_reject=True, keyword_hit=None),
    ]
    outcomes[0].top_relevance_score = 0.60
    outcomes[1].top_relevance_score = 0.70
    outcomes[2].top_relevance_score = 0.20
    outcomes[3].top_relevance_score = 0.30

    analysis = threshold_sweep(outcomes)

    best = analysis["best"]
    assert isinstance(best, dict)
    assert best["total_errors"] == 0
    # 最优区间是 [0.31, 0.60]，取最低的那个：够用，且对分布漂移更宽容
    assert best["threshold"] == pytest.approx(0.31)
    assert analysis["answerable_min_score"] == 0.60
    assert analysis["reject_max_score"] == 0.30


def test_threshold_sweep_reports_errors_at_the_current_setting() -> None:
    """阈值太低会把"该拒"的放进来，扫描里必须看得见。"""
    outcomes = [
        _outcome(case_id="ok"),
        _outcome(case_id="rej", should_reject=True, keyword_hit=None),
    ]
    outcomes[0].top_relevance_score = 0.55
    outcomes[1].top_relevance_score = 0.52

    analysis = threshold_sweep(outcomes, grid=[0.50, 0.55])

    at_fifty = next(row for row in analysis["grid"] if row["threshold"] == 0.50)
    assert at_fifty["false_accepts"] == 1
    assert at_fifty["false_accept_ids"] == ["rej"]
    at_fifty_five = next(row for row in analysis["grid"] if row["threshold"] == 0.55)
    assert at_fifty_five["total_errors"] == 0


def test_threshold_sweep_without_scores_returns_no_advice() -> None:
    """没有分数就没有建议——基于两三条样本的"最佳阈值"比不给还糟。"""
    assert threshold_sweep([_outcome()])["best"] is None
    assert threshold_sweep([_outcome(), _outcome()])["best"] is None


def test_failed_cases_are_counted_and_excluded_from_latency() -> None:
    """执行失败的用例计入 error_count，但不污染延迟统计（0ms 会拉低所有百分位）。"""
    outcomes = [
        _outcome(case_id="ok", latency_ms=100.0),
        _outcome(case_id="fail", latency_ms=0.0, error="GENERATION_ERROR: boom"),
    ]

    summary = summarize(outcomes)

    assert summary["error_count"] == 1
    assert summary["p50_latency_ms"] == 100.0


# ---------------------------------------------------------------------------
# 报告结构
# ---------------------------------------------------------------------------
def test_case_outcome_dict_has_required_detail_fields() -> None:
    """``details`` 的字段名是给下游（人、脚本、看板）读的，不能悄悄改。"""
    payload = _outcome().to_dict()

    assert set(payload) >= _REQUIRED_DETAIL_FIELDS
    assert payload["latency_ms"] == 100.0


def test_faithfulness_field_only_appears_when_evaluated() -> None:
    """没跑的指标不要出现在报告里——空字段最容易被误读成"通过了"。"""
    assert "faithfulness" not in _outcome().to_dict()

    judged = _outcome()
    judged.faithfulness = True
    assert judged.to_dict()["faithfulness"] is True
