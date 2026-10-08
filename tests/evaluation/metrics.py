"""评测指标：纯函数，不碰 Pipeline，可以单独测。

**指标为什么值得单独成模块**

指标算错比系统答错更危险：系统答错你能看见，指标算错你会拿着一个错误的数字去做
优化决策。所以这里全部是"输入一组结果、输出一个数字"的纯函数，每条都有对应的
单元测试（``tests/unit/test_evaluation.py``）。

**几个口径先说清楚**

- **Recall@K**：对一条用例，期望课程编号里**任意一个**出现在前 K 条检索结果中即算
  命中。分母只包含"有期望课程且不该拒答"的用例——拒答题的期望本来就是"什么都别答"，
  把它算进召回率只会稀释指标。
- **keyword_hit**：一条用例的 ``expected_answer_keywords`` **全部**出现才算命中；
  另外单独给 ``keyword_coverage``（命中比例），用来区分"完全答偏"和"只差一个词"。
- **百分位**：用线性插值（与 numpy 默认一致）。样本只有 30 条时 p95 实际上是
  第二大值那种量级，**不要**把 30 条样本的 p95 当成稳定的线上 SLA 来引用——
  报告里也据此标注了样本量。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Final

#: 报告里固定输出的 Recall 档位。
RECALL_KS: Final[tuple[int, ...]] = (3, 5, 10)


@dataclass
class CaseOutcome:
    """一条用例的完整评测结果（报告 ``details`` 的元素）。"""

    id: str
    question: str
    type: str
    expected_route: str
    actual_route: str
    route_correct: bool
    expected_course_ids: list[str]
    retrieved_course_ids: list[str]
    should_reject: bool
    actual_rejected: bool
    keyword_hit: bool | None
    latency_ms: float
    cached: bool = False
    keyword_coverage: float | None = None
    recall_hits: dict[str, bool] = field(default_factory=dict)
    retrieval_hit: bool | None = None
    top_relevance_score: float | None = None
    """最高一条检索分数。**调阈值时唯一有用的证据**：它就是"该拒不拒"那条用例
    被放行的原因，也是判断"能不能靠抬阈值拦下它"的唯一依据。"""
    faithfulness: bool | None = None
    answer: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        """转成写进 ``report.json`` 的结构（字段名与阶段 7 的规格一致）。"""
        payload: dict[str, object] = {
            "id": self.id,
            "question": self.question,
            "type": self.type,
            "expected_route": self.expected_route,
            "actual_route": self.actual_route,
            "route_correct": self.route_correct,
            "expected_course_id": self.expected_course_ids[0] if self.expected_course_ids else None,
            "expected_course_ids": self.expected_course_ids,
            "retrieved_course_ids": self.retrieved_course_ids,
            "retrieval_hit": self.retrieval_hit,
            "top_relevance_score": self.top_relevance_score,
            "recall_hits": self.recall_hits,
            "should_reject": self.should_reject,
            "actual_rejected": self.actual_rejected,
            "keyword_hit": self.keyword_hit,
            "keyword_coverage": self.keyword_coverage,
            "cached": self.cached,
            "latency_ms": self.latency_ms,
            "answer": self.answer,
        }
        if self.faithfulness is not None:
            payload["faithfulness"] = self.faithfulness
        if self.error is not None:
            payload["error"] = self.error
        return payload


def rate(numerator: int, denominator: int) -> float | None:
    """算比例；分母为 0 时返回 ``None``（"无数据"和"0%"是两件事）。"""
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def percentile(values: Sequence[float], percent: float) -> float | None:
    """线性插值百分位。

    :param values: 样本；为空返回 ``None``。
    :param percent: 0~100 的百分位。
    """
    if not values:
        return None
    if not 0.0 <= percent <= 100.0:
        raise ValueError(f"percent 必须落在 [0, 100] 内，当前为 {percent}")

    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 3)

    position = (len(ordered) - 1) * (percent / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    interpolated = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    return round(float(interpolated), 3)


def recall_at_k(outcomes: Iterable[CaseOutcome], k: int) -> float | None:
    """Recall@K：期望课程出现在前 K 条的比例。

    只统计"有期望课程"的用例（``recall_hits`` 里有对应档位的）。
    """
    evaluated = [o for o in outcomes if str(k) in o.recall_hits]
    if not evaluated:
        return None
    hits = sum(1 for o in evaluated if o.recall_hits[str(k)])
    return rate(hits, len(evaluated))


def summarize(outcomes: Sequence[CaseOutcome]) -> dict[str, object]:
    """把一堆用例结果汇总成报告里的 ``summary``。

    :param outcomes: 全部用例结果。
    :return: 指标字典；分母为 0 的项返回 ``None``（而不是 0 或 1）。
    """
    total = len(outcomes)
    latencies = [o.latency_ms for o in outcomes if o.error is None]

    route_correct = sum(1 for o in outcomes if o.route_correct)
    reject_cases = [o for o in outcomes if o.should_reject]
    answered_cases = [o for o in outcomes if not o.should_reject]
    faq_cases = [o for o in outcomes if o.type == "faq"]
    keyword_cases = [o for o in outcomes if o.keyword_hit is not None]
    faithfulness_cases = [o for o in outcomes if o.faithfulness is not None]

    summary: dict[str, object] = {
        "total": total,
        "route_accuracy": rate(route_correct, total),
        "rejection_accuracy": rate(
            sum(1 for o in reject_cases if o.actual_rejected), len(reject_cases)
        ),
        # 反向指标：不该拒答的题被拒了多少。只看 rejection_accuracy 会掩盖
        # "阈值太高、什么都拒"这种表现——那个模型的拒答准确率同样是 100%。
        "over_rejection_rate": rate(
            sum(1 for o in answered_cases if o.actual_rejected), len(answered_cases)
        ),
        "faq_hit_rate": rate(sum(1 for o in faq_cases if o.cached), len(faq_cases)),
        "keyword_hit_rate": rate(
            sum(1 for o in keyword_cases if o.keyword_hit), len(keyword_cases)
        ),
        "keyword_coverage": _mean_or_none([o.keyword_coverage for o in keyword_cases]),
        "average_latency_ms": _mean_or_none(latencies),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "error_count": sum(1 for o in outcomes if o.error is not None),
    }

    for k in RECALL_KS:
        summary[f"recall_at_{k}"] = recall_at_k(outcomes, k)

    if faithfulness_cases:
        summary["faithfulness"] = rate(
            sum(1 for o in faithfulness_cases if o.faithfulness), len(faithfulness_cases)
        )
        summary["faithfulness_evaluated"] = len(faithfulness_cases)
    else:
        summary["faithfulness"] = None

    summary["rejection_case_count"] = len(reject_cases)
    summary["recall_case_count"] = sum(1 for o in outcomes if o.recall_hits)
    summary["faq_case_count"] = len(faq_cases)
    return summary


#: 阈值扫描的默认网格：0.30 ~ 0.80，步长 0.01。
DEFAULT_THRESHOLD_GRID: Final[tuple[float, ...]] = tuple(
    round(0.30 + 0.01 * index, 2) for index in range(51)
)


def threshold_sweep(
    outcomes: Sequence[CaseOutcome],
    *,
    grid: Sequence[float] | None = None,
) -> dict[str, object]:
    """扫一遍相关性阈值，看哪个取值误判最少。

    **为什么这件事值得做成一个指标**

    ``RELEVANCE_THRESHOLD`` 是整个系统里最"玄"的一个数：它直接决定"该拒的拒没拒、
    该答的答没答"，而它没法靠读代码判断好坏。但换个阈值**不需要重新检索**——检索
    分数与阈值无关，只与问题和语料有关。所以只要每条用例记下 ``top_relevance_score``，
    就能一次把整条阈值曲线算出来，不用跑几十次评测。

    :param outcomes: 评测结果（只用到 ``top_relevance_score`` 与 ``should_reject``）。
    :param grid: 待扫的阈值；``None`` 时用 :data:`DEFAULT_THRESHOLD_GRID`。
    :return: ``{"grid": [...], "best": {...}, "current": {...}}``。样本不足时
        ``best`` 为 ``None``（宁可不给建议，也不要基于 3 条用例指手画脚）。
    """
    scored = [o for o in outcomes if o.top_relevance_score is not None and o.error is None]
    if len(scored) < 2:
        return {"grid": [], "best": None, "current": None, "scored_cases": len(scored)}

    thresholds = list(grid) if grid is not None else list(DEFAULT_THRESHOLD_GRID)
    rows: list[dict[str, object]] = []
    for threshold in thresholds:
        # 分数不达标的会被拒答；达标才可能进入生成。
        false_rejects = [
            o for o in scored if not o.should_reject and o.top_relevance_score < threshold
        ]
        false_accepts = [
            o for o in scored if o.should_reject and o.top_relevance_score >= threshold
        ]
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "false_rejects": len(false_rejects),
                "false_accepts": len(false_accepts),
                "total_errors": len(false_rejects) + len(false_accepts),
                "false_reject_ids": [o.id for o in false_rejects],
                "false_accept_ids": [o.id for o in false_accepts],
            }
        )

    best_errors = min(int(row["total_errors"]) for row in rows)  # type: ignore[arg-type]
    best_rows = [row for row in rows if row["total_errors"] == best_errors]
    # 同样是"零误判"，选最高的那个阈值不行——越高越容易误拒没见过的问法。
    # 取最优区间里**最低**的阈值：够用，且对分布漂移更宽容。
    best = min(best_rows, key=lambda row: float(row["threshold"]))  # type: ignore[arg-type]

    return {
        "grid": rows,
        "best": best,
        "best_range": [
            float(best_rows[0]["threshold"]),  # type: ignore[arg-type]
            float(best_rows[-1]["threshold"]),  # type: ignore[arg-type]
        ],
        "scored_cases": len(scored),
        "answerable_min_score": min(
            (o.top_relevance_score for o in scored if not o.should_reject), default=None
        ),
        "reject_max_score": max(
            (o.top_relevance_score for o in scored if o.should_reject), default=None
        ),
    }


def keyword_coverage(answer: str, keywords: Sequence[str]) -> float | None:
    """计算关键词覆盖率：命中的关键词比例（大小写不敏感）。"""
    if not keywords:
        return None
    haystack = answer.casefold()
    hits = sum(1 for keyword in keywords if keyword.casefold() in haystack)
    return round(hits / len(keywords), 4)


def keywords_all_present(answer: str, keywords: Sequence[str]) -> bool | None:
    """关键词是否**全部**出现；没有关键词时返回 ``None``（不参与统计）。"""
    if not keywords:
        return None
    haystack = answer.casefold()
    return all(keyword.casefold() in haystack for keyword in keywords)


def _mean_or_none(values: Sequence[float | None]) -> float | None:
    """只对非 ``None`` 取平均，并保留 3 位小数。"""
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return round(fmean(present), 3)
