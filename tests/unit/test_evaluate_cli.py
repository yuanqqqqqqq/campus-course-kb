"""评测链路里"没被测试盯住"的那两段：命令行入口与执行器的汇总逻辑。

背景：``tests/unit/test_evaluation.py`` 覆盖了**指标口径**与**评测集校验**，
但 ``scripts/evaluate.py``（CLI 参数、报告落盘、摘要打印、阈值建议排版）和
``tests/evaluation/runner.py`` 的纯函数（caveats、配置快照、忠实度判定）此前
只有"手工跑脚本"这一种验证方式。手工跑能发现"整个跑不起来"，发现不了
"某个分支算错了"——所以这里把不需要真实 Chroma / 模型的那些部分补上。

不测的部分（诚实标注）：``run_evaluation`` 的端到端执行需要真实索引与 Embedding
模型，属于手工验证（每个阶段都会跑一次 ``python scripts/evaluate.py``）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate import _format_threshold_advice, _parse_args, _problem_cases, main
from src.config import settings
from tests.evaluation.runner import EvaluationConfig, _caveats


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def test_parse_args_defaults_to_mock_and_faq_on() -> None:
    """默认零成本：假 LLM + 启用 FAQ。"""
    args = _parse_args([])

    assert args.llm == "mock"
    assert args.disable_faq is False
    assert args.top_k == 5
    assert args.output == Path("report.json")


def test_parse_args_reads_the_experiment_switches() -> None:
    """对比实验用的开关要真的接上。"""
    args = _parse_args(["--top-k", "10", "--disable-faq", "--tag", "k=10", "--limit", "5"])

    assert (args.top_k, args.disable_faq, args.tag, args.limit) == (10, True, "k=10", 5)


def test_parse_args_uses_real_llm_when_the_env_flag_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ENABLE_REAL_LLM_EVAL=true 时默认切到真实 LLM（仍需显式配 Key）。"""
    monkeypatch.setattr(settings, "enable_real_llm_eval", True)

    assert _parse_args([]).llm == "real"


def test_real_mode_without_api_key_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """--llm real 但没配 Key：在构造配置时就报错，而不是跑到一半 500。"""
    monkeypatch.setattr(settings, "deepseek_api_key", __import__("pydantic").SecretStr(""))

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        EvaluationConfig(llm_mode="real")


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_is_rejected(top_k: int) -> None:
    """top_k 必须为正整数。"""
    with pytest.raises(ValueError, match="top_k"):
        EvaluationConfig(top_k=top_k)


def test_invalid_llm_mode_is_rejected() -> None:
    """模式只有 mock / real 两种。"""
    with pytest.raises(ValueError, match="llm_mode"):
        EvaluationConfig(llm_mode="half")


# ---------------------------------------------------------------------------
# 摘要与问题清单的排版
# ---------------------------------------------------------------------------
def test_problem_cases_lists_every_failure_kind() -> None:
    """路由错、该拒不拒、不该拒却拒、关键词未全中、执行失败都要能被列出来。"""
    details = [
        {"id": "a", "question": "q1", "route_correct": False, "expected_route": "faq",
         "actual_route": "course_query", "should_reject": False, "actual_rejected": False,
         "keyword_hit": True},
        {"id": "b", "question": "q2", "route_correct": True, "should_reject": True,
         "actual_rejected": False, "keyword_hit": None},
        {"id": "c", "question": "q3", "route_correct": True, "should_reject": False,
         "actual_rejected": True, "keyword_hit": None},
        {"id": "d", "question": "q4", "route_correct": True, "should_reject": False,
         "actual_rejected": False, "keyword_hit": False, "keyword_coverage": 0.5},
        {"id": "e", "question": "q5", "route_correct": True, "should_reject": False,
         "actual_rejected": False, "keyword_hit": True, "error": "GENERATION_ERROR: x"},
    ]

    problems = _problem_cases(details)

    assert len(problems) == 5
    assert any("路由 faq→course_query" in line for line in problems)
    assert any("该拒答却没拒" in line for line in problems)
    assert any("不该拒答却拒了" in line for line in problems)
    assert any("关键词未全中" in line for line in problems)
    assert any("执行失败" in line for line in problems)


def test_problem_cases_is_empty_when_everything_passes() -> None:
    """全绿时不给噪音。"""
    details = [
        {"id": "a", "question": "q", "route_correct": True, "should_reject": False,
         "actual_rejected": False, "keyword_hit": True, "error": None}
    ]

    assert _problem_cases(details) == []


def test_threshold_advice_formats_a_recommendation() -> None:
    """阈值建议要把"当前误判"和"建议阈值"讲清楚。"""
    analysis = {
        "grid": [
            {"threshold": 0.5, "false_rejects": 0, "false_accepts": 2, "total_errors": 2},
            {"threshold": 0.57, "false_rejects": 0, "false_accepts": 1, "total_errors": 1},
        ],
        "best": {"threshold": 0.57, "false_rejects": 0, "false_accepts": 1, "total_errors": 1},
        "best_range": [0.57, 0.58],
        "answerable_min_score": 0.5783,
        "reject_max_score": 0.5754,
    }

    rendered = _format_threshold_advice(analysis, {"relevance_threshold": 0.5})

    assert "当前 0.5" in rendered
    assert "最低建议 0.57" in rendered
    assert "余量 0.0029" in rendered


def test_threshold_advice_handles_missing_samples() -> None:
    """样本不足时不给建议，也不崩。"""
    assert "样本不足" in _format_threshold_advice({"best": None}, {})
    assert "样本不足" in _format_threshold_advice(None, {})


# ---------------------------------------------------------------------------
# 报告落盘
# ---------------------------------------------------------------------------
def test_main_writes_a_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI 能把报告写到指定路径（用一份假的 report 顶替真实评测）。"""
    fake_report = {
        "generated_at": "2026-10-08T00:00:00+00:00",
        "config": {"top_k": 5, "faq_enabled": True, "llm_mode": "mock",
                   "classification": "rules-only", "index_documents": 17,
                   "relevance_threshold": 0.5, "faq_match_threshold": 0.85},
        "summary": {"total": 1, "route_accuracy": 1.0},
        "details": [],
        "caveats": ["示例"],
    }
    monkeypatch.setattr("scripts.evaluate.run_evaluation", lambda config: fake_report)
    target = tmp_path / "report.json"

    exit_code = main(["--output", str(target)])

    assert exit_code == 0
    assert json.loads(target.read_text(encoding="utf-8"))["summary"]["total"] == 1


def test_main_reports_a_bad_dataset_without_a_stack_trace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """评测集坏了要给可读提示与退出码 2，而不是丢一段堆栈。"""
    broken = tmp_path / "eval.json"
    broken.write_text("{不是 JSON", encoding="utf-8")

    exit_code = main(["--dataset", str(broken), "--output", str(tmp_path / "r.json")])

    assert exit_code == 2
    assert "评测无法开始" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# 报告里的 caveats
# ---------------------------------------------------------------------------
def test_caveats_explain_what_mock_mode_cannot_measure() -> None:
    """mock 模式必须写明"测不到 LLM 的写作与分类能力"，不能让人误读数字。"""
    notes = "\n".join(_caveats(EvaluationConfig(), "rules-only"))

    assert "ContextEchoLLM" in notes
    assert "rules-only" in notes
    assert "不含 HTTP" in notes or "网络往返" in notes
    assert "样本量" in notes


def test_caveats_mention_the_disabled_faq_fast_path() -> None:
    """关掉 FAQ 时必须点明 faq_hit_rate 必然为 0。"""
    notes = "\n".join(_caveats(EvaluationConfig(use_faq=False), "rules-only"))

    assert "faq_hit_rate" in notes
