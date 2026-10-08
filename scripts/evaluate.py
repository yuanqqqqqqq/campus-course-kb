"""RAG 评测入口：跑评测集、打印指标、写出 ``report.json``。

用法：

.. code-block:: bash

    python scripts/evaluate.py                      # 默认：top_k=5、启用 FAQ、假 LLM（零成本）
    python scripts/evaluate.py --top-k 3            # 只改检索条数
    python scripts/evaluate.py --disable-faq        # 关掉 FAQ 快路径，对比延迟
    python scripts/evaluate.py --llm real           # 真实 DeepSeek（需 .env 配好 Key，会计费）
    python scripts/evaluate.py --output report_k10.json --tag top_k=10

**默认不花任何 API 费用**：分类走真实规则层，生成走把资料拼成答案的假模型。
代价是"LLM 分类"与"LLM 写作"这两段不进入度量——报告里的 ``caveats`` 会写清楚，
``--llm real`` 才是完整链路。

具体怎么算在 ``tests/evaluation/`` 里，本文件只负责解析参数、打印和落盘。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 直接以脚本方式运行时（python scripts/evaluate.py），sys.path[0] 是 scripts/ 而不是
# 项目根目录，`import src...` 会失败。这里补上项目根目录，让脚本在任意工作目录下都能跑。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.utils.exceptions import AppError  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from tests.evaluation.dataset import DEFAULT_EVAL_SET_PATH  # noqa: E402
from tests.evaluation.runner import EvaluationConfig, run_evaluation  # noqa: E402

logger = get_logger(__name__)

#: 报告里按这个顺序打印指标；``None`` 表示该项"没有可统计的样本"，不是 0。
_SUMMARY_ROWS: tuple[tuple[str, str], ...] = (
    ("total", "用例总数"),
    ("route_accuracy", "Route Accuracy（路由准确率）"),
    ("recall_at_3", "Recall@3"),
    ("recall_at_5", "Recall@5"),
    ("recall_at_10", "Recall@10"),
    ("rejection_accuracy", "Rejection Accuracy（拒答准确率）"),
    ("over_rejection_rate", "Over-Rejection Rate（该答却拒的比例，越低越好）"),
    ("faq_hit_rate", "FAQ Hit Rate"),
    ("keyword_hit_rate", "Keyword Hit Rate（关键词全中的比例）"),
    ("keyword_coverage", "Keyword Coverage（关键词命中比例均值）"),
    ("faithfulness", "Faithfulness（需 --llm real + ENABLE_LLM_EVALUATION）"),
    ("average_latency_ms", "平均延迟（毫秒）"),
    ("p50_latency_ms", "P50 延迟（毫秒）"),
    ("p95_latency_ms", "P95 延迟（毫秒）"),
    ("error_count", "执行失败条数"),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="校园课程知识库 RAG 评测（默认零 API 成本）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--top-k", type=int, default=5, help="每次检索的候选文档数")
    parser.add_argument(
        "--disable-faq",
        action="store_true",
        help="关闭 FAQ 快路径（use_faq=false），用于对比延迟与 FAQ 命中率",
    )
    parser.add_argument(
        "--llm",
        choices=("mock", "real"),
        default="real" if settings.enable_real_llm_eval else "mock",
        help="生成与分类用假实现还是真实 DeepSeek（real 需要 DEEPSEEK_API_KEY）",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVAL_SET_PATH,
        help="评测集路径",
    )
    parser.add_argument("--output", type=Path, default=Path("report.json"), help="报告输出路径")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（调试用）")
    parser.add_argument(
        "--enable-llm-evaluation",
        action="store_true",
        help="启用 LLM Judge 指标（faithfulness），仅在 --llm real 下生效",
    )
    parser.add_argument("--tag", default=None, help="给这次运行打个标签，写进报告便于对比")
    parser.add_argument("--verbose", action="store_true", help="打印逐条请求的 INFO 日志")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """跑一次评测并返回进程退出码。"""
    args = _parse_args(argv)
    # 默认只打到 WARNING：30 条用例 × 每条几行路由日志会把指标淹没掉。
    setup_logging("INFO" if args.verbose else "WARNING")

    config = EvaluationConfig(
        top_k=args.top_k,
        use_faq=not args.disable_faq,
        llm_mode=args.llm,
        dataset_path=args.dataset,
        limit=args.limit,
        enable_llm_evaluation=args.enable_llm_evaluation or settings.enable_llm_evaluation,
        metadata={"tag": args.tag} if args.tag else {},
    )

    try:
        report = run_evaluation(config)
    except AppError as exc:
        # 评测脚本自己出的问题（评测集坏了、索引没建、没配 Key）要给出可操作的提示，
        # 而不是丢一段堆栈——这类失败几乎都是"环境没准备好"，不是代码 bug。
        print(f"\n评测无法开始：[{exc.code}] {exc.message}", file=sys.stderr)
        if exc.details:
            print(f"  上下文：{exc.details}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"\n参数不合法：{exc}", file=sys.stderr)
        return 2

    _write_report(report, args.output)
    _print_summary(report, args.output)
    return 0


def _write_report(report: dict[str, object], output: Path) -> None:
    """把报告写成 JSON（``ensure_ascii=False``：中文问题与答案直接可读）。"""
    target = output if output.is_absolute() else PROJECT_ROOT / output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logger.info("报告已写入 | path=%s", target)


def _print_summary(report: dict[str, object], output: Path) -> None:
    """把 summary 打成一张表，并列出需要关注的用例。"""
    summary = report.get("summary") or {}
    config = report.get("config") or {}
    details = report.get("details") or []

    assert isinstance(summary, dict) and isinstance(config, dict)
    assert isinstance(details, list)

    print()
    print("=" * 62)
    print(f"RAG 评测报告 | top_k={config.get('top_k')} "
          f"faq={config.get('faq_enabled')} llm={config.get('llm_mode')} "
          f"分类={config.get('classification')}")
    print(f"语料：{config.get('index_documents')} 个文档切片，"
          f"阈值 relevance={config.get('relevance_threshold')} "
          f"faq_match={config.get('faq_match_threshold')}")
    print("=" * 62)
    for key, label in _SUMMARY_ROWS:
        print(f"  {label:<46}{_format_value(summary.get(key))}")

    print("-" * 62)
    print(_format_threshold_advice(report.get("threshold_analysis"), config))

    problems = _problem_cases(details)
    if problems:
        print("-" * 62)
        print(f"需要关注的用例（共 {len(problems)} 条）：")
        for line in problems[:8]:
            print(f"  {line}")
    print("=" * 62)
    print(f"报告已写入：{output}")
    caveats = report.get("caveats") or []
    if caveats:
        print("读这份报告前请先看 caveats（写在报告里）：")
        for note in caveats:
            print(f"  - {note}")
    print()


def _format_threshold_advice(analysis: object, config: dict) -> str:
    """把阈值扫描结果压成两三行可执行的建议。"""
    if not isinstance(analysis, dict) or analysis.get("best") is None:
        return "  阈值分析：样本不足，不做建议。"

    best = analysis["best"]
    assert isinstance(best, dict)
    current = config.get("relevance_threshold")
    answerable_min = analysis.get("answerable_min_score")
    reject_max = analysis.get("reject_max_score")

    lines = [
        f"  阈值分析：当前 {current} → 该答却拒 {_current_value(analysis, current, 'false_rejects')} 条、"
        f"该拒却答 {_current_value(analysis, current, 'false_accepts')} 条",
        f"            最低建议 {best['threshold']} → 总误判 {best['total_errors']} 条"
        f"（区间 {analysis.get('best_range')}）",
        f"            分数分布：可答最低 {answerable_min}，应拒最高 {reject_max}"
        f"（余量 {_margin(answerable_min, reject_max)}）",
    ]
    return "\n".join(lines)


def _current_value(analysis: dict, current: object, key: str) -> object:
    """在当前阈值那一行里取指标。"""
    for row in analysis.get("grid") or []:
        if isinstance(row, dict) and row.get("threshold") == current:
            return row.get(key)
    return "N/A"


def _margin(answerable_min: object, reject_max: object) -> str:
    """算两类样本之间的分数余量——余量太小时，抬阈值只是在赌运气。"""
    if not isinstance(answerable_min, (int, float)) or not isinstance(reject_max, (int, float)):
        return "N/A"
    return f"{round(float(answerable_min) - float(reject_max), 4)}"


def _format_value(value: object) -> str:
    """格式化指标：``None`` 说明该项没有样本可统计。"""
    if value is None:
        return "N/A（无样本）"
    if isinstance(value, float):
        return f"{value}"
    return str(value)


def _problem_cases(details: list[object]) -> list[str]:
    """挑出最该看的用例：路由错、该拒没拒、该答却拒、关键词没中。"""
    problems: list[str] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        case_id = item.get("id")
        question = item.get("question")
        reasons: list[str] = []
        if item.get("error"):
            reasons.append(f"执行失败 {item['error']}")
        if not item.get("route_correct"):
            reasons.append(f"路由 {item.get('expected_route')}→{item.get('actual_route')}")
        if item.get("should_reject") and not item.get("actual_rejected"):
            reasons.append("该拒答却没拒")
        if not item.get("should_reject") and item.get("actual_rejected"):
            reasons.append("不该拒答却拒了")
        if item.get("keyword_hit") is False:
            reasons.append(f"关键词未全中 {item.get('keyword_coverage')}")
        if reasons:
            problems.append(f"{case_id} 「{question}」：{'；'.join(reasons)}")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
