"""性能压测：量三条路径的延迟与吞吐。

用法：

.. code-block:: bash

    python scripts/benchmark.py                       # 默认：健康检查 / FAQ / RAG 各 30 次
    python scripts/benchmark.py --requests 100        # 加大样本
    python scripts/benchmark.py --concurrency 8       # 并发 8，量吞吐
    python scripts/benchmark.py --label "FAQ OFF"     # 给这次结果打个标签
    ENABLE_REAL_LLM_BENCHMARK=true python scripts/benchmark.py --real-llm   # 真实 DeepSeek

**测的是什么**

用 :class:`~fastapi.testclient.TestClient` 在**进程内**发真实 HTTP 请求：路由、依赖
注入、Pydantic 校验、JSON 序列化、SSE 分帧都会跑到，只有网络栈（socket、TCP、
uvicorn 的协议解析）不参与。所以这里的数字比 `curl`/`wrk` 看到的**小**，适合用来比较
"不同配置之间谁快"（相对值可信），不适合当成线上 SLA 引用（绝对值偏乐观）。

**为什么要压测**：项目的几个关键取舍都建立在"省一次调用"上（FAQ 命中不调 LLM、
证据不足不调 LLM）。这些取舍值不值，得用数字说话——所以这里专门对比
"FAQ ON / FAQ OFF" 与 "top_k=3/5/10"。

**默认不花钱**：RAG 路径用假生成层（``ContextEchoLLM``），只跑真实的 Embedding +
Chroma + 规则分类。要测真实链路加 ``--real-llm``（需 ``DEEPSEEK_API_KEY``，
并且用 ``ENABLE_REAL_LLM_BENCHMARK=true`` 显式开启）。
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# 直接以脚本方式运行时补上项目根目录（理由同 scripts/evaluate.py）。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.api.dependencies import get_faq_cache, get_pipeline  # noqa: E402
from src.config import settings  # noqa: E402
from src.main import app  # noqa: E402
from src.retrieval.relevance import RelevanceChecker  # noqa: E402
from src.retrieval.vectorstore import VectorStore  # noqa: E402
from src.routing.classifier import IntentClassifier  # noqa: E402
from src.routing.faq_cache import FAQCache  # noqa: E402
from src.services.pipeline import ChatPipeline  # noqa: E402
from src.utils.logging import get_logger, setup_logging  # noqa: E402
from tests.evaluation.metrics import percentile  # noqa: E402
from tests.fakes import ContextEchoLLM  # noqa: E402

logger = get_logger(__name__)

#: 三条被测路径的代表性问题。刻意都取自评测集的稳定用例：
#: FAQ 走快路径、RAG 走完整检索链路、拒答走"检索到但分数不够"。
HEALTH_PATH = "/api/health"
FAQ_QUESTION = "CS101几学分"
RAG_QUESTION = "数据结构与算法的先修课是什么"
REFUSAL_QUESTION = "食堂几点开门"


@dataclass
class BenchResult:
    """一条路径的压测结果。"""

    name: str
    total: int
    errors: int
    latencies_ms: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def error_rate(self) -> float:
        """错误率（HTTP 非 2xx 或抛异常）。"""
        return round(self.errors / self.total, 4) if self.total else 0.0

    @property
    def throughput(self) -> float:
        """吞吐（请求/秒），按整段耗时算。"""
        return round(self.total / self.wall_seconds, 2) if self.wall_seconds > 0 else 0.0

    def summary(self) -> dict[str, object]:
        """平均 / P50 / P95 / P99 与吞吐。"""
        return {
            "路径": self.name,
            "请求数": self.total,
            "错误数": self.errors,
            "错误率": self.error_rate,
            "平均延迟_ms": round(stats.fmean(self.latencies_ms), 3) if self.latencies_ms else None,
            "P50_ms": percentile(self.latencies_ms, 50),
            "P95_ms": percentile(self.latencies_ms, 95),
            "P99_ms": percentile(self.latencies_ms, 99),
            "吞吐_req_s": self.throughput,
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="校园课程知识库性能压测（默认不调用真实 LLM）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--requests", type=int, default=30, help="每条路径的请求数")
    parser.add_argument("--concurrency", type=int, default=1, help="并发线程数")
    parser.add_argument("--top-k", type=int, default=5, help="RAG 路径的检索条数")
    parser.add_argument("--disable-faq", action="store_true", help="关闭 FAQ 快路径")
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="用真实 DeepSeek（需 ENABLE_REAL_LLM_BENCHMARK=true 且配好 Key）",
    )
    parser.add_argument("--label", default="", help="结果标签，写进 JSON 便于对比")
    parser.add_argument("--output", type=Path, default=None, help="把结果写成 JSON")
    parser.add_argument("--warmup", type=int, default=3, help="正式计时前的预热请求数")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印逐条请求的 INFO 日志（默认只打结果表，否则几十次请求的日志会把结果淹没）",
    )
    return parser.parse_args(argv)


def _build_pipeline(*, use_real_llm: bool, top_k: int) -> ChatPipeline:
    """装配被测链路：真实检索 + （默认）假生成层。"""
    from src.generation.llm import LLMService

    vector_store = VectorStore()
    faq_cache = FAQCache()
    faq_cache.load()

    llm = LLMService() if use_real_llm else ContextEchoLLM()
    classifier = IntentClassifier(llm=llm.complete if use_real_llm else None)
    judge = llm.judge_relevance if use_real_llm and settings.enable_llm_relevance_check else None

    return ChatPipeline(
        classifier=classifier,
        faq_cache=faq_cache,
        vector_store=vector_store,
        relevance=RelevanceChecker(judge=judge),
        llm=llm,
    )


def _make_requester(client: TestClient, question: str, top_k: int, use_faq: bool) -> Callable[[], None]:
    """造一个"发一次请求"的函数（返回后延迟由调用方计时）。"""

    def _request() -> None:
        response = client.post(
            "/api/chat",
            json={"question": question, "use_faq": use_faq, "top_k": top_k},
        )
        if response.status_code != 200:
            # 让上层记成错误，而不是把 500 也当成功计时
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:120]}")

    return _request


def _bench(name: str, request: Callable[[], None], *, total: int, concurrency: int) -> BenchResult:
    """跑一条路径并统计。

    并发模式用线程池打同一个进程内的 ASGI 应用：**这不是严格意义的并发压测**
    （没有真实网络，且 Python 侧仍有 GIL），结果用来观察"并发下会不会明显劣化"，
    不适合拿来标定容量。
    """
    result = BenchResult(name=name, total=total, errors=0)
    started = time.perf_counter()

    if concurrency <= 1:
        for _ in range(total):
            request_started = time.perf_counter()
            try:
                request()
                result.latencies_ms.append((time.perf_counter() - request_started) * 1000)
            except Exception as exc:
                result.errors += 1
                logger.warning("压测请求失败 | path=%s error=%s", name, exc)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = []
            for _ in range(total):
                futures.append(pool.submit(_timed, request, result))
            for future in futures:
                future.result()

    result.wall_seconds = time.perf_counter() - started
    return result


def _timed(request: Callable[[], None], result: BenchResult) -> None:
    """线程池里跑单次请求并记录（延迟列表由主线程汇总，这里只 append）。"""
    started = time.perf_counter()
    try:
        request()
        result.latencies_ms.append((time.perf_counter() - started) * 1000)
    except Exception as exc:
        result.errors += 1
        logger.warning("压测请求失败 | path=%s error=%s", result.name, exc)


def main(argv: list[str] | None = None) -> int:
    """跑压测并打印结果。"""
    args = _parse_args(argv)
    setup_logging("INFO" if args.verbose else "WARNING")

    if args.requests < 1 or args.concurrency < 1 or args.top_k < 1:
        print("参数不合法：--requests / --concurrency / --top-k 都必须为正整数", file=sys.stderr)
        return 2

    use_real_llm = args.real_llm
    if use_real_llm and not settings.enable_real_llm_benchmark:
        print(
            "拒绝执行：--real-llm 会把几百次请求打到真实 DeepSeek 上。"
            "确认要这么做请设置 ENABLE_REAL_LLM_BENCHMARK=true。",
            file=sys.stderr,
        )
        return 2
    if use_real_llm and not settings.has_deepseek_api_key:
        print("--real-llm 需要 DEEPSEEK_API_KEY，但当前未配置。", file=sys.stderr)
        return 2

    pipeline = _build_pipeline(use_real_llm=use_real_llm, top_k=args.top_k)
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_faq_cache] = lambda: pipeline.faq_cache

    results: list[BenchResult] = []
    try:
        with TestClient(app) as client:
            # 应用启动时 lifespan 会按 settings.log_level 重新配置日志（默认 INFO），
            # 所以要在**进入上下文之后**再压一次级别，否则每条请求的日志都会刷出来。
            setup_logging("INFO" if args.verbose else "WARNING")
            _warmup(client, pipeline, args.warmup, args.top_k, use_faq=not args.disable_faq)

            results.append(
                _bench(
                    "GET /api/health",
                    lambda: _assert_ok(client.get(HEALTH_PATH)),
                    total=args.requests,
                    concurrency=args.concurrency,
                )
            )
            results.append(
                _bench(
                    "POST /api/chat (FAQ 命中)" if not args.disable_faq else "POST /api/chat (FAQ OFF)",
                    _make_requester(client, FAQ_QUESTION, args.top_k, not args.disable_faq),
                    total=args.requests,
                    concurrency=args.concurrency,
                )
            )
            results.append(
                _bench(
                    "POST /api/chat (RAG)",
                    _make_requester(client, RAG_QUESTION, args.top_k, not args.disable_faq),
                    total=args.requests,
                    concurrency=args.concurrency,
                )
            )
            results.append(
                _bench(
                    "POST /api/chat (拒答)",
                    _make_requester(client, REFUSAL_QUESTION, args.top_k, not args.disable_faq),
                    total=args.requests,
                    concurrency=args.concurrency,
                )
            )
    finally:
        app.dependency_overrides.clear()

    _print_report(results, args, use_real_llm)
    if args.output:
        _write_json(results, args, use_real_llm)
    return 0


def _assert_ok(response: object) -> None:
    """健康检查必须 200，否则算错误。"""
    status_code = getattr(response, "status_code", 0)
    if status_code != 200:
        raise RuntimeError(f"HTTP {status_code}")


def _warmup(
    client: TestClient, pipeline: ChatPipeline, times: int, top_k: int, *, use_faq: bool
) -> None:
    """预热：把模型加载、Chroma 打开这些一次性开销排除在统计之外。"""
    if times <= 0:
        return
    started = time.perf_counter()
    for _ in range(times):
        client.post(
            "/api/chat", json={"question": RAG_QUESTION, "use_faq": use_faq, "top_k": top_k}
        )
    logger.info("预热完成 | times=%d 耗时 %.2fs", times, time.perf_counter() - started)
    print(f"预热 {times} 次（加载 Embedding 模型等一次性开销，不计入统计）…")


def _print_report(results: Sequence[BenchResult], args: argparse.Namespace, use_real_llm: bool) -> None:
    """打印结果表。"""
    print()
    print("=" * 92)
    print(
        f"性能压测 | 每条路径 {args.requests} 次 并发 {args.concurrency} | "
        f"LLM={'真实 DeepSeek' if use_real_llm else '假实现（ContextEchoLLM）'} | "
        f"top_k={args.top_k} faq={'OFF' if args.disable_faq else 'ON'}"
        + (f" | {args.label}" if args.label else "")
    )
    print("=" * 92)
    header = f"{'路径':<28}{'平均':>10}{'P50':>10}{'P95':>10}{'P99':>10}{'吞吐':>10}{'错误率':>9}"
    print(header)
    print("-" * 92)
    for result in results:
        summary = result.summary()
        print(
            f"{summary['路径']:<28}"
            f"{_fmt(summary['平均延迟_ms']):>10}{_fmt(summary['P50_ms']):>10}"
            f"{_fmt(summary['P95_ms']):>10}{_fmt(summary['P99_ms']):>10}"
            f"{_fmt(summary['吞吐_req_s']):>10}{summary['错误率']:>9}"
        )
    print("-" * 92)
    print("延迟单位毫秒。进程内测量，不含网络往返；并发模式受 GIL 影响，只用于看劣化趋势。")
    if not use_real_llm:
        print("RAG 路径用的是假生成层，所以它反映的是「检索 + 装配」的成本，不含模型推理。")
    print()


def _fmt(value: object) -> str:
    """数值格式化（``None`` 显示成 -）。"""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _write_json(results: Sequence[BenchResult], args: argparse.Namespace, use_real_llm: bool) -> None:
    """把结果写成 JSON，便于跨配置对比。"""
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "label": args.label,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "top_k": args.top_k,
            "faq_enabled": not args.disable_faq,
            "llm": "real" if use_real_llm else "mock",
        },
        "results": [result.summary() for result in results],
        "caveats": [
            "进程内（TestClient）测量，不含网络往返与 uvicorn 协议开销，绝对值偏乐观；相对比较可信。",
            "默认使用假生成层，RAG 路径不含真实模型推理耗时。",
            f"每条路径样本 {args.requests} 个，P99 在样本少时波动大。",
        ],
    }
    target = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"结果已写入：{target}")


if __name__ == "__main__":
    raise SystemExit(main())
