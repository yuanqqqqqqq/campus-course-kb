"""评测执行器：逐条驱动**真实的**问答链路，收集结果并汇总成报告。

**为什么在进程内跑，而不是发 HTTP 请求给 /api/chat**

三个理由，按重要性排序：

1. **Recall@K 需要看到检索结果本身**。`/api/chat` 的响应里只有去重后的 ``sources``，
   既没有"前 K 条分别是什么"，也没法在一次请求里同时给出 K=3/5/10 的排序。召回率是
   评测里最该有、也最容易测歪的指标，不能为了走 HTTP 而放弃它。
2. **Mock LLM 只有进程内才可能**。阶段 7 的硬性要求是"评测尽量不产生 API 成本"，
   而要让"真实 Chroma + 真实 Embedding + 假 LLM"这个组合成立，就必须能在构造
   Pipeline 时把它替换掉——HTTP 服务里换不了。
3. **分类器与生成层的选择要可控**（下面"两种模式"一节）。

代价是：**这样测不到 HTTP 层的开销**（序列化、SSE 分帧、网络往返）。所以本模块算出的
延迟是"链路内部耗时"，比 `curl` 看到的小；要量端到端性能应当另外压测接口。
这一点写进报告的 ``caveats`` 里，不让人误读。

**两种模式**

``--llm mock``（默认）
    分类用**真实的规则层**（``IntentClassifier(llm=None)``：高置信度 FAQ 规则短路，
    其余落到规则候选或默认 course_query），生成用 :class:`~tests.fakes.ContextEchoLLM`。
    零 API 成本。此时 Route Accuracy 衡量的是**规则层**的表现，而"LLM 分类"与
    "LLM 写作"这两段没有进入度量——报告里标注 ``classification: rules-only``。

``--llm real``（需 ``DEEPSEEK_API_KEY``）
    分类与生成都走真实 DeepSeek，指标才完整。成本约等于评测集条数 × 一次生成
    （FAQ 命中与拒答不产生调用）。

**预热**

第一次检索要加载 Embedding 模型（本机实测约 5 秒），第一次生成要建 LLM 客户端。
这些是**一次性启动开销**，混进延迟统计会把 p95 直接顶到几秒、让所有 K 之间的对比
失去意义。所以正式计时前先空跑一轮（``warmup``），报告里也标注了它是预热过的。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from langchain_core.documents import Document

from src.config import settings
from src.generation.llm import LLMService, parse_yes_no
from src.generation.prompts import FAITHFULNESS_JUDGE_PROMPT, format_context
from src.retrieval.relevance import RelevanceChecker
from src.retrieval.vectorstore import VectorStore
from src.routing.classifier import IntentClassifier
from src.routing.faq_cache import FAQCache
from src.schemas.chat import ChatRequest
from src.services.pipeline import REFUSAL_REPLY, ChatPipeline
from src.utils.exceptions import AppError
from src.utils.logging import get_logger
from tests.evaluation.dataset import EvalCase, EvalSet, load_eval_set
from tests.evaluation.metrics import (
    RECALL_KS,
    CaseOutcome,
    keyword_coverage,
    keywords_all_present,
    summarize,
    threshold_sweep,
)
from tests.fakes import ContextEchoLLM

logger = get_logger(__name__)

#: Recall 只算到这些档位里最大的那个，一次检索切片即可。
MAX_RECALL_K: Final[int] = max(RECALL_KS)

#: 模式名 → 报告里的说明。
_LLM_MODE_MOCK: Final[str] = "mock"
_LLM_MODE_REAL: Final[str] = "real"


@dataclass
class EvaluationConfig:
    """一次评测的全部参数（对应 ``scripts/evaluate.py`` 的命令行开关）。"""

    top_k: int = 5
    use_faq: bool = True
    llm_mode: str = _LLM_MODE_MOCK
    dataset_path: Path | None = None
    limit: int | None = None
    enable_llm_evaluation: bool = False
    collection_name: str | None = None
    persist_dir: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"top_k 必须为正整数，当前为 {self.top_k}")
        if self.llm_mode not in {_LLM_MODE_MOCK, _LLM_MODE_REAL}:
            raise ValueError(f"llm_mode 只能是 mock 或 real，当前为 {self.llm_mode!r}")
        if self.llm_mode == _LLM_MODE_REAL and not settings.has_deepseek_api_key:
            raise ValueError(
                "llm_mode=real 需要 DEEPSEEK_API_KEY，但当前未配置。"
                "请在 .env 中填入，或改用默认的 mock 模式（零成本）。"
            )


def run_evaluation(config: EvaluationConfig | None = None) -> dict[str, object]:
    """跑完一次评测，返回可直接写进 ``report.json`` 的字典。

    :param config: 评测参数；``None`` 时用默认值（top_k=5、启用 FAQ、mock LLM）。
    """
    resolved = config or EvaluationConfig()
    eval_set = load_eval_set(resolved.dataset_path)

    vector_store = VectorStore(
        persist_dir=resolved.persist_dir, collection_name=resolved.collection_name
    )
    faq_cache = FAQCache()
    faq_cache.load()

    llm: Any
    if resolved.llm_mode == _LLM_MODE_REAL:
        llm = LLMService()
        classifier = IntentClassifier(llm=llm.complete)
        classification = "llm"
    else:
        llm = ContextEchoLLM()
        # 关键：**不注入 LLM**，于是分类只由真实规则层决定。规则层是生产代码的一部分，
        # 没命中的问题按设计落到默认的 course_query —— 这不是"假分类器"，是真实的降级路径。
        classifier = IntentClassifier(llm=None)
        classification = "rules-only"

    judge = (
        llm.judge_relevance
        if resolved.llm_mode == _LLM_MODE_REAL and settings.enable_llm_relevance_check
        else None
    )
    pipeline = ChatPipeline(
        classifier=classifier,
        faq_cache=faq_cache,
        vector_store=vector_store,
        relevance=RelevanceChecker(judge=judge),
        llm=llm,
    )

    cases = eval_set.cases[: resolved.limit] if resolved.limit else eval_set.cases
    _warmup(pipeline, vector_store)

    outcomes: list[CaseOutcome] = []
    for index, case in enumerate(cases, start=1):
        logger.info("[%d/%d] 评测 %s | %s", index, len(cases), case.id, case.question)
        outcomes.append(_evaluate_case(case, pipeline, vector_store, resolved, llm))

    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": _config_snapshot(resolved, eval_set, pipeline, vector_store, classification),
        "summary": summarize(outcomes),
        # 阈值分析单独一块：它用的分数与阈值无关，所以一次评测就能给出整条曲线，
        # 不必为了标定阈值把评测跑二十遍。
        "threshold_analysis": threshold_sweep(outcomes),
        "details": [outcome.to_dict() for outcome in outcomes],
        "caveats": _caveats(resolved, classification),
    }
    return report


# ---------------------------------------------------------------------------
# 单条用例
# ---------------------------------------------------------------------------
def _evaluate_case(
    case: EvalCase,
    pipeline: ChatPipeline,
    vector_store: VectorStore,
    config: EvaluationConfig,
    llm: Any,
) -> CaseOutcome:
    """跑一条用例：请求链路 + 独立检索 + 指标判定。"""
    request = ChatRequest(question=case.question, use_faq=config.use_faq, top_k=config.top_k)

    answer = ""
    actual_route = ""
    cached = False
    latency_ms = 0.0
    error: str | None = None
    try:
        response = pipeline.run(request)
        answer = response.answer
        actual_route = response.route.value
        cached = response.cached
        latency_ms = response.latency_ms
    except AppError as exc:
        # 评测不该因为一条用例炸掉整轮。错误记进 details，并计入 error_count：
        # 一条 500 不能悄悄变成"路由错了"这种看起来正常的失败。
        logger.error("用例执行失败 | id=%s code=%s error=%s", case.id, exc.code, exc)
        answer = f"[{exc.code}] {exc.message}"
        error = f"{exc.code}: {exc.message}"

    # Recall 单独跑一次检索，与请求本身解耦：
    # 1) 响应里看不到"前 K 条分别是什么"；2) 一次 K=10 的检索就能切片出 3/5/10 三档。
    hits = _retrieve_hits(vector_store, case.question)
    retrieved_ids = _course_ids(hits)
    recall_hits = (
        {str(k): any(cid in retrieved_ids[:k] for cid in case.expected_ids) for k in RECALL_KS}
        if case.expected_ids
        else {}
    )

    actual_rejected = answer == REFUSAL_REPLY
    verdict = _MaybeFaithfulness(config=config, llm=llm, error=error, rejected=actual_rejected,
                                should_reject=case.should_reject)
    return CaseOutcome(
        id=case.id,
        question=case.question,
        type=case.type.value,
        expected_route=case.expected_route.value,
        actual_route=actual_route,
        route_correct=actual_route == case.expected_route.value,
        expected_course_ids=case.expected_ids,
        retrieved_course_ids=retrieved_ids[: config.top_k],
        retrieval_hit=(
            any(cid in retrieved_ids[: config.top_k] for cid in case.expected_ids)
            if case.expected_ids
            else None
        ),
        recall_hits=recall_hits,
        top_relevance_score=round(hits[0][1], 4) if hits else None,
        should_reject=case.should_reject,
        actual_rejected=actual_rejected,
        keyword_hit=keywords_all_present(answer, case.expected_answer_keywords),
        keyword_coverage=keyword_coverage(answer, case.expected_answer_keywords),
        cached=cached,
        latency_ms=latency_ms,
        faithfulness=verdict.judge(case.question, answer, hits, config.top_k),
        answer=answer,
        error=error,
    )


@dataclass
class _MaybeFaithfulness:
    """忠实度判定的开关与依赖。

    单独成一个小对象，是为了让"这次到底该不该判"在一处说清楚：假模型不判（没有意义）、
    拒答与出错的用例不判（没有回答可判）、开关关着不判。散在调用点写三个 if 很容易
    漏掉一个，然后报告里出现一个看起来正常的假数字。
    """

    config: EvaluationConfig
    llm: Any
    error: str | None
    rejected: bool
    should_reject: bool

    def judge(
        self,
        question: str,
        answer: str,
        hits: Sequence[tuple[Document, float]],
        top_k: int,
    ) -> bool | None:
        """需要且可以判时调用 LLM Judge，否则返回 ``None``。"""
        if not self.config.enable_llm_evaluation:
            return None
        if self.config.llm_mode != _LLM_MODE_REAL:
            return None
        if self.error is not None or self.rejected or self.should_reject:
            return None
        documents = [document for document, _score in hits[:top_k]]
        return evaluate_faithfulness(self.llm, question, answer, documents)


def _retrieve_hits(
    vector_store: VectorStore, question: str
) -> list[tuple[Document, float]]:
    """独立跑一次 K=10 的检索，返回 ``(文档, 分数)``（按相关性降序）。

    一次取到 :data:`MAX_RECALL_K` 条，3/5/10 三档都从这一份结果里切片，保证三档之间
    口径一致（分别检索三次的话，Chroma 返回顺序的微小差异会被误读成指标变化）。
    """
    return list(vector_store.similarity_search_with_relevance(question, k=MAX_RECALL_K))


def _course_ids(hits: Sequence[tuple[Document, float]]) -> list[str]:
    """从检索结果里取出课程编号序列（保留重复，便于观察同一门课命中了几段）。"""
    ids: list[str] = []
    for document, _score in hits:
        course_id = (document.metadata or {}).get("course_id")
        if course_id:
            ids.append(str(course_id))
    return ids


def _warmup(pipeline: ChatPipeline, vector_store: VectorStore) -> None:
    """预热：把模型加载与连接建立的一次性开销排除在延迟统计之外。"""
    started = time.perf_counter()
    try:
        vector_store.similarity_search_with_relevance("预热", k=1)
        pipeline.run(ChatRequest(question="预热", use_faq=False, top_k=1))
    except AppError as exc:
        # 预热失败（例如索引为空）不算致命：真正的报错会在逐条评估里体现，
        # 而且那时每条都会带上错误码，比在预热阶段中断更有信息量。
        logger.warning("预热未成功（继续评测） | code=%s error=%s", exc.code, exc)
    logger.info("预热完成 | 耗时 %.2fs（不计入延迟统计）", time.perf_counter() - started)


# ---------------------------------------------------------------------------
# 报告附加信息
# ---------------------------------------------------------------------------
def _config_snapshot(
    config: EvaluationConfig,
    eval_set: EvalSet,
    pipeline: ChatPipeline,
    vector_store: VectorStore,
    classification: str,
) -> dict[str, object]:
    """记录这次报告的运行条件——没有它，两份报告之间无法比较。"""
    snapshot: dict[str, object] = {
        "dataset": str(config.dataset_path or "tests/eval_set.json"),
        "dataset_version": eval_set.version,
        "case_count": len(eval_set.cases),
        "top_k": config.top_k,
        "faq_enabled": config.use_faq,
        "llm_mode": config.llm_mode,
        "classification": classification,
        "enable_llm_evaluation": config.enable_llm_evaluation,
        "embedding_model": settings.embedding_model,
        "collection": vector_store.collection_name,
        "index_documents": vector_store.count(),
        "relevance_threshold": pipeline.relevance.threshold,
        "faq_match_threshold": pipeline.faq_cache.threshold,
        "recall_ks": list(RECALL_KS),
    }
    snapshot.update(config.metadata)
    return snapshot


def _caveats(config: EvaluationConfig, classification: str) -> list[str]:
    """把"这份报告不能说明什么"写清楚。

    评测报告最常见的误用，是拿一个只测了半条链路的数字去下全链路的结论。
    """
    notes: list[str] = []
    if config.llm_mode == _LLM_MODE_MOCK:
        notes.append(
            "生成层用的是把检索资料拼成答案的假实现（ContextEchoLLM），"
            "因此 keyword_hit_rate 反映的是**检索到的资料里有没有那条事实**，"
            "而不是模型的写作与幻觉控制能力。要度量后者请用 --llm real。"
        )
        notes.append(
            f"意图分类用的是真实规则层（classification={classification}）："
            "FAQ 规则可短路，其余落到规则候选或默认 course_query。"
            "「LLM 能否更准地判意图」不在本轮度量范围内。"
        )
    notes.append(
        "延迟是链路内部耗时（Pipeline 自测），不含 HTTP 序列化、SSE 分帧与网络往返，"
        "因此会小于用 curl / 压测工具看到的端到端耗时。"
    )
    notes.append(
        f"样本量 {config.limit or '全部'} 条，p95 实际只由少数样本决定，"
        "不要当成线上 SLA 引用；要稳定结论需要上百条量级的评测集。"
    )
    if not config.use_faq:
        notes.append("本轮关闭了 FAQ 快路径（use_faq=false），faq_hit_rate 必然为 0。")
    if config.enable_llm_evaluation:
        notes.append(
            "faithfulness 由 LLM Judge 给出，同一个回答在不同次运行可能被判成不同结果，"
            "相邻两次报告之间的微小波动不要当真。"
        )
    return notes


def evaluate_faithfulness(
    llm: LLMService,
    question: str,
    answer: str,
    documents: Sequence[object],
) -> bool | None:
    """用 LLM Judge 判断回答是否只依据资料（可选指标）。

    只在 ``--llm real`` 且 ``ENABLE_LLM_EVALUATION=true`` 时被调用：假模型判忠实度
    没有意义（它本来就只会抄资料），所以宁可不给这个数字。

    :return: ``True`` / ``False``；判不出来时返回 ``None``。
    """
    text_documents = [document for document in documents if hasattr(document, "page_content")]
    if not text_documents:
        return None
    prompt = FAITHFULNESS_JUDGE_PROMPT.format(
        context=format_context(text_documents),  # type: ignore[arg-type]
        question=question,
        answer=answer,
    )
    try:
        return parse_yes_no(llm.complete(prompt))
    except AppError as exc:
        logger.warning("忠实度判定失败 | code=%s error=%s", exc.code, exc)
        return None
