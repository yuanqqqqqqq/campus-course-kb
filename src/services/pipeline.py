"""业务编排：把意图、FAQ、检索、相关性、生成串成一条问答链路。

**唯一链路**

.. code-block:: text

    问题
      ↓  IntentClassifier.classify()
      ├── chitchat ────────────→ 固定引导话术（不碰 FAQ / Embedding / Chroma / LLM）
      ├── faq + use_faq ─→ FAQCache.match_faq()
      │                        ├── 命中 → 直接返回（cached=true，零外部调用）
      │                        └── 未命中 → 继续往下（**不是拒答**）
      └── 其他 ────────────────→ 继续往下
      ↓  VectorStore.similarity_search_with_relevance()   （原问题，top_k 来自请求）
      ↓  RelevanceChecker.check()
      ├── 不相关 ──────────────→ 固定拒答话术（**不调用生成 LLM**）
      └── 相关 ────────────────→ LLMService.generate() → 回答 + 来源

**每条分支都在省一次调用**：闲聊省掉检索与生成，FAQ 命中省掉检索与生成，证据不足
省掉生成。这三条加起来，绝大多数请求根本不会走到最贵的那个环节。

**为什么 :meth:`ChatPipeline.prepare` 与 :meth:`ChatPipeline.run` 是分开的**

``run`` 用于一次性返回的 ``POST /api/chat``；流式接口 ``POST /api/chat/stream`` 必须
先把"分类、FAQ、检索、相关性"全部跑完——这些阶段要么直接给出答案（闲聊 / FAQ 命中 /
拒答），要么才知道该把哪些资料交给模型。而一旦开始往响应体里写 SSE 帧，就再也改不了
HTTP 状态码了。所以准备阶段必须能在流开始**之前**抛错，让 API 层返回正常的 4xx/5xx。

**错误如何向上传**

本模块**不吞异常、不返回"系统错误"字符串**。每一段都包在 :func:`_stage` 里，
失败时按类型记日志（预期内的业务异常 vs 真正的代码 bug）后原样抛出，由 API 层
按类型映射 HTTP 状态码（见 ``src/main.py`` 的异常处理器）。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Final, Protocol

from langchain_core.documents import Document

from src.config import settings
from src.generation.llm import LLMService, dedupe_sources, source_from_document
from src.retrieval.relevance import RelevanceChecker
from src.routing.classifier import IntentClassifier
from src.routing.faq_cache import FAQCache, FAQMatch
from src.schemas.chat import ChatRequest, ChatResponse, IntentType, Source, TokenUsage
from src.utils.exceptions import AppError, IndexNotReadyError
from src.utils.logging import get_logger, question_for_log

logger = get_logger(__name__)

#: 闲聊意图的引导话术。
#:
#: 不走 LLM：闲聊是唯一"答案与知识库无关"的分支，为一句寒暄付一次调用的钱没有
#: 意义，而且固定话术还有个好处——它会主动告诉用户该问什么。
CHITCHAT_REPLY: Final[str] = (
    "你好，我是校园课程知识库助手。你可以这样问我："
    "「CS201 的先修课是什么」「数据结构与算法怎么考核」「这门课主要讲什么」。"
)

#: 证据不足时的拒答话术。
#:
#: 这是**业务分支**，不是错误：检索到的资料不足以支撑回答时，宁可明说没找到，
#: 也不给模型一个编造的机会。补充"换个说法 / 确认课程编号"是给用户的下一步动作。
REFUSAL_REPLY: Final[str] = (
    "知识库中暂未找到与该问题相关的课程资料，无法回答。"
    "可以换个说法，或确认一下课程编号（例如 CS201）。"
)

#: FAQ 命中时来源的 ``type`` 取值。
FAQ_SOURCE_TYPE: Final[str] = "faq"

#: FAQ 命中时来源的 ``source`` 取值（FAQ 条目的"文件"就是它自己）。
FAQ_SOURCE_FILE: Final[str] = "faq.json"


class VectorStoreLike(Protocol):
    """Pipeline 对向量库的最小要求（真实实现是 :class:`~src.retrieval.vectorstore.VectorStore`）。

    写成 Protocol 有两个作用：一是让"注入假向量库"这件事在类型上说得通，二是把
    依赖面钉死——编排层只用到这两个方法，别的都不该碰。
    """

    def similarity_search_with_relevance(
        self,
        query: str,
        k: int | None = ...,
    ) -> list[tuple[Document, float]]:
        """返回 ``(文档, relevance_score)``，分数口径见 retrieval 层。"""
        ...

    def count(self) -> int:
        """索引里的文档条数；为 0 表示索引还没建。"""
        ...


@dataclass(frozen=True)
class PreparedAnswer:
    """准备阶段的结果：要么是可以直接返回的现成答案，要么是待生成的资料。

    ``answer`` 为 ``None`` 表示"需要调用生成层"，此时 ``question`` 与 ``documents``
    才有意义。把它显式表达出来（而不是用一个空字符串当标记），是为了让"忘了生成"
    这种错误在类型和代码上都看得出来。

    另外几个字段（``top_k`` / ``retrieval_count`` / ``is_relevant`` 等）纯粹是**给日志
    用的**：排查"为什么这条答得不好"时，需要知道当时检索了几条、最高分多少、判定
    结果是什么——这些信息如果只存在于中间变量里，日志就只能记一句"问答完成"。
    """

    route: IntentType
    cached: bool
    answer: str | None
    sources: list[Source] = field(default_factory=list)
    question: str = ""
    documents: list[Document] = field(default_factory=list)
    relevance_score: float | None = None
    top_k: int = 0
    retrieval_count: int = 0
    is_relevant: bool | None = None
    relevance_reason: str = ""

    @property
    def needs_generation(self) -> bool:
        """是否还需要调用 LLM 生成。"""
        return self.answer is None

    @property
    def relevance_label(self) -> str:
        """相关性判定的结果标签，写进日志用。"""
        if self.cached:
            return "not_run(faq_hit)"
        if self.route is IntentType.CHITCHAT:
            return "not_run(chitchat)"
        if self.is_relevant is None:
            return "not_run"
        return "relevant" if self.is_relevant else "insufficient"


@dataclass
class ChatPipeline:
    """问答链路的编排者。

    :param classifier: 意图分类器。
    :param faq_cache: FAQ 快路径。
    :param vector_store: 向量库（只需实现 :class:`VectorStoreLike`）。
    :param relevance: 相关性判定器。
    :param llm: 生成层。

    所有依赖都是构造注入的：真实实现由 :func:`build_pipeline` 组装，测试注入假实现。
    这些对象都是**有状态且构建昂贵**的（Chroma 连接、Embedding 模型、HTTP 客户端），
    因此必须在应用启动时建一次、全程复用，绝不能每个请求新建一份。
    """

    classifier: IntentClassifier
    faq_cache: FAQCache
    vector_store: VectorStoreLike
    relevance: RelevanceChecker
    llm: LLMService

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def run(self, request: ChatRequest) -> ChatResponse:
        """跑完整条链路并返回一次性响应。

        :param request: 已经过 Pydantic 校验的请求。
        :return: :class:`~src.schemas.chat.ChatResponse`。
        :raises AppError: 各阶段的预期失败，由 API 层映射成 HTTP 状态码。
        :raises Exception: 真正的代码 bug 原样向上传播，由 FastAPI 兜底成 500。
        """
        started_at = perf_counter()
        prepared = self.prepare(request)

        usage = None
        llm_latency_ms: float | None = None
        if prepared.needs_generation:
            generation_started = perf_counter()
            with _stage("generation"):
                generated = self.llm.generate(prepared.question, prepared.documents)
            llm_latency_ms = _elapsed_ms(generation_started)
            answer = generated.answer
            usage = generated.usage
        else:
            answer = prepared.answer or ""

        latency_ms = _elapsed_ms(started_at)
        _log_completion(prepared, request, latency_ms, llm_latency_ms, usage)
        return ChatResponse(
            answer=answer,
            sources=prepared.sources,
            route=prepared.route,
            cached=prepared.cached,
            latency_ms=latency_ms,
        )

    def prepare(self, request: ChatRequest) -> PreparedAnswer:
        """跑完"分类 → FAQ → 检索 → 相关性"，返回现成答案或待生成资料。

        流式接口与 :meth:`run` 共用这一步：区别只在拿到结果之后是直接返回，还是
        把它交给 :meth:`stream_answer` 逐片段产出。

        :raises IndexNotReadyError: 向量库是空的（还没建库）。
        :raises ConfigurationError: 配置或鉴权问题。
        :raises RetrievalError: 检索失败。
        """
        question = request.question.strip()
        # 请求里没写 top_k 就用服务端配置（默认 5）。**不要**在 ChatRequest 里写死
        # 默认值：那样运维改了 TOP_K 却不生效，而启动日志还打印着改后的值。
        top_k = request.top_k if request.top_k is not None else settings.top_k

        with _stage("intent"):
            intent = self.classifier.classify(question)

        # ---- 闲聊：直接给引导话术，不碰 FAQ / Embedding / Chroma / LLM ----
        if intent.intent is IntentType.CHITCHAT:
            logger.info("问答路由 | route=chitchat 直接返回引导话术")
            return PreparedAnswer(
                route=intent.intent,
                cached=False,
                answer=CHITCHAT_REPLY,
                question=question,
                top_k=top_k,
            )

        # ---- FAQ：命中即返回；未命中继续走 RAG（不拒答） ----
        if intent.intent is IntentType.FAQ and request.use_faq:
            with _stage("faq"):
                match = self.faq_cache.match_faq(question)
            if match is not None:
                logger.info("问答路由 | route=faq cached=true faq_id=%s", match.entry.id)
                return PreparedAnswer(
                    route=intent.intent,
                    cached=True,
                    answer=match.entry.answer,
                    sources=[_faq_source(match)],
                    question=question,
                    top_k=top_k,
                )
            logger.info("问答路由 | route=faq 未命中 FAQ，继续走 RAG")

        # ---- 检索 ----
        with _stage("retrieval"):
            hits = self.vector_store.similarity_search_with_relevance(question, k=top_k)
            if not hits and self.vector_store.count() == 0:
                # 区分"没匹配上"和"库压根是空的"：前者是正常的业务结果（拒答），
                # 后者是运维问题（忘记建库），报出来比装成一次拒答有用得多。
                raise IndexNotReadyError(
                    "向量库为空，无法检索。请先运行 scripts/ingest.py 建库。",
                    details={"collection": settings.chroma_collection_name},
                )

        # ---- 相关性：不达标直接拒答，不调用生成 LLM ----
        with _stage("relevance"):
            verdict = self.relevance.check(question, hits)

        if not verdict.is_relevant:
            logger.info(
                "问答路由 | route=%s 证据不足，拒答 | %s",
                intent.intent.value,
                verdict.reason,
            )
            return PreparedAnswer(
                route=intent.intent,
                cached=False,
                answer=REFUSAL_REPLY,
                question=question,
                relevance_score=verdict.best_score,
                top_k=top_k,
                retrieval_count=len(hits),
                is_relevant=verdict.is_relevant,
                relevance_reason=verdict.reason,
            )

        # ---- 相关：交给生成层 ----
        documents = [document for document, _score in hits]
        logger.info(
            "问答路由 | route=%s 进入生成 | retrieval_count=%d best_score=%.4f",
            intent.intent.value,
            len(documents),
            verdict.best_score or 0.0,
        )
        return PreparedAnswer(
            route=intent.intent,
            cached=False,
            answer=None,
            sources=_sources_with_scores(hits),
            question=question,
            documents=documents,
            relevance_score=verdict.best_score,
            top_k=top_k,
            retrieval_count=len(hits),
            is_relevant=verdict.is_relevant,
            relevance_reason=verdict.reason,
        )

    def stream_answer(self, prepared: PreparedAnswer) -> Iterator[str]:
        """把准备结果转成文本片段流，供 API 层包成 SSE。

        现成答案（闲聊 / FAQ 命中 / 拒答）也走同一条路：产出**一个**片段。这样
        API 层只有一种分支——无论是哪种路由，都是"先来若干片段，然后结束"。

        :raises AppError: 生成阶段的失败。此时响应头早已发出，调用方只能发一条
            错误事件，不能再改状态码。
        """
        if not prepared.needs_generation:
            yield prepared.answer or ""
            return

        with _stage("generation_stream"):
            yield from self.llm.generate_stream(prepared.question, prepared.documents)


def build_pipeline(
    *,
    classifier: IntentClassifier | None = None,
    faq_cache: FAQCache | None = None,
    vector_store: VectorStoreLike | None = None,
    relevance: RelevanceChecker | None = None,
    llm: LLMService | None = None,
) -> ChatPipeline:
    """组装生产用的 :class:`ChatPipeline`。

    每个组件都允许外部注入（测试、对比实验用），缺省时才按配置构造。**所有构造都是
    惰性的**： ``VectorStore`` 不在构造时连接 Chroma，``LLMService`` 与
    ``EmbeddingService`` 也不在构造时建客户端或加载模型。所以本函数可以在应用启动时
    放心调用，不会拖慢启动、更不会因为缺 API Key 而让服务起不来。

    唯一会真正读盘的是 FAQ：:meth:`~src.routing.faq_cache.FAQCache.load` 会读
    ``data/faq.json``。文件坏了就启动失败——这正是我们想要的（见 FAQCache 的文档）。
    """
    resolved_llm = llm if llm is not None else LLMService()
    resolved_faq = faq_cache if faq_cache is not None else _load_faq_cache()

    if relevance is None:
        # 开启 LLM 复核时把复核回调接到生成层；关闭时（默认）不注入任何回调，
        # 这条路径上自然不会产生额外调用。
        judge = resolved_llm.judge_relevance if settings.enable_llm_relevance_check else None
        relevance = RelevanceChecker(judge=judge)

    if vector_store is None:
        vector_store = _build_vector_store()

    return ChatPipeline(
        # 分类器要的是"给一段提示词、还我一段原始文本"的回调。把生成层的
        # LLMService.complete 直接接上即可：提示词由路由层自己拼（那是它的业务），
        # 这里只借用"能拿到原始文本"这一个能力。
        classifier=classifier
        if classifier is not None
        else IntentClassifier(llm=resolved_llm.complete),
        faq_cache=resolved_faq,
        vector_store=vector_store,
        relevance=relevance,
        llm=resolved_llm,
    )


# ---------------------------------------------------------------------------
# 组装辅助
# ---------------------------------------------------------------------------
def _load_faq_cache() -> FAQCache:
    """构造并加载 FAQ 缓存。"""
    cache = FAQCache()
    cache.load()
    return cache


def _build_vector_store() -> VectorStoreLike:
    """构造向量库。延迟导入：``chromadb`` 的导入不便宜，而没有它的时候
    （例如只测 FAQ 路径）不该被迫加载。"""
    from src.retrieval.vectorstore import VectorStore

    return VectorStore()


def _faq_source(match: FAQMatch) -> Source:
    """把 FAQ 命中转成一条来源。

    ``name`` 留空是刻意的：FAQ 条目里没有课程名，为它编一个（或去猜）不如老实留空。
    """
    return Source(
        course_id=match.entry.course_id,
        name=None,
        type=FAQ_SOURCE_TYPE,
        source=FAQ_SOURCE_FILE,
        section=match.entry.id,
        score=None,
    )


def _sources_with_scores(hits: Sequence[tuple[Document, float]]) -> list[Source]:
    """把 ``(文档, 分数)`` 检索结果转成去重后的来源列表。

    用携带相关性分数的构造方式（而不是生成层的
    :func:`~src.generation.llm.build_sources`），是为了让响应里的来源能按相关度
    排序、前端也能显示"这条有多相关"。检索结果已按分数降序，去重保留首条即最高分。
    """
    return dedupe_sources(
        source_from_document(document, score=score) for document, score in hits
    )


def _format_score(score: float | None) -> str:
    """把分数格式化成日志里好读的形式。"""
    return "无" if score is None else f"{score:.4f}"


def _log_completion(
    prepared: PreparedAnswer,
    request: ChatRequest,
    latency_ms: float,
    llm_latency_ms: float | None,
    usage: TokenUsage | None,
) -> None:
    """打一条**结构化**的完成日志。

    这条日志是排查单个请求的主入口，所以字段是照着"想知道什么"挑的，而不是照着重
    要性挑的：走了哪条路（route/cached）、当时检索条件是什么（top_k）、检索回来几条
    （retrieval_count）、证据够不够（relevance/best_score）、有没有花钱调模型
    （llm_called/llm_latency_ms/tokens）、一共多久（latency_ms）。

    问题文本走 :func:`~src.utils.logging.question_for_log`：默认只记长度，
    ``LOG_REQUEST_CONTENT=true`` 时才记原文。
    """
    logger.info(
        "问答完成 | route=%s cached=%s top_k=%d retrieval_count=%d relevance=%s "
        "best_score=%s llm_called=%s llm_latency_ms=%s latency_ms=%.1f tokens=%s question=%s",
        prepared.route.value,
        str(prepared.cached).lower(),
        prepared.top_k or request.top_k,
        prepared.retrieval_count,
        prepared.relevance_label,
        _format_score(prepared.relevance_score),
        str(llm_latency_ms is not None).lower(),
        "无" if llm_latency_ms is None else f"{llm_latency_ms:.1f}",
        latency_ms,
        usage.total_tokens if usage is not None else "无",
        question_for_log(request.question),
    )


def _elapsed_ms(started_at: float) -> float:
    """把 :func:`time.perf_counter` 的起点换算成毫秒耗时。

    用 ``perf_counter`` 而不是 ``time.time``：前者是单调时钟，不受系统时间被
    NTP 校正、手动改时间或夏令时影响——用 ``time.time`` 测延迟，偶尔会得到一个
    负数或几万毫秒，而这种脏数据会安静地混进监控指标里。
    """
    return round((perf_counter() - started_at) * 1000, 3)


# ---------------------------------------------------------------------------
# 阶段埋点
# ---------------------------------------------------------------------------
@contextmanager
def _stage(name: str) -> Iterator[None]:
    """标记"当前在跑哪一段"，失败时按类型记日志后**原样抛出**。

    分类：

    - :class:`~src.utils.exceptions.AppError`（预期内的业务失败）——
      记 ERROR 带上错误码，抛给 API 层映射状态码；
    - 其他异常（大概率是代码 bug）——记 ERROR 带堆栈，抛给 FastAPI 兜底成 500。

    刻意**不写** ``except Exception: return "系统错误"``：那会把"向量库连不上"
    和"代码写错了"变成同一种含糊的回复，线上除了猜没有别的办法。
    """
    try:
        yield
    except AppError as exc:
        logger.error(
            "Pipeline 阶段失败（预期内） | stage=%s code=%s error=%s",
            name,
            exc.code,
            exc,
        )
        raise
    except Exception:
        logger.exception("Pipeline 阶段失败（未预期，按 500 处理） | stage=%s", name)
        raise
