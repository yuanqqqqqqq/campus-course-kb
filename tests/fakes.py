"""测试共用的假实现（不许联网、不许读真语料）。

放在 ``tests/fakes.py`` 而不是 conftest 里，是因为 unit 与 integration 两处都要用到
同一批假对象；写两份的话，改了一个接口就会漏掉另一个测试文件。

**为什么必须用假的**

``pytest`` 一次要跑几百条用例，其中任何一条真的去连 DeepSeek 或加载 BGE 模型，
整套测试就会从"秒级"变成"分钟级"，而且一旦网络抖动就红一片——那种红没有任何信息量。
这里用假实现换掉的正是三类"慢且不稳"的依赖：网络（LLM）、模型（Embedding）、
磁盘索引（Chroma）。真正的逻辑（意图路由、FAQ 匹配、阈值判定、Prompt 拼装）仍然跑
真实代码。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence

from langchain_core.documents import Document

from src.generation.llm import build_sources
from src.routing.classifier import ClassifiedBy, IntentResult
from src.schemas.chat import GenerationResult, IntentType, TokenUsage
from src.utils.exceptions import AppError, GenerationError


class FakeIntentClassifier:
    """按问题给出预设意图的假分类器（替代 :class:`IntentClassifier`）。

    默认对未登记的问题返回 ``course_query``——这是真实分类器在拿不准时的兜底方向，
    假实现跟着它走，测试里的分支才和生产一致。

    :param intents: ``{问题: 意图}`` 映射。
    :param default: 未登记问题的意图。
    """

    def __init__(
        self,
        intents: Mapping[str, IntentType] | None = None,
        default: IntentType = IntentType.COURSE_QUERY,
    ) -> None:
        self._intents = dict(intents or {})
        self._default = default
        self.questions: list[str] = []

    def classify(self, question: str) -> IntentResult:
        """记录问题并返回预设意图。"""
        self.questions.append(question)
        intent = self._intents.get(question, self._default)
        return IntentResult(
            intent=intent,
            confidence=0.9,
            classified_by=ClassifiedBy.RULE,
            reason="测试用的假分类器",
        )


class FakeVectorStore:
    """假向量库（替代 :class:`~src.retrieval.vectorstore.VectorStore`）。

    :param hits: ``(文档, relevance_score)`` 列表，按原样返回。
    :param total: ``count()`` 的返回值；默认等于 ``len(hits)``，显式传 0 可以模拟
        "索引还没建"。
    :param error: 设了就让检索抛这个异常，用来测错误传播。
    """

    def __init__(
        self,
        hits: Sequence[tuple[Document, float]] | None = None,
        total: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._hits = list(hits or [])
        self._total = len(self._hits) if total is None else total
        self._error = error
        self.queries: list[tuple[str, int | None]] = []

    def similarity_search_with_relevance(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """记录查询参数并返回预设结果。"""
        self.queries.append((query, k))
        if self._error is not None:
            raise self._error
        return list(self._hits)

    def count(self) -> int:
        """索引条数。"""
        return self._total


class FakeLLMService:
    """假生成层（替代 :class:`~src.generation.llm.LLMService`）。

    :param answer: 非流式回答。
    :param pieces: 流式片段；``None`` 时按标点切分 ``answer``。
    :param error: 设了就让生成抛这个异常。
    """

    def __init__(
        self,
        answer: str = "根据课程资料（CS201），该课程的先修课程是程序设计基础（CS101）。",
        pieces: Sequence[str] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._answer = answer
        self._pieces = list(pieces) if pieces is not None else [answer]
        self._error = error
        self.calls: list[tuple[str, list[Document]]] = []
        self.stream_calls: list[tuple[str, list[Document]]] = []

    @property
    def call_count(self) -> int:
        """生成被调用的总次数（含流式）。"""
        return len(self.calls) + len(self.stream_calls)

    def generate(self, question: str, context: Sequence[Document]) -> GenerationResult:
        """返回预设回答，并记录收到的资料。"""
        self.calls.append((question, list(context)))
        if self._error is not None:
            raise self._error
        return GenerationResult(
            answer=self._answer,
            sources=build_sources(context),
            usage=TokenUsage(
                prompt_tokens=100, completion_tokens=20, total_tokens=120, estimated=False
            ),
            model="fake-model",
        )

    def generate_stream(self, question: str, context: Sequence[Document]) -> Iterator[str]:
        """逐个产出预设片段。"""
        self.stream_calls.append((question, list(context)))
        if self._error is not None:
            raise self._error
        yield from self._pieces


def make_document(
    *,
    course_id: str = "CS201",
    name: str = "数据结构与算法",
    content: str = "先修课程：CS101",
    source: str = "course_syllabus.md",
    section: str = "课程基本信息",
    doc_type: str = "syllabus",
) -> Document:
    """造一份与真实检索结果同构的文档。"""
    return Document(
        page_content=content,
        metadata={
            "course_id": course_id,
            "name": name,
            "source": source,
            "section_title": section,
            "type": doc_type,
        },
    )


class ContextEchoLLM:
    """**把检索到的资料原样拼成答案**的假生成层（阶段 7 评测默认用它）。

    存在的理由：评测要能重复跑、且默认不花钱，但又不能随便编一个回答——那样
    "答案关键词命中率"就变成了在测假实现自己。这个假模型的做法是
    "把 Context 里最相关的那几条抄出来并标上出处"，于是指标变成了一件有意义的事：

    - 关键词命中率 ⇒ **检索到的资料里到底有没有那条事实**（证据质量）
    - 忠实度（开 LLM Judge 时） ⇒ 回答是否只用了资料

    必须承认它**测不了**"真实模型的写作能力与幻觉倾向"——那两件事只有
    ``--llm real``（真实 DeepSeek）才能度量，报告里会明确标注当前用的是哪一种。
    """

    def __init__(
        self, *, max_documents: int | None = None, model_name: str = "mock-context-echo"
    ) -> None:
        self._max_documents = max_documents
        self.model_name = model_name
        self.calls: list[str] = []

    @property
    def call_count(self) -> int:
        """被调用次数。"""
        return len(self.calls)

    def generate(self, question: str, context: Sequence[Document]) -> GenerationResult:
        """把资料拼成答案（带出处），不调用任何外部服务。"""
        self.calls.append(question)
        answer = self._compose(context)
        return GenerationResult(
            answer=answer,
            sources=build_sources(context),
            usage=None,
            model=self.model_name,
        )

    def generate_stream(self, question: str, context: Sequence[Document]) -> Iterator[str]:
        """逐句产出，供流式链路使用。"""
        self.calls.append(question)
        answer = self._compose(context)
        pieces = [piece for piece in answer.split("。") if piece]
        for index, piece in enumerate(pieces):
            yield piece if index == len(pieces) - 1 else f"{piece}。"

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """兜底实现：评测链路里分类器不注入 LLM，正常不会走到这里。"""
        raise NotImplementedError(
            "ContextEchoLLM 不参与意图分类：评测默认用规则分类（见 evaluate.py 的说明）"
        )

    def _compose(self, context: Sequence[Document]) -> str:
        """把资料拼成一段带出处的答案。

        **默认读整个 Context，而不是前 N 条。** 这一点当初写错过一次：只读前 3 条时，
        「这条事实在不在检索结果里」会被"它排第 3 还是第 5"左右——同一个检索结果，
        改一个纯属虚构的 N 就能让关键词命中率上下跳。指标的语义是"资料里有没有这条
        事实"，那就该把资料全读进去；排序敏感性由 Recall@K 负责度量。
        """
        if not context:
            return "知识库中暂未找到相关信息。"

        selected = context if self._max_documents is None else context[: self._max_documents]
        blocks: list[str] = []
        for document in selected:
            metadata = document.metadata or {}
            name = str(metadata.get("name") or "").strip()
            course_id = str(metadata.get("course_id") or "").strip()
            header = f"根据《{name}》（{course_id}）课程资料" if name or course_id else "根据课程资料"
            body = " ".join(document.page_content.split())
            blocks.append(f"{header}：{body}。")
        return " ".join(blocks)


def make_error(message: str = "假的失败") -> AppError:
    """造一个项目自有的业务异常（默认是 :class:`~src.utils.exceptions.GenerationError`）。"""
    return GenerationError(message)


__all__ = [
    "ContextEchoLLM",
    "FakeIntentClassifier",
    "FakeLLMService",
    "FakeVectorStore",
    "make_document",
    "make_error",
]
