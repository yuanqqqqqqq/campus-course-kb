"""向量库层：基于 ChromaDB 的索引建立与相似度检索。

本模块只负责"把向量写进去、把近邻取出来"，**不做相关性判断**（那是
:mod:`src.retrieval.relevance` 的事），也不生成任何自然语言。

**距离空间为什么显式钉死为 cosine**

Chroma 的默认距离是平方欧氏距离，不是余弦。实测（详见
:mod:`src.retrieval.relevance` 的模块文档）默认配置下 `|A-B|²` 会是返回的
"分数"，与余弦相似度毫无关系。文本检索关心的是方向而非模长，因此这里在建
集合时显式写入 ``hnsw:space = "cosine"``。

**这个设置一旦写错，整条链路的阈值都会失去意义，而且不会报错。**
所以本模块做了一件防御：打开集合后会核对它实际的距离空间，不一致就抛异常，
而不是带着错误的距离继续跑。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.relevance import to_relevance_score
from src.utils.exceptions import RetrievalError
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: 本向量库使用的距离空间。**改动它必须同步修改 relevance.to_relevance_score。**
DISTANCE_SPACE: Final[str] = "cosine"

#: Chroma 集合创建时写入的配置。
COLLECTION_METADATA: Final[dict[str, str]] = {"hnsw:space": DISTANCE_SPACE}

#: 单次写入的文档条数上限。
#:
#: Chroma 对单次 add 的批量大小有上限，语料规模变大后一次性提交会被拒绝。
#: 分批写入还附带一个好处：进度可见，而不是卡在那里没有任何输出。
_BATCH_SIZE: Final[int] = 500

#: 已知可用的 metadata 过滤键。
#:
#: 仅用于**发现拼写错误**：Chroma 对不存在的过滤键不会报错，只会安静地返回
#: 空结果——那种"检索突然什么都查不到"的问题很难定位。这里不校验 ``section``
#: 的具体取值，因为大纲里未收录的章节标题会原样作为 section 键（见 loader）。
KNOWN_FILTERABLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "course_id",
        "name",
        "credits",
        "type",
        "source",
        "instructor",
        "semester",
        "prerequisites",
        "textbooks",
        "tags",
        "section",
        "section_title",
        "assessment_exam",
        "assessment_homework",
        "assessment_project",
        "chunk_index",
        "chunk_total",
    }
)


class VectorStore:
    """课程知识库的向量索引。

    典型用法：

    .. code-block:: python

        store = VectorStore()
        store.build_index(documents)

        hits = store.similarity_search_with_relevance("数据结构的先修课是什么？")
        for document, score in hits:
            print(score, document.metadata["course_id"])

    :param embedding_service: Embedding 组件。``None`` 时新建一个默认实例。
    :param embeddings: 直接指定 Embedding 实现，优先于 ``embedding_service``。
        测试里注入假实现时使用。
    :param persist_dir: 持久化目录。``None`` 时取 ``settings.chroma_persist_dir``。
    :param collection_name: 集合名。``None`` 时取 ``settings.chroma_collection_name``。
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        embeddings: Embeddings | None = None,
        persist_dir: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._embeddings = embeddings
        self._persist_dir = Path(persist_dir) if persist_dir is not None else settings.chroma_persist_dir
        self._collection_name = (
            collection_name if collection_name is not None else settings.chroma_collection_name
        )
        self._vectorstore: Chroma | None = None

        if not str(self._collection_name).strip():
            raise RetrievalError("Chroma 集合名不能为空，请检查 CHROMA_COLLECTION_NAME 配置。")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    @property
    def collection_name(self) -> str:
        """当前集合名。"""
        return self._collection_name

    @property
    def persist_dir(self) -> Path:
        """当前持久化目录。"""
        return self._persist_dir

    @property
    def vectorstore(self) -> Chroma:
        """底层 :class:`~langchain_chroma.Chroma` 实例（惰性创建）。"""
        return self._get_store()

    def build_index(self, documents: Sequence[Document]) -> None:
        """用给定文档**重建**整个索引。

        :param documents: 阶段 2 产出的文档列表。
        :raises RetrievalError: 文档列表为空。

        这是**全量重建**语义，不是增量追加：会先删除同名集合再重新写入。
        这样"改了语料重新跑一遍"的结果永远等于语料本身，而不会混入上一版
        残留的文档——增量追加最容易出的一类事故，就是删掉的课程还能被检索到。

        需要增量追加时请直接使用 ``vectorstore.add_documents``。
        """
        if not documents:
            raise RetrievalError(
                "没有文档可建库。若语料目录为空，请先确认 data/raw 下有可解析的文件。"
            )

        self.reset_index()

        store = self._get_store()
        total = len(documents)
        for start in range(0, total, _BATCH_SIZE):
            batch = list(documents[start : start + _BATCH_SIZE])
            # 显式传入稳定 ID，而不是让 Chroma 自己生成随机 ID：
            # 这样同一份语料重复写入是幂等的（覆盖而不是新增），两次索引也能
            # 按 ID 逐条比对。
            store.add_documents(batch, ids=[document_id(item) for item in batch])
            logger.info("索引进度 | 已写入 %d/%d", min(start + len(batch), total), total)

        logger.info(
            "索引建立完成 | collection=%s documents=%d persist_dir=%s",
            self._collection_name,
            self.count(),
            self._persist_dir,
        )

    def reset_index(self) -> None:
        """删除并重建一个空集合。

        底层集合被删除后，原来那个 :class:`Chroma` 实例持有的集合句柄就失效了，
        因此这里把缓存的实例一并丢弃，下一次访问时重新创建。
        """
        if self._vectorstore is not None:
            self._vectorstore.delete_collection()
            self._vectorstore = None
            logger.info("已删除既有集合 | collection=%s", self._collection_name)
        self._get_store()

    def similarity_search(
        self,
        query: str,
        k: int | None = None,
        filter: Mapping[str, Any] | None = None,
    ) -> list[Document]:
        """返回与查询最相近的文档（不含分数）。

        :param query: 查询文本。
        :param k: 返回条数。``None`` 时取 ``settings.top_k``。
        :param filter: metadata 过滤条件，例如 ``{"course_id": "CS201"}``。
        """
        _warn_unknown_filter_keys(filter)
        resolved_k = self._resolve_k(k)
        return self._get_store().similarity_search(
            query, k=resolved_k, filter=_normalize_filter(filter)
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int | None = None,
        filter: Mapping[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """返回 ``(文档, 原始距离)``。

        .. warning::

           返回的是 **Chroma 的原始距离**，不是相关性分数。在 cosine 空间下
           它等于 ``1 - 余弦相似度``，取值 ``[0, 2]``，**越小越相关**，与阈值
           的方向正好相反。

           需要与阈值比较时请改用 :meth:`similarity_search_with_relevance`，
           或自行调用 :func:`~src.retrieval.relevance.to_relevance_score` 换算。

           保留这个方法是因为"看看原始距离分布"在调阈值时很有用——直接看
           归一化后的分数会把距离的分布信息压掉。
        """
        _warn_unknown_filter_keys(filter)
        resolved_k = self._resolve_k(k)
        return self._get_store().similarity_search_with_score(
            query, k=resolved_k, filter=_normalize_filter(filter)
        )

    def similarity_search_with_relevance(
        self,
        query: str,
        k: int | None = None,
        filter: Mapping[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """返回 ``(文档, relevance_score)``。

        ``relevance_score`` 已由 :func:`~src.retrieval.relevance.to_relevance_score`
        归一化成余弦相似度，取值 ``[0, 1]``，**越大越相关**，可直接与
        ``settings.relevance_threshold`` 比较。

        这是 Pipeline 与评测脚本应当使用的检索入口。
        """
        hits = self.similarity_search_with_score(query, k=k, filter=filter)
        return [(document, to_relevance_score(distance)) for document, distance in hits]

    def count(self) -> int:
        """集合内的文档条数。"""
        return int(self._get_store()._collection.count())

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _get_store(self) -> Chroma:
        """惰性创建并缓存底层 Chroma 实例。"""
        if self._vectorstore is None:
            self._vectorstore = self._create_store()
        return self._vectorstore

    def _create_store(self) -> Chroma:
        """创建 Chroma 实例，并核对它实际使用的距离空间。"""
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        try:
            store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._resolve_embeddings(),
                persist_directory=str(self._persist_dir),
                collection_metadata=dict(COLLECTION_METADATA),
            )
        except (AttributeError, TypeError, ImportError):
            # 这几类几乎只可能是**我们自己的代码/依赖出了问题**（例如 SDK 签名变了、
            # 参数类型传错）。原样抛出，绝不翻译成"Chroma 打不开"——那会给出
            # "删掉持久化目录重建"这种误导性的运维指引，让人删掉一个正常的索引。
            logger.exception("构造 Chroma 客户端时出现代码层面的错误 | collection=%s", self._collection_name)
            raise
        except Exception as exc:
            # Chroma 对"同名集合已存在但配置不同"的处理方式在不同版本间变过
            # （有时静默复用、有时直接报错）。这里不赌它的行为：任何一种都
            # 转成同一条可操作的提示。
            raise RetrievalError(
                f"打开 Chroma 集合 {self._collection_name!r} 失败：{exc}\n"
                f"若持久化目录 {self._persist_dir} 中的同名集合是用旧配置建立的，"
                f"请删除该目录后重新建库。",
                details={"collection": self._collection_name, "persist_dir": str(self._persist_dir)},
            ) from exc

        _assert_distance_space(store, self._collection_name, self._persist_dir)
        return store

    def _resolve_embeddings(self) -> Embeddings:
        """取得 Embedding 实现：注入的优先，否则按配置构造。"""
        if self._embeddings is not None:
            return self._embeddings
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service.get_embeddings()

    @staticmethod
    def _resolve_k(k: int | None) -> int:
        """解析本次检索的 k 值。"""
        resolved = k if k is not None else settings.top_k
        if resolved < 1:
            raise RetrievalError(f"检索条数 k 必须为正整数，当前为 {resolved}。")
        return resolved


# ---------------------------------------------------------------------------
# 模块级辅助
# ---------------------------------------------------------------------------
def _assert_distance_space(store: Chroma, collection_name: str, persist_dir: Path) -> None:
    """核对集合实际使用的距离空间，不一致就报错。

    这是本模块最重要的一处防御。``get_or_create_collection`` 在集合**已存在**
    时会直接复用，**完全忽略**本次传入的 ``collection_metadata``。于是会出现
    这种情况：集合是很早以前用默认的 L2 空间建的，代码却按余弦去解释距离，
    阈值全错，而整个链路一声不吭。

    与其等到有人疑惑"为什么明明相关的内容被拒答"，不如在打开集合时就失败。
    """
    collection = getattr(store, "_collection", None)
    if collection is None:  # pragma: no cover - 取决于 langchain-chroma 内部实现
        logger.debug("无法读取集合元数据，跳过距离空间核对 | collection=%s", collection_name)
        return

    metadata = getattr(collection, "metadata", None) or {}
    actual = metadata.get("hnsw:space")
    if actual != DISTANCE_SPACE:
        raise RetrievalError(
            f"集合 {collection_name!r} 实际使用的距离空间是 {actual!r}，"
            f"而代码按 {DISTANCE_SPACE!r} 解释距离——两者不一致会让相关性阈值完全失效。\n"
            f"这通常发生在集合是用旧版本代码（或默认 L2 空间）建立的。\n"
            f"解决办法：删除持久化目录 {persist_dir} 后重新建库"
            f"（集合的距离空间一旦建立就无法修改）。",
            details={
                "collection": collection_name,
                "expected_space": DISTANCE_SPACE,
                "actual_space": actual,
                "persist_dir": str(persist_dir),
            },
        )


def _normalize_filter(filter_: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """把直观的多字段过滤条件转成 Chroma 认识的写法。

    Chroma 的 ``where`` **只允许一个运算符**：``{"course_id": "CS201", "type": "course"}``
    会被直接抛 ``ValueError: Expected where to have exactly one operator``。
    多条件必须写成 ``{"$and": [{"course_id": "CS201"}, {"type": "course"}]}``。

    这个限制很反直觉，而报错又发生在检索时刻。调用方自然会把相关条件写成
    ``{...多个键...}``，所以这里替他们转换，而不是把 Chroma 的方言泄漏到
    业务代码里。

    已经显式使用了 ``$and`` / ``$or`` 的条件原样透传，不做二次包装。
    """
    if not filter_:
        return None

    conditions = dict(filter_)
    if len(conditions) <= 1:
        return conditions
    if any(str(key).startswith("$") for key in conditions):
        # 调用方自己组织了逻辑运算符。此时多键混合是写法错误，交给 Chroma 报错，
        # 报错信息比这里猜测意图更有用。
        return conditions
    return {"$and": [{key: value} for key, value in conditions.items()]}


def _warn_unknown_filter_keys(filter_: Mapping[str, Any] | None) -> None:
    """过滤键不在已知集合内时告警（仅告警，不拦截）。"""
    if not filter_:
        return
    # $and / $or 是逻辑运算符，不是 metadata 字段名，不参与校验
    unknown = [
        key
        for key in filter_
        if not str(key).startswith("$") and key not in KNOWN_FILTERABLE_KEYS
    ]
    if unknown:
        logger.warning(
            "过滤条件包含未知的 metadata 键 | unknown=%s | 已知键=%s。"
            "Chroma 对未知键不会报错，只会返回空结果——如果检索突然查不到东西，"
            "先检查这里是不是拼错了。",
            unknown,
            sorted(KNOWN_FILTERABLE_KEYS),
        )


def document_id(document: Document) -> str:
    """为文档生成稳定的 ID。

    用内容与来源的哈希而不是 ``uuid4``：同样的语料重复建库会得到同样的 ID，
    便于比对两次索引是否一致；也避免 Chroma 自动生成随机 ID 导致重复入库时
    无法识别出是同一份内容。

    metadata 里不含文档正文，但含 course_id / section / chunk_index，与正文
    一起足以唯一标识一个切片。
    """
    fingerprint = "\u0000".join(
        [
            document.page_content,
            str(document.metadata.get("source", "")),
            str(document.metadata.get("course_id", "")),
            str(document.metadata.get("section", "")),
            str(document.metadata.get("chunk_index", "")),
        ]
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:32]
