"""``src.retrieval.vectorstore`` 的单元测试。

全部使用假的查表 Embedding，不下载模型、不访问网络、不调用任何真实 API。
每个用例用独立的临时目录与集合名，互不干扰。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.retrieval.relevance import to_relevance_score
from src.retrieval.vectorstore import (
    COLLECTION_METADATA,
    DISTANCE_SPACE,
    VectorStore,
    document_id,
)
from src.utils.exceptions import RetrievalError
from tests.conftest import TableEmbeddings

#: 与查询的余弦相似度分别精确等于 1.0 / 0.7071 / 0.0 的一组向量。
#: 有了确定值，才能断言"最相关的排在第一条"而不只是"返回了三条"。
_QUERY_KEY = "查询文本"
_VECTORS: dict[str, Sequence[float]] = {
    _QUERY_KEY: [1.0, 0.0, 0.0, 0.0],
    "最相关": [1.0, 0.0, 0.0, 0.0],
    "部分相关": [0.70710678, 0.70710678, 0.0, 0.0],
    "不相关": [0.0, 0.0, 1.0, 0.0],
}

#: 期望的余弦距离（= 1 - 余弦相似度）。
_EXPECTED_DISTANCE: dict[str, float] = {
    "最相关": 0.0,
    "部分相关": 1.0 - 0.70710678,
    "不相关": 1.0,
}

_COLLECTION_NAME = "test_collection"


def _documents() -> list[Document]:
    """构造三条内容可控、metadata 完整的文档。"""
    return [
        Document(
            page_content="最相关",
            metadata={
                "course_id": "CS201",
                "name": "数据结构与算法",
                "type": "course",
                "source": "courses.json",
                "section": "course_profile",
                "credits": 4.0,
            },
        ),
        Document(
            page_content="部分相关",
            metadata={
                "course_id": "CS101",
                "name": "程序设计基础",
                "type": "course",
                "source": "courses.json",
                "section": "course_profile",
                "credits": 4.0,
            },
        ),
        Document(
            page_content="不相关",
            metadata={
                "course_id": "CS201",
                "name": "数据结构与算法",
                "type": "syllabus",
                "source": "course_syllabus.md",
                "section": "assessment",
                "credits": 4.0,
            },
        ),
    ]


@pytest.fixture
def embeddings(table_embeddings: Callable[[dict[str, Sequence[float]]], TableEmbeddings]) -> Embeddings:
    """查表假 Embedding。"""
    return table_embeddings(dict(_VECTORS))


@pytest.fixture
def store(embeddings: Embeddings, tmp_path: Path) -> VectorStore:
    """已建好索引的向量库。"""
    vector_store = VectorStore(
        embeddings=embeddings,
        persist_dir=tmp_path / "chroma",
        collection_name=_COLLECTION_NAME,
    )
    vector_store.build_index(_documents())
    return vector_store


@pytest.fixture
def empty_store(embeddings: Embeddings, tmp_path: Path) -> VectorStore:
    """空索引（集合存在但没有文档）。"""
    return VectorStore(
        embeddings=embeddings,
        persist_dir=tmp_path / "chroma",
        collection_name="empty_collection",
    )


# ---------------------------------------------------------------------------
# 建库
# ---------------------------------------------------------------------------
def test_build_index_writes_all_documents(store: VectorStore) -> None:
    """文档应当全部写入。"""
    assert store.count() == 3


def test_collection_uses_cosine_space(store: VectorStore) -> None:
    """集合必须显式配置为 cosine 空间。

    默认的 L2 空间会让"距离"的含义完全不同，而这不会报任何错。
    """
    assert store.vectorstore._collection.metadata == COLLECTION_METADATA
    assert COLLECTION_METADATA["hnsw:space"] == DISTANCE_SPACE == "cosine"


def test_build_index_is_full_rebuild(store: VectorStore) -> None:
    """重复建库应当是重建，不是追加 —— 否则删掉的课程还能被检索到。"""
    store.build_index(_documents())

    assert store.count() == 3


def test_build_index_replaces_stale_documents(store: VectorStore) -> None:
    """语料变化后重建，旧文档必须消失。"""
    store.build_index(
        [
            Document(
                page_content="最相关",
                metadata={"course_id": "CS999", "name": "新课程", "type": "course", "source": "new.json"},
            )
        ]
    )

    assert store.count() == 1
    results = store.similarity_search(_QUERY_KEY, k=10)
    assert {document.metadata["course_id"] for document in results} == {"CS999"}


def test_build_index_with_empty_list_raises(empty_store: VectorStore) -> None:
    """空文档列表要报错，不能安静地建出一个空索引。

    空索引之后所有的检索都会返回空，而拒答机制会让它看起来像"正常工作"。
    """
    with pytest.raises(RetrievalError, match="没有文档可建库"):
        empty_store.build_index([])


def test_reset_index_empties_the_collection(store: VectorStore) -> None:
    """reset 之后集合应当为空，且仍可继续使用。"""
    store.reset_index()

    assert store.count() == 0
    assert store.similarity_search(_QUERY_KEY, k=5) == []


def test_index_persists_across_instances(embeddings: Embeddings, tmp_path: Path) -> None:
    """持久化目录里的索引应当能被新实例读到。"""
    persist_dir = tmp_path / "chroma"
    first = VectorStore(
        embeddings=embeddings, persist_dir=persist_dir, collection_name=_COLLECTION_NAME
    )
    first.build_index(_documents())

    second = VectorStore(
        embeddings=embeddings, persist_dir=persist_dir, collection_name=_COLLECTION_NAME
    )

    assert second.count() == 3


def test_persist_directory_is_created_on_first_use(embeddings: Embeddings, tmp_path: Path) -> None:
    """持久化目录不存在时应当自动创建（在首次真正使用时）。"""
    target = tmp_path / "deep" / "nested" / "chroma"
    nested = VectorStore(embeddings=embeddings, persist_dir=target, collection_name="nested_col")

    nested.build_index(_documents())

    assert target.is_dir()


def test_construction_alone_has_no_side_effects(embeddings: Embeddings, tmp_path: Path) -> None:
    """只构造对象不应产生任何磁盘副作用。

    加载模型、连接向量库这类动作都是惰性的，这样"构造一个服务"就可以放心
    写在配置装配阶段，不必担心它顺带写盘或发起网络请求。
    """
    target = tmp_path / "untouched" / "chroma"

    VectorStore(embeddings=embeddings, persist_dir=target, collection_name="lazy_col")

    assert not target.exists()


def test_empty_collection_name_rejected(embeddings: Embeddings, tmp_path: Path) -> None:
    """集合名不能为空。"""
    with pytest.raises(RetrievalError, match="集合名不能为空"):
        VectorStore(embeddings=embeddings, persist_dir=tmp_path, collection_name="   ")


# ---------------------------------------------------------------------------
# 距离空间守卫
# ---------------------------------------------------------------------------
def test_existing_collection_with_wrong_distance_space_is_rejected(
    embeddings: Embeddings, tmp_path: Path
) -> None:
    """打开一个用 L2 空间建的旧集合时必须报错，而不是按余弦去解释它的距离。

    这是本模块最重要的一处防御：``get_or_create_collection`` 在集合已存在时
    会忽略本次传入的配置，于是代码按余弦解释、数据却是 L2 距离，阈值全错而
    整条链路一声不吭。
    """
    persist_dir = tmp_path / "chroma"
    legacy_name = "legacy_l2_collection"
    Chroma(
        collection_name=legacy_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "l2"},
    )

    legacy = VectorStore(
        embeddings=embeddings, persist_dir=persist_dir, collection_name=legacy_name
    )

    with pytest.raises(RetrievalError, match="距离空间") as excinfo:
        legacy.count()

    assert excinfo.value.details["expected_space"] == "cosine"
    assert excinfo.value.details["actual_space"] == "l2"


# ---------------------------------------------------------------------------
# similarit_search：结果与排序
# ---------------------------------------------------------------------------
def test_similarity_search_returns_documents(store: VectorStore) -> None:
    """基本检索应当返回 Document 对象。"""
    results = store.similarity_search(_QUERY_KEY, k=3)

    assert len(results) == 3
    assert all(isinstance(document, Document) for document in results)


def test_similarity_search_sorted_by_relevance(store: VectorStore) -> None:
    """最相关的必须排在第一条。"""
    results = store.similarity_search(_QUERY_KEY, k=3)

    assert [document.page_content for document in results] == ["最相关", "部分相关", "不相关"]


def test_similarity_search_respects_k(store: VectorStore) -> None:
    """k 限制返回条数。"""
    assert len(store.similarity_search(_QUERY_KEY, k=2)) == 2
    assert len(store.similarity_search(_QUERY_KEY, k=1)) == 1


def test_similarity_search_preserves_metadata(store: VectorStore) -> None:
    """metadata 必须原样带回来，一个键都不能少。

    引用来源、过滤、去重全靠它；丢了 course_id 的检索结果等于没有用。
    """
    results = store.similarity_search(_QUERY_KEY, k=1)
    metadata = results[0].metadata

    for key in ("course_id", "name", "credits", "type", "source", "section"):
        assert key in metadata, f"metadata 缺少 {key}"

    assert metadata["course_id"] == "CS201"
    assert metadata["credits"] == 4.0


def test_similarity_search_on_empty_index_returns_empty(empty_store: VectorStore) -> None:
    """空索引返回空列表，不抛异常 —— 由上层决定是否拒答。"""
    assert empty_store.similarity_search(_QUERY_KEY, k=5) == []


@pytest.mark.parametrize("bad_k", [0, -1])
def test_invalid_k_rejected(store: VectorStore, bad_k: int) -> None:
    """k 必须为正整数。"""
    with pytest.raises(RetrievalError, match="必须为正整数"):
        store.similarity_search(_QUERY_KEY, k=bad_k)


def test_default_k_comes_from_settings(
    store: VectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不传 k 时取 settings.top_k，而不是写死的默认值。"""
    from src.retrieval import vectorstore as vectorstore_module

    monkeypatch.setattr(vectorstore_module.settings, "top_k", 2, raising=False)

    assert len(store.similarity_search(_QUERY_KEY)) == 2


# ---------------------------------------------------------------------------
# metadata 过滤
# ---------------------------------------------------------------------------
def test_filter_by_course_id(store: VectorStore) -> None:
    """按 course_id 过滤只返回该课程的文档。"""
    results = store.similarity_search(_QUERY_KEY, k=10, filter={"course_id": "CS201"})

    assert len(results) == 2
    assert {document.metadata["course_id"] for document in results} == {"CS201"}


def test_filter_by_type(store: VectorStore) -> None:
    """按 type 区分"整门课"与"大纲章节"。"""
    results = store.similarity_search(_QUERY_KEY, k=10, filter={"type": "syllabus"})

    assert len(results) == 1
    assert results[0].metadata["section"] == "assessment"


def test_combined_filter(store: VectorStore) -> None:
    """多个条件同时生效。

    这里传的是直观的 ``{"course_id": ..., "type": ...}`` 写法。Chroma 原生
    只接受单运算符的 ``where``，多键会被它直接拒绝，因此 VectorStore 内部会
    自动包装成 ``{"$and": [...]}``——这条用例守的就是那层转换。
    """
    results = store.similarity_search(
        _QUERY_KEY, k=10, filter={"course_id": "CS201", "type": "course"}
    )

    assert len(results) == 1
    assert results[0].page_content == "最相关"


def test_three_way_filter(store: VectorStore) -> None:
    """三个及以上条件同样支持。"""
    results = store.similarity_search(
        _QUERY_KEY, k=10, filter={"course_id": "CS201", "type": "syllabus", "section": "assessment"}
    )

    assert len(results) == 1
    assert results[0].page_content == "不相关"


def test_combined_filter_with_scores(store: VectorStore) -> None:
    """带分数的检索也走同一套过滤转换。"""
    hits = store.similarity_search_with_relevance(
        _QUERY_KEY, k=10, filter={"course_id": "CS201", "type": "syllabus"}
    )

    assert len(hits) == 1
    assert hits[0][0].metadata["section"] == "assessment"


def test_explicit_and_operator_passes_through(store: VectorStore) -> None:
    """显式写 $and 时不再二次包装。"""
    results = store.similarity_search(
        _QUERY_KEY,
        k=10,
        filter={"$and": [{"course_id": "CS201"}, {"type": "course"}]},
    )

    assert len(results) == 1
    assert results[0].page_content == "最相关"


def test_explicit_or_operator_passes_through(store: VectorStore) -> None:
    """$or 同样透传，且不被当成未知 metadata 键告警。"""
    results = store.similarity_search(
        _QUERY_KEY,
        k=10,
        filter={"$or": [{"course_id": "CS101"}, {"course_id": "CS999"}]},
    )

    assert len(results) == 1
    assert results[0].metadata["course_id"] == "CS101"


def test_operator_keys_do_not_trigger_unknown_key_warning(
    store: VectorStore, caplog: pytest.LogCaptureFixture
) -> None:
    """逻辑运算符不是字段名，不该触发拼写告警。

    注意 ``$or`` 至少需要两个子表达式（Chroma 的硬性要求），
    所以这里的过滤条件写了两项。
    """
    with caplog.at_level(logging.WARNING, logger="src.retrieval.vectorstore"):
        store.similarity_search(
            _QUERY_KEY, k=10, filter={"$or": [{"course_id": "CS101"}, {"course_id": "CS999"}]}
        )

    assert caplog.text == ""


def test_filter_with_no_match_returns_empty(store: VectorStore) -> None:
    """过滤条件没有命中时返回空列表，而不是忽略过滤条件。"""
    assert store.similarity_search(_QUERY_KEY, k=10, filter={"course_id": "NOT_EXIST"}) == []


def test_filter_works_with_scores(store: VectorStore) -> None:
    """带分数的检索同样支持过滤。"""
    hits = store.similarity_search_with_score(_QUERY_KEY, k=10, filter={"type": "course"})

    assert len(hits) == 2
    assert all(document.metadata["type"] == "course" for document, _score in hits)


def test_unknown_filter_key_warns(
    store: VectorStore, caplog: pytest.LogCaptureFixture
) -> None:
    """过滤键拼错时告警。

    Chroma 对未知键不报错，只会返回空结果 —— 那是"检索突然什么都查不到"
    这类问题里最难定位的一种。
    """
    with caplog.at_level(logging.WARNING, logger="src.retrieval.vectorstore"):
        store.similarity_search(_QUERY_KEY, k=5, filter={"courseId": "CS201"})

    assert "courseId" in caplog.text


def test_known_filter_key_does_not_warn(store: VectorStore, caplog: pytest.LogCaptureFixture) -> None:
    """已知的过滤键不应产生告警。"""
    with caplog.at_level(logging.WARNING, logger="src.retrieval.vectorstore"):
        store.similarity_search(_QUERY_KEY, k=5, filter={"course_id": "CS201"})

    assert caplog.text == ""


# ---------------------------------------------------------------------------
# 分数语义
# ---------------------------------------------------------------------------
def test_search_with_score_returns_raw_distance(store: VectorStore) -> None:
    """``similarity_search_with_score`` 返回的是**距离**：越小越相关。

    这条断言把这个容易搞反的语义钉死。数值与手算的余弦距离逐条比对。
    """
    hits = store.similarity_search_with_score(_QUERY_KEY, k=3)

    assert [document.page_content for document, _score in hits] == ["最相关", "部分相关", "不相关"]
    for document, distance in hits:
        assert distance == pytest.approx(_EXPECTED_DISTANCE[document.page_content], abs=1e-5)


def test_raw_distances_are_ascending(store: VectorStore) -> None:
    """原始距离升序 —— 再次确认"越小越相关"的方向。"""
    distances = [distance for _document, distance in store.similarity_search_with_score(_QUERY_KEY, k=3)]

    assert distances == sorted(distances)


def test_search_with_relevance_returns_similarity(store: VectorStore) -> None:
    """``similarity_search_with_relevance`` 返回的是**相似度**：越大越相关。"""
    hits = store.similarity_search_with_relevance(_QUERY_KEY, k=3)

    assert [document.page_content for document, _score in hits] == ["最相关", "部分相关", "不相关"]
    for document, score in hits:
        expected = to_relevance_score(_EXPECTED_DISTANCE[document.page_content])
        assert score == pytest.approx(expected, abs=1e-5)


def test_relevance_scores_stay_within_unit_interval(store: VectorStore) -> None:
    """relevance_score 必须落在 [0, 1] 内，可直接与阈值比较。"""
    for _document, score in store.similarity_search_with_relevance(_QUERY_KEY, k=3):
        assert 0.0 <= score <= 1.0


def test_relevance_and_distance_are_consistent(store: VectorStore) -> None:
    """两种口径必须严格对应 ``relevance = 1 - distance``，否则调用方会算错。"""
    by_distance = {
        document.page_content: distance
        for document, distance in store.similarity_search_with_score(_QUERY_KEY, k=3)
    }
    by_relevance = {
        document.page_content: score
        for document, score in store.similarity_search_with_relevance(_QUERY_KEY, k=3)
    }

    assert set(by_distance) == set(by_relevance)
    for key, score in by_relevance.items():
        assert score == pytest.approx(to_relevance_score(by_distance[key]), abs=1e-9)
        assert score <= 1.0


def test_perfect_match_scores_one(store: VectorStore) -> None:
    """完全相同方向的向量应当得满分 1.0。"""
    hits = store.similarity_search_with_relevance(_QUERY_KEY, k=1)

    assert hits[0][1] == pytest.approx(1.0, abs=1e-6)
    assert hits[0][1] == pytest.approx(math.cos(0.0), abs=1e-9)


# ---------------------------------------------------------------------------
# 文档 ID
# ---------------------------------------------------------------------------
def test_document_id_is_stable() -> None:
    """同样的文档必须得到同样的 ID。"""
    document = _documents()[0]

    assert document_id(document) == document_id(_documents()[0])
    assert len(document_id(document)) == 32


def test_document_id_differs_for_different_documents() -> None:
    """不同文档的 ID 必须不同，否则重建索引时会互相覆盖。"""
    ids = {document_id(document) for document in _documents()}

    assert len(ids) == 3


def test_document_id_distinguishes_same_text_in_different_sources() -> None:
    """正文相同但来源/章节不同时，ID 也必须不同 —— 它们是两份独立的证据。"""
    base = {"course_id": "CS201", "name": "数据结构与算法", "type": "course"}
    first = Document(page_content="相同正文", metadata={**base, "source": "a.json", "section": "x"})
    second = Document(page_content="相同正文", metadata={**base, "source": "b.md", "section": "y"})

    assert document_id(first) != document_id(second)
