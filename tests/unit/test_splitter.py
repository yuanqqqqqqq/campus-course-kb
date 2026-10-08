"""``src.ingestion.splitter`` 的单元测试。

关注四件事：该切的时候切、不该切的时候不切、切完以后 course_id 与课程标题
都还在、任何情况下都不篡改调用方传进来的文档。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.documents import Document

from src.ingestion.splitter import CourseDocumentSplitter
from src.schemas.course import Course

#: 造一段"由完整句子组成、没有换行"的中文正文，用于验证切点落在句末。
_SENTENCE = "这是第{n}句用于测试的句子，内容本身没有实际含义但长度接近真实文本。"


def _long_body(sentences: int = 60) -> str:
    """生成足够长的正文。"""
    return "".join(_SENTENCE.format(n=index) for index in range(sentences))


def _long_document_body(sentences: int = 60, prefix: str = "") -> str:
    """带前缀的正文（用于区分段落）。"""
    return prefix + _long_body(sentences)


def _extract_body(piece: Document) -> str:
    """剥掉切片的上下文前缀，只留被切分的正文部分。

    前缀 = 课程抬头 + ``## 教学内容``，这些是每个切片都会重复的内容，
    断言句边界时要把它们排除掉。
    """
    return piece.page_content.split("## 教学内容", maxsplit=1)[1].strip()


@pytest.fixture
def splitter() -> CourseDocumentSplitter:
    """默认参数的切分器。"""
    return CourseDocumentSplitter()


# ---------------------------------------------------------------------------
# 不该切的时候不切
# ---------------------------------------------------------------------------
def test_short_document_is_returned_unchanged(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """长度合适的章节必须原样保留，不能为了"统一"而硬切一刀。"""
    document = build_course_document(body="这是一段长度合适的正文。")

    pieces = splitter.split([document])

    assert len(pieces) == 1
    assert pieces[0].page_content == document.page_content


def test_short_document_gets_chunk_locator(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """未切分的文档也要补上定位字段，让下游逻辑不必区分两种情况。"""
    document = build_course_document(body="短正文。")

    piece = splitter.split([document])[0]

    assert piece.metadata["chunk_index"] == 0
    assert piece.metadata["chunk_total"] == 1


def test_document_exactly_at_limit_is_not_split(
    build_course_document: Callable[..., Document],
) -> None:
    """边界情况：正文长度正好等于 chunk_size 时不切。"""
    document = build_course_document(body="短")
    splitter = CourseDocumentSplitter(chunk_size=len(document.page_content), chunk_overlap=10)

    pieces = splitter.split([document])

    assert len(pieces) == 1
    assert pieces[0].page_content == document.page_content


# ---------------------------------------------------------------------------
# 过长文档的二次切分
# ---------------------------------------------------------------------------
def test_long_document_is_split(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """超长章节必须被切开。"""
    document = build_course_document(body=_long_document_body())

    pieces = splitter.split([document])

    assert len(pieces) > 1


def test_every_piece_respects_chunk_size(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """任何切片都不能超过 chunk_size —— 超出会直接顶爆 Embedding 的输入窗口。"""
    document = build_course_document(body=_long_document_body())

    pieces = splitter.split([document])

    for piece in pieces:
        assert len(piece.page_content) <= splitter.chunk_size, (
            f"第 {piece.metadata['chunk_index']} 片长度 {len(piece.page_content)} 超限"
        )


def test_split_prefers_sentence_boundaries(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """切点应落在句子边界上，而不是把一句话劈成两半。

    正文刻意不含换行，迫使分块器只能退到句号层级切分；因此每个切片去掉
    上下文前缀之后都应当以句号结尾。

    这里曾经出过一个真实缺陷：``keep_separator`` 传 ``True`` 时，分隔符会被
    贴到**下一片**的开头，于是每片都以半句话结尾、下一片以孤零零的句号开头。
    改成 ``"end"`` 之后才符合预期，这条用例就是那个回归的防线。
    """
    document = build_course_document(body=_long_document_body())

    pieces = splitter.split([document])
    assert len(pieces) > 1

    for piece in pieces:
        body = _extract_body(piece)
        assert body.endswith("。"), f"切片未落在句末：…{body[-30:]!r}"


def test_split_pieces_do_not_start_with_stray_separator(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """切片不能以孤立的标点开头。

    分块器把分隔符留在前一片末尾，后一片才不会以"。"、"、"这种没有主语
    的标点开场。
    """
    document = build_course_document(body=_long_document_body())

    pieces = splitter.split([document])

    assert len(pieces) > 1
    for piece in pieces:
        body = _extract_body(piece)
        assert body[0] not in "。！？；，、", f"切片以孤立标点开头：{body[:20]!r}"


def test_no_sentence_is_cut_in_half(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """每一句都必须完整地落在某一个切片里，不能被拦腰截断。"""
    document = build_course_document(body=_long_document_body(sentences=40))

    pieces = splitter.split([document])
    joined = "".join(_extract_body(piece) for piece in pieces)

    # 重叠会让某些句子重复出现，但不允许出现"缺了后半句"的句子
    for index in range(40):
        sentence = _SENTENCE.format(n=index)
        assert sentence in joined, f"句子 {index} 被切断了"


def test_chunk_index_and_total_are_consistent(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """索引应从 0 连续递增，总数一致，且没有任何正文丢失。"""
    document = build_course_document(body=_long_document_body())

    pieces = splitter.split([document])
    total = len(pieces)

    assert [piece.metadata["chunk_index"] for piece in pieces] == list(range(total))
    assert all(piece.metadata["chunk_total"] == total for piece in pieces)

    # 重叠会让还原不精确，但每句话都应当至少出现在某一片里
    joined = "\n".join(piece.page_content for piece in pieces)
    for index in range(20):
        assert f"这是第{index}句" in joined


# ---------------------------------------------------------------------------
# 切分后上下文不能丢
# ---------------------------------------------------------------------------
def test_split_pieces_keep_course_context_header(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """每个子片段都要重新贴上课程抬头。"""
    document = build_course_document(body=_long_document_body())

    pieces = splitter.split([document])

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.page_content.startswith("课程：数据结构与算法（CS201）")
        assert "先修课程：CS101" in piece.page_content


def test_split_pieces_keep_course_id(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """course_id 是引用主键，任何情况下都不能丢。"""
    document = build_course_document(course_id="CS302", body=_long_document_body())

    pieces = splitter.split([document])

    assert len(pieces) > 1
    assert all(piece.metadata["course_id"] == "CS302" for piece in pieces)


def test_split_pieces_keep_section_metadata(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """section / section_title 要完整继承。"""
    document = build_course_document(
        section="content", section_title="教学内容", body=_long_document_body()
    )

    pieces = splitter.split([document])

    for piece in pieces:
        assert piece.metadata["section"] == "content"
        assert piece.metadata["section_title"] == "教学内容"


def test_all_original_metadata_is_preserved(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """原始 metadata 一个键都不能少。"""
    document = build_course_document(
        body=_long_document_body(), extra_metadata={"custom_key": "custom_value"}
    )

    pieces = splitter.split([document])

    assert len(pieces) > 1
    for piece in pieces:
        for key, value in document.metadata.items():
            assert piece.metadata[key] == value, f"{key} 在切分后丢失或被改写"


def test_header_length_does_not_eat_into_limit(
    build_course_document: Callable[..., Document],
) -> None:
    """抬头很长时，正文可用宽度要相应收缩，总长度仍不超限。"""
    document = build_course_document(
        name="这是一个名字特别长的课程名称用于测试抬头占用空间的边界情况",
        body=_long_document_body(),
    )
    splitter = CourseDocumentSplitter(chunk_size=300, chunk_overlap=30)

    pieces = splitter.split([document])

    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece.page_content) <= 300


def test_structured_course_document_uses_header_only_prefix(
    splitter: CourseDocumentSplitter, course_payload: Callable[..., dict[str, Any]]
) -> None:
    """结构化课程文档的正文里没有 "## 课程总览" 这一行，前缀应退化到只有课程抬头。

    这是 :func:`~src.ingestion.splitter._resolve_context_prefix` 的两级降级
    路径。若实现得不对，抬头会被重复拼进每个切片（正文里出现两遍课程编号），
    而这种情况不会被"长度不超限"之类的断言发现。
    """
    course = Course.model_validate(course_payload())
    document = Document(
        page_content=course.to_document_text(source="courses.json") + "\n\n" + _long_body(),
        metadata=course.to_metadata(source="courses.json"),
    )

    pieces = splitter.split([document])

    assert len(pieces) > 1
    for piece in pieces:
        # 课程抬头在每片里恰好出现一次，不能重复
        assert piece.page_content.count("课程：数据结构与算法（CS201）") == 1


# ---------------------------------------------------------------------------
# 无副作用
# ---------------------------------------------------------------------------
def test_input_documents_are_not_mutated(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """切分不能就地改写调用方传进来的文档。

    这种副作用在流水线里极难排查：出问题的现场早就在几步之前被改掉了。
    """
    document = build_course_document(body=_long_document_body())
    original_metadata = dict(document.metadata)
    original_content = document.page_content

    splitter.split([document])

    assert document.metadata == original_metadata
    assert "chunk_index" not in document.metadata
    assert document.page_content == original_content


def test_returned_documents_are_new_objects(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """返回的应当是新对象，不是入参本身。"""
    document = build_course_document(body="短正文。")

    piece = splitter.split([document])[0]

    assert piece is not document


# ---------------------------------------------------------------------------
# 边界与参数校验
# ---------------------------------------------------------------------------
def test_empty_input_returns_empty(splitter: CourseDocumentSplitter) -> None:
    """空输入返回空列表，不报错。"""
    assert splitter.split([]) == []


def test_multiple_documents_are_all_processed(
    splitter: CourseDocumentSplitter, build_course_document: Callable[..., Document]
) -> None:
    """多文档输入时每篇都独立处理，索引互不影响。"""
    documents = [
        build_course_document(body="短正文一。"),
        build_course_document(course_id="CS302", body=_long_document_body()),
        build_course_document(course_id="CS301", body="短正文二。"),
    ]

    pieces = splitter.split(documents)

    # 每篇的第一个切片 index 都应当是 0
    starts = [piece for piece in pieces if piece.metadata["chunk_index"] == 0]
    assert len(starts) == 3
    assert {piece.metadata["course_id"] for piece in pieces} == {"CS201", "CS302", "CS301"}


def test_document_without_course_identity_still_splits(
    splitter: CourseDocumentSplitter,
) -> None:
    """没有课程身份信息时不拼抬头，但正文仍要能切分（不能崩）。"""
    document = Document(page_content=_long_body(), metadata={"source": "loose.md"})

    pieces = splitter.split([document])

    assert len(pieces) > 1
    assert all(piece.page_content.strip() for piece in pieces)


@pytest.mark.parametrize("bad_chunk_size", [0, -1, -100])
def test_non_positive_chunk_size_rejected(bad_chunk_size: int) -> None:
    """chunk_size 必须为正。"""
    with pytest.raises(ValueError, match="chunk_size 必须为正数"):
        CourseDocumentSplitter(chunk_size=bad_chunk_size)


def test_negative_overlap_rejected() -> None:
    """overlap 不能为负。"""
    with pytest.raises(ValueError, match="chunk_overlap 不能为负数"):
        CourseDocumentSplitter(chunk_size=100, chunk_overlap=-1)


def test_overlap_not_smaller_than_chunk_size_rejected() -> None:
    """overlap 必须小于 chunk_size，否则切分会陷入死循环。"""
    with pytest.raises(ValueError, match="必须小于"):
        CourseDocumentSplitter(chunk_size=100, chunk_overlap=100)


def test_too_small_chunk_size_for_header_raises(
    build_course_document: Callable[..., Document],
) -> None:
    """chunk_size 小到装不下抬头时，要报错而不是产出一堆碎片。"""
    document = build_course_document(name="名字很长的课程" * 5, body=_long_document_body())
    splitter = CourseDocumentSplitter(chunk_size=80, chunk_overlap=10)

    with pytest.raises(ValueError, match="不足以切分正文"):
        splitter.split([document])


def test_properties_expose_configuration() -> None:
    """公开的配置属性应当与构造参数一致。"""
    splitter = CourseDocumentSplitter(chunk_size=500, chunk_overlap=50)

    assert splitter.chunk_size == 500
    assert splitter.chunk_overlap == 50
