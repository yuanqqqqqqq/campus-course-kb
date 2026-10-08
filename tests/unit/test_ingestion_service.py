"""``src.ingestion.service`` 的单元测试。

除端到端结果外，重点验证两个**编排层独有**的职责：依赖注入真的生效、以及
出口一致性校验真的会拦住缺字段的文档。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document

from src.config import PROJECT_ROOT
from src.ingestion.loader import DocumentLoader
from src.ingestion.service import REQUIRED_METADATA_KEYS, IngestionService
from src.ingestion.splitter import CourseDocumentSplitter
from src.utils.exceptions import DocumentLoadError, IngestionError

_MARKDOWN = """\
# 数据结构与算法

- 课程编号：CS201
- 学分：4
- 授课教师：张伟
- 开课学期：2026-2027-1
- 先修课程：CS101

## 课程简介

本课程介绍基本数据结构。

## 考核方式

- 期末考试：60%
"""


class _StubLoader:
    """返回固定文档的假加载器，用于隔离上游。"""

    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def load_directory(self, directory: str | Path) -> list[Document]:
        return self._documents


class _StubSplitter:
    """原样返回（或返回预置结果）的假切分器。"""

    def __init__(self, documents: list[Document] | None = None) -> None:
        self._documents = documents

    def split(self, documents: list[Document]) -> list[Document]:
        return self._documents if self._documents is not None else documents


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------
def test_load_and_split_returns_documents(
    raw_dir: Path, course_payload: Callable[..., dict[str, Any]]
) -> None:
    """一条完整链路：目录 → Document 列表。"""
    (raw_dir / "courses.json").write_text(
        json.dumps([course_payload()], ensure_ascii=False), encoding="utf-8"
    )
    (raw_dir / "syllabus.md").write_text(_MARKDOWN, encoding="utf-8")

    documents = IngestionService().load_and_split(raw_dir)

    assert documents
    assert all(isinstance(document, Document) for document in documents)
    assert {document.metadata["source"] for document in documents} == {
        "courses.json",
        "syllabus.md",
    }


def test_every_document_has_required_metadata(
    raw_dir: Path, course_payload: Callable[..., dict[str, Any]]
) -> None:
    """出口校验是本服务的核心保证，每个文档都要带齐必备字段。"""
    (raw_dir / "courses.json").write_text(
        json.dumps([course_payload()], ensure_ascii=False), encoding="utf-8"
    )
    (raw_dir / "syllabus.md").write_text(_MARKDOWN, encoding="utf-8")

    documents = IngestionService().load_and_split(raw_dir)

    for document in documents:
        for key in REQUIRED_METADATA_KEYS:
            assert str(document.metadata.get(key, "")).strip(), f"{document.metadata} 缺 {key}"


def test_splitter_is_applied(raw_dir: Path) -> None:
    """服务确实调用了切分器，而不是把加载结果直接返回。"""
    long_markdown = _MARKDOWN + "\n## 教学内容\n\n" + "这是一句用于撑长正文的测试句子。" * 200
    (raw_dir / "syllabus.md").write_text(long_markdown, encoding="utf-8")

    documents = IngestionService().load_and_split(raw_dir)

    assert any(int(document.metadata["chunk_total"]) > 1 for document in documents)


# ---------------------------------------------------------------------------
# 出口校验
# ---------------------------------------------------------------------------
def test_missing_course_id_is_rejected() -> None:
    """缺 course_id 的文档必须让整条流水线失败。"""
    broken = Document(page_content="正文", metadata={"name": "X", "type": "course", "source": "x"})
    service = IngestionService(
        loader=_StubLoader([broken]),  # type: ignore[arg-type]
        splitter=_StubSplitter(),
    )

    with pytest.raises(IngestionError, match="course_id") as excinfo:
        service.load_and_split("ignored")

    assert excinfo.value.details["missing"] == ["course_id"]
    assert excinfo.value.details["index"] == 0


@pytest.mark.parametrize("missing_key", ["course_id", "name", "type", "source"])
def test_each_required_key_is_enforced(missing_key: str) -> None:
    """四个必备字段逐个验证，避免校验逻辑漏掉某一项。"""
    metadata: dict[str, Any] = {
        "course_id": "CS201",
        "name": "数据结构与算法",
        "type": "course",
        "source": "courses.json",
    }
    metadata[missing_key] = "   "  # 纯空白等同于缺失
    broken = Document(page_content="正文", metadata=metadata)
    service = IngestionService(
        loader=_StubLoader([broken]),  # type: ignore[arg-type]
        splitter=_StubSplitter(),
    )

    with pytest.raises(IngestionError, match=missing_key):
        service.load_and_split("ignored")


def test_blank_metadata_value_counted_as_missing() -> None:
    """纯空白值不算"有值"，否则 course_id=" " 会一路混进向量库。"""
    broken = Document(
        page_content="正文",
        metadata={"course_id": " ", "name": "N", "type": "course", "source": "s"},
    )
    service = IngestionService(
        loader=_StubLoader([broken]),  # type: ignore[arg-type]
        splitter=_StubSplitter(),
    )

    with pytest.raises(IngestionError, match="course_id"):
        service.load_and_split("ignored")


#: ChromaDB 不接受的 metadata 取值形态。
_NON_SCALAR_VALUES: list[Any] = [["CS101"], {"exam": 1}, None]


@pytest.mark.parametrize("bad_value", _NON_SCALAR_VALUES)
def test_non_scalar_metadata_is_rejected(bad_value: Any) -> None:
    """metadata 里的 list / dict / None 必须在接入阶段就被拦下。

    ``MetadataValue`` 只是类型注解，运行时拦不住任何东西。如果不在这里把关，
    问题会一直潜伏到阶段 3 建库时才以 Chroma 的报错形式出现，定位成本高得多。
    """
    document = Document(
        page_content="正文",
        metadata={
            "course_id": "CS201",
            "name": "数据结构与算法",
            "type": "course",
            "source": "courses.json",
            "prerequisites": bad_value,
        },
    )
    service = IngestionService(
        loader=_StubLoader([document]),  # type: ignore[arg-type]
        splitter=_StubSplitter(),
    )

    with pytest.raises(IngestionError, match="非标量取值"):
        service.load_and_split("ignored")


# ---------------------------------------------------------------------------
# 依赖注入
# ---------------------------------------------------------------------------
def test_injected_dependencies_are_exposed() -> None:
    """注入的组件应当能被取回，方便调用方确认实际生效的实现。"""
    loader = DocumentLoader()
    splitter = CourseDocumentSplitter(chunk_size=123, chunk_overlap=1)
    service = IngestionService(loader=loader, splitter=splitter)

    assert service.loader is loader
    assert service.splitter is splitter


def test_default_dependencies_are_created() -> None:
    """不注入时应当自建默认实现，直接可用。"""
    service = IngestionService()

    assert isinstance(service.loader, DocumentLoader)
    assert isinstance(service.splitter, CourseDocumentSplitter)


# ---------------------------------------------------------------------------
# 异常透传
# ---------------------------------------------------------------------------
def test_loader_errors_propagate(raw_dir: Path) -> None:
    """上游加载错误要原样抛出，不能被编排层吞掉或换成模糊的错误。"""
    (raw_dir / "broken.json").write_text("{ not json", encoding="utf-8")

    with pytest.raises(DocumentLoadError, match="不是合法 JSON"):
        IngestionService().load_and_split(raw_dir)


def test_empty_directory_error_propagates(tmp_path: Path) -> None:
    """空目录的报错也要传出来。"""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(DocumentLoadError, match="没有解析出任何文档"):
        IngestionService().load_and_split(empty)


# ---------------------------------------------------------------------------
# 样例语料
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_shipped_sample_data_end_to_end() -> None:
    """验收标准：``IngestionService().load_and_split("data/raw")`` 能跑通。"""
    documents = IngestionService().load_and_split(PROJECT_ROOT / "data" / "raw")

    assert len(documents) > 0

    course_ids = {str(document.metadata["course_id"]) for document in documents}
    assert {"CS101", "CS201", "CS301"} <= course_ids

    # 每门课至少要能被引用到，且正文非空
    for document in documents:
        assert document.page_content.strip()
        assert "course_id" in document.metadata
        assert "chunk_total" in document.metadata
