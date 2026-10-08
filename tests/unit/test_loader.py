"""``src.ingestion.loader`` 的单元测试。

约定：所有会写文件的用例都用 ``tmp_path``，不依赖仓库里的 ``data/raw``；
只有最后一条"样例语料自检"用例会去读真实数据，用于发现样例数据与加载器
脱节的问题。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.config import PROJECT_ROOT
from src.ingestion.loader import SUPPORTED_SUFFIXES, DocumentLoader
from src.schemas.course import DocumentType
from src.utils.exceptions import DocumentLoadError, UnsupportedFormatError

# ---------------------------------------------------------------------------
# 构造辅助
# ---------------------------------------------------------------------------
_MARKDOWN_HEADER = """\
# 数据结构与算法

- 课程编号：CS201
- 学分：4
- 授课教师：张伟
- 开课学期：2026-2027-1
- 先修课程：CS101
"""


def _write_json(directory: Path, name: str, payload: object) -> Path:
    path = directory / name
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _write_text(directory: Path, name: str, content: str) -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def loader() -> DocumentLoader:
    """被测加载器。"""
    return DocumentLoader()


# ---------------------------------------------------------------------------
# JSON：三种受支持的形态
# ---------------------------------------------------------------------------
def test_load_json_array_form(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """课程对象数组形态。"""
    _write_json(raw_dir, "courses.json", [course_payload()])

    documents = loader.load_file(raw_dir / "courses.json")

    assert len(documents) == 1
    assert documents[0].metadata["course_id"] == "CS201"


def test_load_json_wrapped_form(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """``{"courses": [...]}`` 包装形态。"""
    payload = {"note": "演示数据", "courses": [course_payload(), course_payload(course_id="CS301")]}
    _write_json(raw_dir, "courses.json", payload)

    documents = loader.load_file(raw_dir / "courses.json")

    assert [doc.metadata["course_id"] for doc in documents] == ["CS201", "CS301"]


def test_load_json_single_object_form(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """单个课程对象形态。"""
    _write_json(raw_dir, "one.json", course_payload())

    documents = loader.load_file(raw_dir / "one.json")

    assert len(documents) == 1
    assert documents[0].metadata["course_id"] == "CS201"


def test_load_json_one_document_per_course(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """每门课生成一个 Document，不是整个文件一个。"""
    _write_json(
        raw_dir,
        "courses.json",
        [course_payload(course_id=f"CS{i}") for i in range(5)],
    )

    documents = loader.load_file(raw_dir / "courses.json")

    assert len(documents) == 5


# ---------------------------------------------------------------------------
# JSON：metadata 与正文
# ---------------------------------------------------------------------------
def test_json_metadata_is_complete(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """结构化字段必须进入 metadata，而不是只塞进正文。"""
    _write_json(raw_dir, "courses.json", [course_payload()])

    metadata = loader.load_file(raw_dir / "courses.json")[0].metadata

    assert metadata["course_id"] == "CS201"
    assert metadata["name"] == "数据结构与算法"
    assert metadata["credits"] == 4
    assert metadata["type"] == DocumentType.COURSE.value
    assert metadata["source"] == "courses.json"
    assert metadata["instructor"] == "张伟"
    assert metadata["prerequisites"] == "CS101"


def test_json_page_content_keeps_structured_info(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """正文里也要有课程信息——Embedding 只看正文，不看 metadata。"""
    _write_json(raw_dir, "courses.json", [course_payload()])

    content = loader.load_file(raw_dir / "courses.json")[0].page_content

    assert "数据结构与算法" in content
    assert "CS201" in content
    assert "CS101" in content  # 先修课程
    assert "系统讲授线性表" in content  # 简介


# ---------------------------------------------------------------------------
# JSON：错误处理
# ---------------------------------------------------------------------------
def test_invalid_json_reports_position(loader: DocumentLoader, raw_dir: Path) -> None:
    """JSON 语法错误要给出行列位置，而不是一句"解析失败"。"""
    _write_text(raw_dir, "broken.json", '{"courses": [ }')

    with pytest.raises(DocumentLoadError, match="不是合法 JSON") as excinfo:
        loader.load_file(raw_dir / "broken.json")

    assert "行" in str(excinfo.value)
    assert excinfo.value.details["file"].endswith("broken.json")


def test_empty_json_file_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """空文件必须报错，不能被当成"没有课程"悄悄跳过。"""
    _write_text(raw_dir, "empty.json", "   \n\n  ")

    with pytest.raises(DocumentLoadError, match="空文件"):
        loader.load_file(raw_dir / "empty.json")


def test_invalid_course_record_reports_index(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """某条课程数据非法时，错误信息要能定位到是第几条、哪个字段。"""
    payload = [course_payload(), course_payload(), course_payload(credits=-3)]
    _write_json(raw_dir, "courses.json", payload)

    with pytest.raises(DocumentLoadError, match="第 2 条") as excinfo:
        loader.load_file(raw_dir / "courses.json")

    assert excinfo.value.details["index"] == 2
    assert "credits" in str(excinfo.value)


def test_missing_course_field_reports_field_name(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """缺字段时要报出字段名，方便直接改数据。"""
    _write_json(raw_dir, "courses.json", [course_payload(name=...)])

    with pytest.raises(DocumentLoadError, match="name"):
        loader.load_file(raw_dir / "courses.json")


def test_unrecognized_json_shape_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """结构无法识别时给出"支持哪些写法"的可操作提示。"""
    _write_json(raw_dir, "weird.json", {"foo": "bar"})

    with pytest.raises(DocumentLoadError, match="结构无法识别"):
        loader.load_file(raw_dir / "weird.json")


def test_json_courses_field_must_be_list(loader: DocumentLoader, raw_dir: Path) -> None:
    """``courses`` 字段必须是数组，写成对象要报错。"""
    _write_json(raw_dir, "courses.json", {"courses": {"course_id": "CS201"}})

    with pytest.raises(DocumentLoadError, match="应当是数组"):
        loader.load_file(raw_dir / "courses.json")


def test_json_record_must_be_object(loader: DocumentLoader, raw_dir: Path) -> None:
    """数组元素必须是对象。"""
    _write_json(raw_dir, "courses.json", ["CS201", "CS301"])

    with pytest.raises(DocumentLoadError, match="应当是 JSON 对象"):
        loader.load_file(raw_dir / "courses.json")


# ---------------------------------------------------------------------------
# Markdown：结构与 metadata
# ---------------------------------------------------------------------------
def test_markdown_splits_into_sections(loader: DocumentLoader, raw_dir: Path) -> None:
    """每个二级标题各成一个 Document，外加一节合成的基本信息。"""
    content = (
        _MARKDOWN_HEADER
        + "\n## 课程简介\n\n本课程介绍基本数据结构。\n"
        + "\n## 考核方式\n\n- 期末考试：60%\n- 平时作业：40%\n"
    )
    _write_text(raw_dir, "syllabus.md", content)

    documents = loader.load_file(raw_dir / "syllabus.md")

    assert [doc.metadata["section"] for doc in documents] == [
        "basic_info",
        "description",
        "assessment",
    ]


def test_markdown_metadata_is_complete(loader: DocumentLoader, raw_dir: Path) -> None:
    """大纲抬头里的结构化字段要解析进 metadata。"""
    content = _MARKDOWN_HEADER + "\n## 课程简介\n\n正文。\n"
    _write_text(raw_dir, "syllabus.md", content)

    metadata = loader.load_file(raw_dir / "syllabus.md")[0].metadata

    assert metadata["course_id"] == "CS201"
    assert metadata["name"] == "数据结构与算法"
    assert metadata["type"] == DocumentType.SYLLABUS.value
    assert metadata["source"] == "syllabus.md"
    assert metadata["credits"] == 4
    assert metadata["instructor"] == "张伟"
    assert metadata["semester"] == "2026-2027-1"
    assert metadata["prerequisites"] == "CS101"
    assert metadata["section"] == "basic_info"


def test_markdown_known_title_is_normalized(loader: DocumentLoader, raw_dir: Path) -> None:
    """别名表命中的标题归一化成 ASCII 键，便于下游过滤。"""
    content = _MARKDOWN_HEADER + "\n## 课程概述\n\n正文。\n"
    _write_text(raw_dir, "syllabus.md", content)

    sections = [doc.metadata["section"] for doc in loader.load_file(raw_dir / "syllabus.md")]

    assert "description" in sections  # 课程概述 → description
    assert "课程概述" not in sections


def test_markdown_unknown_title_kept_verbatim(loader: DocumentLoader, raw_dir: Path) -> None:
    """别名表没命中的标题原样作键。

    宁可键名不统一，也不要把"实验安排"硬塞进某个已知类别——那会造成误分类，
    而误分类在检索时是静默的错误。
    """
    content = _MARKDOWN_HEADER + "\n## 实验安排\n\n实验一：顺序表。\n"
    _write_text(raw_dir, "syllabus.md", content)

    documents = loader.load_file(raw_dir / "syllabus.md")
    sections = {doc.metadata["section"]: doc for doc in documents}

    assert "实验安排" in sections
    assert sections["实验安排"].metadata["section_title"] == "实验安排"


def test_markdown_explicit_basic_info_section_is_not_duplicated(
    loader: DocumentLoader, raw_dir: Path
) -> None:
    """大纲里已经手写了「课程基本信息」章节时，不再合成一节。"""
    content = _MARKDOWN_HEADER + "\n## 课程基本信息\n\n- 课程性质：专业核心课\n\n## 课程简介\n\n正文。\n"
    _write_text(raw_dir, "syllabus.md", content)

    sections = [doc.metadata["section"] for doc in loader.load_file(raw_dir / "syllabus.md")]

    assert sections.count("basic_info") == 1


def test_markdown_every_section_carries_course_context(
    loader: DocumentLoader, raw_dir: Path
) -> None:
    """每节正文都要带课程抬头，否则切片会失去上下文。"""
    content = _MARKDOWN_HEADER + "\n## 考核方式\n\n- 期末考试：60%\n"
    _write_text(raw_dir, "syllabus.md", content)

    for document in loader.load_file(raw_dir / "syllabus.md"):
        assert "数据结构与算法（CS201）" in document.page_content
        assert "先修课程：CS101" in document.page_content


def test_markdown_section_heading_appears_in_content(loader: DocumentLoader, raw_dir: Path) -> None:
    """章节标题要写进正文，Embedding 才能感知"这是考核方式"。"""
    content = _MARKDOWN_HEADER + "\n## 考核方式\n\n- 期末考试：60%\n"
    _write_text(raw_dir, "syllabus.md", content)

    sections = {doc.metadata["section"]: doc for doc in loader.load_file(raw_dir / "syllabus.md")}

    assert "## 考核方式" in sections["assessment"].page_content
    assert "期末考试：60%" in sections["assessment"].page_content


def test_markdown_multiple_courses_in_one_file(loader: DocumentLoader, raw_dir: Path) -> None:
    """一个文件里写多门课也要能正确切分。"""
    content = (
        "# 程序设计基础\n\n- 课程编号：CS101\n- 学分：4\n- 授课教师：李明\n- 开课学期：2026-2027-1\n- 先修课程：无\n"
        "\n## 课程简介\n\n编程入门课。\n"
        "\n# 数据结构与算法\n\n- 课程编号：CS201\n- 学分：4\n- 授课教师：张伟\n"
        "- 开课学期：2026-2027-1\n- 先修课程：CS101\n"
        "\n## 课程简介\n\n数据结构课。\n"
    )
    _write_text(raw_dir, "all.md", content)

    documents = loader.load_file(raw_dir / "all.md")

    assert {doc.metadata["course_id"] for doc in documents} == {"CS101", "CS201"}
    cs101 = [doc for doc in documents if doc.metadata["course_id"] == "CS101"]
    # "无" 被解析成空列表 → 抬头显示"先修课程：无"
    assert "先修课程：无" in cs101[0].page_content


def test_markdown_prerequisite_separators(loader: DocumentLoader, raw_dir: Path) -> None:
    """先修课程支持顿号、逗号等多种分隔写法。"""
    header = _MARKDOWN_HEADER.replace("先修课程：CS101", "先修课程：CS101、CS202，CS301")
    _write_text(raw_dir, "syllabus.md", header + "\n## 课程简介\n\n正文。\n")

    metadata = loader.load_file(raw_dir / "syllabus.md")[0].metadata

    assert metadata["prerequisites"] == "CS101,CS202,CS301"


def test_markdown_explicit_course_name_wins_over_h1(loader: DocumentLoader, raw_dir: Path) -> None:
    """一级标题写成"教学大纲"时，课程名应取抬头里显式声明的「课程名称」。"""
    content = (
        "# 教学大纲\n\n- 课程编号：CS201\n- 课程名称：数据结构与算法\n- 学分：4\n"
        "- 授课教师：张伟\n- 开课学期：2026-2027-1\n"
        "\n## 课程简介\n\n正文。\n"
    )
    _write_text(raw_dir, "syllabus.md", content)

    metadata = loader.load_file(raw_dir / "syllabus.md")[0].metadata

    assert metadata["name"] == "数据结构与算法"


def test_markdown_skips_empty_sections(loader: DocumentLoader, raw_dir: Path) -> None:
    """只有标题没有正文的章节不产出文档（它只会稀释检索结果）。"""
    content = _MARKDOWN_HEADER + "\n## 课程简介\n\n正文。\n\n## 空章节\n\n\n## 考核方式\n\n- 期末：60%\n"
    _write_text(raw_dir, "syllabus.md", content)

    sections = [doc.metadata["section"] for doc in loader.load_file(raw_dir / "syllabus.md")]

    assert "空章节" not in sections
    assert "assessment" in sections


def test_markdown_deep_headings_stay_in_parent_section(
    loader: DocumentLoader, raw_dir: Path
) -> None:
    """三级标题不单独成节，原样留在所属二级章节正文里。"""
    content = (
        _MARKDOWN_HEADER
        + "\n## 教学内容\n\n### 第一章 绪论\n\n- 基本概念\n\n### 第二章 线性表\n\n- 顺序存储\n"
    )
    _write_text(raw_dir, "syllabus.md", content)

    documents = loader.load_file(raw_dir / "syllabus.md")
    content_doc = next(doc for doc in documents if doc.metadata["section"] == "content")

    assert "### 第一章 绪论" in content_doc.page_content
    assert "### 第二章 线性表" in content_doc.page_content
    assert not any(doc.metadata["section"] == "第一章 绪论" for doc in documents)


# ---------------------------------------------------------------------------
# Markdown：错误处理
# ---------------------------------------------------------------------------
def test_markdown_without_course_id_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """没有课程编号的大纲无法引用，必须报错并指明是哪一门。"""
    content = "# 数据结构与算法\n\n- 学分：4\n\n## 课程简介\n\n正文。\n"
    _write_text(raw_dir, "syllabus.md", content)

    with pytest.raises(DocumentLoadError, match="课程编号") as excinfo:
        loader.load_file(raw_dir / "syllabus.md")

    assert excinfo.value.details["course_title"] == "数据结构与算法"


def test_markdown_without_h1_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """没有一级标题就确定不了课程边界。"""
    _write_text(raw_dir, "syllabus.md", "## 课程简介\n\n正文。\n")

    with pytest.raises(DocumentLoadError, match="找不到一级标题"):
        loader.load_file(raw_dir / "syllabus.md")


def test_markdown_empty_file_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """空文件必须报错。"""
    _write_text(raw_dir, "syllabus.md", "\n   \n")

    with pytest.raises(DocumentLoadError, match="空文件"):
        loader.load_file(raw_dir / "syllabus.md")


def test_markdown_with_only_headings_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """所有章节都没有正文时，应当报"没产出非空章节"而不是静默返回空列表。

    这里需要显式写出一个空的「课程基本信息」章节：否则加载器会自动用抬头
    合成一节，文档列表就不为空了。这也是该分支唯一可达的路径。
    """
    content = (
        _MARKDOWN_HEADER
        + "\n## 课程基本信息\n\n## 课程简介\n\n## 考核方式\n"
    )
    _write_text(raw_dir, "syllabus.md", content)

    with pytest.raises(DocumentLoadError, match="没有产出任何非空章节"):
        loader.load_file(raw_dir / "syllabus.md")


def test_markdown_invalid_credits_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """学分写成非数字要报错。"""
    content = _MARKDOWN_HEADER.replace("学分：4", "学分：四") + "\n## 课程简介\n\n正文。\n"
    _write_text(raw_dir, "syllabus.md", content)

    with pytest.raises(DocumentLoadError, match="学分字段无法解析"):
        loader.load_file(raw_dir / "syllabus.md")


def test_non_utf8_file_gives_actionable_error(loader: DocumentLoader, raw_dir: Path) -> None:
    """GBK 编码的文件要提示"另存为 UTF-8"，而不是抛裸的 UnicodeDecodeError。"""
    path = raw_dir / "gbk.md"
    path.write_bytes("# 数据结构\n\n- 课程编号：CS201\n".encode("gbk"))

    with pytest.raises(DocumentLoadError, match="不是 UTF-8 编码"):
        loader.load_file(path)


# ---------------------------------------------------------------------------
# 文件与目录层面
# ---------------------------------------------------------------------------
def test_unsupported_suffix_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """不支持的后缀必须明确报错，并列出支持的格式。"""
    _write_text(raw_dir, "notes.txt", "一些说明")

    with pytest.raises(UnsupportedFormatError, match="不支持的文件格式") as excinfo:
        loader.load_file(raw_dir / "notes.txt")

    assert excinfo.value.details["suffix"] == ".txt"


def test_missing_file_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """文件不存在时的报错要带上路径。"""
    with pytest.raises(DocumentLoadError, match="文件不存在"):
        loader.load_file(raw_dir / "nope.json")


def test_directory_not_found_rejected(loader: DocumentLoader, tmp_path: Path) -> None:
    """目录不存在时要报错，而不是返回空列表。"""
    with pytest.raises(DocumentLoadError, match="目录不存在"):
        loader.load_directory(tmp_path / "no-such-dir")


def test_passing_file_as_directory_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """把文件当目录传进去要报错。"""
    path = _write_text(raw_dir, "x.md", _MARKDOWN_HEADER)

    with pytest.raises(DocumentLoadError, match="期望传入目录"):
        loader.load_directory(path)


def test_empty_directory_rejected(loader: DocumentLoader, raw_dir: Path) -> None:
    """没有解析出任何文档时要报错——空索引在建库阶段很难被发现。"""
    with pytest.raises(DocumentLoadError, match="没有解析出任何文档"):
        loader.load_directory(raw_dir)


def test_directory_skips_unsupported_files_with_warning(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """目录里混入无关文件时跳过并告警，但不中断整条接入流程。"""
    _write_json(raw_dir, "courses.json", [course_payload()])
    _write_text(raw_dir, "readme.txt", "说明文件")

    with caplog.at_level(logging.WARNING, logger="src.ingestion.loader"):
        documents = loader.load_directory(raw_dir)

    assert len(documents) == 1
    assert "readme.txt" in caplog.text


def test_directory_ignores_gitkeep_silently(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """.gitkeep 这类占位文件应当静默忽略，不产生告警噪音。"""
    _write_json(raw_dir, "courses.json", [course_payload()])
    _write_text(raw_dir, ".gitkeep", "")

    with caplog.at_level(logging.WARNING, logger="src.ingestion.loader"):
        documents = loader.load_directory(raw_dir)

    assert len(documents) == 1
    assert ".gitkeep" not in caplog.text


def test_directory_result_order_is_deterministic(
    loader: DocumentLoader,
    raw_dir: Path,
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """多次加载同一目录，结果顺序必须一致。"""
    _write_json(raw_dir, "b_courses.json", [course_payload(course_id="CS201")])
    _write_json(raw_dir, "a_courses.json", [course_payload(course_id="CS101")])

    first = [doc.metadata["course_id"] for doc in loader.load_directory(raw_dir)]
    second = [doc.metadata["course_id"] for doc in loader.load_directory(raw_dir)]

    assert first == second == ["CS101", "CS201"]


def test_markdown_extension_alias(
    loader: DocumentLoader,
    raw_dir: Path,
) -> None:
    """.markdown 与 .md 等价。"""
    _write_text(raw_dir, "syllabus.markdown", _MARKDOWN_HEADER + "\n## 课程简介\n\n正文。\n")

    documents = loader.load_file(raw_dir / "syllabus.markdown")

    assert documents[0].metadata["course_id"] == "CS201"


def test_supported_suffixes_constant_is_intact() -> None:
    """支持的后缀集合是公开契约，改动需同步测试与文档。"""
    assert frozenset({".json", ".md", ".markdown"}) == SUPPORTED_SUFFIXES


# ---------------------------------------------------------------------------
# 样例语料自检
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_shipped_sample_data_loads(loader: DocumentLoader) -> None:
    """仓库自带的 data/raw 必须能被当前加载器解析。

    这条用例的价值在于：样例数据与加载器是两拨人（两次提交）在不同时间写的，
    一旦字段名或结构发生漂移，这里会立刻红。
    """
    documents = loader.load_directory(PROJECT_ROOT / "data" / "raw")

    course_ids = {str(doc.metadata["course_id"]) for doc in documents}
    assert {"CS101", "CS201", "CS301"} <= course_ids

    for document in documents:
        for key in ("course_id", "name", "type", "source"):
            assert str(document.metadata.get(key, "")).strip(), f"{key} 不能为空"
        assert document.page_content.strip()
