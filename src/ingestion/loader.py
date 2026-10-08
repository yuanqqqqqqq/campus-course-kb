"""文档加载层：把 ``data/raw`` 下的原始文件读成 LangChain ``Document``。

支持的格式：

``.json``
    结构化课程目录。文件可以是三种形态之一：课程对象数组、``{"courses": [...]}``
    包装对象、单个课程对象。每条记录用 :class:`~src.schemas.course.Course`
    校验后渲染成**一门课一个 Document**。

``.md`` / ``.markdown``
    教学大纲。按课程结构做语义分块（而不是按固定字符数硬切）：一个 ``#`` 一级
    标题代表一门课，其下 ``##`` 二级标题各成为一个 Document。

本模块只负责"读 + 解析 + 生成 Document"，**不做二次切分**（那是 splitter 的
职责），也不碰 Embedding 和向量库。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from langchain_core.documents import Document
from pydantic import ValidationError

from src.schemas.common import MetadataValue
from src.schemas.course import (
    Course,
    DocumentType,
    build_context_header,
    join_list_values,
)
from src.utils.exceptions import DocumentLoadError, UnsupportedFormatError
from src.utils.logging import get_logger
from src.utils.text import as_text, format_validation_error, simplify_validation_errors

logger = get_logger(__name__)

#: 支持的文件后缀（小写）。
SUPPORTED_SUFFIXES: Final[frozenset[str]] = frozenset({".json", ".md", ".markdown"})

#: 扫描目录时**静默**忽略的文件名。
#:
#: 这些要么是版本控制占位文件，要么是编辑器 / 操作系统残留。对它们报警只会
#: 制造噪音，掩盖真正需要关注的问题。
_IGNORED_FILENAMES: Final[frozenset[str]] = frozenset(
    {".gitkeep", ".gitignore", ".ds_store", "thumbs.db", "desktop.ini"}
)

#: 一级标题（``# 数据结构与算法``）。
_H1_PATTERN: Final[re.Pattern[str]] = re.compile(r"^#\s+(?P<title>\S.*?)\s*$")

#: 二级标题（``## 课程简介``）。注意 ``^##\s`` 不会误匹配 ``### 子标题``。
_H2_PATTERN: Final[re.Pattern[str]] = re.compile(r"^##\s+(?P<title>\S.*?)\s*$")

#: 大纲抬头里的键值行（``- 课程编号：CS201``），全角 / 半角冒号都接受。
_BULLET_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*[-*+]\s*(?P<key>[^:：]+?)\s*[:：]\s*(?P<value>.*?)\s*$"
)

#: 抬头字段的中文别名 → :class:`Course` 字段名。
_HEADER_FIELD_ALIASES: Final[dict[str, str]] = {
    "课程编号": "course_id",
    "课程号": "course_id",
    "课程代码": "course_id",
    "课程名称": "name",
    "课程名": "name",
    "学分": "credits",
    "授课教师": "instructor",
    "任课教师": "instructor",
    "教师": "instructor",
    "开课学期": "semester",
    "学期": "semester",
    "先修课程": "prerequisites",
    "先修课": "prerequisites",
    "前置课程": "prerequisites",
}

#: 二级标题 → 归一化的 ASCII section 键。
#:
#: 归一化的意义：筛选条件不用写中文字符串，且"课程简介 / 简介 / 课程概述"这类
#: 同义标题会落到同一个键上。原始标题仍会完整保存在 ``section_title`` 里，
#: 所以归一化不会丢失任何信息。
_SECTION_ALIASES: Final[dict[str, str]] = {
    "课程基本信息": "basic_info",
    "基本信息": "basic_info",
    "课程信息": "basic_info",
    "课程简介": "description",
    "简介": "description",
    "课程概述": "description",
    "课程介绍": "description",
    "教学目标": "objectives",
    "课程目标": "objectives",
    "学习目标": "objectives",
    "教学内容": "content",
    "课程内容": "content",
    "教学大纲": "content",
    "教学内容与安排": "content",
    "考核方式": "assessment",
    "考核": "assessment",
    "成绩评定": "assessment",
    "考核与成绩": "assessment",
    "参考教材": "textbooks",
    "教材": "textbooks",
    "推荐教材": "textbooks",
    "参考书目": "textbooks",
}

#: 合成"课程基本信息"章节时使用的标题与键。
_BASIC_INFO_TITLE: Final[str] = "课程基本信息"
_BASIC_INFO_KEY: Final[str] = "basic_info"

#: 表示"没有先修课程"的写法。
_PREREQUISITE_NONE_MARKERS: Final[frozenset[str]] = frozenset(
    {"无", "無", "没有", "none", "n/a", "na", "-", "--", "/"}
)

#: 拆分先修课程列表时接受的分隔符（兼容中英文标点）。
_PREREQUISITE_SEPARATORS: Final[re.Pattern[str]] = re.compile(r"[,，、;；/|]+")

#: 写入文件时使用的编码。读取时用 utf-8-sig，可同时正确处理带 BOM 的 UTF-8
#: （Windows 记事本另存为 UTF-8 时会加 BOM）。
_ENCODING: Final[str] = "utf-8-sig"


@dataclass
class _CourseBlock:
    """Markdown 中一个"课程块"的解析结果。"""

    title: str
    """一级标题文本，作为课程名的兜底来源。"""

    header_lines: list[str] = field(default_factory=list)
    """一级标题之后、第一个二级标题之前的所有原始行（通常是大纲抬头）。"""

    header_fields: dict[str, object] = field(default_factory=dict)
    """从抬头里解析出的结构化字段，键是 :class:`Course` 的字段名。"""

    sections: list[tuple[str, str]] = field(default_factory=list)
    """二级标题章节，元素为 ``(标题, 正文)``。"""


class DocumentLoader:
    """把原始文件加载成 ``Document`` 列表。

    典型用法：

    .. code-block:: python

        loader = DocumentLoader()
        documents = loader.load_directory("data/raw")
    """

    def __init__(self) -> None:
        #: 后缀 → 处理函数。新增格式时只需在这里注册，无需改动 load_file 的骨架。
        self._handlers: dict[str, Callable[[Path], list[Document]]] = {
            ".json": self._load_json_file,
            ".md": self._load_markdown_file,
        }
        self._handlers[".markdown"] = self._load_markdown_file

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def load_directory(self, directory: str | Path) -> list[Document]:
        """加载目录下所有支持的文件。

        :param directory: 目录路径。
        :return: ``Document`` 列表，顺序按**文件名排序**，保证多次运行结果一致
                 （向量库建库顺序影响不了检索，但可复现的输入更容易排查问题）。
        :raises DocumentLoadError: 目录不存在、不是目录，或目录下没有任何可用文档。
        :raises UnsupportedFormatError: 单个文件后缀不支持（仅在显式调用
            :meth:`load_file` 时抛出；扫描目录时未知后缀会被跳过并记 WARNING）。
        """
        root = Path(directory)
        if not root.exists():
            raise DocumentLoadError(f"目录不存在：{root}")
        if not root.is_dir():
            raise DocumentLoadError(f"期望传入目录，但传入的是文件：{root}")

        files = sorted(path for path in root.iterdir() if path.is_file())
        documents: list[Document] = []
        skipped: list[str] = []

        for path in files:
            if self._should_ignore(path):
                logger.debug("跳过忽略名单中的文件 | file=%s", path.name)
                continue
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                skipped.append(path.name)
                continue
            documents.extend(self.load_file(path))

        if skipped:
            # 不直接报错：数据目录里常有说明文档之类的无关文件，为此中断整条
            # 接入流程代价太大。但必须让人看见，否则会出现"文件明明放进去了，
            # 检索却没有"这种极难排查的情况。
            logger.warning(
                "以下文件后缀不受支持，已跳过 | files=%s | supported=%s",
                skipped,
                sorted(SUPPORTED_SUFFIXES),
            )

        if not documents:
            raise DocumentLoadError(
                f"目录 {root} 下没有解析出任何文档。"
                f"请确认目录内含 {sorted(SUPPORTED_SUFFIXES)} 文件且内容非空。",
                details={"directory": str(root), "scanned_files": len(files)},
            )

        logger.info("目录加载完成 | directory=%s files=%d documents=%d", root, len(files), len(documents))
        return documents

    def load_file(self, path: str | Path) -> list[Document]:
        """加载单个文件。

        :param path: 文件路径。
        :raises UnsupportedFormatError: 后缀不在 :data:`SUPPORTED_SUFFIXES` 内。
        :raises DocumentLoadError: 文件不存在、为空、编码错误或解析失败。
        """
        file_path = Path(path)
        if not file_path.exists():
            raise DocumentLoadError(f"文件不存在：{file_path}")
        if not file_path.is_file():
            raise DocumentLoadError(f"期望传入文件，但传入的是目录：{file_path}")

        suffix = file_path.suffix.lower()
        handler = self._handlers.get(suffix)
        if handler is None:
            raise UnsupportedFormatError(
                f"不支持的文件格式：{file_path.name}（后缀 {suffix or '无'}）。"
                f"当前支持：{sorted(SUPPORTED_SUFFIXES)}",
                details={"file": str(file_path), "suffix": suffix},
            )
        return handler(file_path)

    # ------------------------------------------------------------------
    # 目录扫描辅助
    # ------------------------------------------------------------------
    @staticmethod
    def _should_ignore(path: Path) -> bool:
        """判断文件是否属于"静默忽略"名单（隐藏文件、系统残留等）。"""
        name = path.name
        if name.startswith("."):
            return True
        return name.lower() in _IGNORED_FILENAMES

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    def _load_json_file(self, path: Path) -> list[Document]:
        """把课程目录 JSON 转成"一门课一个 Document"。"""
        payload = self._read_json(path)
        records = self._extract_course_records(payload, path)

        documents: list[Document] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise DocumentLoadError(
                    f"{path.name} 第 {index} 条记录应当是 JSON 对象，实际是 {type(record).__name__}",
                    details={"file": str(path), "index": index},
                )
            try:
                course = Course.model_validate(dict(record))
            except ValidationError as exc:
                raise DocumentLoadError(
                    f"{path.name} 第 {index} 条课程数据不符合 Course 模型：{format_validation_error(exc)}",
                    details={
                        "file": str(path),
                        "index": index,
                        "errors": simplify_validation_errors(exc),
                    },
                ) from exc

            documents.append(
                Document(
                    page_content=course.to_document_text(source=path.name),
                    metadata=course.to_metadata(source=path.name),
                )
            )

        logger.info("JSON 加载完成 | file=%s courses=%d", path.name, len(documents))
        return documents

    @staticmethod
    def _read_json(path: Path) -> object:
        """读取并解析 JSON，所有失败路径都转成带定位信息的 :class:`DocumentLoadError`。"""
        raw_text = _read_text(path)
        if not raw_text.strip():
            raise DocumentLoadError(
                f"{path.name} 是空文件，无法解析课程数据。",
                details={"file": str(path)},
            )
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise DocumentLoadError(
                f"{path.name} 不是合法 JSON：第 {exc.lineno} 行第 {exc.colno} 列 {exc.msg}",
                details={
                    "file": str(path),
                    "line": exc.lineno,
                    "column": exc.colno,
                    "reason": exc.msg,
                },
            ) from exc

    @staticmethod
    def _extract_course_records(payload: object, path: Path) -> Sequence[object]:
        """从三种受支持的 JSON 形态里取出课程记录序列。

        接受 ``[...]``、``{"courses": [...]}``、``{...单个课程...}`` 三种写法，
        是因为不同来源的导出的确长得不一样，而在加载器里兼容一次比让使用者
        手工改数据划算得多。
        """
        if isinstance(payload, list):
            return payload

        if isinstance(payload, Mapping):
            if "courses" in payload:
                courses = payload["courses"]
                if not isinstance(courses, list):
                    raise DocumentLoadError(
                        f"{path.name} 的 'courses' 字段应当是数组，实际是 {type(courses).__name__}",
                        details={"file": str(path)},
                    )
                return courses
            if "course_id" in payload:
                return [payload]

        raise DocumentLoadError(
            f"{path.name} 的结构无法识别。支持的写法：课程对象数组、"
            f"包含 'courses' 数组的对象、单个课程对象。",
            details={"file": str(path), "top_level_type": type(payload).__name__},
        )

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    def _load_markdown_file(self, path: Path) -> list[Document]:
        """把教学大纲 Markdown 按课程结构拆成多个 Document。"""
        text = _read_text(path)
        if not text.strip():
            raise DocumentLoadError(
                f"{path.name} 是空文件，无法解析教学大纲。",
                details={"file": str(path)},
            )

        blocks = _split_into_course_blocks(text)
        if not blocks:
            raise DocumentLoadError(
                f"{path.name} 中找不到一级标题（以 '# ' 开头的行），无法确定课程边界。",
                details={"file": str(path)},
            )

        documents: list[Document] = []
        for block in blocks:
            documents.extend(self._build_markdown_documents(block, path))

        if not documents:
            raise DocumentLoadError(
                f"{path.name} 解析后没有产出任何非空章节。请检查大纲是否只有标题、没有正文。",
                details={"file": str(path), "course_blocks": len(blocks)},
            )

        logger.info("Markdown 加载完成 | file=%s sections=%d", path.name, len(documents))
        return documents

    def _build_markdown_documents(self, block: _CourseBlock, path: Path) -> list[Document]:
        """把一个课程块展开成若干章节 Document。"""
        course_id = as_text(block.header_fields.get("course_id"))
        if not course_id:
            raise DocumentLoadError(
                f"{path.name} 的课程块「{block.title}」缺少「课程编号」。"
                f"course_id 是检索与引用的主键，无法从标题推断，请在大纲抬头中补上。",
                details={"file": str(path), "course_title": block.title},
            )

        base_metadata = self._build_markdown_metadata(block, course_id, path)
        header = build_context_header(base_metadata)

        documents: list[Document] = []

        # 抬头本身也作为一个可检索章节：像"这门课几学分""先修课是什么"这类问题，
        # 命中它比命中"教学内容"要准得多。
        existing_keys = {_normalize_section_key(title) for title, _ in block.sections}
        if _BASIC_INFO_KEY not in existing_keys:
            basic_body = "\n".join(block.header_lines).strip()
            if basic_body:
                documents.append(
                    Document(
                        page_content=_compose_page_content(header, _BASIC_INFO_TITLE, basic_body),
                        metadata={
                            **base_metadata,
                            "section": _BASIC_INFO_KEY,
                            "section_title": _BASIC_INFO_TITLE,
                        },
                    )
                )
            else:
                logger.debug(
                    "课程块抬头没有内容，跳过合成的基本信息章节 | file=%s course=%s",
                    path.name,
                    course_id,
                )

        for title, body in block.sections:
            if not body.strip():
                # 只有标题没有正文的章节没有可检索内容，生成出来的切片只会稀释
                # 检索结果（它可能因为包含课程名而拿到不低的相似度）。
                logger.debug("跳过空章节 | file=%s course=%s section=%s", path.name, course_id, title)
                continue
            documents.append(
                Document(
                    page_content=_compose_page_content(header, title, body),
                    metadata={
                        **base_metadata,
                        "section": _normalize_section_key(title),
                        "section_title": title,
                    },
                )
            )

        return documents

    @staticmethod
    def _build_markdown_metadata(
        block: _CourseBlock, course_id: str, path: Path
    ) -> dict[str, MetadataValue]:
        """组装 Markdown 文档的公共 metadata（不含 section 相关字段）。

        课程名优先取抬头里显式写明的「课程名称」——因为一级标题未必就是课程名
        （有人会写成"教学大纲"）。取不到时才退回一级标题。
        """
        name = as_text(block.header_fields.get("name")) or block.title

        metadata: dict[str, MetadataValue] = {
            "course_id": course_id,
            "name": name,
            "type": DocumentType.SYLLABUS.value,
            "source": path.name,
        }

        credits = block.header_fields.get("credits")
        if isinstance(credits, float):
            metadata["credits"] = credits

        instructor = as_text(block.header_fields.get("instructor"))
        if instructor:
            metadata["instructor"] = instructor

        semester = as_text(block.header_fields.get("semester"))
        if semester:
            metadata["semester"] = semester

        # 只有真的解析到了先修课程这一项才写键。写空字符串会让抬头显示
        # "先修课程：无"——而这里的"没解析到"并不等于"没有先修课"。
        if "prerequisites" in block.header_fields:
            prerequisites = block.header_fields["prerequisites"]
            values = prerequisites if isinstance(prerequisites, list) else []
            metadata["prerequisites"] = join_list_values([str(item) for item in values])

        return metadata


# ---------------------------------------------------------------------------
# Markdown 解析辅助
# ---------------------------------------------------------------------------
def _split_into_course_blocks(text: str) -> list[_CourseBlock]:
    """按一级标题把 Markdown 切成课程块，并进一步解析抬头与章节。

    一级标题之前的内容会被丢弃——那通常是文件级说明或注释，不属于任何课程。
    如果有非空内容落在那里，记一条 WARNING，避免"写了却没生效"。
    """
    blocks: list[_CourseBlock] = []
    current_title: str | None = None
    current_lines: list[str] = []
    preamble_lines: list[str] = []

    for line in text.splitlines():
        match = _H1_PATTERN.match(line)
        if match:
            if current_title is not None:
                blocks.append(_parse_course_block(current_title, current_lines))
            current_title = match.group("title")
            current_lines = []
        elif current_title is None:
            preamble_lines.append(line)
        else:
            current_lines.append(line)

    if current_title is not None:
        blocks.append(_parse_course_block(current_title, current_lines))

    if any(line.strip() for line in preamble_lines):
        logger.warning(
            "Markdown 首个一级标题之前存在内容，这部分不会进入知识库 | lines=%d",
            sum(1 for line in preamble_lines if line.strip()),
        )

    return blocks


def _parse_course_block(title: str, lines: list[str]) -> _CourseBlock:
    """把课程块的原始行拆成"抬头"和"二级章节"。"""
    block = _CourseBlock(title=title)

    header_lines: list[str] = []
    current_section_title: str | None = None
    current_section_lines: list[str] = []

    for line in lines:
        match = _H2_PATTERN.match(line)
        if match:
            if current_section_title is not None:
                block.sections.append(
                    (current_section_title, "\n".join(current_section_lines).strip())
                )
            current_section_title = match.group("title")
            current_section_lines = []
        elif current_section_title is None:
            header_lines.append(line)
        else:
            current_section_lines.append(line)

    if current_section_title is not None:
        block.sections.append((current_section_title, "\n".join(current_section_lines).strip()))

    # 抬头里首尾的空行没有信息量，去掉；中间的空行保留原样。
    while header_lines and not header_lines[0].strip():
        header_lines.pop(0)
    while header_lines and not header_lines[-1].strip():
        header_lines.pop()

    block.header_lines = header_lines
    block.header_fields = _parse_header_fields(header_lines)
    return block


def _parse_header_fields(header_lines: Sequence[str]) -> dict[str, object]:
    """从抬头行里解析出结构化字段。

    ``###`` 及更深的标题不参与切分，会原样留在所属二级章节的正文里——本文档的
    数据粒度是二级章节，把子标题也切开会把一段完整论述拆散。
    """
    fields: dict[str, object] = {}

    for line in header_lines:
        match = _BULLET_PATTERN.match(line)
        if match is None:
            continue
        field_name = _HEADER_FIELD_ALIASES.get(match.group("key").strip())
        if field_name is None:
            # 未知键（比如"课程类型：专业必修"）保留在正文里即可，不报错。
            continue
        fields[field_name] = match.group("value").strip()

    if "credits" in fields:
        fields["credits"] = _parse_credits(str(fields["credits"]))

    if "prerequisites" in fields:
        fields["prerequisites"] = _parse_prerequisites(str(fields["prerequisites"]))

    return fields


def _parse_credits(raw: str) -> float:
    """把学分文本转成 float。"""
    try:
        return float(raw)
    except ValueError as exc:
        raise DocumentLoadError(f"学分字段无法解析为数字：{raw!r}") from exc


def _parse_prerequisites(raw: str) -> list[str]:
    """把先修课程文本拆成编号列表。"""
    normalized = raw.strip()
    if not normalized or normalized.lower() in _PREREQUISITE_NONE_MARKERS:
        return []
    parts = _PREREQUISITE_SEPARATORS.split(normalized)
    return [part.strip() for part in parts if part.strip()]


def _normalize_section_key(title: str) -> str:
    """把二级标题归一化成 ASCII section 键。

    命中别名表就用归一化键（便于过滤）；未命中则原样返回标题，宁可键名不统一
    也不要丢信息——把"实验安排"硬塞进某个已知类别只会造成误分类。
    """
    stripped = title.strip()
    return _SECTION_ALIASES.get(stripped, stripped)


def _compose_page_content(header: str, section_title: str, body: str) -> str:
    """拼装最终正文：课程抬头 + 章节标题 + 章节正文。

    章节标题一定要写进正文，而不是只放在 metadata 里：Embedding 只看正文，
    没有标题的话"考核方式"这一节的内容会缺少"这是考核方式"这层语义。
    """
    parts: list[str] = []
    if header:
        parts.append(header)
    parts.append(f"## {section_title}")
    if body:
        parts.append(body)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------
def _read_text(path: Path) -> str:
    """以 UTF-8 读取文本文件，编码错误时给出可操作的提示。"""
    try:
        return path.read_text(encoding=_ENCODING)
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"{path.name} 不是 UTF-8 编码，无法读取（{exc.reason}，位置 {exc.start}）。"
            f"请用编辑器把文件另存为 UTF-8 后重试。",
            details={"file": str(path), "encoding": _ENCODING},
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(
            f"{path.name} 读取失败：{exc.strerror or exc}",
            details={"file": str(path)},
        ) from exc




