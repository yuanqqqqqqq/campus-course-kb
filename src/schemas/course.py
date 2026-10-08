"""课程领域模型：课程数据结构 + 文档元数据契约。

本模块在"课程"这一实体的职责范围内做三件事：

1. 定义并校验课程数据（:class:`Course` / :class:`Assessment`）；
2. 定义 :class:`DocumentType`，区分"整门课"与"大纲章节"两类文档；
3. 提供把课程实体渲染成 *(a)* 文档正文、*(b)* 检索元数据的**唯一实现**，
   保证 JSON 加载器与 Markdown 加载器产出同构的结果。

**为什么渲染逻辑放在 schema 层而不是 loader 层**

向量库（ChromaDB）对 ``metadata`` 只接受标量值（``str`` / ``int`` / ``float`` /
``bool``），不接受 list、dict、None。这条约束如果散落在各个加载器里，迟早会有
一处违反，然后在建库时才炸。集中放在 :meth:`Course.to_metadata` 里消化掉，
约束就只有一个出口。

同理，:func:`build_context_header` 也放在这里：Loader 用它拼正文、Splitter 用它
在二次切分后把抬头贴回子片段，两边必须算出完全一致的结果。让它们调用同一个函数
是唯一可靠的保证方式（放在 loader 里会让 splitter 反向依赖 loader）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schemas.common import MetadataValue
from src.utils.text import as_text

#: 组合列表型字段时使用的分隔符。
#:
#: metadata 里存的是标量，所以 list 必须压平。用半角逗号而不是中文顿号 "",
#: 是为了让下游（评测脚本、SQL 式过滤）能直接 ``split(",")`` 还原。
_LIST_SEPARATOR: Final[str] = ","

#: 未填写先修课程时，抬头里显示的占位文案。
_NO_PREREQUISITE_TEXT: Final[str] = "无"

#: 来自结构化 JSON 的"整门课"文档使用的 section 标识。
SECTION_KEY_COURSE_PROFILE: Final[str] = "course_profile"
SECTION_TITLE_COURSE_PROFILE: Final[str] = "课程总览"


class DocumentType(str, Enum):
    """文档粒度的判别标签，写入 ``metadata["type"]``。

    两类文档并存是有意为之：``courses.json`` 给出的是**每门课一条**的概览，
    而教学大纲 Markdown 给出的是**每门课多段**的细节。检索时两种粒度都有用，
    用 ``type`` 区分后可以按需过滤。
    """

    COURSE = "course"
    SYLLABUS = "syllabus"


class Assessment(BaseModel):
    """考核方式构成，单位为百分制占比。

    三个字段都可为 ``None``（表示该课程未设置该考核项），但**必须显式给出**：
    数据里漏写一个字段会直接报错，而不是被静默当成"没有"。这是刻意的严格
    选择——考核占比缺失会直接影响"这门课怎么考核"这类问题的答案质量。

    注意：本模型**不校验三项之和等于 100**。现实中还有出勤、答辩等未建模的
    考核项，强行要求求和等于 100 会把合法数据挡在门外。需要校验时应在数据
    准备环节单独做。
    """

    model_config = ConfigDict(extra="forbid")

    exam: float | None = Field(ge=0, le=100, description="期末考试占比（百分制）")
    homework: float | None = Field(ge=0, le=100, description="平时作业占比（百分制）")
    project: float | None = Field(ge=0, le=100, description="实验 / 项目占比（百分制）")

    def as_pairs(self) -> list[tuple[str, str, float]]:
        """返回 ``[(metadata 键, 中文标签, 数值)]``，只包含已设置的考核项。

        供 :meth:`Course.to_metadata` 与正文渲染共用，避免两处重复维护
        "字段名 → 中文标签"的映射。
        """
        candidates: list[tuple[str, str, float | None]] = [
            ("assessment_exam", "期末考试", self.exam),
            ("assessment_homework", "平时作业", self.homework),
            ("assessment_project", "实验项目", self.project),
        ]
        return [(key, label, value) for key, label, value in candidates if value is not None]


class Course(BaseModel):
    """一门课程的完整结构化描述。

    对应 ``data/raw/courses.json`` 中的一条记录。校验策略偏严格：字段名写错、
    类型不对、必要字段缺失都会在加载阶段报错，而不是带着脏数据建完库之后再
    靠人肉排查检索效果。
    """

    model_config = ConfigDict(extra="forbid")

    course_id: str = Field(
        min_length=1,
        max_length=32,
        description="课程编号，例如 CS201。是贯穿全文档的主键，不可为空。",
    )
    name: str = Field(min_length=1, max_length=200, description="课程名称。")
    credits: float = Field(
        gt=0,
        le=30,
        description="学分。必须大于 0；上界 30 只是为了防止录入时多打一个 0 这类笔误。",
    )
    prerequisites: list[str] = Field(
        default_factory=list,
        description="先修课程编号列表，填的是其他课程的 course_id。无先修课时为空列表。",
    )
    instructor: str = Field(min_length=1, max_length=100, description="授课教师。")
    semester: str = Field(min_length=1, max_length=50, description="开课学期，例如 2026-2027-1。")
    assessment: Assessment = Field(description="考核方式构成。")
    description: str = Field(min_length=1, description="课程简介。不可为空。")
    textbooks: list[str] = Field(default_factory=list, description="参考教材列表。")
    tags: list[str] = Field(
        default_factory=list,
        description="标签，例如「核心课」「算法」。用于召回后的过滤与结果归组。",
    )

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    @field_validator("course_id", "name", "instructor", "semester", "description", mode="after")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        """去掉首尾空白，并拒绝"全是空格"的字符串。

        ``min_length=1`` 拦不住 ``"   "``，所以必须额外做一次判定——否则
        ``course_id=" "`` 会带着一个空白主键一路混到向量库里。
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("不能为空字符串或纯空白字符")
        return stripped

    @field_validator("prerequisites", "textbooks", "tags", mode="after")
    @classmethod
    def _normalize_string_list(cls, values: list[str]) -> list[str]:
        """列表字段统一做：去首尾空白 → 丢弃空项 → 按原顺序去重。

        去重是为了应付 ``["CS101", " CS101 "]`` 这种脏数据；保持原顺序而不是
        排序，是为了让先修课的书写顺序（通常有教学先后含义）不被破坏。
        """
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in values:
            item = raw.strip()
            if not item or item in seen:
                continue
            seen.add(item)
            cleaned.append(item)
        return cleaned

    @field_validator("course_id", mode="after")
    @classmethod
    def _keep_course_id_verbatim(cls, value: str) -> str:
        """课程编号只去首尾空白，**不改变大小写**。

        曾考虑统一转成大写以规避 ``cs201`` / ``CS201`` 这种重复，但那样会让
        入库的编号与教务原始数据不一致，反而难以对账。保持原样，大小写不一致
        的问题应在数据准备阶段解决。
        """
        return value

    # ------------------------------------------------------------------
    # 渲染：结构化数据 → 检索元数据
    # ------------------------------------------------------------------
    def to_metadata(self, source: str) -> dict[str, MetadataValue]:
        """把课程压平成 Chroma 可接受的标量 metadata。

        :param source: 来源文件名，写入 ``metadata["source"]``，用于回溯。

        列表压平成逗号分隔的字符串（空列表 → 空字符串），``None`` 的考核项
        直接**不写入该键**（而不是写 ``"None"`` 或 ``0``，那会让"未设置"和
        "占比为 0"变得无法区分）。
        """
        metadata: dict[str, MetadataValue] = {
            "course_id": self.course_id,
            "name": self.name,
            "credits": self.credits,
            "type": DocumentType.COURSE.value,
            "source": source,
            "instructor": self.instructor,
            "semester": self.semester,
            "prerequisites": _LIST_SEPARATOR.join(self.prerequisites),
            "textbooks": _LIST_SEPARATOR.join(self.textbooks),
            "tags": _LIST_SEPARATOR.join(self.tags),
            "section": SECTION_KEY_COURSE_PROFILE,
            "section_title": SECTION_TITLE_COURSE_PROFILE,
        }
        for key, _label, value in self.assessment.as_pairs():
            metadata[key] = value
        return metadata

    # ------------------------------------------------------------------
    # 渲染：结构化数据 → 文档正文
    # ------------------------------------------------------------------
    def to_document_text(self, source: str = "") -> str:
        """渲染成用于 Embedding 的正文。

        :param source: 透传给 :meth:`to_metadata`，仅影响抬头里的先修课程行。

        结构上分两段：先是由 metadata 推导出的**统一抬头**
        （课程名 / 编号 / 学分 / 教师 / 学期 / 先修课），再是描述性正文
        （简介 / 考核方式 / 教材 / 标签）。抬头部分必须与
        :func:`build_context_header` 的输出完全一致，Splitter 才能正确地
        剥离与还原。
        """
        header = build_context_header(self.to_metadata(source))
        body = self._render_body()
        if not body:
            return header
        return f"{header}\n\n{body}"

    def _render_body(self) -> str:
        """渲染抬头之外的自然语言正文。"""
        blocks: list[str] = []

        blocks.append(f"课程简介\n{self.description}")

        assessment_pairs = self.assessment.as_pairs()
        if assessment_pairs:
            joined = "、".join(f"{label} {value:g}%" for _key, label, value in assessment_pairs)
            blocks.append(f"考核方式\n{joined}")
        else:
            blocks.append("考核方式\n未公布")

        if self.textbooks:
            blocks.append("参考教材\n" + "\n".join(f"《{title}》" for title in self.textbooks))

        if self.tags:
            blocks.append("标签\n" + "、".join(self.tags))

        return "\n\n".join(blocks)


def build_context_header(metadata: Mapping[str, object]) -> str:
    """根据 metadata 生成统一的"课程上下文抬头"。

    抬头会被前置到每个 Document 的 ``page_content`` 开头，作用有两个：

    1. **让每一段切片自带上下文。** 否则检索出来的可能是一条孤零零的
       "期末考试：60%"，向量模型和 LLM 都不知道它说的是哪门课。
    2. **让 Splitter 能在二次切分后把抬头贴回每个子片段。**

    抬头完全由 metadata 推导，只要 metadata 不变，抬头就稳定，因此 Loader 与
    Splitter 各自计算也不会出现不一致。

    :param metadata: 至少包含 ``course_id`` / ``name``；其余字段缺失时对应的
        行会被省略。``course_id`` 与 ``name`` 都取不到时返回空字符串，调用方
        应当据此跳过抬头拼接。
    """
    course_id = as_text(metadata.get("course_id"))
    name = as_text(metadata.get("name"))
    if not course_id and not name:
        return ""

    title = f"课程：{name}（{course_id}）" if course_id else f"课程：{name}"

    details: list[str] = []

    credits = metadata.get("credits")
    # bool 是 int 的子类，先排除掉，避免 True 被格式化成"学分：1"
    if isinstance(credits, (int, float)) and not isinstance(credits, bool):
        details.append(f"学分：{credits:g}")

    instructor = as_text(metadata.get("instructor"))
    if instructor:
        details.append(f"授课教师：{instructor}")

    semester = as_text(metadata.get("semester"))
    if semester:
        details.append(f"开课学期：{semester}")

    # 先修课程要么不出现（该来源没解析到这项），要么出现且明确写出"无"。
    # 直接省略空值会让"CS101 没有先修课"这条事实无法被检索到。
    if "prerequisites" in metadata:
        prerequisites = as_text(metadata.get("prerequisites"))
        details.append(f"先修课程：{prerequisites or _NO_PREREQUISITE_TEXT}")

    if not details:
        return title
    return f"{title}\n" + " | ".join(details)



def join_list_values(values: Sequence[str]) -> str:
    """把字符串列表压平成 metadata 标量，与 :meth:`Course.to_metadata` 口径一致。"""
    return _LIST_SEPARATOR.join(values)
