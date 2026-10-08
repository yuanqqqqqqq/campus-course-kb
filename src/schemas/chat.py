"""问答业务数据契约：意图、来源、生成结果、HTTP 请求与响应。

本模块只放**纯数据模型**：不依赖 LangChain、不发起任何调用。把 ``Document`` 适配
成 :class:`Source` 的代码放在生成层（:func:`src.generation.llm.source_from_document`），
这样 schemas 层永远只有一个依赖（Pydantic），任何模块都能安全地引用它。

**为什么 :class:`IntentType` 在这里而不是路由层**

它同时是"内部判定的结果"和"对外响应里的一个字段"（``ChatResponse.route``）。
枚举属于数据契约，按本项目既有约定（``DocumentType`` 就在 ``schemas/course.py``）
放在 schemas；路由层再从 schemas 导入。反过来（schemas 依赖 routing）会让最底层
反向依赖上层，那正是分层要避免的。
"""

from __future__ import annotations

from enum import Enum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.schemas.common import MetadataValue

#: FAQ 单条问法的最大长度。
#:
#: 同时约束"接口写入"与"文件加载"两条路径：``POST /api/faq`` 无需鉴权且默认落盘，
#: 一条超长 pattern 会写进 ``data/faq.json``，并在之后**每次 FAQ 未命中时**参与全量
#: 相似度比对——那是一条不花钱就能拖慢所有问答的路径。
MAX_PATTERN_CHARS: Final[int] = 200


class IntentType(str, Enum):
    """四类意图。取值即 LLM 提示词里允许出现的字符串。

    - :attr:`FAQ`            —— 答案是某个固定的课程属性，应先查 FAQ 快路径
    - :attr:`COURSE_QUERY`   —— 需要检索课程语料才能回答
    - :attr:`PROCESS_QUERY`  —— 教务流程类问题（选课 / 学分认定 / 毕业要求）
    - :attr:`CHITCHAT`       —— 寒暄或与校园课程无关
    """

    FAQ = "faq"
    COURSE_QUERY = "course_query"
    PROCESS_QUERY = "process_query"
    CHITCHAT = "chitchat"


class Source(BaseModel):
    """一条答案来源，用于回答里的引用与前端展示。

    字段全部可选，因为来源质量取决于语料：Markdown 大纲一定有 ``course_id``，
    而临时加进来的纯文本资料可能什么都没有——宁可留空，也不要编一个"未知课程"。
    """

    model_config = ConfigDict(extra="forbid")

    course_id: str | None = Field(default=None, description="课程编号，例如 CS201。")
    name: str | None = Field(default=None, description="课程名称，例如 数据结构与算法。")
    type: str | None = Field(
        default=None,
        description=(
            "来源类型。取自文档 metadata['type']，目前有 course（结构化课程目录）与 "
            "syllabus（教学大纲）；FAQ 快路径命中时为 faq。"
        ),
    )
    source: str | None = Field(
        default=None,
        description="文档来源文件名（对应 metadata['source']），例如 course_syllabus.md。",
    )
    section: str | None = Field(
        default=None,
        description="章节标题，例如「考核方式」。取自 metadata['section_title']，缺失时退到 ['section']。",
    )
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "该来源的相关性分数，口径与 retrieval 层一致（[0,1] 的余弦相似度，越大越相关）。"
            "传原始距离会在这里被拒绝——两者方向相反，混用会让分数彻底失去意义。"
        ),
    )


class TokenUsage(BaseModel):
    """一次 LLM 调用的 token 用量。

    **``estimated=True`` 的含义**：用本地分词器（tiktoken 之类）数出来的近似值。
    DeepSeek 的 tokenizer 与 tiktoken 并不一致，把它当精确值去做成本核算会系统性
    偏差，所以任何估算值都必须带上这个标记。当前实现只填服务端回报的数字
    （``estimated=False``），拿不到就整个 ``usage`` 置 ``None``——不猜。
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0, description="输入 token 数（含系统提示词、资料与问题）。")
    completion_tokens: int = Field(ge=0, description="输出 token 数。")
    total_tokens: int = Field(ge=0, description="总 token 数。")
    estimated: bool = Field(
        default=False,
        description="是否为本地估算值。服务端回报的计数为 False。",
    )


class GenerationResult(BaseModel):
    """一次完整生成的结果（非流式）。

    比接口文档里声明的最低要求多了 ``usage`` 与 ``model`` 两个字段，理由和
    ``IntentResult`` 一样：排查线上问题（走的是哪个模型、这次花了多少 token）
    时，只有 ``answer`` 是看不出任何东西的。
    """

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, description="模型生成的回答，已去除首尾空白。")
    sources: list[Source] = Field(
        default_factory=list,
        description="回答依据的来源列表，已按课程 / 文件 / 章节去重。",
    )
    usage: TokenUsage | None = Field(
        default=None,
        description="token 用量；服务端没有回报时为 None（不估算）。",
    )
    model: str | None = Field(default=None, description="本次调用使用的模型名。")


# ---------------------------------------------------------------------------
# HTTP 请求 / 响应
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """``POST /api/chat`` 与 ``POST /api/chat/stream`` 的请求体。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="用户问题。纯空白会被拒绝（422），而不是当成一个空问题去检索。",
        examples=["数据结构的先修课是什么？"],
    )
    use_faq: bool = Field(
        default=True,
        description=(
            "是否允许走 FAQ 快路径。置 false 时即使意图判定为 faq 也直接走 RAG——"
            "调参、对比实验与排查「FAQ 是不是答错了」时用得上。"
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description=(
            "本次检索的候选文档数。**留空表示用服务端的 TOP_K 配置**（默认 5）——"
            "写死默认值会让运维改了 TOP_K 却不生效，还会让启动日志与实际行为对不上。"
        ),
    )

    @field_validator("question", mode="after")
    @classmethod
    def _reject_blank_question(cls, value: str) -> str:
        """去首尾空白并拒绝纯空白。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("question 不能为空字符串或纯空白字符")
        return stripped


class ChatResponse(BaseModel):
    """``POST /api/chat`` 的响应体。字段与契约一一对应，不额外夹带。"""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(description="回答文本。可能是课程资料生成的答案，也可能是拒答或引导话术。")
    sources: list[Source] = Field(
        default_factory=list,
        description="回答依据的来源；拒答、闲聊与 FAQ 命中时为 FAQ 条目来源。",
    )
    route: IntentType = Field(description="本次命中的意图，决定走了哪条链路。")
    cached: bool = Field(
        description="是否由 FAQ 快路径直接返回（true 表示没调用 Embedding / Chroma / LLM）。",
    )
    latency_ms: float = Field(ge=0.0, description="端到端耗时（毫秒），用 perf_counter 测量。")


class FAQCreateRequest(BaseModel):
    """``POST /api/faq`` 的请求体：动态新增一条 FAQ。"""

    model_config = ConfigDict(extra="forbid")

    patterns: list[str] = Field(
        min_length=1,
        max_length=20,
        description="唤起这条答案的问法，至少一条。建议写成完整问句并带上课程编号。",
        examples=[["CS101 多少学分", "程序设计基础学分是多少"]],
    )
    answer: str = Field(
        min_length=1,
        max_length=2000,
        description="命中后原样返回的答案（不经过 LLM 改写）。",
    )
    course_id: str | None = Field(default=None, max_length=32, description="关联课程编号。")

    @field_validator("patterns", mode="after")
    @classmethod
    def _clean_patterns(cls, values: list[str]) -> list[str]:
        """去空白、丢空项、限制单条长度；一条不剩就报错。

        长度上限是防御性的：这个接口无需鉴权，超长 pattern 会让之后每一次
        FAQ 未命中都多花一次 O(len) 的相似度计算。
        """
        cleaned = [pattern.strip() for pattern in values if pattern.strip()]
        if not cleaned:
            raise ValueError("patterns 至少要有一条非空问法")
        too_long = [pattern for pattern in cleaned if len(pattern) > MAX_PATTERN_CHARS]
        if too_long:
            raise ValueError(f"单条问法不能超过 {MAX_PATTERN_CHARS} 个字符")
        return cleaned

    @field_validator("answer", mode="after")
    @classmethod
    def _reject_blank_answer(cls, value: str) -> str:
        """答案不能是纯空白。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("answer 不能为空字符串或纯空白字符")
        return stripped

    @field_validator("course_id", mode="after")
    @classmethod
    def _blank_course_id_becomes_none(cls, value: str | None) -> str | None:
        """空串归一成 ``None``。"""
        if value is None:
            return None
        return value.strip() or None


class FAQCreateResponse(BaseModel):
    """``POST /api/faq`` 的响应体：新增后的完整条目。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="自动分配的条目 id，形如 faq_008。")
    patterns: list[str] = Field(description="实际写入的问法（已去重、去空白）。")
    answer: str = Field(description="实际写入的答案。")
    course_id: str | None = Field(default=None, description="关联课程编号。")
    saved: bool = Field(description="是否已落盘到 FAQ 文件（false 表示仅本次进程内生效）。")


def metadata_text(metadata: dict[str, MetadataValue], *keys: str) -> str | None:
    """按顺序取第一个非空的 metadata 文本值，全空则返回 ``None``。

    供来源适配使用：``section_title`` 与 ``section`` 是两个不同粒度的键，
    取值时总是"先具体、后笼统"。
    """
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None
