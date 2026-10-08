"""评测集的数据模型与加载。

**为什么用 Pydantic 严格校验评测集**

评测集写错字段名（``expected_course_id`` 写成 ``course_id``）时，最典型的症状不是
报错，而是**指标看起来很正常**——只是所有题的"期望课程"都成了空，Recall 分母变成 0
或 1，然后你会拿着一个毫无意义的数字去调阈值。所以这里的策略和 ``Course`` 一样：
字段写错、类型不对、缺必要字段，一律在加载阶段报错。

**字段与判定的关系**

``type`` 是"题目类型"，只用于分组统计（FAQ 命中率的分母就是 ``type == faq`` 的那些）；
``expected_route`` 是"期望路由"，参与 Route Accuracy。两者刻意分开：

- 「数据结构的学分是多少」这类问题在真实链路上会命中 FAQ 规则 → 走 FAQ 分支（大概率
  FAQ 库里没有 → 继续 RAG）。如果强行按"事实查询"把 ``expected_route`` 写成
  ``course_query``，那测的就不是系统行为，而是一厢情愿。所以：**类型按问法分，路由按
  系统应有的行为写**。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from src.schemas.chat import IntentType
from src.utils.exceptions import AppError
from src.utils.text import format_validation_error, simplify_validation_errors

#: 评测集默认位置。
DEFAULT_EVAL_SET_PATH: Final[Path] = Path(__file__).resolve().parent.parent / "eval_set.json"


class EvalDataError(AppError):
    """评测集读取或校验失败。"""

    default_code = "EVAL_DATA_ERROR"


class QuestionType(str, Enum):
    """题目类型，只用于分组统计。"""

    FACTUAL = "factual"
    """事实查询：某门课的学分 / 教师 / 考核方式等。"""

    RELATION = "relation"
    """关系查询：课程之间的先修关系、哪些课依赖某门课。"""

    PROCESS = "process"
    """流程查询：选课、缓考、重修等教务流程。"""

    FAQ = "faq"
    """应与 FAQ 库对应的问法（FAQ 命中率的分母）。"""

    REJECTION = "rejection"
    """知识库明确没有答案，应当拒答。"""


class EvalCase(BaseModel):
    """评测集里的一条用例。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, description="唯一标识，例如 eval_001。")
    question: str = Field(min_length=1, description="用户问题原文。")
    type: QuestionType = Field(description="题目类型，用于分组统计。")
    expected_route: IntentType = Field(description="期望命中的意图（Route Accuracy 的依据）。")
    expected_course_id: str | None = Field(
        default=None, description="期望检索到的课程编号（单值）。"
    )
    expected_course_ids: list[str] = Field(
        default_factory=list,
        description=(
            "期望检索到的课程编号（多值，用于「哪些课程需要 CS101」这类问题）。"
            "Recall@K 判定为：**任意一个**期望课程出现在前 K 条即算命中。"
        ),
    )
    should_reject: bool = Field(
        default=False,
        description="本题是否应当拒答。为 true 时不计入 Recall 与关键词命中。",
    )
    expected_answer_keywords: list[str] = Field(
        default_factory=list,
        description="回答里应当出现的字面关键词（全部出现才算命中）。",
    )
    note: str | None = Field(default=None, description="备注，例如为什么期望拒答。")

    @field_validator("expected_course_ids", mode="after")
    @classmethod
    def _dedupe_ids(cls, values: list[str]) -> list[str]:
        """去空白、去重、保持顺序。"""
        cleaned: list[str] = []
        for raw in values:
            item = raw.strip()
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned

    @model_validator(mode="after")
    def _check_consistency(self) -> EvalCase:
        """跨字段一致性检查：把"这条用例其实没法判定"的写法挡在加载阶段。"""
        if self.type is QuestionType.FAQ and self.expected_route is not IntentType.FAQ:
            raise ValueError("type=faq 的用例，expected_route 必须是 faq")
        if self.should_reject and self.expected_ids:
            raise ValueError("应当拒答的用例不该有期望课程——拒答的前提就是没有依据")
        if not self.should_reject and not self.expected_answer_keywords:
            raise ValueError(
                "可回答的用例必须给出 expected_answer_keywords，"
                "否则关键词命中率无从计算，这条用例也等于没测"
            )
        return self

    @property
    def expected_ids(self) -> list[str]:
        """合并单值与多值后的期望课程编号。"""
        ids = [self.expected_course_id] if self.expected_course_id else []
        for course_id in self.expected_course_ids:
            if course_id not in ids:
                ids.append(course_id)
        return ids


class EvalSet(BaseModel):
    """一份完整的评测集。"""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(default="1", description="评测集版本，改动题目时递增。")
    description: str | None = Field(default=None, description="这份评测集覆盖什么。")
    cases: list[EvalCase] = Field(min_length=1, description="用例列表。")

    @model_validator(mode="after")
    def _check_unique_ids(self) -> EvalSet:
        """id 重复会让报告里同一条出现两次，先拦住。"""
        seen: set[str] = set()
        duplicates: list[str] = []
        for case in self.cases:
            if case.id in seen:
                duplicates.append(case.id)
            seen.add(case.id)
        if duplicates:
            raise ValueError(f"评测集里 id 重复：{duplicates}")
        return self

    @property
    def reject_cases(self) -> list[EvalCase]:
        """应当拒答的用例。"""
        return [case for case in self.cases if case.should_reject]

    @property
    def faq_cases(self) -> list[EvalCase]:
        """FAQ 类型用例（FAQ 命中率的分母）。"""
        return [case for case in self.cases if case.type is QuestionType.FAQ]

    @property
    def recall_cases(self) -> list[EvalCase]:
        """参与 Recall 计算的用例（有期望课程且不应拒答）。"""
        return [case for case in self.cases if case.expected_ids and not case.should_reject]

    @property
    def keyword_cases(self) -> list[EvalCase]:
        """参与关键词命中率计算的用例。"""
        return [case for case in self.cases if case.expected_answer_keywords]


def load_eval_set(path: str | Path | None = None) -> EvalSet:
    """加载并校验评测集。

    兼容两种写法：``{"version": ..., "cases": [...]}`` 与裸数组 ``[...]``。后者是
    手写评测集时最自然的形式，兼容它一行代码的事，比让人改数据划算。

    :param path: 文件路径；``None`` 时用 :data:`DEFAULT_EVAL_SET_PATH`。
    :raises EvalDataError: 文件不存在、不是合法 JSON、结构不符、用例校验不过、id 重复。
    """
    target = Path(path) if path is not None else DEFAULT_EVAL_SET_PATH
    if not target.exists():
        raise EvalDataError(f"评测集不存在：{target}")

    try:
        text = target.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError) as exc:
        raise EvalDataError(f"评测集读取失败：{target}（{exc}）") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvalDataError(
            f"评测集不是合法 JSON：第 {exc.lineno} 行第 {exc.colno} 列 {exc.msg}",
            details={"path": str(target)},
        ) from exc

    normalized = _as_eval_set_payload(payload, target)
    try:
        return EvalSet.model_validate(normalized)
    except ValidationError as exc:
        raise EvalDataError(
            f"评测集校验失败：{format_validation_error(exc)}",
            details={"path": str(target), "errors": simplify_validation_errors(exc)},
        ) from exc


def _as_eval_set_payload(payload: object, path: Path) -> Mapping[str, object]:
    """把两种受支持的形态归一成 :class:`EvalSet` 的入参。"""
    if isinstance(payload, list):
        return {"cases": payload}
    if isinstance(payload, Mapping) and "cases" in payload:
        return payload
    raise EvalDataError(
        f"{path.name} 的结构无法识别。支持两种写法："
        f'{{"version": "1", "cases": [...]}} 包装对象，或直接的用例数组 [...]。',
        details={"path": str(path), "top_level_type": type(payload).__name__},
    )


def course_ids_from_hits(hits: Sequence[object]) -> list[str]:
    """从检索结果里取出课程编号（按相关性顺序，保留重复以便看排序）。

    只读 ``Document.metadata['course_id']``；取不到就跳过——评测不该因为某条资料
    缺 metadata 就整体失败。
    """
    ids: list[str] = []
    for hit in hits:
        document = hit[0] if isinstance(hit, tuple) else hit
        metadata = getattr(document, "metadata", None) or {}
        course_id = metadata.get("course_id")
        if course_id:
            ids.append(str(course_id))
    return ids



