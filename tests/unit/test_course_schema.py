"""``src.schemas.course`` 的单元测试。

覆盖三类关注点：字段校验是否足够严、metadata 是否满足向量库的标量约束、
正文渲染是否与抬头契约一致。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from src.schemas.course import (
    Assessment,
    Course,
    DocumentType,
    build_context_header,
    join_list_values,
)

#: 所有受支持的标量 metadata 类型。
_SCALAR_TYPES = (str, int, float, bool)


def test_valid_course_can_be_created(course_payload: Callable[..., dict[str, Any]]) -> None:
    """一份合法数据应当能顺利建成 Course。"""
    course = Course.model_validate(course_payload())

    assert course.course_id == "CS201"
    assert course.name == "数据结构与算法"
    assert course.credits == 4
    assert course.prerequisites == ["CS101"]
    assert course.assessment.exam == 60


def test_credits_accepts_fractional_value(course_payload: Callable[..., dict[str, Any]]) -> None:
    """学分允许小数（例如 3.5 学分）。"""
    course = Course.model_validate(course_payload(credits=3.5))
    assert course.credits == 3.5


@pytest.mark.parametrize("bad_credits", [0, 0.0, -1, -0.5])
def test_non_positive_credits_rejected(
    course_payload: Callable[..., dict[str, Any]], bad_credits: float
) -> None:
    """学分必须大于 0。"""
    with pytest.raises(ValidationError, match="credits"):
        Course.model_validate(course_payload(credits=bad_credits))


def test_absurd_credits_rejected(course_payload: Callable[..., dict[str, Any]]) -> None:
    """明显是录入笔误的学分应当被拦下。"""
    with pytest.raises(ValidationError, match="credits"):
        Course.model_validate(course_payload(credits=400))


@pytest.mark.parametrize("field", ["course_id", "name", "description", "instructor", "semester"])
@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_required_text_rejected(
    course_payload: Callable[..., dict[str, Any]], field: str, blank: str
) -> None:
    """必填文本字段拒绝空串与纯空白。

    ``min_length=1`` 拦不住 ``"   "``，所以这条用例真正验证的是额外的
    "去空白后非空"判定。
    """
    with pytest.raises(ValidationError, match=field):
        Course.model_validate(course_payload(**{field: blank}))


@pytest.mark.parametrize(
    "missing_field",
    ["course_id", "name", "credits", "instructor", "semester", "assessment", "description"],
)
def test_missing_required_field_rejected(
    course_payload: Callable[..., dict[str, Any]], missing_field: str
) -> None:
    """缺失任何必填字段都应报错（用 ... 作为哨兵表示"删掉该字段"）。"""
    with pytest.raises(ValidationError, match=missing_field):
        Course.model_validate(course_payload(**{missing_field: ...}))


@pytest.mark.parametrize("list_field", ["prerequisites", "textbooks", "tags"])
def test_list_fields_default_to_empty(
    course_payload: Callable[..., dict[str, Any]], list_field: str
) -> None:
    """列表字段不填时应当是空列表，而不是 None。"""
    course = Course.model_validate(course_payload(**{list_field: ...}))
    assert getattr(course, list_field) == []


def test_unknown_field_rejected(course_payload: Callable[..., dict[str, Any]]) -> None:
    """多余字段应当报错，避免把字段名拼错的数据当成合法数据放行。"""
    with pytest.raises(ValidationError, match=r"extra_forbidden|Extra inputs"):
        Course.model_validate(course_payload(difficulty="hard"))


def test_assessment_requires_all_three_keys() -> None:
    """考核方式的三个字段必须显式给出，漏写不能被当成"没有这一项"。"""
    with pytest.raises(ValidationError, match="project"):
        Assessment.model_validate({"exam": 60, "homework": 40})


@pytest.mark.parametrize("bad_value", [-1, 101, 999])
def test_assessment_value_out_of_range_rejected(bad_value: float) -> None:
    """考核占比必须在 0~100 之间。"""
    with pytest.raises(ValidationError, match="exam"):
        Assessment.model_validate({"exam": bad_value, "homework": 0, "project": 0})


def test_assessment_allows_none() -> None:
    """显式写 null 表示"没有这一考核项"，是合法数据。"""
    assessment = Assessment.model_validate({"exam": 70, "homework": 30, "project": None})
    assert assessment.project is None
    assert len(assessment.as_pairs()) == 2


def test_string_lists_are_normalized(course_payload: Callable[..., dict[str, Any]]) -> None:
    """列表字段去空白、丢空项、按原顺序去重。"""
    course = Course.model_validate(
        course_payload(prerequisites=[" CS101 ", "", "CS102", "CS101", "   "])
    )

    # 保持书写顺序（教学先后有含义），只是去掉重复与空项
    assert course.prerequisites == ["CS101", "CS102"]


def test_course_id_case_is_preserved(course_payload: Callable[..., dict[str, Any]]) -> None:
    """课程编号不做大小写归一化，保证与数据源可对账。"""
    course = Course.model_validate(course_payload(course_id="cs201"))
    assert course.course_id == "cs201"


# ---------------------------------------------------------------------------
# metadata 契约
# ---------------------------------------------------------------------------
def test_metadata_contains_required_keys(course_payload: Callable[..., dict[str, Any]]) -> None:
    """检索所依赖的必备键必须齐全。"""
    metadata = Course.model_validate(course_payload()).to_metadata(source="courses.json")

    for key in ("course_id", "name", "credits", "type", "source"):
        assert key in metadata, f"metadata 缺少必备键 {key}"

    assert metadata["type"] == DocumentType.COURSE.value
    assert metadata["source"] == "courses.json"


def test_metadata_values_are_scalars(course_payload: Callable[..., dict[str, Any]]) -> None:
    """所有 metadata 取值必须是标量。

    这是整个项目最关键的一条约束：ChromaDB 不接受 list / dict / None，
    一旦违反只会在阶段 3 建库时炸掉，所以在这里就钉死。
    """
    metadata = Course.model_validate(course_payload()).to_metadata(source="courses.json")

    for key, value in metadata.items():
        assert isinstance(value, _SCALAR_TYPES), f"{key} 的类型是 {type(value).__name__}，不是标量"
        assert value is not None, f"{key} 是 None，Chroma 不接受"


def test_metadata_flattens_lists(course_payload: Callable[..., dict[str, Any]]) -> None:
    """列表字段压平成逗号分隔的字符串。"""
    metadata = Course.model_validate(
        course_payload(prerequisites=["CS101", "CS102"], tags=["算法", "核心课"])
    ).to_metadata(source="courses.json")

    assert metadata["prerequisites"] == "CS101,CS102"
    assert metadata["tags"] == "算法,核心课"
    # 压平后可无损还原
    assert str(metadata["prerequisites"]).split(",") == ["CS101", "CS102"]


def test_metadata_omits_unset_assessment_items(
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """未设置的考核项不写键，避免"未设置"与"占比为 0"混淆。"""
    metadata = Course.model_validate(
        course_payload(assessment={"exam": 70, "homework": 30, "project": None})
    ).to_metadata(source="courses.json")

    assert metadata["assessment_exam"] == 70
    assert metadata["assessment_homework"] == 30
    assert "assessment_project" not in metadata


def test_metadata_empty_list_becomes_empty_string(
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """空列表压平成空字符串，而不是 "[]" 或 None。"""
    metadata = Course.model_validate(course_payload(prerequisites=[])).to_metadata(source="s.json")

    assert metadata["prerequisites"] == ""


# ---------------------------------------------------------------------------
# 正文渲染
# ---------------------------------------------------------------------------
def test_document_text_starts_with_context_header(
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """正文必须以便于切分器识别的抬头开头。"""
    course = Course.model_validate(course_payload())
    text = course.to_document_text(source="courses.json")

    header = build_context_header(course.to_metadata(source="courses.json"))
    assert text.startswith(header)
    assert "课程：数据结构与算法（CS201）" in header
    assert "先修课程：CS101" in header


def test_document_text_contains_soft_fields(course_payload: Callable[..., dict[str, Any]]) -> None:
    """简介、考核方式、教材、标签都要出现在正文里，否则 Embedding 看不到。"""
    text = Course.model_validate(course_payload()).to_document_text(source="courses.json")

    assert "课程简介" in text
    assert "系统讲授线性表" in text
    assert "期末考试 60%" in text
    assert "《数据结构（C 语言版）》" in text
    assert "专业核心课、算法" in text


def test_document_text_renders_missing_assessment(
    course_payload: Callable[..., dict[str, Any]],
) -> None:
    """三项考核全为空时，正文要明说"未公布"而不是留空白。"""
    text = Course.model_validate(
        course_payload(assessment={"exam": None, "homework": None, "project": None})
    ).to_document_text(source="courses.json")

    assert "考核方式\n未公布" in text


# ---------------------------------------------------------------------------
# build_context_header
# ---------------------------------------------------------------------------
def test_context_header_without_course_identity_returns_empty() -> None:
    """拿不到课程编号和名称时返回空串，调用方据此跳过抬头拼接。"""
    assert build_context_header({}) == ""
    assert build_context_header({"section": "content"}) == ""


def test_context_header_marks_absent_prerequisites() -> None:
    """解析到了先修课程但值为空时，明确写出"无"。

    否则"CS101 没有先修课"这条事实在知识库里根本检索不到。
    """
    header = build_context_header({"course_id": "CS101", "name": "程序设计基础", "prerequisites": ""})
    assert "先修课程：无" in header


def test_context_header_omits_prerequisites_when_key_absent() -> None:
    """压根没解析到先修课程这项时，不显示这一行，避免误导成"没有先修课"。"""
    header = build_context_header({"course_id": "CS101", "name": "程序设计基础"})

    assert "先修课程" not in header
    assert header == "课程：程序设计基础（CS101）"


def test_context_header_formats_integral_credits_without_decimal() -> None:
    """4.0 学分显示为 "4"，不是 "4.0"。"""
    header = build_context_header({"course_id": "CS201", "name": "数据结构与算法", "credits": 4.0})
    assert "学分：4" in header
    assert "学分：4.0" not in header


def test_context_header_keeps_fractional_credits() -> None:
    """3.5 学分要保留小数。"""
    header = build_context_header({"course_id": "CS301", "name": "数据库系统", "credits": 3.5})
    assert "学分：3.5" in header


def test_context_header_handles_name_only() -> None:
    """只有课程名时也能生成抬头，不应崩溃或漏掉课程名。"""
    assert build_context_header({"name": "数据结构与算法"}) == "课程：数据结构与算法"


def test_join_list_values_matches_course_metadata_convention() -> None:
    """独立的压平函数与 Course.to_metadata 口径必须一致。"""
    assert join_list_values(["CS101", "CS102"]) == "CS101,CS102"
    assert join_list_values([]) == ""
