"""``src.schemas.chat`` 的单元测试：接口契约的守门员。

这个模块是**对外接口的唯一契约**（Request / Response / Source / TokenUsage），
它的校验器决定了"什么请求能进到业务逻辑里"。以前它没有专属测试文件，只被别的
测试当构造器用——等于契约本身没人盯着。

重点覆盖三类容易悄悄坏掉的东西：

1. **边界值**（空问题、超长、top_k 越界、score 越界）；
2. **校验器**（去空白、去重、长度上限——这些"顺手做的事"改坏了不会报错）；
3. **默认值语义**（``top_k=None`` 表示"用服务端配置"，不是 5）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.chat import (
    MAX_PATTERN_CHARS,
    ChatRequest,
    ChatResponse,
    FAQCreateRequest,
    GenerationResult,
    IntentType,
    Source,
    TokenUsage,
    metadata_text,
)


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------
def test_top_k_defaults_to_none_meaning_use_server_setting() -> None:
    """``top_k`` 的默认值是 ``None`` 而不是 5。

    写死 5 会让运维改了 ``TOP_K`` 却不生效，而启动日志还打印着改后的值——
    这种"配置说了话但没算数"的情况最难排查。
    """
    request = ChatRequest(question="问题")

    assert request.top_k is None
    assert request.use_faq is True


@pytest.mark.parametrize("value", [1, 50])
def test_top_k_accepts_the_documented_range(value: int) -> None:
    """边界值 1 与 50 可用。"""
    assert ChatRequest(question="问题", top_k=value).top_k == value


@pytest.mark.parametrize("value", [0, -1, 51, 1000])
def test_top_k_rejects_out_of_range(value: int) -> None:
    """越界直接 422。"""
    with pytest.raises(ValidationError):
        ChatRequest(question="问题", top_k=value)


def test_question_is_stripped_and_blank_is_rejected() -> None:
    """去首尾空白；纯空白按空问题拒绝。"""
    assert ChatRequest(question="  先修课是什么？  ").question == "先修课是什么？"

    with pytest.raises(ValidationError):
        ChatRequest(question="   \n\t ")


def test_question_length_is_bounded() -> None:
    """上限 1000 字符（防粘贴一整篇文章）。"""
    assert len(ChatRequest(question="问" * 1000).question) == 1000

    with pytest.raises(ValidationError):
        ChatRequest(question="问" * 1001)


def test_unknown_fields_are_rejected() -> None:
    """字段拼错要被发现，而不是被静默忽略。"""
    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"question": "问题", "top_K": 5})


# ---------------------------------------------------------------------------
# Source / TokenUsage / GenerationResult
# ---------------------------------------------------------------------------
def test_source_score_must_be_a_relevance_score() -> None:
    """score 限定在 ``[0,1]``：传向量库的原始距离（可到 2）会被拒。"""
    assert Source(score=1.0).score == 1.0

    with pytest.raises(ValidationError):
        Source(score=1.4)


def test_source_allows_missing_metadata() -> None:
    """来源字段全部可选（纯文本资料没有 metadata 是允许的语料形态）。"""
    source = Source()

    assert source.course_id is None and source.name is None and source.score is None


def test_token_usage_rejects_negative_counts() -> None:
    """计数不能为负。"""
    assert TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0).estimated is False

    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1, completion_tokens=0, total_tokens=0)


def test_generation_result_requires_a_non_empty_answer() -> None:
    """空回答不是一个合法的生成结果（"没答"应当由上层显式表达）。"""
    assert GenerationResult(answer="有内容").sources == []

    with pytest.raises(ValidationError):
        GenerationResult(answer="")


def test_chat_response_matches_the_documented_shape() -> None:
    """响应字段就是文档里那五个，不多不少。"""
    response = ChatResponse(
        answer="答案", sources=[], route=IntentType.FAQ, cached=True, latency_ms=1.5
    )

    assert set(response.model_dump()) == {
        "answer",
        "sources",
        "route",
        "cached",
        "latency_ms",
    }


# ---------------------------------------------------------------------------
# FAQCreateRequest
# ---------------------------------------------------------------------------
def test_faq_request_cleans_patterns() -> None:
    """去空白、丢空项。"""
    request = FAQCreateRequest(patterns=["  CS101 几学分  ", "", "   "], answer="4 学分")

    assert request.patterns == ["CS101 几学分"]


def test_faq_request_requires_at_least_one_pattern() -> None:
    """一条有效的问法都没有就没有意义。"""
    with pytest.raises(ValidationError):
        FAQCreateRequest(patterns=["   "], answer="答案")

    with pytest.raises(ValidationError):
        FAQCreateRequest(patterns=[], answer="答案")


def test_faq_request_bounds_each_pattern_length() -> None:
    """单条问法有长度上限。

    ``patterns`` 是**无需鉴权**的写入口（未配令牌时），而每条 pattern 都会参与之后
    每一次 FAQ 未命中的相似度比对——不封顶等于给了一条免费拖慢全站的路径。
    """
    assert FAQCreateRequest(patterns=["问" * MAX_PATTERN_CHARS], answer="答案").patterns

    with pytest.raises(ValidationError):
        FAQCreateRequest(patterns=["问" * (MAX_PATTERN_CHARS + 1)], answer="答案")


def test_faq_request_rejects_blank_answer() -> None:
    """答案不能是纯空白。"""
    with pytest.raises(ValidationError):
        FAQCreateRequest(patterns=["问法"], answer="   ")


def test_faq_request_normalizes_blank_course_id() -> None:
    """空串课程编号归一成 None。"""
    assert FAQCreateRequest(patterns=["问法"], answer="答案", course_id="  ").course_id is None
    assert FAQCreateRequest(patterns=["问法"], answer="答案", course_id=" CS201 ").course_id == "CS201"


def test_faq_request_caps_pattern_count() -> None:
    """条数上限 20。"""
    with pytest.raises(ValidationError):
        FAQCreateRequest(patterns=[f"问法{index}" for index in range(21)], answer="答案")


# ---------------------------------------------------------------------------
# metadata_text
# ---------------------------------------------------------------------------
def test_metadata_text_picks_the_first_present_key() -> None:
    """按顺序取第一个非空值（section_title 优先于 section）。"""
    metadata = {"section_title": "考核方式", "section": "assessment"}

    assert metadata_text(metadata, "section_title", "section") == "考核方式"
    assert metadata_text({"section": "assessment"}, "section_title", "section") == "assessment"


def test_metadata_text_skips_none_and_blank() -> None:
    """``None`` 与空串都跳过；全空返回 ``None``。"""
    assert metadata_text({"a": None, "b": "有值"}, "a", "b") == "有值"
    assert metadata_text({"a": "   "}, "a") is None
    assert metadata_text({}, "a", "b") is None


def test_metadata_text_stringifies_scalars() -> None:
    """数字型 metadata 也能取（Chroma 只存标量）。"""
    assert metadata_text({"credits": 4.0}, "credits") == "4.0"
