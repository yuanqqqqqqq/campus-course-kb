"""``src.routing.classifier`` 的单元测试。

重点验证四件事：

1. **只有高置信度 FAQ 规则能跳过 LLM**，其余规则命中都要过 LLM 这一关；
2. LLM 响应必须是"合法 JSON 对象 + 合法 intent + [0,1] 内的 confidence"才算数，
   其余一律退化，**不猜测**；
3. 退化顺序是 规则候选 → 默认 course_query，绝不退到 chitchat；
4. 阈值、兜底意图都是可配置的（而不是写死在代码里）。

测试全部用假 LLM 回调，不联网、不依赖 API Key。
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from src.config import settings
from src.routing.classifier import (
    DEFAULT_INTENT,
    FALLBACK_CONFIDENCE,
    ClassifiedBy,
    IntentClassifier,
    IntentResult,
    IntentType,
)
from src.utils.exceptions import ConfigurationError

#: 一个命不中任何规则的问题，用来观察"纯 LLM / 纯兜底"的行为。
NEUTRAL_QUESTION = "图书馆几点关门"

#: 命中 FAQ 规则（"几学分"）的问题。
FAQ_QUESTION = "CS101 几学分？"

#: 命中 course_query 规则（"主要讲"、"讲什么"）但**不允许短路**的问题。
COURSE_QUESTION = "数据结构与算法主要讲什么"


class RecordingLLM:
    """假 LLM 分类回调：记录提示词，返回预设响应（或按预设抛异常）。

    :param response: 返回给调用方的响应。传 :class:`Exception` 实例表示"调用抛异常"。
    """

    def __init__(self, response: object) -> None:
        self._response = response
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        """被调用的次数。"""
        return len(self.prompts)

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if isinstance(self._response, Exception):
            raise self._response
        assert isinstance(self._response, str)
        return self._response


@pytest.fixture
def make_llm() -> Callable[[object], RecordingLLM]:
    """返回一个"造假 LLM"的工厂。"""

    def _factory(response: object) -> RecordingLLM:
        return RecordingLLM(response)

    return _factory


def _json(intent: str, confidence: object = 0.9) -> str:
    """造一段规范的 LLM 响应。"""
    return f'{{"intent": "{intent}", "confidence": {confidence}}}'


# ---------------------------------------------------------------------------
# 规则层：只有 FAQ 能短路
# ---------------------------------------------------------------------------
def test_high_confidence_faq_rule_short_circuits_without_llm(make_llm) -> None:
    """FAQ 规则达到阈值时直接给结论，LLM 一次都不该被调用。"""
    llm = make_llm(_json("chitchat", 0.99))

    result = IntentClassifier(llm=llm).classify(FAQ_QUESTION)

    assert result.intent is IntentType.FAQ
    assert result.classified_by is ClassifiedBy.RULE
    assert result.confidence >= 0.9
    assert llm.calls == 0, "FAQ 短路后仍调用了 LLM"


def test_faq_rule_beats_course_code_rule(make_llm) -> None:
    """"CS101 几学分" 同时命中 FAQ 规则与课号正则，应当稳定取 FAQ（置信度更高）。"""
    llm = make_llm(_json("course_query", 0.99))

    result = IntentClassifier(llm=llm).classify("CS101几学分")

    assert result.intent is IntentType.FAQ
    assert llm.calls == 0


def test_course_query_keyword_does_not_short_circuit(make_llm) -> None:
    """关键词命中 course_query 时**必须**再过一次 LLM —— 关键词不足以定论。"""
    llm = make_llm(_json("course_query", 0.96))

    result = IntentClassifier(llm=llm).classify(COURSE_QUESTION)

    assert llm.calls == 1
    assert result.intent is IntentType.COURSE_QUERY
    assert result.classified_by is ClassifiedBy.LLM


def test_chitchat_keyword_does_not_short_circuit(make_llm) -> None:
    """寒暄关键词同样不短路：把正经问题误判成闲聊的代价太大。"""
    llm = make_llm(_json("chitchat", 0.9))

    result = IntentClassifier(llm=llm).classify("你好啊")

    assert llm.calls == 1
    assert result.intent is IntentType.CHITCHAT
    assert result.classified_by is ClassifiedBy.LLM


def test_rule_threshold_is_configurable(make_llm) -> None:
    """把阈值提到 1.0，FAQ 规则（0.95）就不再短路，应当落到 LLM。"""
    llm = make_llm(_json("faq", 0.99))

    result = IntentClassifier(llm=llm, rule_threshold=1.0).classify(FAQ_QUESTION)

    assert llm.calls == 1
    assert result.classified_by is ClassifiedBy.LLM


def test_default_rule_threshold_comes_from_settings() -> None:
    """默认阈值取自配置，而不是写死在代码里。"""
    assert IntentClassifier().rule_threshold == settings.intent_rule_threshold


def test_matched_keywords_are_recorded_and_boost_confidence(make_llm) -> None:
    """多个关键词同时命中时置信度略高，且命中的词会被记录（便于排查）。"""
    llm = make_llm(_json("faq", 0.99))

    result = IntentClassifier(llm=llm).classify("这门课几学分？平时分怎么算？")

    assert set(result.matched_keywords) >= {"几学分", "平时分"}
    assert 0.95 < result.confidence <= 1.0
    assert llm.calls == 0


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_invalid_rule_threshold_raises(threshold: float) -> None:
    """阈值越界属于配置错误，构造时就报。"""
    with pytest.raises(ConfigurationError):
        IntentClassifier(rule_threshold=threshold)


# ---------------------------------------------------------------------------
# LLM 层：严格 JSON
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("intent", ["course_query", "faq", "process_query", "chitchat"])
def test_llm_pure_json_is_used(make_llm, intent: str) -> None:
    """规范的 JSON 直接采用，四类意图都能过。"""
    llm = make_llm(_json(intent, 0.88))

    result = IntentClassifier(llm=llm).classify(NEUTRAL_QUESTION)

    assert result.intent is IntentType(intent)
    assert result.confidence == pytest.approx(0.88)
    assert result.classified_by is ClassifiedBy.LLM
    assert llm.calls == 1


def test_llm_json_in_code_fence_is_used(make_llm) -> None:
    """模型把 JSON 包进 Markdown 代码块是常见偏离，应当被剥掉后照常解析。"""
    llm = make_llm('```json\n{"intent": "process_query", "confidence": 0.91}\n```')

    result = IntentClassifier(llm=llm).classify(NEUTRAL_QUESTION)

    assert result.intent is IntentType.PROCESS_QUERY
    assert result.classified_by is ClassifiedBy.LLM


def test_llm_json_with_surrounding_prose_is_recovered(make_llm) -> None:
    """JSON 前后夹了人话时做最后一次补救（并记 WARNING）。"""
    llm = make_llm('分类结果如下：\n{"intent": "chitchat", "confidence": 0.7}\n以上。')

    result = IntentClassifier(llm=llm).classify(NEUTRAL_QUESTION)

    assert result.intent is IntentType.CHITCHAT
    assert result.classified_by is ClassifiedBy.LLM


def test_llm_intent_is_case_insensitive(make_llm) -> None:
    """``intent`` 的大小写不影响解析。"""
    llm = make_llm('{"intent": "FAQ", "confidence": 0.9}')

    assert IntentClassifier(llm=llm).classify(NEUTRAL_QUESTION).intent is IntentType.FAQ


def test_llm_extra_keys_are_ignored(make_llm) -> None:
    """多余字段（模型爱补个 reason）不影响解析——只取契约里的两个字段。"""
    llm = make_llm('{"intent": "faq", "confidence": 0.9, "reason": "看起来像学分问题"}')

    result = IntentClassifier(llm=llm).classify(NEUTRAL_QUESTION)

    assert result.intent is IntentType.FAQ


def test_llm_numeric_string_confidence_is_accepted(make_llm) -> None:
    """``"0.9"`` 这种数字字符串照收（模型偶尔会给字符串）。"""
    llm = make_llm('{"intent": "faq", "confidence": "0.9"}')

    assert IntentClassifier(llm=llm).classify(NEUTRAL_QUESTION).confidence == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("response", "reason_hint"),
    [
        ("course_query", "自然语言，没有 JSON"),
        ("", "空响应"),
        ("我不知道该怎么分类。", "没有 JSON 对象"),
        ('[{"intent": "faq", "confidence": 0.9}]', "顶层是数组不是对象"),
        ('{"intent": "course", "confidence": 0.9}', "intent 取值不在枚举内"),
        ('{"intent": "", "confidence": 0.9}', "intent 为空"),
        ('{"intent": "faq"}', "缺 confidence"),
        ('{"intent": "faq", "confidence": 1.4}', "confidence 超上界"),
        ('{"intent": "faq", "confidence": -0.2}', "confidence 低于下界"),
        ('{"intent": "faq", "confidence": "high"}', "confidence 不是数字"),
        ('{"intent": "faq", "confidence": true}', "confidence 是布尔值"),
        ("null", "顶层是 null"),
    ],
)
def test_unparseable_llm_response_falls_back(make_llm, response: str, reason_hint: str) -> None:
    """任何不合规的响应都必须退到安全默认路由，而不是猜一个意图。"""
    llm = make_llm(response)

    result = IntentClassifier(llm=llm).classify(NEUTRAL_QUESTION)

    assert result.intent is DEFAULT_INTENT, reason_hint
    assert result.classified_by is ClassifiedBy.FALLBACK, reason_hint
    assert result.confidence == FALLBACK_CONFIDENCE, reason_hint
    assert "无法解析" in result.reason, reason_hint


def test_unparseable_response_falls_back_to_rule_candidate(make_llm) -> None:
    """有规则候选时优先用候选，而不是直接落到默认意图。"""
    llm = make_llm("我不太确定")

    result = IntentClassifier(llm=llm).classify(COURSE_QUESTION)

    assert result.intent is IntentType.COURSE_QUERY
    assert result.classified_by is ClassifiedBy.RULE
    assert result.confidence > FALLBACK_CONFIDENCE
    assert "候选" in result.reason


# ---------------------------------------------------------------------------
# 失败与退化
# ---------------------------------------------------------------------------
def test_llm_exception_falls_back_to_rule_candidate(make_llm) -> None:
    """LLM 调用抛异常时不能把整条链路带崩，退到规则候选。"""
    llm = make_llm(RuntimeError("connection reset"))

    result = IntentClassifier(llm=llm).classify(COURSE_QUESTION)

    assert result.intent is IntentType.COURSE_QUERY
    assert result.classified_by is ClassifiedBy.RULE
    assert "失败" in result.reason


def test_llm_exception_without_candidate_uses_default(make_llm) -> None:
    """没有规则候选时退到默认意图（course_query → RAG）。"""
    llm = make_llm(TimeoutError("timeout"))

    result = IntentClassifier(llm=llm).classify(NEUTRAL_QUESTION)

    assert result.intent is DEFAULT_INTENT
    assert result.classified_by is ClassifiedBy.FALLBACK


def test_llm_failure_never_returns_chitchat(make_llm) -> None:
    """退化路径绝不能落到 chitchat —— 失败方向必须是"去检索"。"""
    llm = make_llm(RuntimeError("boom"))

    for question in (NEUTRAL_QUESTION, FAQ_QUESTION, COURSE_QUESTION, "你好"):
        result = IntentClassifier(llm=llm).classify(question)
        assert result.intent is not IntentType.CHITCHAT


def test_chitchat_rule_candidate_is_discarded_on_failure(make_llm) -> None:
    """LLM 不可用时，寒暄关键词命中不能让问题降级成闲聊。

    "谢谢啦"只命中寒暄规则。照搬候选会得到一个 chitchat 结论，而闲聊回复里没有
    任何课程信息；退到 course_query 去检索至少还有可能答对。
    """
    llm = make_llm(RuntimeError("boom"))
    classifier = IntentClassifier(llm=llm)

    result = classifier.classify("谢谢啦")

    assert result.intent is DEFAULT_INTENT
    assert result.classified_by is ClassifiedBy.FALLBACK

    # 反过来，"你好"开头的正经问题本来就该走 FAQ 规则，压根轮不到寒暄规则说话
    faq_result = classifier.classify("你好，数据结构的先修课是什么")

    assert faq_result.intent is IntentType.FAQ
    assert llm.calls == 1, "FAQ 规则短路后不该再调用 LLM"


def test_no_llm_injected_uses_rule_candidate() -> None:
    """未接入生成层（阶段 5 之前）时，规则候选仍可用，且不抛异常。"""
    result = IntentClassifier().classify(COURSE_QUESTION)

    assert result.intent is IntentType.COURSE_QUERY
    assert result.classified_by is ClassifiedBy.RULE


def test_no_llm_injected_without_candidate_uses_default() -> None:
    """既没有 LLM 也没有规则命中时落到默认意图，置信度为 0。"""
    classifier = IntentClassifier()

    assert classifier.has_llm is False
    result = classifier.classify(NEUTRAL_QUESTION)

    assert result.intent is DEFAULT_INTENT
    assert result.classified_by is ClassifiedBy.FALLBACK
    assert result.confidence == FALLBACK_CONFIDENCE


def test_fallback_intent_is_overridable() -> None:
    """兜底意图可配置——默认 course_query 是决策，不是硬编码。"""
    classifier = IntentClassifier(fallback_intent=IntentType.CHITCHAT)

    result = classifier.classify(NEUTRAL_QUESTION)

    assert classifier.fallback_intent is IntentType.CHITCHAT
    assert result.intent is IntentType.CHITCHAT


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_question_uses_default_route(question: str) -> None:
    """空问题不发 LLM 调用，直接兜底。"""
    llm = RecordingLLM(_json("faq", 0.99))

    result = IntentClassifier(llm=llm).classify(question)

    assert result.intent is DEFAULT_INTENT
    assert result.classified_by is ClassifiedBy.FALLBACK
    assert llm.calls == 0


def test_llm_is_called_at_most_once(make_llm) -> None:
    """本层不做重试（重试是生成层的职责），一次分类只发一次请求。"""
    llm = make_llm(_json("process_query", 0.8))

    IntentClassifier(llm=llm).classify("选课系统怎么用")

    assert llm.calls == 1


# ---------------------------------------------------------------------------
# 提示词与结果模型
# ---------------------------------------------------------------------------
def test_prompt_contains_json_example_and_question(make_llm) -> None:
    """提示词里的 JSON 示例不能被 ``str.format`` 吃掉，问题要原样带入。"""
    llm = make_llm(_json("course_query", 0.9))

    IntentClassifier(llm=llm).classify(COURSE_QUESTION)

    prompt = llm.prompts[0]
    assert '{"intent": "course_query", "confidence": 0.96}' in prompt
    assert COURSE_QUESTION in prompt
    assert "{question}" not in prompt
    for intent in IntentType:
        assert intent.value in prompt


def test_prompt_is_not_sent_when_faq_short_circuits(make_llm) -> None:
    """短路时连提示词都不该拼——这条路径上不该产生任何 LLM 相关开销。"""
    llm = make_llm(_json("faq", 0.9))

    IntentClassifier(llm=llm).classify(FAQ_QUESTION)

    assert llm.prompts == []


def test_intent_result_rejects_out_of_range_confidence() -> None:
    """结果模型自己守住 confidence 的取值范围。"""
    with pytest.raises(ValidationError):
        IntentResult(intent=IntentType.FAQ, confidence=1.5)

    with pytest.raises(ValidationError):
        IntentResult(intent=IntentType.FAQ, confidence=-0.1)
