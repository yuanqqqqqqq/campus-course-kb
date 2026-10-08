"""意图路由：把用户问题归到四类之一，决定后续走 FAQ 快路径还是检索链路。

**这一层存在的意义是"少花钱、少出错"**

如果什么判断都交给 LLM，一次问答会变成"分类 LLM + 相关性 LLM + 生成 LLM"
三次串行调用，延迟与费用翻倍，而且每一步都可能把上一步的结论带偏。本项目把
判断拆成两层：

===========  ====================================  ================
层           依据                                  成本
===========  ====================================  ================
规则路由     关键词 / 课号正则（纯字符串匹配）      零
LLM 分类     模型对整句的理解                      一次网络调用
===========  ====================================  ================

规则层只回答一个问题：**"这个问句该不该去翻 FAQ 库"**。只有 FAQ 类规则允许在
高置信度下直接给出结论，其余意图一律交给 LLM。

**为什么只有 FAQ 能短路**

因为 FAQ 的最终判定不是关键词，而是下一步
:class:`~src.routing.faq_cache.FAQCache` 的精确 / 模糊匹配——关键词只负责决定
"要不要去翻这本册子"，翻不到就继续走 RAG（见 FAQCache 的模块文档）。
``course_query`` / ``process_query`` 没有这样一道高精度的兜底校验：关键词说
"这是课程问题"并不足以下结论，所以它们只能作为候选。

**LLM 的接入方式：注入回调，而不是在本层 new 一个客户端**

本层通过构造参数接收 ``Callable[[str], str]``——给它一段提示词，它返回模型原始
文本。真正的 DeepSeek 客户端属于生成层（阶段 5）：把 HTTP 细节、重试、Key 的
读取写进路由层，会让"换个模型"变成"改路由"。这与
:class:`~src.retrieval.relevance.RelevanceChecker` 注入 ``judge`` 回调是同一种
做法，同时让测试可以注入假分类器，无需联网。

**规则候选不会喂给 LLM。** 把"关键词命中 course_query"作为提示塞进提示词，模型
容易被锚定，规则层就悄悄变成了决策层。规则候选只在 LLM 不可用或响应不可解析时
用作兜底。

**失败时退到 course_query（RAG 链路），而不是 chitchat。** 走检索最坏是"答得
不准"，而错误的闲聊路由会把一个正经的课程问题变成一句寒暄。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from src.config import settings
from src.generation.prompts import INTENT_CLASSIFIER_PROMPT
from src.schemas.chat import IntentType
from src.utils.exceptions import ConfigurationError

#: ``IntentType`` 定义在 :mod:`src.schemas.chat`（它同时是对外响应里的字段，
#: 枚举属于数据契约，按 ``DocumentType`` 的先例放在 schemas）。这里重新导出，
#: 使 ``from src.routing.classifier import IntentType`` 依旧可用。
__all__ = [
    "DEFAULT_INTENT",
    "ClassifiedBy",
    "IntentClassifier",
    "IntentResult",
    "IntentType",
    "RuleMatch",
]
from src.utils.logging import get_logger, question_for_log

logger = get_logger(__name__)

#: confidence 的合法下界。
MIN_CONFIDENCE: Final[float] = 0.0

#: confidence 的合法上界。
MAX_CONFIDENCE: Final[float] = 1.0

#: 多条规则同时命中同一意图时的置信度加成（每条 +0.02，封顶 1.0）。
#:
#: "这几学分？怎么考核？"同时命中两条 FAQ 规则，比只命中一条更可信。加成刻意
#: 做得很小：它只是顺带的确认，不足以把一个低置信度的规则推过短路阈值。
_MULTI_HIT_BONUS: Final[float] = 0.02

#: 课程编号形态（``CS101`` / ``MA101`` / ``SE3010``）。匹配前会先去掉空白，
#: 所以 ``CS 101`` 也能认出。
COURSE_CODE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]{2,4}\d{3,4}")

#: 日志里打印模型原始响应的最大长度，避免把整段胡说八道写进日志。
_LOG_PREVIEW_CHARS: Final[int] = 200


#: 无法判断时的安全默认意图。
#:
#: 选 ``COURSE_QUERY`` 而不是 ``CHITCHAT``：默认走检索最坏是答得不准，默认走闲聊
#: 会把正经的课程问题变成"你好呀"，这是不可接受的失败方向。
DEFAULT_INTENT: Final[IntentType] = IntentType.COURSE_QUERY

#: 兜底路由的置信度。没有任何证据支持这个判断，所以是 0 而不是某个"看起来还行"的值。
FALLBACK_CONFIDENCE: Final[float] = 0.0

#: 送入 LLM 的分类提示词模板。
#:
#: 文本的**唯一出处**是 :data:`src.generation.prompts.INTENT_CLASSIFIER_PROMPT`：
#: 提示词集中在一个模块里，才能在一处调优、一处做版本管理。这里只做引用，不复制
#: 文本——复制两份的后果是改了一处另一处不生效，而且不会报错。
#: （``routing`` 依赖 ``generation.prompts`` 只是引用一个字符串常量，不触发
#: ``generation.llm`` 的导入。）
_PROMPT_TEMPLATE: Final[str] = INTENT_CLASSIFIER_PROMPT


class ClassifiedBy(str, Enum):
    """这条意图结论是怎么来的。"""

    RULE = "rule"
    """规则命中（高置信度 FAQ 短路，或 LLM 不可用时的候选兜底）。"""

    LLM = "llm"
    """LLM 返回了合法 JSON。"""

    FALLBACK = "fallback"
    """没有任何可用依据，落到 :data:`DEFAULT_INTENT`。"""


class IntentResult(BaseModel):
    """一次意图分类的结果。

    ``intent`` 与 ``confidence`` 是契约字段，其余字段**只用于诊断**（写日志、
    排查"为什么走了 RAG"）：线上出问题时，只看一个 ``course_query`` 是完全看不出
    它来自规则、LLM 还是兜底的。
    """

    model_config = ConfigDict(extra="forbid")

    intent: IntentType = Field(description="分类结果。")
    confidence: float = Field(
        ge=MIN_CONFIDENCE,
        le=MAX_CONFIDENCE,
        description="置信度，0 到 1。兜底结果的置信度是 0。",
    )
    classified_by: ClassifiedBy = Field(
        default=ClassifiedBy.RULE,
        description="结论来源：rule / llm / fallback。",
    )
    reason: str = Field(default="", description="人类可读的判定理由，直接写日志。")
    matched_keywords: list[str] = Field(
        default_factory=list,
        description="命中的规则关键词，仅规则路由会填。",
    )


@dataclass(frozen=True)
class RuleMatch:
    """一条规则命中的结果。"""

    label: str
    """规则名，例如 ``faq_credits``。"""

    intent: IntentType
    """这条规则给出的候选意图。"""

    confidence: float
    """候选置信度（已含多关键词加成）。"""

    matched: tuple[str, ...]
    """命中的关键词或正则命中的文本。"""

    short_circuit: bool
    """该规则是否允许在达到阈值时跳过 LLM。"""


@dataclass(frozen=True)
class _RuleHit:
    """单条规则的一次命中，聚合前的中间产物。"""

    label: str
    base_confidence: float
    short_circuit: bool
    matched: tuple[str, ...]


@dataclass(frozen=True)
class IntentRule:
    """关键词规则：问题文本中出现任一关键词即命中。"""

    label: str
    intent: IntentType
    keywords: tuple[str, ...]
    confidence: float
    short_circuit: bool = False


@dataclass(frozen=True)
class PatternRule:
    """正则规则：用于关键词表达不了的形态（目前只有课程编号）。"""

    label: str
    intent: IntentType
    pattern: re.Pattern[str]
    confidence: float
    short_circuit: bool = False


#: 关键词规则表。**改这张表就等于改路由行为**，所以每条规则都要写清楚它凭什么
#: 得出这个意图。
_KEYWORD_RULES: Final[tuple[IntentRule, ...]] = (
    # --- FAQ：答案是课程的一个属性值，适合预置问答对 ---------------------
    IntentRule(
        "faq_credits",
        IntentType.FAQ,
        ("几学分", "多少学分", "学分是多少", "几个学分", "学分多少", "多少分"),
        0.95,
        short_circuit=True,
    ),
    IntentRule(
        "faq_assessment",
        IntentType.FAQ,
        ("考核方式", "怎么考核", "考试形式", "平时分", "期末占", "成绩怎么算", "怎么给分"),
        0.95,
        short_circuit=True,
    ),
    IntentRule(
        "faq_instructor",
        IntentType.FAQ,
        ("谁教", "谁上这门课", "授课教师", "老师是谁", "哪个老师"),
        0.95,
        short_circuit=True,
    ),
    IntentRule(
        "faq_prerequisite",
        IntentType.FAQ,
        ("先修课", "先修要求", "前置课程", "要先学什么", "有没有先修"),
        0.95,
        short_circuit=True,
    ),
    IntentRule(
        "faq_schedule",
        IntentType.FAQ,
        ("什么时候上课", "上课时间", "在哪个教室", "上课地点", "什么时候开课"),
        0.95,
        short_circuit=True,
    ),
    IntentRule(
        "faq_textbook",
        IntentType.FAQ,
        ("用什么教材", "教材是什么", "参考书", "用的什么书", "指定教材"),
        0.95,
        short_circuit=True,
    ),
    # --- course_query：需要检索语料才能回答，规则只能作为候选 -----------
    IntentRule(
        "course_content",
        IntentType.COURSE_QUERY,
        ("讲什么", "学什么", "主要讲", "课程内容", "课程介绍", "这门课怎么样", "难不难", "难度大不大"),
        0.75,
    ),
    IntentRule(
        "course_relation",
        IntentType.COURSE_QUERY,
        ("有哪些课", "有什么课", "先学哪门", "后续课程", "课程关系"),
        0.75,
    ),
    # --- process_query：教务流程 ---------------------------------------
    IntentRule(
        "process_enroll",
        IntentType.PROCESS_QUERY,
        ("怎么选课", "如何选课", "选课流程", "怎么退课", "怎么报名", "抢课"),
        0.75,
    ),
    IntentRule(
        "process_admin",
        IntentType.PROCESS_QUERY,
        ("学分认定", "绩点", "怎么毕业", "毕业要求", "怎么申请", "流程是什么", "补考", "重修"),
        0.75,
    ),
    # --- chitchat -------------------------------------------------------
    IntentRule(
        "chitchat_social",
        IntentType.CHITCHAT,
        ("你好", "您好", "在吗", "谢谢", "多谢", "再见", "你是谁", "你能做什么", "哈哈"),
        0.75,
    ),
)

#: 正则规则表。
_PATTERN_RULES: Final[tuple[PatternRule, ...]] = (
    # 光是一个课号（"CS101"）几乎只可能是在问这门课。但"问哪一门课"并不能推出
    # "答案能直接从 FAQ 里取"，所以它不短路，只是让规则候选更可信一点。
    PatternRule("course_code", IntentType.COURSE_QUERY, COURSE_CODE_PATTERN, 0.85),
)

#: LLM 分类回调：入参是完整提示词，返回模型原始文本。
#:
#: 返回类型放宽成 ``str`` 而不是某个 schema，是因为"模型一定会给出规范 JSON"这件事
#: 本身需要被验证——解析与失败处理都在本模块完成。
IntentLLM = Callable[[str], str]


class IntentClassifier:
    """规则优先、LLM 兜底的意图分类器。

    :param llm: LLM 分类回调。``None`` 表示未接入生成层（阶段 5 之前的情形），
        此时完全依赖规则候选与安全兜底，不会抛异常。
    :param rule_threshold: 规则短路阈值。``None`` 时取
        ``settings.intent_rule_threshold``。
    :param fallback_intent: 兜底意图。``None`` 时取 :data:`DEFAULT_INTENT`。

    典型用法：

    .. code-block:: python

        classifier = IntentClassifier(llm=deepseek_client.invoke)   # 阶段 5 注入
        result = classifier.classify("数据结构与算法几学分？")
        if result.intent is IntentType.FAQ:
            ...  # 交给 FAQCache.lookup()，未命中继续 RAG
    """

    def __init__(
        self,
        llm: IntentLLM | None = None,
        rule_threshold: float | None = None,
        fallback_intent: IntentType | None = None,
    ) -> None:
        resolved_threshold = (
            rule_threshold if rule_threshold is not None else settings.intent_rule_threshold
        )
        if not MIN_CONFIDENCE <= resolved_threshold <= MAX_CONFIDENCE:
            raise ConfigurationError(
                f"INTENT_RULE_THRESHOLD 必须落在 "
                f"[{MIN_CONFIDENCE}, {MAX_CONFIDENCE}] 内，当前为 {resolved_threshold}。"
            )

        self._llm = llm
        self._rule_threshold = resolved_threshold
        self._fallback_intent = fallback_intent or DEFAULT_INTENT

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def rule_threshold(self) -> float:
        """当前生效的规则短路阈值。"""
        return self._rule_threshold

    @property
    def fallback_intent(self) -> IntentType:
        """当前生效的兜底意图。"""
        return self._fallback_intent

    @property
    def has_llm(self) -> bool:
        """是否注入了 LLM 分类回调。"""
        return self._llm is not None

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def classify(self, question: str) -> IntentResult:
        """判定问题的意图。

        :param question: 用户原始问题。纯空白视为"无法判断"，直接兜底。
        :return: :class:`IntentResult`。**本方法不抛异常**：分类失败绝不能打断整条
            问答链路，退到 RAG 永远是安全路径。配置错误（阈值非法）在构造时就报。

        判定顺序（从便宜到昂贵，一旦能定论就返回）：规则短路 → LLM → 规则候选 → 兜底。
        """
        text = question.strip() if isinstance(question, str) else ""
        if not text:
            return self._fallback_result(None, "问题为空，使用安全默认路由")

        candidate = self._pick_candidate(self._match_rules(text))

        if candidate is not None and self._can_short_circuit(candidate):
            logger.info(
                "意图分类：规则短路，不调用 LLM | intent=%s confidence=%.2f rule=%s keywords=%s",
                candidate.intent.value,
                candidate.confidence,
                candidate.label,
                list(candidate.matched),
            )
            return IntentResult(
                intent=candidate.intent,
                confidence=candidate.confidence,
                classified_by=ClassifiedBy.RULE,
                reason=(
                    f"规则「{candidate.label}」置信度 {candidate.confidence:.2f} "
                    f"达到短路阈值 {self._rule_threshold:.2f}，跳过 LLM 分类"
                ),
                matched_keywords=list(candidate.matched),
            )

        if self._llm is None:
            return self._fallback_result(candidate, "未注入 LLM 分类回调，按规则候选/兜底路由")

        try:
            raw = self._llm(_build_prompt(text))
        except Exception as exc:
            # 这里刻意捕获所有异常：分类是一次"锦上添花"的调用，网络抖动、超时、
            # 额度耗尽都不应该让一个本来能靠检索回答的问题失败。重试是生成层
            # （阶段 5）的职责，本层失败即退化。
            logger.warning(
                "LLM 意图分类调用失败，退化为规则候选/兜底 | error=%s: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return self._fallback_result(
                candidate, f"LLM 分类调用失败（{type(exc).__name__}），退化为规则候选/兜底路由"
            )

        result = self._parse_llm_response(raw)
        if result is None:
            return self._fallback_result(
                candidate, "LLM 分类响应无法解析为合法结果，退化为规则候选/兜底路由"
            )

        logger.info(
            "意图分类：LLM | intent=%s confidence=%.2f question=%s",
            result.intent.value,
            result.confidence,
            question_for_log(text),
        )
        return result

    # ------------------------------------------------------------------
    # 规则层
    # ------------------------------------------------------------------
    @staticmethod
    def _match_rules(question: str) -> tuple[RuleMatch, ...]:
        """跑一遍规则表，**按意图聚合**后返回候选（按置信度降序）。

        "这门课几学分？平时分怎么算？"会同时命中 ``faq_credits`` 与 ``faq_assessment``
        ——它们不是互相竞争的关系，而是同一个判断的两重证据，所以先按意图合并：取最强
        的那条作基准，命中数每多一条加一点置信度，命中的关键词全部记下来（排查"凭什么
        判定 FAQ"全靠它）。

        不同意图之间才是竞争关系，由 :meth:`_pick_candidate` 取最高分。
        """
        text = _normalize_for_rules(question)
        hits_by_intent: dict[IntentType, list[_RuleHit]] = {}

        for rule in _KEYWORD_RULES:
            matched = tuple(keyword for keyword in rule.keywords if keyword in text)
            if matched:
                hits_by_intent.setdefault(rule.intent, []).append(
                    _RuleHit(rule.label, rule.confidence, rule.short_circuit, matched)
                )

        for pattern_rule in _PATTERN_RULES:
            matched = tuple(pattern_rule.pattern.findall(text))
            if matched:
                hits_by_intent.setdefault(pattern_rule.intent, []).append(
                    _RuleHit(
                        pattern_rule.label,
                        pattern_rule.confidence,
                        pattern_rule.short_circuit,
                        matched,
                    )
                )

        matches: list[RuleMatch] = []
        for intent, hits in hits_by_intent.items():
            strongest = max(hits, key=lambda hit: (hit.base_confidence, hit.short_circuit, hit.label))
            matched: list[str] = []
            for hit in hits:
                matched.extend(keyword for keyword in hit.matched if keyword not in matched)
            matches.append(
                RuleMatch(
                    label=strongest.label,
                    intent=intent,
                    confidence=_boost(strongest.base_confidence, len(matched)),
                    matched=tuple(matched),
                    short_circuit=strongest.short_circuit,
                )
            )

        # 排序保证结果可复现：置信度 → 可短路 → 规则名。同分时优先能短路的候选，
        # 这样"CS101 几学分"（faq 0.95 vs course_code 0.85）稳定取 FAQ。
        matches.sort(
            key=lambda match: (match.confidence, match.short_circuit, match.label), reverse=True
        )
        return tuple(matches)

    @staticmethod
    def _pick_candidate(matches: Sequence[RuleMatch]) -> RuleMatch | None:
        """取置信度最高的规则命中作为候选；没有命中返回 ``None``。"""
        return matches[0] if matches else None

    def _can_short_circuit(self, candidate: RuleMatch) -> bool:
        """判断候选是否允许跳过 LLM。

        三个条件缺一不可：规则本身标了可短路、意图是 FAQ、置信度达标。
        第二个条件是硬编码的——它对应模块文档里"只有 FAQ 能短路"那条设计决策，
        不是可以由单条规则自行决定的开关。
        """
        return (
            candidate.short_circuit
            and candidate.intent is IntentType.FAQ
            and candidate.confidence >= self._rule_threshold
        )

    # ------------------------------------------------------------------
    # LLM 层
    # ------------------------------------------------------------------
    def _parse_llm_response(self, raw: object) -> IntentResult | None:
        """把模型原始响应解析成 :class:`IntentResult`；任何不合规都返回 ``None``。

        合规要求（提示词里已经写明，这里做二次把关）：一个 JSON 对象，``intent``
        取值合法，``confidence`` 是 ``[0, 1]`` 内的数字。少了任何一样都返回
        ``None``，由调用方退化——**不做猜测**，猜错的代价是走错链路。
        """
        if not isinstance(raw, str):
            logger.warning("LLM 分类响应不是字符串 | type=%s", type(raw).__name__)
            return None

        text = raw.strip()
        if not text:
            logger.warning("LLM 分类响应为空")
            return None

        payload = _extract_json_object(text)
        if payload is None:
            logger.warning("LLM 分类响应不是合法 JSON | raw=%s", _preview(text))
            return None

        intent = _parse_intent(payload.get("intent"))
        if intent is None:
            logger.warning("LLM 分类响应的 intent 取值非法 | raw=%s", _preview(text))
            return None

        confidence = _parse_confidence(payload.get("confidence"))
        if confidence is None:
            logger.warning("LLM 分类响应的 confidence 非法 | raw=%s", _preview(text))
            return None

        return IntentResult(
            intent=intent,
            confidence=confidence,
            classified_by=ClassifiedBy.LLM,
            reason="LLM 返回了合法的 intent 与 confidence",
        )

    # ------------------------------------------------------------------
    # 兜底
    # ------------------------------------------------------------------
    def _fallback_result(self, candidate: RuleMatch | None, reason: str) -> IntentResult:
        """LLM 不可用 / 响应不合规时的退路。

        有规则候选就用候选（它至少来自真实命中的关键词），连候选都没有才落到
        :data:`DEFAULT_INTENT`。

        **例外：chitchat 候选被刻意丢弃。** 规则只看得到几个字，而一句话里出现
        "你好"完全可能是个正经问题（"你好，数据结构的先修课是什么"）。LLM 不可用时
        已经没有第二道判断了，此时把课程问题降级成闲聊的代价，远大于多跑一次检索。
        """
        if candidate is not None and candidate.intent is not IntentType.CHITCHAT:
            logger.info(
                "意图分类：退化为规则候选 | intent=%s confidence=%.2f rule=%s reason=%s",
                candidate.intent.value,
                candidate.confidence,
                candidate.label,
                reason,
            )
            return IntentResult(
                intent=candidate.intent,
                confidence=candidate.confidence,
                classified_by=ClassifiedBy.RULE,
                reason=f"{reason}（采用规则「{candidate.label}」候选）",
                matched_keywords=list(candidate.matched),
            )

        if candidate is not None:
            logger.info(
                "意图分类：丢弃 chitchat 规则候选，改用默认路由 | rule=%s keywords=%s",
                candidate.label,
                list(candidate.matched),
            )

        logger.warning(
            "意图分类：退化为默认路由 | intent=%s reason=%s",
            self._fallback_intent.value,
            reason,
        )
        return IntentResult(
            intent=self._fallback_intent,
            confidence=FALLBACK_CONFIDENCE,
            classified_by=ClassifiedBy.FALLBACK,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _build_prompt(question: str) -> str:
    """把模板与问题拼成最终提示词。"""
    return _PROMPT_TEMPLATE.format(question=question)


def _normalize_for_rules(question: str) -> str:
    """规则匹配前的归一化：转小写并去掉所有空白。

    只做这两件事：规则用的是子串匹配，标点本来就是子串之外的东西，删不删都不影响
    命中，而删掉能让 "CS 101 几学分" 这种断开的写法也能匹配上课号正则。
    """
    return "".join(question.split()).lower()


def _boost(confidence: float, hits: int) -> float:
    """多条关键词同时命中时小幅提升置信度，封顶 1.0。"""
    return min(MAX_CONFIDENCE, confidence + _MULTI_HIT_BONUS * (hits - 1))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """从模型响应里取出 JSON 对象；取不到返回 ``None``。

    依次尝试三种形态，都是实测中真出现过的：

    1. 纯 JSON —— 提示词要求的形态；
    2. `````json ... ````` 代码块 —— 最常见的一种偏离；
    3. JSON 前后夹了人话（"分类结果如下：{...}"）—— 提示词明确禁止，但模型偶尔仍会
       这么做。这里做最后一次补救，**并打一条 WARNING**：补救本身不该变成常态，
       日志里必须能看见它发生了。三次都失败则返回 ``None``，由调用方退化。

    只接受 JSON **对象**：数组、字符串、数字都不算——契约是"一个对象"。整段响应本身
    是合法 JSON 但不是对象时（例如模型返回了 ``[{...}]``），**直接判失败**，不去
    "从数组里捞一个元素"：那属于猜测，而猜错的代价是走错链路。
    """
    stripped = text.strip()
    candidates = [stripped, _strip_code_fence(stripped)]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])

    for index, candidate in enumerate(candidates):
        body = candidate.strip()
        if not body:
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            if index > 0:
                logger.warning(
                    "LLM 分类响应不是纯 JSON，已从文本中提取 JSON 对象 | chars=%d", len(text)
                )
            return payload
        return None
    return None


def _strip_code_fence(text: str) -> str:
    """剥掉 Markdown 代码块围栏（`````json ... `````）。"""
    if not text.startswith("```"):
        return text
    body = text[3:]
    first_break = body.find("\n")
    if first_break != -1 and body[:first_break].strip().lower() in {
        "",
        "json",
        "jsonc",
        "js",
        "javascript",
        "text",
    }:
        body = body[first_break + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _parse_intent(value: object) -> IntentType | None:
    """把 ``intent`` 字段转成枚举；大小写不敏感，非法取值返回 ``None``。"""
    if not isinstance(value, str):
        return None
    try:
        return IntentType(value.strip().lower())
    except ValueError:
        return None


def _parse_confidence(value: object) -> float | None:
    """把 ``confidence`` 字段转成 ``[0, 1]`` 内的 float；非法返回 ``None``。

    容忍数字字符串（``"0.96"``）——模型偶尔会给字符串；``bool`` 直接拒绝，
    因为 JSON 里的 ``true`` 会被 Python 当成 ``1.0``，那是误读而不是数据。
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        confidence = float(value)
    elif isinstance(value, str):
        try:
            confidence = float(value.strip())
        except ValueError:
            return None
    else:
        return None

    if not MIN_CONFIDENCE <= confidence <= MAX_CONFIDENCE:
        return None
    return confidence


def _preview(text: str) -> str:
    """截断日志里要打印的模型响应。"""
    single_line = text.replace("\n", "\\n")
    if len(single_line) <= _LOG_PREVIEW_CHARS:
        return single_line
    return single_line[:_LOG_PREVIEW_CHARS] + "…"
