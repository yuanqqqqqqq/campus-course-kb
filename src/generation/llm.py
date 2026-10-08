"""生成层：调用 DeepSeek，把检索到的资料变成带出处的答案。

**本模块只做三件事**：把 (问题, 资料) 渲染成消息 → 调模型（带超时与重试）→ 把回答、
来源、token 用量组装成 :class:`~src.schemas.chat.GenerationResult`。

它不认识"课程""FAQ""意图"这些业务概念，也不碰 HTTP 层：**SSE 是 API 层的事**，
本模块只把模型输出成一个个文本片段交给调用方（见 :meth:`LLMService.generate_stream`）。

**重试策略**

只重试"可能自愈"的失败：超时、连接中断、限流、5xx、空回答。鉴权失败、模型名不
存在、请求体不合法这类问题，重试一百次也是同样的结果，直接抛
:class:`~src.utils.exceptions.ConfigurationError`，不浪费两次调用。

次数由 ``LLM_MAX_RETRIES`` 控制（默认 2，即最多尝试 3 次），退避用
``wait_exponential``。**同时把 OpenAI SDK 自带的重试关掉**（``max_retries=0``）：
两层各试 3 次会变成 9 次调用，延迟与费用都会失控。

**流式输出的重试边界**

流式只在"还没吐出任何 token"之前重试。一旦已经把部分内容交给调用方，重试就会把
前半句再说一遍，用户看到的是重复文本——所以那一刻之后的失败直接抛错，由上层决定
是提示"生成中断"还是重来（见 :meth:`LLMService.generate_stream`）。

**token 统计不做本地估算**

服务端在响应里回报了用量就记录（``estimated=False``）；没回报就整个字段置 ``None``。
不用 tiktoken 之类的本地分词器去补——DeepSeek 的 tokenizer 与它们并不一致，补出来的
数字看着精确、实则系统性偏差，拿去算成本是自欺欺人。真要做估算，必须标
``estimated=True``（字段已备好，见 :class:`~src.schemas.chat.TokenUsage`）。
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Final

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
)
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.generation.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    RELEVANCE_JUDGE_PROMPT,
    format_context,
)
from src.schemas.chat import GenerationResult, Source, TokenUsage, metadata_text
from src.utils.exceptions import (
    AppError,
    ConfigurationError,
    GenerationError,
    LLMFormatError,
    LLMResponseError,
    LLMTimeoutError,
)
from src.utils.logging import get_logger, question_for_log

logger = get_logger(__name__)

#: 重试退避：0.5s、1s、2s…（上限 4s）。
#:
#: 退避的上限刻意压得比较低：用户在等一个回答，超过几秒的等待还不如直接告诉他
#: "稍后再试"。真正需要漫长退避的场景（批量建库）不在这里。
_RETRY_BACKOFF_MULTIPLIER: Final[float] = 0.5
_RETRY_BACKOFF_MAX: Final[float] = 4.0

#: 明确"稍后可能就好了"的状态码，只有这些才重试。
#:
#: 用的是一张**可重试白名单**而不是"不可重试黑名单"：黑名单漏掉的状态码（501 未实现、
#: 505 版本不支持这类确定性失败）会被当成可重试，白白多打两次请求、多等几秒退避。
_RETRYABLE_STATUS: Final[frozenset[int]] = frozenset(
    {
        408,  # 请求超时
        409,  # 冲突（对端要求重试）
        425,  # Too Early
        429,  # 限流
        500,  # 上游内部错误
        502,  # 网关错误
        503,  # 暂时不可用
        504,  # 网关超时
    }
)


def _is_retryable_status(status: object) -> bool:
    """状态码是否值得重试。

    非整数状态码（拿不到 ``status_code`` 的情况）按可重试处理：那种场景多半是
    连接层面的异常，重试通常有意义。
    """
    if not isinstance(status, int):
        return True
    return status in _RETRYABLE_STATUS


class LLMService:
    """DeepSeek（OpenAI 兼容接口）调用封装。

    :param client: 注入的 LangChain ChatModel。``None`` 时按配置构造
        :class:`~langchain_openai.ChatOpenAI`。测试与离线场景注入假实现，
        这样"生成逻辑"不需要联网也能验证。
    :param max_retries: 最大重试次数。``None`` 时取 ``settings.llm_max_retries``。
    :param timeout: 单次调用超时（秒）。``None`` 时取 ``settings.llm_timeout``。

    典型用法：

    .. code-block:: python

        service = LLMService()
        result = service.generate("CS201 的先修课是什么", documents)
        for piece in service.generate_stream("CS201 的先修课是什么", documents):
            ...  # 由 API 层包成 SSE
    """

    def __init__(
        self,
        client: Any | None = None,
        max_retries: int | None = None,
        timeout: float | None = None,
    ) -> None:
        resolved_retries = max_retries if max_retries is not None else settings.llm_max_retries
        if resolved_retries < 0:
            raise ConfigurationError(f"LLM_MAX_RETRIES 不能为负数，当前为 {resolved_retries}。")
        resolved_timeout = timeout if timeout is not None else settings.llm_timeout
        if resolved_timeout <= 0:
            raise ConfigurationError(f"LLM_TIMEOUT 必须大于 0，当前为 {resolved_timeout}。")

        self._client = client
        self._max_retries = resolved_retries
        self._timeout = resolved_timeout

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def max_retries(self) -> int:
        """最大重试次数（不含首次）。"""
        return self._max_retries

    @property
    def timeout(self) -> float:
        """单次调用超时（秒）。"""
        return self._timeout

    @property
    def model(self) -> str:
        """当前使用的模型名：注入的 client 能报出自己的名字就用它，否则用配置值。"""
        return getattr(self._client, "model_name", None) or settings.deepseek_model

    @property
    def client(self) -> Any:
        """LLM 客户端（首次访问时才构造）。

        对外暴露只读入口，主要是为了让"客户端到底按哪套配置建的"可被检查——
        否则这段装配逻辑只能靠读代码相信它是对的。
        """
        return self._get_client()

    # ------------------------------------------------------------------
    # 消息构造
    # ------------------------------------------------------------------
    def build_messages(self, question: str, context: Sequence[Document]) -> list[BaseMessage]:
        """构造要发给模型的消息列表。

        结构是刻意的两层：**system 消息只放规则，用户消息只放资料与问题**。
        把规则和资料混在同一条消息里，模型很难分清哪句是"要求"哪句是"内容"——而那
        正是 Prompt Injection 得以生效的前提。

        :param question: 用户问题。
        :param context: 检索到的文档。
        :return: ``[SystemMessage, HumanMessage]``。
        """
        rendered = RAG_USER_PROMPT.format(
            context=format_context(context),
            question=question.strip(),
        )
        return [SystemMessage(content=RAG_SYSTEM_PROMPT), HumanMessage(content=rendered)]

    # ------------------------------------------------------------------
    # 非流式
    # ------------------------------------------------------------------
    def generate(self, question: str, context: Sequence[Document]) -> GenerationResult:
        """生成一个完整回答。

        :param question: 用户问题，不能为空白。
        :param context: 检索到的文档。**允许为空**：闲聊路径本来就没有资料，
            空资料时 system prompt 会让模型说明"知识库中未找到"。检索链路应当更早
            一步就用 ``RelevanceChecker`` 把证据不足的请求挡掉，不该依赖这里的兜底。
        :return: :class:`~src.schemas.chat.GenerationResult`。
        :raises ValueError: 问题为空白（调用方的代码 bug）。
        :raises ConfigurationError: 未配置 API Key，或鉴权 / 模型名等不可重试的问题。
        :raises LLMTimeoutError: 超时且重试用尽。
        :raises GenerationError: 其他调用失败，或重试用尽。
        """
        self._validate_question(question)
        if not context:
            logger.debug(
                "本次生成没有检索资料（闲聊路径或空上下文） | question=%s",
                question_for_log(question),
            )

        messages = self.build_messages(question, context)
        response = self._retrying()(self._invoke_once, messages)
        answer = _extract_text(response).strip()

        logger.info(
            "生成完成 | model=%s context_docs=%d answer_chars=%d",
            self.model,
            len(context),
            len(answer),
        )
        return GenerationResult(
            answer=answer,
            sources=build_sources(context),
            usage=usage_from_message(response),
            model=self.model,
        )

    # ------------------------------------------------------------------
    # 通用调用
    # ------------------------------------------------------------------
    def complete(self, prompt: str, *, system: str | None = None) -> str:
        """按给定提示词调一次模型，返回**原始文本**。

        给"提示词由调用方拼、结果由调用方解析"的场景用——目前是路由层的意图分类
        （:class:`~src.routing.classifier.IntentClassifier` 需要一个
        ``Callable[[str], str]``）。本方法不做任何解析、也不保证返回的是 JSON：
        那是分类器的职责。

        超时、重试与异常翻译与 :meth:`generate` 完全一致（走同一个 ``_invoke_once``）。

        :param prompt: 完整的用户消息内容。
        :param system: 可选的系统消息；``None`` 表示只发一条用户消息。
        :raises ValueError: 提示词为空白。
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt 不能为空")

        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        response = self._retrying()(self._invoke_once, messages)
        return _extract_text(response).strip()

    # ------------------------------------------------------------------
    # 相关性复核
    # ------------------------------------------------------------------
    def judge_relevance(self, question: str, documents: Sequence[Document]) -> bool:
        """让模型判断"这些资料够不够回答问题"，供检索层在分数不达标时复核。

        只在 ``ENABLE_LLM_RELEVANCE_CHECK=true`` 时被 ``RelevanceChecker`` 调用，
        默认关闭——它是"分类 LLM + 复核 LLM + 生成 LLM"里的那第二次调用，开着就等于
        每个不达标的问题都多花一次钱。

        **判不准一律按不相关处理**：模型没回答出明确的 yes 就是没把握，而"没把握"
        在我们这条链路上的正确动作是拒答，不是硬着头皮生成。

        :param question: 用户问题。
        :param documents: 候选文档；为空直接返回 ``False``（没有材料可判）。
        :return: 资料是否足以回答。
        :raises ConfigurationError: 未配置 API Key，或鉴权等不可重试的问题。
        :raises GenerationError: 调用失败（含重试用尽）。
        """
        if not documents:
            return False

        prompt = RELEVANCE_JUDGE_PROMPT.format(
            context=format_context(documents),
            question=question.strip(),
        )
        messages: list[BaseMessage] = [HumanMessage(content=prompt)]
        response = self._retrying()(self._invoke_once, messages)
        verdict = parse_yes_no(_extract_text(response))

        logger.info(
            "LLM 相关性复核 | verdict=%s docs=%d question=%s",
            verdict,
            len(documents),
            question_for_log(question),
        )
        return verdict

    # ------------------------------------------------------------------
    # 流式
    # ------------------------------------------------------------------
    def generate_stream(self, question: str, context: Sequence[Document]) -> Iterator[str]:
        """流式生成，逐片段产出**纯文本**。

        本方法只把模型输出拆成片段，不做任何协议层的事——SSE 的分帧、``data:``
        前缀、``[DONE]`` 标记全部由 API 层负责（阶段 6 的 ``POST /api/chat/stream``）。

        重试只覆盖"拿到第一个非空片段之前"；之后失败直接抛异常，不再重试（原因见
        模块文档）。调用方拿到的片段按顺序拼接就是完整回答。

        :param question: 用户问题，不能为空白。
        :param context: 检索到的文档，允许为空。
        :raises ValueError: 问题为空白。
        :raises ConfigurationError: 配置 / 鉴权问题，不重试。
        :raises LLMResponseError: 整个流没有任何内容。
        :raises GenerationError: 已经吐出片段之后的失败，或首个片段重试用尽。
        """
        self._validate_question(question)
        messages = self.build_messages(question, context)

        stream, first_piece = self._retrying()(self._open_stream, messages)
        yield first_piece

        emitted = len(first_piece)
        usage: TokenUsage | None = None
        try:
            for chunk in stream:
                # 流式响应里会有内容为空的片段（例如只带 role 的首块、只带 usage 的
                # 尾块），直接跳过，不要产出空字符串——上层每收到一个片段就会发一条
                # SSE 事件，空片段只会制造噪音。
                text = _extract_text(chunk)
                if text:
                    emitted += len(text)
                    yield text
                usage = usage_from_message(chunk) or usage
        except Exception as exc:
            # 已经产出过内容，重试会导致重复输出，所以只翻译不重试。
            logger.warning("流式生成中断（已产出 %d 字符，不再重试）", emitted)
            raise _translate_error(exc) from exc
        finally:
            # 正常结束、中途失败、客户端断开（GeneratorExit）都会走这里：
            # 把上游连接及时还回去，而不是等 GC。
            _close_quietly(stream)

        logger.info(
            "流式生成完成 | model=%s chunks_chars=%d context_docs=%d usage=%s",
            self.model,
            emitted,
            len(context),
            usage.model_dump() if usage is not None else "服务端未回报",
        )

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _invoke_once(self, messages: list[BaseMessage]) -> Any:
        """调用一次模型，并把"空回答"也当作可重试的失败。

        空回答被放在这里（而不是 ``generate`` 里）判，是为了让它进入重试：内容为空的
        响应通常是一次抖动，再试一次往往就正常了。
        """
        client = self._get_client()
        try:
            response = client.invoke(messages)
        except Exception as exc:
            raise _translate_error(exc) from exc

        if not _extract_text(response).strip():
            raise LLMResponseError("LLM 返回了空回答", details={"model": self.model})
        return response

    def _open_stream(self, messages: list[BaseMessage]) -> tuple[Iterator[Any], str]:
        """建立流并把"第一个非空片段"读出来。

        重试必须覆盖到这一步：``client.stream()`` 返回的是惰性迭代器，真正发请求是在
        第一次迭代时，所以"拿到迭代器"并不代表调用成功。
        """
        client = self._get_client()
        try:
            stream = client.stream(messages)
            for chunk in stream:
                text = _extract_text(chunk)
                if text:
                    return stream, text
        except Exception as exc:
            raise _translate_error(exc) from exc

        raise LLMResponseError("流式响应没有任何内容", details={"model": self.model})

    def _retrying(self) -> Retrying:
        """构造本次重试用的 :class:`~tenacity.Retrying`。

        每次调用新建一个实例：``Retrying`` 持有自己的重试状态，共享实例在并发下
        会互相干扰。
        """
        return Retrying(
            stop=stop_after_attempt(self._max_retries + 1),
            wait=wait_exponential(multiplier=_RETRY_BACKOFF_MULTIPLIER, max=_RETRY_BACKOFF_MAX),
            # 只重试 GenerationError 家族：超时、连接失败、限流、5xx、空回答。
            # 排除两类：ConfigurationError（Key 错了重试多少次都一样）与
            # LLMFormatError（响应结构解析不了，是确定性失败）。
            retry=retry_if_exception_type(GenerationError)
            & retry_if_not_exception_type(LLMFormatError),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    def _get_client(self) -> Any:
        """返回 LLM 客户端，首次调用时才构造（惰性）。"""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        """按配置构造 ChatOpenAI（DeepSeek 走 OpenAI 兼容接口）。

        ``max_retries=0`` 是必须的：SDK 默认自己会重试 2 次，与本模块的重试叠在一
        起就是 3×3=9 次调用。
        """
        if not settings.has_deepseek_api_key:
            raise ConfigurationError(
                "未配置 DEEPSEEK_API_KEY，无法调用 LLM。请在 .env 中填入后重试。",
                details={
                    "model": settings.deepseek_model,
                    "base_url": settings.deepseek_base_url,
                },
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - langchain-openai 是必需依赖
            raise ConfigurationError(
                f"未安装 langchain-openai，无法调用 LLM：{exc}。"
                f"请执行 pip install -r requirements.txt。"
            ) from exc

        logger.info(
            "初始化 LLM 客户端 | model=%s base_url=%s timeout=%.1fs max_retries=%d",
            settings.deepseek_model,
            settings.deepseek_base_url,
            self._timeout,
            self._max_retries,
        )
        return ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key_value,
            base_url=settings.deepseek_base_url,
            timeout=self._timeout,
            max_retries=0,
        )

    @staticmethod
    def _validate_question(question: str) -> None:
        """空问题属于调用方 bug，直接报错；不要让它变成一个"模型回答了空问题"的请求。"""
        if not question or not question.strip():
            raise ValueError("question 不能为空")


# ---------------------------------------------------------------------------
# 来源适配
# ---------------------------------------------------------------------------
def source_from_document(document: Document, score: float | None = None) -> Source:
    """把一份检索结果转成 :class:`~src.schemas.chat.Source`。

    :param document: 检索命中的文档。
    :param score: relevance_score（[0,1]，越大越相关）。传原始距离会被
        ``Source`` 的字段校验拒绝——两者的方向相反。
    """
    metadata = document.metadata or {}
    return Source(
        course_id=metadata_text(metadata, "course_id"),
        name=metadata_text(metadata, "name"),
        type=metadata_text(metadata, "type"),
        source=metadata_text(metadata, "source"),
        # 章节优先取原始中文标题，缺失时退到归一化键
        section=metadata_text(metadata, "section_title", "section"),
        score=score,
    )


def dedupe_sources(sources: Iterable[Source]) -> list[Source]:
    """按 ``(课程编号, 来源文件, 章节)`` 去重，**保留先出现的**。

    检索会命中同一门课同一章节的多个切片，不去重的话答案后面会挂一长串几乎一样的
    来源。保留先出现的而非随便挑一条，是因为调用方通常已按相关性降序排序——
    第一个就是分数最高的那条（需要分数时请自己构造带 ``score`` 的
    :class:`~src.schemas.chat.Source` 再传进来）。
    """
    seen: set[tuple[str | None, str | None, str | None]] = set()
    unique: list[Source] = []
    for source in sources:
        key = (source.course_id, source.source, source.section)
        if key == (None, None, None):
            # 三条身份信息全都取不到（纯文本资料没有 metadata 是允许的形态）。
            # 此时所有来源的键都相同，去重会把 5 条依据压成 1 条——宁可重复，
            # 也不要少报依据。
            unique.append(source)
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique


def build_sources(context: Sequence[Document]) -> list[Source]:
    """把资料列表转成去重后的来源列表（不含相关性分数）。"""
    return dedupe_sources(source_from_document(document) for document in context)


# ---------------------------------------------------------------------------
# token 用量
# ---------------------------------------------------------------------------
def usage_from_message(message: object) -> TokenUsage | None:
    """从模型响应里读取 token 用量；拿不到就返回 ``None``。

    LangChain 把服务端回报的用量放在 ``usage_metadata`` 里（``input_tokens`` /
    ``output_tokens`` / ``total_tokens``）。这里**不补算、不估算**：字段不全就整体
    返回 ``None``，因为只填一半的用量比没有用量更容易被误用。
    """
    metadata = getattr(message, "usage_metadata", None)
    if not isinstance(metadata, Mapping):
        return None

    prompt_tokens = _as_token_count(metadata.get("input_tokens"))
    completion_tokens = _as_token_count(metadata.get("output_tokens"))
    if prompt_tokens is None or completion_tokens is None:
        return None

    total_tokens = _as_token_count(metadata.get("total_tokens"))
    if total_tokens is None or total_tokens < prompt_tokens + completion_tokens:
        # 总数缺失，或者与明细自相矛盾（少数实现会给 0）。用明细相加更可信——
        # 一个"总数小于两部分之和"的用量，拿去算成本只会更离谱。
        total_tokens = prompt_tokens + completion_tokens

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        # 这些数字来自服务端响应，不是本地估算。
        estimated=False,
    )


# ---------------------------------------------------------------------------
# 异常翻译
# ---------------------------------------------------------------------------
def _translate_error(exc: BaseException) -> BaseException:
    """把底层 SDK 异常翻译成本项目的异常体系，同时决定它是否值得重试。

    翻译表：

    ==========================================  ==========================  ========
    底层异常                                     翻译成                      可重试
    ==========================================  ==========================  ========
    超时（``APITimeoutError`` / ``TimeoutError``）  ``LLMTimeoutError``         是
    连接失败、限流、5xx                            ``GenerationError``         是
    鉴权失败、模型不存在、请求体不合法               ``ConfigurationError``      否
    已经是本项目的异常                            原样返回                     看类型
    其他（大概率是代码 bug）                       原样抛出                     否
    ==========================================  ==========================  ========

    **"其他"一律原样抛出**，不翻译成 ``GenerationError``：那样会把 ``TypeError``
    之类的代码 bug 伪装成一次"调用失败"，重试三次之后以一个含糊的错误码返回 500，
    真正的原因被埋掉。
    """
    if isinstance(exc, AppError):
        return exc

    if isinstance(exc, (APITimeoutError, TimeoutError)):
        return LLMTimeoutError(f"LLM 调用超时：{exc}")

    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        # 不把上游异常原文拼进 message：401 文案里可能带 Key 片段，而 message 会
        # 直接回给调用方。完整原因只进日志。
        logger.error("LLM 鉴权失败 | error=%s", exc)
        return ConfigurationError(
            "LLM 鉴权失败，请检查 DEEPSEEK_API_KEY 是否正确、是否已过期。",
            details={"model": settings.deepseek_model},
        )

    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if not _is_retryable_status(status):
            return ConfigurationError(
                f"LLM 返回 {status}，属于请求或配置问题，重试无意义：{exc}",
                details={"status_code": status, "model": settings.deepseek_model},
            )
        return GenerationError(f"LLM 返回错误状态 {status}：{exc}", details={"status_code": status})

    if isinstance(exc, APIConnectionError):
        return GenerationError(f"LLM 连接失败：{exc}")

    return exc


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------
def _extract_text(message: object) -> str:
    """从 LangChain 消息（或片段）里取出纯文本。

    正常情况下 ``content`` 就是字符串；多模态模型会给一个内容块列表。这里两种都
    支持，其余形态抛 :class:`LLMResponseError`——**不返回 ``str(content)``**，
    那会把一个结构对象打印成 Python 字面量塞进答案里。
    """
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    raise LLMFormatError(
        f"无法从模型响应中取出文本 | content_type={type(content).__name__}",
        details={"content_type": type(content).__name__},
    )


def _close_quietly(stream: object) -> None:
    """尽力关闭上游流，失败不抛。

    对已经耗尽的迭代器调用 ``close()`` 是空操作；对中途被放弃的流（客户端断开、
    生成报错）它才是真正有用的一步——把 httpx 连接还回连接池，而不是等 GC 回收。
    关不掉也不该影响已经产出的回答，所以异常一律吞掉并记 debug。
    """
    close = getattr(stream, "close", None)
    if not callable(close):
        return
    with contextlib.suppress(Exception):  # pragma: no cover - 取决于 SDK 实现
        close()


def _as_token_count(value: object) -> int | None:
    """把 token 计数转成非负整数；非法值返回 ``None``。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def parse_yes_no(text: str) -> bool:
    """把复核回答解析成布尔值：只有明确的 ``yes`` 才算相关。

    宽容处理大小写与 ``yes.`` / ``yes,`` 这类标点，但**不做模糊匹配**：回答里没有
    明确的肯定词就是没把握，按不相关处理。反过来（从一段解释里猜它的倾向）一旦猜错，
    后果是拿不相关的资料去生成答案——比拒答糟得多。
    """
    normalized = text.strip().casefold()
    if not normalized:
        return False
    first_word = normalized.split(maxsplit=1)[0].strip("。.!！?？、,，;；:：～~ '\"\t")
    return first_word in {"yes", "y", "是", "是的", "相关"}
