"""Embedding 组件：与 LLM 完全解耦的独立实现。

**为什么必须解耦**

DeepSeek 只提供对话（生成）模型，不提供 Embedding 模型。Embedding 与 LLM 是
两个独立的服务、两套独立的计费方式，也是两个独立的失败点。把它们混在一层里
会带来一个很实际的后果：换 Embedding 就要动 LLM 的配置，反之亦然。

因此本模块只做一件事——按 :data:`EMBEDDING_PROVIDER` 产出
:class:`~langchain_core.embeddings.Embeddings` 实例：

``local``
    本地 HuggingFace / sentence-transformers 模型（默认）。不需要任何 API
    额度，离线可用，代价是首次要下载模型权重、推理占本地算力。

``openai``
    任意 OpenAI 兼容的 Embedding HTTP 接口（硅基流动、通义、智谱等）。

**模型名一律来自 Settings，本模块不出现任何硬编码的模型名。**

**惰性导入**

``langchain-huggingface`` 与 ``sentence-transformers``（连带 torch）体积很大，
而且只在 ``local`` 模式下才需要。如果在模块顶层 import，那么只使用 ``openai``
模式的环境也必须装 torch。所以两类实现都推迟到真正构造时才 import，缺失时
给出明确的安装指引。
"""

from __future__ import annotations

from typing import Final

from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from src.config import EmbeddingProvider, settings
from src.utils.exceptions import ConfigurationError, EmbeddingError
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: 本地模型缺失依赖时给出的安装提示。
_LOCAL_INSTALL_HINT: Final[str] = (
    "本地 Embedding 需要额外依赖，请执行：\n"
    "    pip install langchain-huggingface sentence-transformers\n"
    "或把 EMBEDDING_PROVIDER 改成 openai 改用兼容接口。"
)


class EmbeddingService:
    """按配置产出 Embedding 实现。

    典型用法：

    .. code-block:: python

        embeddings = EmbeddingService().get_embeddings()
        vectors = embeddings.embed_documents(["数据结构", "操作系统"])

    :param provider: Embedding 提供方。``None`` 时取 ``settings.embedding_provider``。
    :param model: 模型名。``None`` 时取 ``settings.embedding_model``。
    :param api_base_url: 仅 ``openai`` 模式使用，``None`` 时取
        ``settings.embedding_api_base_url``。
    :param api_key: 仅 ``openai`` 模式使用，``None`` 时取
        ``settings.embedding_api_key``。
    """

    def __init__(
        self,
        provider: EmbeddingProvider | str | None = None,
        model: str | None = None,
        api_base_url: str | None = None,
        api_key: str | SecretStr | None = None,
    ) -> None:
        self._provider = _resolve_provider(provider)
        self._model = model if model is not None else settings.embedding_model
        self._api_base_url = (
            api_base_url if api_base_url is not None else settings.embedding_api_base_url
        )
        self._api_key = api_key if api_key is not None else settings.embedding_api_key
        self._embeddings: Embeddings | None = None

        if not self._model.strip():
            raise ConfigurationError("Embedding 模型名不能为空，请检查 EMBEDDING_MODEL 配置。")

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def get_embeddings(self) -> Embeddings:
        """返回 Embedding 实现（同一实例内只构造一次并复用）。

        本地模型加载要几秒到几十秒，重复构造是纯粹的浪费，所以这里做缓存。

        :raises ConfigurationError: ``openai`` 模式缺少 base_url 或 api_key。
        :raises EmbeddingError: 本地模式缺少依赖，或 provider 取值不受支持。
        """
        if self._embeddings is None:
            self._embeddings = self._build()
            logger.info(
                "Embedding 初始化完成 | provider=%s model=%s",
                self._provider.value,
                self._model,
            )
        return self._embeddings

    @property
    def provider(self) -> EmbeddingProvider:
        """当前使用的提供方。"""
        return self._provider

    @property
    def model(self) -> str:
        """当前使用的模型名。"""
        return self._model

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _build(self) -> Embeddings:
        """按 provider 构造具体实现。"""
        if self._provider is EmbeddingProvider.LOCAL:
            return self._build_local()
        if self._provider is EmbeddingProvider.OPENAI:
            return self._build_openai()
        # EmbeddingProvider 是闭合枚举，走到这里说明枚举被改过而这里忘了同步
        raise EmbeddingError(f"不支持的 EMBEDDING_PROVIDER：{self._provider!r}")

    def _build_local(self) -> Embeddings:
        """构造本地 HuggingFace Embedding。"""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - 取决于运行环境是否装了依赖
            raise EmbeddingError(
                f"未安装本地 Embedding 依赖：{exc}。\n{_LOCAL_INSTALL_HINT}",
                details={"provider": self._provider.value, "model": self._model},
            ) from exc

        # normalize_embeddings=True 是必须的，不是优化项：
        #
        # 向量库使用 cosine 空间，而余弦相似度只有在向量已归一化时才与
        # 内积等价。不归一化的话，长文本会因为模长更大而拿到虚高的相似度，
        # 阈值也就失去意义。BGE 系列本身推荐的做法也是归一化后用余弦。
        return HuggingFaceEmbeddings(
            model_name=self._model,
            encode_kwargs={"normalize_embeddings": True},
        )

    def _build_openai(self) -> Embeddings:
        """构造 OpenAI 兼容接口的 Embedding。"""
        if not self._api_base_url:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=openai 时必须配置 EMBEDDING_API_BASE_URL"
                "（例如 https://api.siliconflow.cn/v1）。"
            )
        api_key_value = _secret_value(self._api_key)
        if not api_key_value:
            raise ConfigurationError(
                "EMBEDDING_PROVIDER=openai 时必须配置 EMBEDDING_API_KEY。"
            )

        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:  # pragma: no cover - langchain-openai 是必需依赖
            raise EmbeddingError(f"未安装 langchain-openai：{exc}") from exc

        return OpenAIEmbeddings(
            model=self._model,
            api_key=api_key_value,
            base_url=self._api_base_url,
            # 关掉客户端侧的分词与长度校验。
            #
            # langchain-openai 默认会先用 tiktoken 把文本切成不超过模型上下文
            # 的片段、再分多次请求并加权平均。这套逻辑是为 OpenAI 官方模型写
            # 的，换成第三方的 OpenAI 兼容接口后，分词器、上下文长度、是否支持
            # 批量都可能对不上，轻则分数失真、重则直接报错。
            # 文本长度由上游切分器保证（见 ingestion/splitter.py），这里直接
            # 把原文发给服务端更可控。
            check_embedding_ctx_length=False,
        )


def _resolve_provider(provider: EmbeddingProvider | str | None) -> EmbeddingProvider:
    """把字符串形式的 provider 归一化成枚举。"""
    if provider is None:
        return settings.embedding_provider
    if isinstance(provider, EmbeddingProvider):
        return provider
    try:
        return EmbeddingProvider(provider.strip().lower())
    except ValueError as exc:
        supported = "、".join(item.value for item in EmbeddingProvider)
        raise ConfigurationError(
            f"不支持的 Embedding provider：{provider!r}，可选值为 {supported}。"
        ) from exc


def _secret_value(value: str | SecretStr | None) -> str:
    """取出明文密钥，兼容 ``str`` 与 ``SecretStr`` 两种入参。"""
    if value is None:
        return ""
    if isinstance(value, SecretStr):
        return value.get_secret_value().strip()
    return value.strip()
