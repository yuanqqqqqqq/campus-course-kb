"""``src.retrieval.embeddings`` 的单元测试。

**全部不下载模型、不访问网络、不需要任何 API Key。**

做法是把假的实现塞进 ``sys.modules``，而不是直接 patch 真实包里的类：

* 被测模块用惰性 import 是有意为之（只使用 openai 模式的环境不该被迫装
  torch），因此测试必须能在**依赖没装**的情况下依然验证构造参数；
* ``sys.modules[name] = None`` 是 CPython 的既定行为，能让 ``import name``
  抛 ImportError，正好用来测试"缺少依赖时的报错路径"。
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from pydantic import SecretStr

from src.config import EmbeddingProvider
from src.retrieval.embeddings import EmbeddingService
from src.utils.exceptions import ConfigurationError, EmbeddingError


# ---------------------------------------------------------------------------
# 假实现
# ---------------------------------------------------------------------------
class _ConstructorRecorder:
    """记录构造参数，并返回一个假 Embeddings。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Embeddings:
        self.calls.append(kwargs)
        return DeterministicFakeEmbedding(size=8)

    @property
    def last(self) -> dict[str, Any]:
        """最近一次构造的参数。"""
        assert self.calls, "构造器尚未被调用"
        return self.calls[-1]


@pytest.fixture
def hf_module(monkeypatch: pytest.MonkeyPatch) -> _ConstructorRecorder:
    """注入一个假的 ``langchain_huggingface`` 模块。"""
    recorder = _ConstructorRecorder()
    module = types.ModuleType("langchain_huggingface")
    module.HuggingFaceEmbeddings = recorder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_huggingface", module)
    return recorder


@pytest.fixture
def openai_module(monkeypatch: pytest.MonkeyPatch) -> _ConstructorRecorder:
    """注入一个假的 ``langchain_openai`` 模块。"""
    recorder = _ConstructorRecorder()
    module = types.ModuleType("langchain_openai")
    module.OpenAIEmbeddings = recorder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    return recorder


# ---------------------------------------------------------------------------
# provider 解析
# ---------------------------------------------------------------------------
def test_default_provider_is_local() -> None:
    """默认走本地 Embedding，不依赖任何 API 额度。"""
    assert EmbeddingService().provider is EmbeddingProvider.LOCAL


def test_default_model_comes_from_settings() -> None:
    """模型名来自 Settings，代码里不硬编码模型名。"""
    from src.config import settings

    assert EmbeddingService().model == settings.embedding_model


def test_model_can_be_overridden_explicitly() -> None:
    """显式传入的模型名优先于配置。"""
    assert EmbeddingService(model="bge-m3").model == "bge-m3"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("local", EmbeddingProvider.LOCAL),
        ("LOCAL", EmbeddingProvider.LOCAL),
        ("  Local  ", EmbeddingProvider.LOCAL),
        ("openai", EmbeddingProvider.OPENAI),
        ("OpenAI", EmbeddingProvider.OPENAI),
    ],
)
def test_provider_string_is_normalised(raw: str, expected: EmbeddingProvider) -> None:
    """字符串形式的 provider 应当大小写不敏感、容忍首尾空白。"""
    assert EmbeddingService(provider=raw).provider is expected


def test_enum_provider_is_accepted() -> None:
    """直接传枚举也可以。"""
    assert EmbeddingService(provider=EmbeddingProvider.OPENAI).provider is EmbeddingProvider.OPENAI


@pytest.mark.parametrize("bad", ["gemini", "huggingface", "deepseek", ""])
def test_unsupported_provider_rejected(bad: str) -> None:
    """不支持的 provider 要明确报错并列出可选值。"""
    with pytest.raises(ConfigurationError, match="不支持的 Embedding provider"):
        EmbeddingService(provider=bad)


def test_blank_model_rejected() -> None:
    """模型名为空要报错，不能带着空值去构造。"""
    with pytest.raises(ConfigurationError, match="模型名不能为空"):
        EmbeddingService(model="   ")


def test_deepseek_is_not_an_embedding_provider() -> None:
    """DeepSeek 不提供 Embedding 模型，配成它必须被拒绝。

    这条用例对应项目的一条硬约束：Embedding 与 LLM 解耦，不能指望 DeepSeek
    同时兼任两件事。
    """
    with pytest.raises(ConfigurationError):
        EmbeddingService(provider="deepseek")


# ---------------------------------------------------------------------------
# local 模式
# ---------------------------------------------------------------------------
def test_local_provider_builds_huggingface_embeddings(hf_module: _ConstructorRecorder) -> None:
    """local 模式构造 HuggingFace 实现，模型名取自配置。"""
    from src.config import settings

    embeddings = EmbeddingService(provider="local").get_embeddings()

    assert isinstance(embeddings, Embeddings)
    assert hf_module.last["model_name"] == settings.embedding_model


def test_local_provider_enables_normalisation(hf_module: _ConstructorRecorder) -> None:
    """必须开启向量归一化。

    向量库用的是 cosine 空间，余弦相似度只有在向量归一化后才与内积等价。
    不归一化的话长文本会因为模长更大而拿到虚高的相似度，阈值随之失去意义。
    """
    EmbeddingService(provider="local").get_embeddings()

    assert hf_module.last["encode_kwargs"]["normalize_embeddings"] is True


def test_local_provider_uses_configured_model(hf_module: _ConstructorRecorder) -> None:
    """显式指定的模型名要传到构造函数里。"""
    EmbeddingService(provider="local", model="BAAI/bge-large-zh-v1.5").get_embeddings()

    assert hf_module.last["model_name"] == "BAAI/bge-large-zh-v1.5"


def test_missing_local_dependency_gives_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺少本地依赖时，报错信息必须包含可执行的安装命令。"""
    # sys.modules 中的 None 会让 `import langchain_huggingface` 抛 ImportError
    monkeypatch.setitem(sys.modules, "langchain_huggingface", None)

    with pytest.raises(EmbeddingError, match="pip install langchain-huggingface") as excinfo:
        EmbeddingService(provider="local").get_embeddings()

    assert excinfo.value.details["provider"] == "local"


# ---------------------------------------------------------------------------
# openai 兼容模式
# ---------------------------------------------------------------------------
def test_openai_provider_builds_client(
    openai_module: _ConstructorRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """openai 模式的三个关键参数都要正确传递。"""
    from src.retrieval import embeddings as embeddings_module

    monkeypatch.setattr(
        embeddings_module.settings, "embedding_api_base_url", "https://api.example.com/v1", raising=False
    )
    monkeypatch.setattr(
        embeddings_module.settings, "embedding_api_key", _secret("sk-test"), raising=False
    )

    EmbeddingService(provider="openai").get_embeddings()

    assert openai_module.last["model"] == EmbeddingService(provider="openai").model
    assert openai_module.last["base_url"] == "https://api.example.com/v1"
    assert openai_module.last["api_key"] == "sk-test"


def test_openai_provider_disables_client_side_chunking(
    openai_module: _ConstructorRecorder,
) -> None:
    """必须关掉客户端侧的分词与长度切分。

    langchain-openai 默认用 tiktoken 先切分再多次请求加权平均，这套逻辑是为
    OpenAI 官方模型写的；换成第三方兼容接口后分词器与上下文长度都可能对不上，
    轻则分数失真、重则直接报错。
    """
    EmbeddingService(
        provider="openai",
        model="bge-m3",
        api_base_url="https://api.example.com/v1",
        api_key="sk-test",
    ).get_embeddings()

    assert openai_module.last["check_embedding_ctx_length"] is False


def test_openai_provider_requires_base_url() -> None:
    """缺少 base_url 时报错并指明该配哪个变量。"""
    with pytest.raises(ConfigurationError, match="EMBEDDING_API_BASE_URL"):
        EmbeddingService(provider="openai", api_key="sk-test").get_embeddings()


def test_openai_provider_requires_api_key(openai_module: _ConstructorRecorder) -> None:
    """缺少 api_key 时报错并指明该配哪个变量。"""
    with pytest.raises(ConfigurationError, match="EMBEDDING_API_KEY"):
        EmbeddingService(
            provider="openai", api_base_url="https://api.example.com/v1"
        ).get_embeddings()


def test_openai_provider_accepts_secret_str(
    openai_module: _ConstructorRecorder,
) -> None:
    """密钥用 SecretStr 传入也要能正确取出明文。"""
    EmbeddingService(
        provider="openai",
        api_base_url="https://api.example.com/v1",
        api_key=_secret("sk-from-secret"),
    ).get_embeddings()

    assert openai_module.last["api_key"] == "sk-from-secret"


def test_openai_provider_uses_configured_values(
    openai_module: _ConstructorRecorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """不传参时全部取自 Settings。"""
    from src.retrieval import embeddings as embeddings_module

    monkeypatch.setattr(
        embeddings_module.settings, "embedding_api_base_url", "https://cfg.example.com/v1", raising=False
    )
    monkeypatch.setattr(
        embeddings_module.settings, "embedding_api_key", _secret("sk-cfg"), raising=False
    )
    monkeypatch.setattr(embeddings_module.settings, "embedding_model", "cfg-model", raising=False)

    EmbeddingService(provider="openai").get_embeddings()

    assert openai_module.last["base_url"] == "https://cfg.example.com/v1"
    assert openai_module.last["api_key"] == "sk-cfg"
    assert openai_module.last["model"] == "cfg-model"


# ---------------------------------------------------------------------------
# 缓存与复用
# ---------------------------------------------------------------------------
def test_get_embeddings_caches_instance(hf_module: _ConstructorRecorder) -> None:
    """同一个 service 反复取实例只构造一次。

    本地模型加载要几秒到几十秒，重复构造是纯粹的浪费。
    """
    service = EmbeddingService(provider="local")

    first = service.get_embeddings()
    second = service.get_embeddings()

    assert first is second
    assert len(hf_module.calls) == 1


def test_separate_services_build_separately(hf_module: _ConstructorRecorder) -> None:
    """不同 service 实例各自构造，互不影响。"""
    EmbeddingService(provider="local").get_embeddings()
    EmbeddingService(provider="local").get_embeddings()

    assert len(hf_module.calls) == 2


def test_local_and_openai_services_are_independent(
    hf_module: _ConstructorRecorder, openai_module: _ConstructorRecorder
) -> None:
    """两种 provider 可以在同一进程内共存，互不干扰。"""
    local = EmbeddingService(provider="local", model="m-local")
    remote = EmbeddingService(
        provider="openai", model="m-remote", api_base_url="https://x/v1", api_key="k"
    )

    local.get_embeddings()
    remote.get_embeddings()

    assert hf_module.last["model_name"] == "m-local"
    assert openai_module.last["model"] == "m-remote"


# ---------------------------------------------------------------------------
# 真实模型（集成测试，依赖本地模型文件）
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_real_local_model_embeds_text() -> None:
    """用真实的 BGE 模型编码一段中文，验证维度与归一化。

    需要本地可用的模型文件（HF 缓存，或 EMBEDDING_MODEL 指向的目录）。取不到
    模型时**快速跳过**——先做一次廉价的可用性检查，再决定是否加载，否则这条
    用例会先跑几十秒的下载重试，把整个测试套件拖慢到无法接受。

    可用 ``pytest -m "not integration"`` 完全排除。
    """
    pytest.importorskip("sentence_transformers")

    from src.config import settings

    if not _local_model_available(settings.embedding_model):
        pytest.skip(
            f"本地没有可用的模型文件：{settings.embedding_model}。"
            f"设置 EMBEDDING_MODEL 为本机模型目录，或先联网下载模型。"
        )

    embeddings = EmbeddingService(provider="local").get_embeddings()
    vector = embeddings.embed_query("数据结构与算法的先修课程是什么")

    assert len(vector) == 512, "bge-small-zh-v1.5 的向量维度应为 512"
    norm = sum(value * value for value in vector) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-3), "向量应当已归一化"


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------
def _secret(value: str) -> SecretStr:
    """构造一个 SecretStr，模拟从配置里读出来的密钥。"""
    return SecretStr(value)


def _local_model_available(model: str) -> bool:
    """判断模型是否能**不联网**加载。

    两种情况算可用：模型名本身就是本机存在的目录；或者 HuggingFace 缓存里
    已经有这个仓库的快照。
    """
    if Path(model).is_dir():
        return True
    try:
        from huggingface_hub import constants as hf_constants

        cache_dir = Path(hf_constants.HF_HUB_CACHE) / f"models--{model.replace('/', '--')}"
        return cache_dir.is_dir() and any(cache_dir.glob("snapshots/*"))
    except Exception:
        return False
