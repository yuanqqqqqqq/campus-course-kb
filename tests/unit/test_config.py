"""``src.config`` 的单元测试。

测试一律用 ``Settings(_env_file=None)`` 构造独立实例，并在 fixture 中清掉
相关环境变量，避免开发者本机的 ``.env`` 影响断言结果。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import PROJECT_ROOT, EmbeddingProvider, Settings

#: 需要从进程环境中剔除的变量名，保证测试只反映"默认值 + 显式入参"。
_MANAGED_ENV_VARS: tuple[str, ...] = (
    "APP_NAME",
    "APP_VERSION",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "LLM_TIMEOUT",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_API_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_DIMENSION",
    "CHROMA_PERSIST_DIR",
    "CHROMA_COLLECTION_NAME",
    "FAQ_PATH",
    "FAQ_MATCH_THRESHOLD",
    "FAQ_ADMIN_TOKEN",
    "TOP_K",
    "RELEVANCE_THRESHOLD",
    "ENABLE_LLM_RELEVANCE_CHECK",
    "INTENT_RULE_THRESHOLD",
    "LLM_MAX_RETRIES",
    "LOG_LEVEL",
    "LOG_REQUEST_CONTENT",
    "CORS_ORIGINS",
    "APP_ENV",
    "ENABLE_REAL_LLM_EVAL",
    "ENABLE_LLM_EVALUATION",
    "ENABLE_REAL_LLM_BENCHMARK",
)


@pytest.fixture
def clean_settings() -> Settings:
    """返回一个不受本机 .env 影响的配置实例。"""
    return Settings(_env_file=None)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """把相关环境变量从当前进程中暂时移除。"""
    for name in _MANAGED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_match_project_spec(clean_settings: Settings) -> None:
    """默认值应当与 .env.example 中的约定一致。"""
    assert clean_settings.deepseek_base_url == "https://api.deepseek.com"
    assert clean_settings.deepseek_model == "deepseek-chat"
    assert clean_settings.embedding_provider is EmbeddingProvider.LOCAL
    assert clean_settings.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert clean_settings.top_k == 5
    assert clean_settings.llm_timeout == 30.0
    assert clean_settings.log_level == "INFO"
    assert clean_settings.cors_origins == ["http://localhost:3000"]
    assert clean_settings.enable_llm_relevance_check is False


def test_relevance_threshold_default_is_calibrated(clean_settings: Settings) -> None:
    """默认阈值是实测标定的 0.50，不是早期拍脑袋的 0.6。

    标定依据（bge-small-zh-v1.5，本项目 7 门课语料）：10 条相关问题的 top1
    分数落在 0.573~0.767，8 条无关问题落在 0.261~0.480。取 0.6 会误拒其中
    一条合法问题，0.50 在 18 条查询上零误拒、零误放。
    """
    assert clean_settings.relevance_threshold == 0.50


def test_relative_paths_are_resolved_against_project_root(clean_settings: Settings) -> None:
    """相对路径必须展开成项目根目录下的绝对路径，与启动时的工作目录无关。"""
    assert clean_settings.chroma_persist_dir.is_absolute()
    assert clean_settings.chroma_persist_dir == PROJECT_ROOT / "data" / "chroma"
    assert clean_settings.faq_path == PROJECT_ROOT / "data" / "faq.json"


def test_absolute_path_is_kept_as_is(tmp_path: Path) -> None:
    """绝对路径不应被二次拼接。"""
    custom = tmp_path / "chroma"
    instance = Settings(_env_file=None, chroma_persist_dir=custom)
    assert instance.chroma_persist_dir == custom


def test_cors_origins_accepts_comma_separated_string() -> None:
    """CORS_ORIGINS 支持逗号分隔写法，而不是只支持 JSON 数组。"""
    instance = Settings(
        _env_file=None,
        cors_origins="http://localhost:3000, http://127.0.0.1:5173",
    )
    assert instance.cors_origins == ["http://localhost:3000", "http://127.0.0.1:5173"]


def test_cors_origins_supports_wildcard() -> None:
    """填 * 表示允许所有来源。"""
    instance = Settings(_env_file=None, cors_origins="*")
    assert instance.cors_origins == ["*"]


def test_cors_origins_empty_string_yields_empty_list() -> None:
    """空字符串退化成空列表，而不是 [""]。"""
    instance = Settings(_env_file=None, cors_origins="")
    assert instance.cors_origins == []


def test_base_url_trailing_slash_is_stripped() -> None:
    """结尾斜杠要去掉，避免 SDK 拼接出双斜杠路径。"""
    instance = Settings(_env_file=None, deepseek_base_url="https://api.deepseek.com/")
    assert instance.deepseek_base_url == "https://api.deepseek.com"


@pytest.mark.parametrize(
    "bad_url",
    ["api.deepseek.com", "ftp://api.deepseek.com", ""],
)
def test_invalid_base_url_is_rejected(bad_url: str) -> None:
    """缺协议头或协议非法的地址必须在启动时报错。"""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, deepseek_base_url=bad_url)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("relevance_threshold", 1.5),
        ("relevance_threshold", -0.1),
        ("top_k", 0),
        ("log_level", "VERBOSE"),
        ("embedding_provider", "gemini"),
    ],
)
def test_out_of_range_values_are_rejected(field: str, value: object) -> None:
    """越界或非法取值必须在启动时暴露，而不是运行到一半才出问题。"""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_api_key_is_not_leaked_in_repr() -> None:
    """SecretStr 应当保证密钥不会被 repr / str / 异常堆栈打印出来。"""
    secret = "sk-super-secret-value"
    instance = Settings(_env_file=None, deepseek_api_key=secret)

    assert secret not in repr(instance)
    assert secret not in str(instance)
    assert secret not in str(instance.model_dump())
    # 需要明文时必须显式取值
    assert instance.deepseek_api_key_value == secret


def test_missing_api_key_does_not_block_startup(clean_settings: Settings) -> None:
    """未配置 Key 时配置对象仍可构造，由 /api/health 保持可用、LLM 接口再报错。"""
    assert clean_settings.has_deepseek_api_key is False


def test_has_deepseek_api_key_detects_blank_value() -> None:
    """只有空白的 Key 等同于未配置。"""
    instance = Settings(_env_file=None, deepseek_api_key="   ")
    assert instance.has_deepseek_api_key is False


def test_get_settings_is_cached() -> None:
    """``get_settings`` 应当缓存实例，避免每次请求都重新解析 .env。"""
    from src.config import get_settings

    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()


def test_managed_env_vars_cover_every_setting_field() -> None:
    """隔离清单必须覆盖全部配置字段。

    漏掉一个的后果很隐蔽：开发者的 shell 里恰好导出了那个变量时，某个断言会莫名
    变红，而报错信息完全指不到原因（本机 .env 已被隔离，环境变量却没有）。
    """
    missing = {
        field.upper()
        for field in Settings.model_fields
        if field.upper() not in _MANAGED_ENV_VARS
    }

    assert missing == set(), f"这些配置项没有加进隔离清单：{sorted(missing)}"
