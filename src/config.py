"""应用配置中心。

本项目**所有**可配置项集中在本模块。业务代码禁止直接调用 ``os.getenv`` /
``os.environ``，一律通过 ``from src.config import settings`` 读取，目的是：

1. 配置项有类型与取值范围校验，写错会在启动时立刻报错，而不是运行到一半才炸；
2. 默认值集中可见，不会散落在各个文件里；
3. 敏感信息用 ``SecretStr`` 包装，避免被日志或异常堆栈打印出来。

配置来源优先级（从高到低）：

1. 真实环境变量（容器 / CI / 命令行注入）
2. 项目根目录下的 ``.env`` 文件
3. 本模块中各字段的默认值
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

#: 项目根目录（即包含 ``src/`` ``data/`` ``pyproject.toml`` 的那一层）。
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: 合法的日志级别，取值受 Pydantic 校验，写错会在启动时报错。
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class EmbeddingProvider(str, Enum):
    """Embedding 计算方式。

    DeepSeek 官方不提供 Embedding 模型，因此 Embedding 层必须与 LLM 层解耦，
    通过本枚举在两种实现之间切换：

    - :attr:`LOCAL`  —— 本地 HuggingFace 模型（默认，无需额外 API 额度）
    - :attr:`OPENAI` —— 任意 OpenAI 兼容的 Embedding HTTP 接口
    """

    LOCAL = "local"
    OPENAI = "openai"


class Settings(BaseSettings):
    """全局配置对象。

    字段含义与取值约束见各字段的 ``description``；对应的环境变量名即字段名
    的大写形式（例如 ``deepseek_api_key`` 对应 ``DEEPSEEK_API_KEY``）。
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # 应用元信息
    # ------------------------------------------------------------------
    app_name: str = Field(
        default="Campus Course Knowledge Base",
        description="应用名称，出现在 OpenAPI 文档标题与启动日志中。",
    )
    app_version: str = Field(
        default="0.1.0",
        description="应用版本号，阶段推进时手动维护。",
    )
    app_env: str = Field(
        default="development",
        description=(
            "运行环境标识：development / staging / production。"
            "只用于日志与运维判断（例如容器里设成 production 便于区分环境），"
            "目前不改变任何业务行为——本项目没有「生产环境才开启」的分支。"
        ),
    )

    # ------------------------------------------------------------------
    # DeepSeek LLM（走 OpenAI 兼容接口）
    # ------------------------------------------------------------------
    deepseek_api_key: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "DeepSeek API Key。允许为空，此时应用仍可启动、/api/health 仍可用，"
            "但任何需要调用 LLM 的接口都会抛出 ConfigurationError。"
            "使用 SecretStr 包装，避免被日志或堆栈打印。"
        ),
    )
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek OpenAI 兼容接口的 base_url，不要带结尾斜杠。",
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        description="对话模型名，例如 deepseek-chat / deepseek-reasoner。",
    )
    llm_timeout: float = Field(
        default=30.0,
        gt=0,
        le=300,
        description="单次 LLM 调用的超时时间（秒）。",
    )
    llm_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "单次 LLM 调用失败后的最大重试次数（不含首次，即最多尝试 2+1 次）。"
            "设为 0 表示不重试。只重试可能自愈的失败（超时、连接中断、限流、5xx、空回答），"
            "鉴权失败之类的问题不重试。"
        ),
    )

    # ------------------------------------------------------------------
    # Embedding（与 LLM 完全解耦的独立组件）
    # ------------------------------------------------------------------
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.LOCAL,
        description="Embedding 提供方：local（本地 HuggingFace）或 openai（兼容接口）。",
    )
    embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        description="Embedding 模型名。local 模式下是 HF 模型仓库名；openai 模式下是接口的 model 参数。",
    )
    embedding_api_base_url: str | None = Field(
        default=None,
        description=(
            "仅当 EMBEDDING_PROVIDER=openai 时使用：OpenAI 兼容 Embedding 接口地址"
            "（例如 https://api.siliconflow.cn/v1）。local 模式下忽略。"
        ),
    )
    embedding_api_key: SecretStr | None = Field(
        default=None,
        description="仅当 EMBEDDING_PROVIDER=openai 时使用：该 Embedding 服务的 API Key。",
    )
    embedding_dimension: int | None = Field(
        default=None,
        gt=0,
        description=(
            "可选：向量维度，仅用于**文档化**当前模型产出多少维（本机 bge-small-zh-v1.5 为 512）。"
            "注意：当前代码**不会**用它做校验——写错不会报错。要核对维度请看 "
            "tests/unit/test_embeddings.py 里的断言，或换模型后重跑一次评测。"
        ),
    )

    # ------------------------------------------------------------------
    # 向量库（ChromaDB）
    # ------------------------------------------------------------------
    chroma_persist_dir: Path = Field(
        default=Path("./data/chroma"),
        description="ChromaDB 持久化目录，相对路径按项目根目录解析。",
    )
    chroma_collection_name: str = Field(
        default="campus_courses",
        min_length=1,
        description="ChromaDB collection 名称。",
    )

    # ------------------------------------------------------------------
    # FAQ 快路径
    # ------------------------------------------------------------------
    faq_path: Path = Field(
        default=Path("./data/faq.json"),
        description="FAQ 数据文件路径，相对路径按项目根目录解析。",
    )
    faq_admin_token: SecretStr = Field(
        default=SecretStr(""),
        description=(
            "写入 FAQ 的管理令牌。**默认留空 = 接口不校验**（演示/本地开发用）。"
            "一旦服务对公网或多用户开放，`POST /api/faq` 就是「任何人可写、且内容会"
            "原样广播给所有用户」的口子，必须设置本项：设置后调用方要带请求头 "
            "X-Admin-Token，否则返回 401。启动日志会在未设置时给出 WARNING。"
        ),
    )
    faq_match_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "FAQ 模糊匹配阈值：问题与 patterns 归一化后用 difflib.SequenceMatcher "
            "算相似度，达到该值才算命中。精确匹配不走这个阈值。"
            "注意 0.85 挡不住「只差一个字符」的近似问句（例如只改了课程编号），"
            "所以 patterns 要写成完整问句，别指望模糊匹配兜住错别字。"
        ),
    )

    # ------------------------------------------------------------------
    # 意图路由
    # ------------------------------------------------------------------
    intent_rule_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "规则路由短路所需的最低置信度。只有 FAQ 类规则允许短路"
            "（FAQ 的最终判定交给 FAQ Cache 的精确/模糊匹配）；"
            "其余意图的规则命中只作为候选，仍会调用 LLM 分类。"
        ),
    )

    # ------------------------------------------------------------------
    # 检索与拒答
    # ------------------------------------------------------------------
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="向量检索返回的候选文档数量。",
    )
    relevance_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "相关性阈值，作用在**统一归一化后的 relevance_score**（余弦相似度，"
            "取值 [0,1]，越大越相关）上，见 retrieval/relevance.py。低于该值时："
            "若未开启 LLM 复核则直接拒答；若开启则交给 LLM 复核后再决定。"
            "默认值 0.50 是用 bge-small-zh-v1.5 在本项目 7 门课语料上实测标定的："
            "10 条相关问题的 top1 分数为 0.573~0.767，8 条无关问题为 0.261~0.480，"
            "0.50 落在两者的空隙中间。**换 Embedding 模型后必须重新标定**——"
            "分数分布随模型变化很大。"
        ),
    )
    enable_llm_relevance_check: bool = Field(
        default=False,
        description=(
            "是否在向量分数不达标时额外调用一次 LLM 复核相关性。"
            "默认关闭：向量分数达标就直接放行，避免「分类 LLM + 相关性 LLM + 生成 LLM」"
            "每次全部调用。开启后需要向 RelevanceChecker 注入 judge 回调（阶段 5 提供）。"
        ),
    )

    # ------------------------------------------------------------------
    # 评测（阶段 7）
    # ------------------------------------------------------------------
    enable_llm_evaluation: bool = Field(
        default=False,
        description=(
            "评测时是否启用 LLM Judge 指标（faithfulness：回答能否被 Context 支持）。"
            "默认关闭——30 条评测集每条多一次调用，成本翻倍且会引入评分模型的随机性。"
            "评测脚本的 --llm real 模式下才可能生效。"
        ),
    )
    enable_real_llm_eval: bool = Field(
        default=False,
        description=(
            "评测时是否真的调用 DeepSeek（生成 + 意图分类）。"
            "默认 false：默认用规则分类 + 假生成层，全程零 API 成本，"
            "代价是「LLM 分类」与「LLM 写作」这两段不进入度量（报告里会标注）。"
            "只有显式开启才会花钱，且必须在 .env 里配好 DEEPSEEK_API_KEY。"
        ),
    )
    enable_real_llm_benchmark: bool = Field(
        default=False,
        description=(
            "压测时是否真的调用 DeepSeek。默认 false：压测用假 LLM——"
            "把几百次请求打到真实接口上，结果既受上游限流影响（延迟不可比），"
            "又会产生真实费用。"
        ),
    )

    # ------------------------------------------------------------------
    # 日志与跨域
    # ------------------------------------------------------------------
    log_level: LogLevel = Field(
        default="INFO",
        description="根 logger 级别，同时作用于 uvicorn 的日志。",
    )
    log_request_content: bool = Field(
        default=False,
        description=(
            "是否把用户问题原文写进日志（默认关闭，只记长度）。"
            "日志会被采集、转发、长期留存，用户问了什么属于用户内容，不该默认入库；"
            "排查具体问题时临时打开即可。"
        ),
    )
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description=(
            "允许跨域的来源列表。环境变量中写逗号分隔的字符串，"
            "例如 CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:5173；"
            "填 * 表示允许所有来源（仅建议本地开发使用）。"
        ),
    )

    # ------------------------------------------------------------------
    # 校验器
    # ------------------------------------------------------------------
    @field_validator("deepseek_base_url", "embedding_api_base_url", mode="after")
    @classmethod
    def _strip_trailing_slash(cls, value: str | None) -> str | None:
        """去掉结尾斜杠，避免与 SDK 内部拼接路径时出现双斜杠。"""
        if value is None:
            return None
        stripped = value.strip().rstrip("/")
        if not stripped:
            raise ValueError("URL 不能为空字符串")
        if not stripped.startswith(("http://", "https://")):
            raise ValueError(f"URL 必须以 http:// 或 https:// 开头，当前为 {stripped!r}")
        return stripped

    @field_validator("chroma_persist_dir", "faq_path", mode="after")
    @classmethod
    def _resolve_relative_path(cls, value: Path) -> Path:
        """把相对路径按项目根目录展开成绝对路径。

        这样无论从哪个工作目录启动应用（项目根、scripts/、pytest 根），
        指向的都是同一份数据。
        """
        if value.is_absolute():
            return value
        return (PROJECT_ROOT / value).resolve()

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """把逗号分隔的字符串拆成列表。

        pydantic-settings 默认要求复杂类型用 JSON 语法（``["a","b"]``），
        这里改成更常见的逗号分隔写法。
        """
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw == "*":
                return ["*"]
            return [origin.strip() for origin in raw.split(",") if origin.strip()]
        return value

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------
    @property
    def deepseek_api_key_value(self) -> str:
        """取出明文 API Key。

        只在真正构造 LLM 客户端时调用，不要写进日志。
        """
        return self.deepseek_api_key.get_secret_value()

    @property
    def has_deepseek_api_key(self) -> bool:
        """是否已配置 DeepSeek API Key。"""
        return bool(self.deepseek_api_key.get_secret_value().strip())

    @property
    def has_faq_admin_token(self) -> bool:
        """是否配置了 FAQ 写入令牌。"""
        return bool(self.faq_admin_token.get_secret_value().strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回全局唯一的 :class:`Settings` 实例。

    用 ``lru_cache`` 保证整个进程只解析一次 ``.env``，避免在请求路径上
    反复读取磁盘。测试中如需重新加载，调用 ``get_settings.cache_clear()``。
    """
    return Settings()


#: 全局配置实例。业务代码统一从这里读取配置。
settings: Settings = get_settings()
