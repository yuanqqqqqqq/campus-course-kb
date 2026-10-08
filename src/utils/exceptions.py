"""项目自有的异常体系。

设计目标：让上层（FastAPI 异常处理器、脚本、pytest）能够用 ``except AppError``
一次性接住所有"意料之中的失败"，同时把"代码 bug"（``ValueError`` / ``KeyError``
之类）暴露成 500，而不是被悄悄吞掉。

所有自定义异常都携带机器可读的 ``code``，便于前端与日志聚合按错误码分类。
"""

from __future__ import annotations

from typing import Any, Final


class AppError(Exception):
    """项目内所有预期异常的基类。

    :param message: 给开发者看的错误描述。
    :param code: 机器可读的错误码，默认取类名的大写下划线形式。
    :param details: 附加上下文（入参、命中的文档 id 等），用于排查。

        注意：**它会随错误响应返回给调用方**（见 ``src/main.py`` 的异常处理器），
        所以只放非敏感信息——不要塞密钥、令牌、个人信息或与具体用户绑定的内容。
    """

    #: 默认错误码，子类可覆盖。
    default_code: Final[str] = "APP_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """转成可 JSON 序列化的结构，供异常处理器构造响应体。"""
        return {"code": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:
        if self.details:
            return f"[{self.code}] {self.message} | details={self.details}"
        return f"[{self.code}] {self.message}"


# ---------------------------------------------------------------------------
# 配置类
# ---------------------------------------------------------------------------
class ConfigurationError(AppError):
    """配置缺失或非法，例如未设置 DEEPSEEK_API_KEY 却调用了 LLM 接口。"""

    default_code = "CONFIGURATION_ERROR"


# ---------------------------------------------------------------------------
# 数据层（ingestion）
# ---------------------------------------------------------------------------
class IngestionError(AppError):
    """数据加载 / 切分 / 入库阶段失败。"""

    default_code = "INGESTION_ERROR"


class DocumentLoadError(IngestionError):
    """原始语料读取或解析失败（文件不存在、编码错误、JSON 格式错误等）。"""

    default_code = "DOCUMENT_LOAD_ERROR"


class UnsupportedFormatError(DocumentLoadError):
    """文件后缀不在支持列表内。

    单独成一个子类，是为了让"目录里混进了不认识的文件"这条分支可以被
    精确捕获并单独处理（例如批量扫描时记录后跳过），而不必去解析错误消息。
    """

    default_code = "UNSUPPORTED_FORMAT"


# ---------------------------------------------------------------------------
# 检索层（retrieval）
# ---------------------------------------------------------------------------
class RetrievalError(AppError):
    """向量检索阶段失败（向量库不可用、collection 不存在等）。"""

    default_code = "RETRIEVAL_ERROR"


class EmbeddingError(RetrievalError):
    """Embedding 计算失败（模型下载失败、接口报错、维度不匹配等）。"""

    default_code = "EMBEDDING_ERROR"


class IndexNotReadyError(RetrievalError):
    """向量库为空或未初始化，通常意味着还没跑过 ``scripts/ingest.py``。"""

    default_code = "INDEX_NOT_READY"


# ---------------------------------------------------------------------------
# 意图路由层（routing）
# ---------------------------------------------------------------------------
class RoutingError(AppError):
    """意图路由阶段失败（规则阈值非法、分类器不可用等）。"""

    default_code = "ROUTING_ERROR"


class FAQDataError(RoutingError):
    """FAQ 数据不可用：文件不是合法 JSON、结构不符合 FAQEntry、id 重复等。

    设计成"加载即报错"而不是"跳过坏条目"，是因为 FAQ 快路径一旦静默失效，
    表现是"问同样的问句却走了 RAG"，从日志上几乎看不出异常——这类问题必须
    在加载阶段就暴露。
    """

    default_code = "FAQ_DATA_ERROR"


# ---------------------------------------------------------------------------
# HTTP 层（api）
# ---------------------------------------------------------------------------
class ServiceNotReadyError(AppError):
    """应用还没准备好就收到了请求（例如绕过了 lifespan 启动流程）。

    单独成一个类型而不是抛 FastAPI 的 ``HTTPException``：后者返回 ``{"detail": ...}``，
    与本项目统一的 ``{"code", "message", "details"}`` 结构不一致，客户端得写两套解析。
    """

    default_code = "SERVICE_NOT_READY"


class UnauthorizedError(AppError):
    """写了接口但没带凭证／凭证不对。

    单独成一个类型（而不是抛 FastAPI 的 ``HTTPException``），是为了让错误响应
    保持全项目统一的结构 ``{"code", "message", "details"}``——混用会让客户端
    不得不写两套解析逻辑。
    """

    default_code = "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# 生成层（generation）
# ---------------------------------------------------------------------------
class GenerationError(AppError):
    """调用 LLM 失败。"""

    default_code = "GENERATION_ERROR"


class LLMTimeoutError(GenerationError):
    """LLM 调用超时，且重试后仍未成功。"""

    default_code = "LLM_TIMEOUT"


class LLMResponseError(GenerationError):
    """LLM 返回了无法解析的响应（空内容、非法 JSON 等）。"""

    default_code = "LLM_RESPONSE_ERROR"


class LLMFormatError(LLMResponseError):
    """模型响应的**结构**无法解析。

    与"空回答"刻意分开：空回答通常是一次抖动，值得重试；结构解析不了是
    **确定性**失败，重试三次结果一样，只会把延迟与费用翻倍。
    :mod:`src.generation.llm` 的重试谓词会显式排除本类型。
    """

    default_code = "LLM_FORMAT_ERROR"


# ---------------------------------------------------------------------------
# 业务编排（services）
# ---------------------------------------------------------------------------
class NoRelevantContextError(AppError):
    """检索结果不足以支撑回答，按拒答策略处理。

    注意：这不是"错误"，而是一条正常的业务分支——Pipeline 捕获它之后
    直接返回固定拒答话术，**不会**调用 LLM。保留成异常是为了让调用方
    无法忽略"证据不足"这一事实。
    """

    default_code = "NO_RELEVANT_CONTEXT"


class PipelineError(AppError):
    """Pipeline 编排过程中的其他失败。"""

    default_code = "PIPELINE_ERROR"
