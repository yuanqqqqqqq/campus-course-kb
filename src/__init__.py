"""校园课程知识库 RAG 问答系统。

分层结构（上层可依赖下层，下层不得反向依赖上层）：

- ``config``       —— 全局配置，被所有层依赖
- ``utils``        —— 日志与异常，被所有层依赖
- ``schemas``      —— Pydantic 数据契约
- ``ingestion``    —— 语料加载与切分
- ``retrieval``    —— Embedding、向量库、相关性判断
- ``routing``      —— 意图识别与 FAQ 缓存
- ``generation``   —— Prompt 与 LLM 调用
- ``services``     —— Pipeline 编排
- ``api``          —— FastAPI 路由（只做 HTTP 层）
"""

__version__ = "0.1.0"
