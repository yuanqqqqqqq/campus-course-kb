"""测试包。

分为 ``unit``（纯逻辑、无外部依赖）与 ``integration``（走 FastAPI TestClient、
需要真实文件/向量库）两层。测试代码不放进 ``src``，避免被生产镜像打包。
"""
