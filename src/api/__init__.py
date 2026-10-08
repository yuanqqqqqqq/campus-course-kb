"""HTTP 层：只做路由、入参校验与响应封装，不含业务逻辑（阶段 6 已完成）。

- ``health``        —— 探活，不碰任何业务组件
- ``chat``          —— ``POST /api/chat``、``POST /api/chat/stream``（SSE）
- ``faq``           —— ``POST /api/faq``，动态新增 FAQ
- ``dependencies``  —— 从 ``app.state`` 取应用级单例的依赖函数

异常到状态码的映射集中在 ``src/main.py``，本层不写 ``try/except``。与其它包一致，
``__init__`` 不导入子模块。
"""
