"""通用工具层：日志与异常。

本包不导入子模块，调用方按需显式导入：

.. code-block:: python

    from src.utils.logging import get_logger
    from src.utils.exceptions import AppError

这样可以避免"只想拿个 logger，却把整个异常体系连同配置一起加载"。
"""
