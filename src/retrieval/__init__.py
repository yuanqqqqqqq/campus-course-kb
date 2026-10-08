"""检索层：Embedding、向量库、相关性判断（阶段 3 已完成）。

职责边界：

- ``embeddings``  —— Embedding 组件，支持 local(HuggingFace) / openai(兼容接口)；
  惰性导入，只使用 openai 模式的环境无需安装 torch
- ``vectorstore`` —— 只负责 Chroma 的读写与过滤，不做相关性判断；
  负责把集合钉死在 cosine 距离空间，并核对既有集合的空间是否一致
- ``relevance``   —— 只负责把距离换算成 relevance_score 并做阈值判定；
  不检索、不生成任何自然语言

本包不导入子模块，调用方按需显式导入，以免"只想用 Embedding"却把向量库
一起加载进来。
"""
