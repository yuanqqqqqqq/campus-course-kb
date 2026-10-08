"""接入编排层：把"加载"和"切分"串成一条流水线。

.. code-block:: text

    directory
        ↓
    DocumentLoader       加载 + 解析 + 生成原始 Document
        ↓
    CourseDocumentSplitter   按需二次切分
        ↓
    list[Document]       交给向量库建库（阶段 3）

本模块只负责**编排**：决定按什么顺序调用谁、在什么时候做一致性校验。具体的
读取逻辑在 loader、切分策略在 splitter，互不重复。

除编排之外，这里还守一道**出口校验**：流水线产出的每个文档必须带齐
``course_id`` / ``name`` / ``type`` / ``source``。上游 loader 理论上都会写入
这些字段，但流水线的价值恰恰在于"上游改坏了，这里立刻报错"，而不是等到建库
之后发现某些切片无法引用、无法回溯来源。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from langchain_core.documents import Document

from src.ingestion.loader import DocumentLoader
from src.ingestion.splitter import CourseDocumentSplitter
from src.utils.exceptions import IngestionError
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: 每个文档都必须具备的 metadata 字段。
#:
#: - ``course_id`` —— 引用与去重的主键
#: - ``name``      —— 回答里展示课程名
#: - ``type``      —— 区分"整门课"与"大纲章节"
#: - ``source``    —— 回溯到原始文件
REQUIRED_METADATA_KEYS: Final[tuple[str, ...]] = ("course_id", "name", "type", "source")

#: metadata 允许的取值类型。
#:
#: ChromaDB 只接受标量 metadata，list / dict / None 一律拒绝。``MetadataValue``
#: 类型别名只在静态检查阶段起作用，运行时什么都拦不住；所以这里再把关一次，
#: 让问题在**接入阶段**暴露，而不是等到阶段 3 建库时才炸。
_ALLOWED_METADATA_TYPES: Final[tuple[type, ...]] = (str, int, float, bool)


class IngestionService:
    """语料接入服务。

    :param loader: 文档加载器。默认新建一个 :class:`DocumentLoader`。
    :param splitter: 文档切分器。默认使用 :class:`CourseDocumentSplitter` 的
        默认参数。显式传入可以让调用方（或测试）替换任一环节的实现，而无需
        改动本类。
    """

    def __init__(
        self,
        loader: DocumentLoader | None = None,
        splitter: CourseDocumentSplitter | None = None,
    ) -> None:
        self._loader = loader if loader is not None else DocumentLoader()
        self._splitter = splitter if splitter is not None else CourseDocumentSplitter()

    @property
    def loader(self) -> DocumentLoader:
        """当前使用的加载器。"""
        return self._loader

    @property
    def splitter(self) -> CourseDocumentSplitter:
        """当前使用的切分器。"""
        return self._splitter

    def load_and_split(self, directory: str | Path) -> list[Document]:
        """加载目录下的语料并完成切分。

        :param directory: 语料目录，通常是 ``data/raw``。
        :return: 可直接送入向量库的文档列表。
        :raises DocumentLoadError: 目录 / 文件层面的问题（不存在、为空、格式错误等）。
        :raises IngestionError: 产出文档缺少必需 metadata，说明上游存在缺陷。
        """
        raw_documents = self._loader.load_directory(directory)
        documents = self._splitter.split(raw_documents)
        self._assert_required_metadata(documents)

        logger.info(
            "语料接入完成 | directory=%s raw=%d chunks=%d courses=%d",
            directory,
            len(raw_documents),
            len(documents),
            len({str(doc.metadata.get("course_id", "")) for doc in documents}),
        )
        return documents

    @staticmethod
    def _assert_required_metadata(documents: Sequence[Document]) -> None:
        """校验每个文档的 metadata 既完整又满足向量库的类型约束，不合格即报错。

        这里刻意用抛异常而不是记日志：缺字段的文档一旦进入向量库，检索结果里
        就会出现没有课程编号、无法引用、也无法回溯的"幽灵片段"，而那时候再想
        定位是哪一步弄丢的已经很难了。

        校验两项：

        1. 四个必备键存在且非空白；
        2. 所有取值都是标量——ChromaDB 不接受 list / dict / None。
        """
        for index, document in enumerate(documents):
            missing = [
                key
                for key in REQUIRED_METADATA_KEYS
                if not str(document.metadata.get(key, "")).strip()
            ]
            if missing:
                raise IngestionError(
                    f"第 {index} 个文档缺少必需 metadata 字段：{'、'.join(missing)}",
                    details={
                        "index": index,
                        "missing": missing,
                        "metadata": {str(k): str(v) for k, v in document.metadata.items()},
                        "page_content_head": document.page_content[:120],
                    },
                )

            invalid = [
                f"{key}={value!r}（{type(value).__name__}）"
                for key, value in document.metadata.items()
                if not isinstance(value, _ALLOWED_METADATA_TYPES)
            ]
            if invalid:
                raise IngestionError(
                    f"第 {index} 个文档的 metadata 含非标量取值，ChromaDB 无法存储："
                    f"{'、'.join(invalid)}。请在对应的 to_metadata 里压平成字符串。",
                    details={"index": index, "invalid": invalid},
                )
