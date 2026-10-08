"""切分层：对过长的课程章节做二次切分。

**为什么不直接对所有文档无脑切**

上游 :class:`~src.ingestion.loader.DocumentLoader` 已经按课程结构做好了语义
分块——每个二级章节就是一个天然完整的语义单元。对这类"长度本来就合适"的
文档再切一刀，只会把一段完整论述拆成两个互不完整、各自都说不清话的片段，
同时让同一个知识点在向量库里出现多次，干扰检索排序。

所以本模块的策略是：**只在必要时切**，并且切完之后保证每个子片段依然
"自报家门"。

**上下文前缀的剥离与还原**

Loader 生成的正文形如::

    课程：数据结构与算法（CS201）
    学分：4 | 授课教师：张老师 | 开课学期：2026-2027-1 | 先修课程：CS101

    ## 教学内容

    （很长的正文……）

切分时先把开头这段"课程抬头 + 章节标题"摘下来，只切剩下的正文，然后再把
同一个前缀贴回每个子片段。

**为什么章节标题也要算进前缀**：底层分块器最优先按空行切分，标题与正文之间
正好就是一个空行。如果标题留在待切正文里，第一个切片会只剩"## 教学内容"
这一行、正文为空——一个纯标题切片对检索毫无价值，还会稀释真正的答案。把标题
挪进前缀后，每个子片段都同时带着"哪门课"和"哪一节"，既消灭了空切片，也让
片段自身更完整。

前缀中的课程抬头通过 :func:`~src.schemas.course.build_context_header` 从
metadata 重新推导，与 Loader 用的是同一个函数，因此两侧结果必然一致；章节
标题直接取 ``metadata["section_title"]``。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.schemas.course import build_context_header
from src.utils.logging import get_logger
from src.utils.text import as_text

logger = get_logger(__name__)

#: 切分正文时可用的最小宽度。
#:
#: 低于这个值时，抬头本身几乎占满整个 chunk，切出来的片段会碎到没有语义。
#: 与其产出一堆垃圾切片，不如直接报错让人把 chunk_size 调大。
_MIN_BODY_CHUNK_SIZE: Final[int] = 50

#: 中文优先的分隔符序列，从粗到细。
#:
#: ``RecursiveCharacterTextSplitter`` 会依次尝试这些分隔符：先用段落切，段落
#: 仍然超长时才退到换行，再退到句号、分号……这样能保证切点尽量落在语义边界上，
#: 而不是硬生生截断在半句话中间。
_SEPARATORS: Final[list[str]] = [
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    " ",
    "",
]

#: 写入每个片段的 metadata 键。
_CHUNK_INDEX_KEY: Final[str] = "chunk_index"
_CHUNK_TOTAL_KEY: Final[str] = "chunk_total"


class CourseDocumentSplitter:
    """课程文档切分器。

    :param chunk_size: 单个片段的**总长度上限**（含抬头），按字符数计。
        默认 800 —— 中文场景下约合 500~600 token，既能装下一段完整的
        考核方式说明或若干条教学内容，又不至于让检索粒度太粗。
    :param chunk_overlap: 相邻片段的重叠字符数。保留少量重叠是为了避免
        关键句正好被切在边界上而两边都读不完整。
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正数，当前为 {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap 不能为负数，当前为 {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})，"
                f"否则切分会陷入死循环。"
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    @property
    def chunk_size(self) -> int:
        """单个片段的长度上限。"""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """相邻片段的重叠长度。"""
        return self._chunk_overlap

    def split(self, documents: list[Document]) -> list[Document]:
        """按需切分文档列表。

        :param documents: 上游 Loader 产出的文档。
        :return: 切分后的文档列表。长度合适的文档会**原样保留**（仅补上
            ``chunk_index`` / ``chunk_total`` 两个定位字段），不会被切开。

        原始 metadata 一律完整保留；``course_id`` 与课程标题通过抬头机制
        出现在每个子片段的正文里，不会因切分而丢失。
        """
        result: list[Document] = []
        split_count = 0

        for document in documents:
            pieces = self._split_one(document)
            # 注意：这里用"是否超过 1 片"而不是"是否真的调用了切分器"来判断，
            # 两者在本实现中等价，但前者更贴合调用方的直觉。
            if len(pieces) > 1:
                split_count += 1
            result.extend(pieces)

        logger.info(
            "分块完成 | input=%d output=%d split_documents=%d chunk_size=%d overlap=%d",
            len(documents),
            len(result),
            split_count,
            self._chunk_size,
            self._chunk_overlap,
        )
        return result

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _split_one(self, document: Document) -> list[Document]:
        """切分单个文档。"""
        metadata = dict(document.metadata)
        page_content = document.page_content

        # 长度没有超限 → 不切。这是本模块最重要的分支：绝大多数章节都会走这里。
        if len(page_content) <= self._chunk_size:
            return [_with_chunk_metadata(document, index=0, total=1)]

        prefix = _resolve_context_prefix(page_content, metadata)
        body = page_content[len(prefix) :].lstrip("\n") if prefix else page_content
        if not body.strip():
            # 整个文档就是一段上下文前缀，没有正文可切。
            return [_with_chunk_metadata(document, index=0, total=1)]

        footprint = _prefix_footprint(prefix)
        available = self._chunk_size - footprint
        if available < _MIN_BODY_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size({self._chunk_size}) 减去上下文前缀({footprint} 字符)"
                f"后只剩 {available} 字符，不足以切分正文。请调大 chunk_size。"
            )

        pieces = [piece.strip() for piece in self._build_splitter(available).split_text(body)]
        pieces = [piece for piece in pieces if piece]

        if len(pieces) <= 1:
            # 理论上不该发生（长度已超限），但真出现了也不能把正文弄丢：
            # 退化成不切，正文保持原样。
            logger.debug("二次切分未产生有效片段，按原文档返回 | course=%s", metadata.get("course_id"))
            return [_with_chunk_metadata(document, index=0, total=1)]

        total = len(pieces)
        results: list[Document] = []
        for index, piece in enumerate(pieces):
            content = f"{prefix}\n\n{piece}" if prefix else piece
            results.append(
                _with_chunk_metadata(document, index=index, total=total, page_content=content)
            )
        return results

    def _build_splitter(self, body_chunk_size: int) -> RecursiveCharacterTextSplitter:
        """构造用于正文的底层分块器。

        每次按实际可用宽度新建实例，而不是在 ``__init__`` 里建一个复用：
        不同文档的抬头长度不同，"留给正文的宽度"也就不同，复用会导致
        部分片段超出 chunk_size 上限。
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=body_chunk_size,
            chunk_overlap=min(self._chunk_overlap, max(body_chunk_size - 1, 0)),
            length_function=len,
            # 必须是 "end" 而不是 True。
            #
            # keep_separator=True 走的是"分隔符贴到下一片开头"的分支，切出来的
            # 片段会以半句话结尾、下一片以孤零零的句号开头（"…接近真实文本" /
            # "。这是第 8 句…"）。传 "end" 才是把分隔符留在前一片末尾，
            # 于是每片都以完整的句子收尾，下一片也从完整句子开头。
            keep_separator="end",
            separators=_SEPARATORS,
        )


# ---------------------------------------------------------------------------
# 模块级辅助
# ---------------------------------------------------------------------------
def _prefix_footprint(prefix: str) -> int:
    """上下文前缀在正文里占用的字符数（含与正文之间的空行）。"""
    return len(prefix) + 2 if prefix else 0


def _resolve_context_prefix(page_content: str, metadata: Mapping[str, object]) -> str:
    """找出正文开头那段"每个子片段都应当重复"的上下文前缀。

    优先尝试"课程抬头 + 章节标题"（Markdown 大纲文档的形态），退化到只有
    课程抬头（结构化课程文档的形态：它只有总览，没有 ``## 章节`` 这一层）。

    两种都不是时返回空串——比如人为改写过的文档、或 metadata 里没有课程身份
    的文档。此时不剥离也不重复贴，宁可少做一步，也不能错误地砍掉正文开头或
    把前缀重复拼进去。
    """
    header = build_context_header(metadata)
    if not header:
        return ""

    section_title = as_text(metadata.get("section_title"))
    candidates = (
        [f"{header}\n\n## {section_title}", header] if section_title else [header]
    )
    for candidate in candidates:
        if page_content.startswith(candidate):
            return candidate

    logger.debug(
        "正文未以预期上下文前缀开头，跳过前缀剥离 | header=%r",
        header[:40],
    )
    return ""



def _with_chunk_metadata(
    document: Document,
    *,
    index: int,
    total: int,
    page_content: str | None = None,
) -> Document:
    """复制文档并补上分块定位信息。

    返回新对象而不是原地修改：``split`` 的入参是调用方的文档列表，就地改
    metadata 会让"输入"在调用后变成另一个样子，这种副作用在流水线里很难查。
    """
    metadata = dict(document.metadata)
    metadata[_CHUNK_INDEX_KEY] = index
    metadata[_CHUNK_TOTAL_KEY] = total
    return Document(
        page_content=page_content if page_content is not None else document.page_content,
        metadata=metadata,
    )
