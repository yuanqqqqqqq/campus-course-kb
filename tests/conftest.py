"""跨测试文件共享的 fixture。

放在 ``tests/`` 根下，``unit/`` 与 ``integration/`` 的子测试都能用到。
这里的工厂 fixture 一律**复用被测代码本身的函数**（例如
:func:`~src.schemas.course.build_context_header`）来构造数据，而不是手写一份
"长得像"的字符串——否则被测逻辑一旦变化，测试里的假数据会跟着失真，
测试就变成了自说自话。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from src.retrieval.relevance import RelevanceChecker
from src.routing.faq_cache import FAQCache
from src.schemas.chat import IntentType
from src.schemas.course import DocumentType, build_context_header
from src.services.pipeline import ChatPipeline
from tests.fakes import FakeIntentClassifier, FakeLLMService, FakeVectorStore

#: 一份合法的课程 JSON 载荷，供各测试按需覆盖字段。
_DEFAULT_COURSE_PAYLOAD: dict[str, Any] = {
    "course_id": "CS201",
    "name": "数据结构与算法",
    "credits": 4,
    "prerequisites": ["CS101"],
    "instructor": "张伟",
    "semester": "2026-2027-1",
    "assessment": {"exam": 60, "homework": 20, "project": 20},
    "description": "系统讲授线性表、栈与队列、树与图等基本数据结构。",
    "textbooks": ["数据结构（C 语言版）"],
    "tags": ["专业核心课", "算法"],
}


@pytest.fixture
def course_payload() -> Callable[..., dict[str, Any]]:
    """返回一个"造课程字典"的工厂，可覆盖任意字段。

    .. code-block:: python

        course_payload(credits=0)          # 造一份学分非法的数据
        course_payload(course_id="CS999")  # 只改编号
    """

    def _factory(**overrides: Any) -> dict[str, Any]:
        payload = dict(_DEFAULT_COURSE_PAYLOAD)
        # 浅拷贝会让 assessment 被跨用例污染，这里单独复制一层
        payload["assessment"] = dict(_DEFAULT_COURSE_PAYLOAD["assessment"])
        for key, value in overrides.items():
            if value is ...:  # 用 ... 表示"删掉这个字段"，测试缺字段场景
                payload.pop(key, None)
            else:
                payload[key] = value
        return payload

    return _factory


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """一个干净的空语料目录。"""
    directory = tmp_path / "raw"
    directory.mkdir()
    return directory


#: 假 Embedding 的向量维度。取值只要能跑起来即可，与真实模型无关。
FAKE_EMBEDDING_SIZE = 16


class TableEmbeddings(Embeddings):
    """按文本查表返回固定向量的假 Embedding。

    存在的意义：检索测试需要**已知的相似度关系**才能断言排序与分数。用随机
    向量的话，只能断言"返回了 k 条"，无法断言"最相关的那条排在第一"。
    构造时给出几个向量，就能精确算出期望的余弦相似度。

    未登记的文本走确定性哈希兜底（同样的文本永远得到同样的向量），这样过滤、
    计数之类的测试不必为每段文本登记向量。

    :param table: ``{文本: 向量}``。所有向量维度必须一致。
    """

    def __init__(self, table: dict[str, Sequence[float]]) -> None:
        if not table:
            raise ValueError("table 不能为空")
        widths = {len(vector) for vector in table.values()}
        if len(widths) != 1:
            raise ValueError(f"table 中所有向量维度必须一致，实际出现 {sorted(widths)}")

        self._table: dict[str, list[float]] = {
            text: [float(value) for value in vector] for text, vector in table.items()
        }
        self._size = widths.pop()

    @property
    def size(self) -> int:
        """向量维度。"""
        return self._size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入。"""
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """嵌入单条查询。"""
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        """查表；未登记则由文本哈希确定性地生成一个向量。"""
        if text in self._table:
            return list(self._table[text])
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index] / 255.0 for index in range(self._size)]


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """框架自带的确定性假 Embedding，用于只关心结构、不关心语义的测试。"""
    return DeterministicFakeEmbedding(size=FAKE_EMBEDDING_SIZE)


@pytest.fixture
def table_embeddings() -> Callable[[dict[str, Sequence[float]]], TableEmbeddings]:
    """返回一个"造查表 Embedding"的工厂。

    .. code-block:: python

        embeddings = table_embeddings({"问题": [1.0, 0.0], "答案": [1.0, 0.0]})
    """

    def _factory(table: dict[str, Sequence[float]]) -> TableEmbeddings:
        return TableEmbeddings(table)

    return _factory


@pytest.fixture
def build_course_document() -> Callable[..., Document]:
    """返回一个"造课程文档"的工厂，产出与 Loader 同构的 Document。

    正文严格按 Loader 的规则拼装：``抬头 + 空行 + ## 章节标题 + 空行 + 正文``。
    这样切分器测试验证的就是真实输入形态，而不是一个理想化的假样本。
    """

    def _factory(
        *,
        course_id: str = "CS201",
        name: str = "数据结构与算法",
        section: str = "content",
        section_title: str = "教学内容",
        body: str = "正文内容。",
        doc_type: str = DocumentType.SYLLABUS.value,
        source: str = "course_syllabus.md",
        credits: float = 4.0,
        instructor: str = "张伟",
        semester: str = "2026-2027-1",
        prerequisites: str = "CS101",
        extra_metadata: dict[str, Any] | None = None,
    ) -> Document:
        metadata: dict[str, Any] = {
            "course_id": course_id,
            "name": name,
            "type": doc_type,
            "source": source,
            "credits": credits,
            "instructor": instructor,
            "semester": semester,
            "prerequisites": prerequisites,
            "section": section,
            "section_title": section_title,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        header = build_context_header(metadata)
        page_content = f"{header}\n\n## {section_title}\n\n{body}"
        return Document(page_content=page_content, metadata=metadata)

    return _factory


# ---------------------------------------------------------------------------
# 阶段 6：问答链路（Pipeline / API）的共享件
# ---------------------------------------------------------------------------
#: 测试用的一份 FAQ 数据。刻意保持"条目少、问法完整"，让命中与未命中都好构造。
_FAQ_ENTRIES: list[dict[str, object]] = [
    {
        "id": "faq_001",
        "patterns": ["CS201 的考核方式是什么"],
        "answer": "CS201《数据结构与算法》的考核方式：期末考试 60%、平时作业 20%、实验项目 20%。",
        "course_id": "CS201",
    },
    {
        "id": "faq_002",
        "patterns": ["CS101 几学分"],
        "answer": "CS101《程序设计基础》为 4 学分。",
        "course_id": "CS101",
    },
]

FAQ_PAYLOAD: dict[str, object] = {"questions": _FAQ_ENTRIES}


def _default_faq_intents() -> dict[str, IntentType]:
    """从 FAQ 数据推导"默认世界设定"：文件里写了的问法就判成 ``faq`` 意图。

    这样 :func:`make_pipeline` 造出来的链路是自洽的（分类器与 FAQ 文件对得上），
    测试想改某一条意图时再显式传自己的假分类器，不必每个用例都重复搭一遍。
    """
    intents: dict[str, IntentType] = {}
    for entry in _FAQ_ENTRIES:
        patterns = entry["patterns"]
        assert isinstance(patterns, list)
        for pattern in patterns:
            intents[str(pattern)] = IntentType.FAQ
    return intents


@pytest.fixture
def faq_cache(tmp_path: Path) -> FAQCache:
    """指向临时文件的、已加载的 FAQ 缓存。

    用临时文件而不是 ``data/faq.json``：测试会调 ``/api/faq`` 往缓存里写东西，
    绝不能污染仓库里的真实数据。
    """
    path = tmp_path / "faq.json"
    path.write_text(json.dumps(FAQ_PAYLOAD, ensure_ascii=False), encoding="utf-8")
    cache = FAQCache(path=path)
    cache.load()
    return cache


@pytest.fixture
def make_pipeline(faq_cache: FAQCache) -> Callable[..., ChatPipeline]:
    """返回一个"造问答链路"的工厂，只替换想控制的那几个组件。

    .. code-block:: python

        pipeline = make_pipeline(vector_store=FakeVectorStore(hits=[...]))

    未指定的部件用假实现 + 真实的相关性判定器（后者是纯逻辑，没有理由造假）。
    """
    default_faq = faq_cache

    def _factory(
        *,
        classifier: Any | None = None,
        faq_cache: FAQCache | None = None,
        vector_store: Any | None = None,
        relevance: RelevanceChecker | None = None,
        llm: Any | None = None,
        threshold: float = 0.5,
    ) -> ChatPipeline:
        return ChatPipeline(
            classifier=classifier or FakeIntentClassifier(_default_faq_intents()),
            faq_cache=default_faq if faq_cache is None else faq_cache,
            vector_store=vector_store or FakeVectorStore(),
            relevance=relevance or RelevanceChecker(threshold=threshold),
            llm=llm or FakeLLMService(),
        )

    return _factory
