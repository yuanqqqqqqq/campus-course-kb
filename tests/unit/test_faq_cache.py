"""``src.routing.faq_cache`` 的单元测试。

重点验证四件事：

1. **匹配**：精确匹配对大小写/标点/空格免疫；模糊匹配按 ``FAQ_MATCH_THRESHOLD``
   判定，阈值可配置；
2. **数据校验**：JSON 非法、结构不对、字段写错、id 重复都要在加载阶段报错；
3. **读写**：``add_faq`` 立即生效但不自动落盘；``save`` / ``load`` 能往返；
4. **验收链路**：FAQ 命中 → 直接返回答案且不碰 LLM / Embedding / Chroma；
   FAQ 未命中 → 继续走 RAG，**不拒答**。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import settings
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.vectorstore import VectorStore
from src.routing.classifier import IntentClassifier, IntentType
from src.routing.faq_cache import (
    EXACT_MATCH_SCORE,
    FAQCache,
    FAQEntry,
    MatchMethod,
    normalize_question,
)
from src.utils.exceptions import ConfigurationError, FAQDataError

#: 一份合法的 FAQ 数据（两条），供各测试按需覆盖。
FAQ_PAYLOAD: dict[str, object] = {
    "questions": [
        {
            "id": "faq_001",
            "patterns": ["CS201 的考核方式是什么", "数据结构与算法怎么考核"],
            "answer": "期末考试 60%、平时作业 20%、实验项目 20%。",
            "course_id": "CS201",
        },
        {
            "id": "faq_002",
            "patterns": ["CS101 几学分"],
            "answer": "CS101《程序设计基础》为 4 学分。",
            "course_id": "CS101",
        },
    ]
}


def _write(path: Path, payload: object) -> Path:
    """把任意载荷写入文件（包括故意写坏的内容）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def faq_path(tmp_path: Path) -> Path:
    """一份写在临时目录里的合法 FAQ 文件。"""
    return _write(tmp_path / "faq.json", FAQ_PAYLOAD)


@pytest.fixture
def cache(faq_path: Path) -> FAQCache:
    """已加载好的缓存实例。"""
    instance = FAQCache(path=faq_path)
    instance.load()
    return instance


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    ["CS101几学分", "cs101 几学分", "CS101，几学分！", " CS101\n几学分？ ", "CS101；；几学分、"],
)
def test_normalize_question_ignores_case_space_and_punctuation(text: str) -> None:
    """大小写、空白、中英文标点都不影响归一化结果。"""
    assert normalize_question(text) == "cs101几学分"


def test_normalize_question_of_pure_punctuation_is_empty() -> None:
    """全是标点的输入归一化后是空串 —— 调用方需要据此判未命中。"""
    assert normalize_question("？？？ ！！！ ") == ""


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------
def test_load_reads_entries_and_patterns(cache: FAQCache, faq_path: Path) -> None:
    """加载后条目数与 pattern 数都对得上。"""
    assert len(cache) == 2
    assert [entry.id for entry in cache.entries] == ["faq_001", "faq_002"]
    assert cache.path == faq_path
    assert cache.entries[0].course_id == "CS201"


def test_load_accepts_bare_list(tmp_path: Path) -> None:
    """裸数组写法也接受（兼容早期的 data/faq.json）。"""
    path = _write(tmp_path / "faq.json", FAQ_PAYLOAD["questions"])
    instance = FAQCache(path=path)
    instance.load()

    assert len(instance) == 2


def test_missing_file_gives_empty_cache(tmp_path: Path) -> None:
    """文件不存在不是错误：视为"还没建 FAQ"，匹配一律未命中并记 WARNING。"""
    instance = FAQCache(path=tmp_path / "not-there.json")
    instance.load()

    assert len(instance) == 0
    assert instance.lookup("CS101 几学分") is None


def test_empty_file_raises(tmp_path: Path) -> None:
    """空文件属于"有文件但不可用"，直接报错。"""
    path = tmp_path / "faq.json"
    path.write_text("   \n", encoding="utf-8")
    instance = FAQCache(path=path)

    with pytest.raises(FAQDataError, match="空文件"):
        instance.load()


def test_invalid_json_raises_with_position(tmp_path: Path) -> None:
    """非法 JSON 的报错要带行列号，否则排查不了手写文件里的漏逗号。"""
    path = tmp_path / "faq.json"
    path.write_text('{"questions": [ {"id": "faq_001",} ]}', encoding="utf-8")
    instance = FAQCache(path=path)

    with pytest.raises(FAQDataError, match="不是合法 JSON"):
        instance.load()


@pytest.mark.parametrize(
    "payload",
    [
        {"faqs": []},  # 顶层键名写错
        {"questions": {}},  # questions 不是数组
        [{"id": "faq_001", "patterns": ["x"]}],  # 缺 answer
        [{"id": "faq_001", "answer": "x"}],  # 缺 patterns
        [{"id": "faq_001", "patterns": [], "answer": "x"}],  # patterns 为空
        [{"id": "faq_001", "patterns": ["   "], "answer": "x"}],  # patterns 全是空白
        [{"id": "faq_001", "patterns": ["x"], "answer": "   "}],  # answer 全是空白
        [{"id": "faq_001", "patterns": ["x"], "answer": "y", "cours_id": "CS101"}],  # 字段名写错
        [{"id": "faq_001", "patterns": ["x"], "answer": "y", "id2": 1}],  # 多余字段
        ["不是对象"],  # 元素不是对象
        "questions",  # 顶层不是对象也不是数组
    ],
)
def test_structural_problems_raise(tmp_path: Path, payload: object) -> None:
    """结构不合规一律在加载阶段报错，不静默跳过。"""
    path = _write(tmp_path / "faq.json", payload)
    instance = FAQCache(path=path)

    with pytest.raises(FAQDataError):
        instance.load()


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    """id 重复会让"改哪一条"变得含糊，直接报错。"""
    path = _write(
        tmp_path / "faq.json",
        {
            "questions": [
                {"id": "faq_001", "patterns": ["A"], "answer": "答案 A"},
                {"id": "faq_001", "patterns": ["B"], "answer": "答案 B"},
            ]
        },
    )
    instance = FAQCache(path=path)

    with pytest.raises(FAQDataError, match="id 重复"):
        instance.load()


def test_unloaded_cache_refuses_to_match(faq_path: Path) -> None:
    """忘了 load() 必须报错，而不是静默"永远未命中"。"""
    instance = FAQCache(path=faq_path)

    with pytest.raises(FAQDataError, match="尚未加载"):
        instance.lookup("CS101 几学分")
    with pytest.raises(FAQDataError, match="尚未加载"):
        instance.save()
    with pytest.raises(FAQDataError, match="尚未加载"):
        instance.add_faq(["问法"], "答案")


def test_default_threshold_comes_from_settings() -> None:
    """默认阈值取自配置，而不是写死在代码里。"""
    assert FAQCache().threshold == settings.faq_match_threshold


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_invalid_threshold_raises(threshold: float) -> None:
    """阈值越界属于配置错误，构造时就报。"""
    with pytest.raises(ConfigurationError):
        FAQCache(threshold=threshold)


# ---------------------------------------------------------------------------
# 匹配
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question",
    ["CS101 几学分", "cs101几学分", "CS101，几学分？", "  cs101 几学分！ "],
)
def test_exact_match_is_punctuation_and_case_insensitive(cache: FAQCache, question: str) -> None:
    """精确匹配：大小写、空白、标点怎么写都命中同一条。"""
    match = cache.match_faq(question)

    assert match is not None
    assert match.entry.id == "faq_002"
    assert match.method is MatchMethod.EXACT
    assert match.score == EXACT_MATCH_SCORE
    assert cache.lookup(question) is match.entry


def test_fuzzy_match_hits_above_threshold(cache: FAQCache) -> None:
    """"CS201 的考核方式"少了"是什么"，相似度 0.8696，达到 0.85 应当命中。"""
    question = "CS201的考核方式"

    match = cache.match_faq(question)

    assert match is not None
    assert match.entry.id == "faq_001"
    assert match.method is MatchMethod.FUZZY
    assert match.score == pytest.approx(0.8696, abs=1e-3)
    assert match.score >= cache.threshold


def test_fuzzy_match_is_threshold_driven(cache: FAQCache, faq_path: Path) -> None:
    """同一个问题，把阈值提到 0.9 就不再模糊命中了 —— 阈值确实在起作用。"""
    strict = FAQCache(path=faq_path, threshold=0.9)
    strict.load()

    assert cache.match_faq("CS201的考核方式") is not None
    assert strict.match_faq("CS201的考核方式") is None
    # 精确匹配不受阈值影响
    assert strict.match_faq("CS101 几学分") is not None


def test_fuzzy_match_misses_unrelated_question(cache: FAQCache) -> None:
    """完全不相关的问题判未命中，且**不抛异常**（未命中是正常分支）。"""
    assert cache.lookup("图书馆几点关门") is None
    assert cache.match_faq("食堂今天有什么菜") is None


def test_fuzzy_match_rejects_a_different_course_code(cache: FAQCache) -> None:
    """只差一个课号的问句必须**未命中**，而不是命中另一个课的答案。

    背景（这条曾经是"已记录的局限"，现在是"必须被拦住的行为"）：
    ``"CS202的考核方式是什么"`` 与 pattern ``"CS201的考核方式是什么"`` 的相似度是
    **0.9231**，远高于 0.85 的阈值——纯靠相似度，问 CS202 会拿到 CS201 的考核方式，
    而且是以 ``cached=true`` 直接返回，不检索、不生成、不过相关性闸门。

    现在有课号一致性校验兜住它：问句里点了名的课程，必须与条目的 ``course_id`` 一致。
    """
    question = "CS202的考核方式是什么"
    ratio = SequenceMatcher(None, normalize_question(question), "cs201的考核方式是什么").ratio()

    # 相似度确实很高——这正是需要额外校验的原因，而不是可以放宽阈值的理由
    assert ratio == pytest.approx(0.9231, abs=1e-3)
    assert cache.match_faq(question) is None
    assert cache.lookup(question) is None


@pytest.mark.parametrize("question", ["CS202 的考核方式是什么", "CS401 几学分", "CS999 几学分"])
def test_a_question_naming_another_course_falls_through_to_rag(
    cache: FAQCache, question: str
) -> None:
    """语料里有、但 FAQ 库里没有的课：应当继续走 RAG，而不是拿别的课的答案。"""
    assert cache.match_faq(question) is None


def test_consistency_check_keeps_matching_the_same_course(cache: FAQCache) -> None:
    """课号一致的问句照常命中（含大小写差异）。"""
    assert cache.match_faq("CS201的考核方式").entry.id == "faq_001"
    assert cache.match_faq("cs201 的考核方式是什么").entry.id == "faq_001"


def test_questions_without_a_code_are_unaffected(cache: FAQCache) -> None:
    """问句里没有课号时无从校验，交给相似度与阈值（原有行为不变）。

    ``"数据结构与算法怎么考核的"`` 比 pattern 多一个字、且不含课号 —— 这类改写必须
    照常命中（统一测试 fixture 里的 pattern 是"数据结构与算法怎么考核"）。
    """
    match = cache.match_faq("数据结构与算法怎么考核的")

    assert match is not None
    assert match.entry.id == "faq_001"
    assert match.method is MatchMethod.FUZZY


def test_generic_entry_still_matches_a_course_specific_question(tmp_path: Path) -> None:
    """没有 ``course_id`` 的通用条目不受课号校验影响。

    数据作者常常写一条课程相关的问法却忘了填 ``course_id``；这时校验无从判断，
    放行是正确选择（拦下它等于把一条正确的 FAQ 变成永远命不中的死数据）。
    """
    path = _write(
        tmp_path / "faq.json",
        {
            "questions": [
                {"id": "faq_g1", "patterns": ["CS202 这门课怎么考核"], "answer": "闭卷笔试。"},
            ]
        },
    )
    cache = FAQCache(path=path)
    cache.load()

    match = cache.match_faq("CS202这门课怎么考核")

    assert match is not None
    assert match.entry.id == "faq_g1"


def test_blank_question_does_not_match_everything(cache: FAQCache) -> None:
    """空问题（或纯标点）判未命中 —— 两个空串的最相似度是 1.0，不拦住会误命中。"""
    assert cache.lookup("") is None
    assert cache.lookup("   ") is None
    assert cache.lookup("！！！？？？") is None


# ---------------------------------------------------------------------------
# 新增与保存
# ---------------------------------------------------------------------------
def test_add_faq_takes_effect_immediately(cache: FAQCache) -> None:
    """新增后立刻可查，且 id 按 faq_NNN 递增。"""
    entry = cache.add_faq(
        patterns=["MA101 几学分", "高等数学几学分"],
        answer="MA101《高等数学（上）》为 5 学分。",
        course_id="MA101",
    )

    assert entry.id == "faq_003"
    assert len(cache) == 3
    assert cache.lookup("MA101 几学分") is entry
    assert cache.lookup("MA101，几学分！") is entry


def test_add_faq_does_not_touch_disk(cache: FAQCache, faq_path: Path) -> None:
    """``add_faq`` 只改内存 —— 落盘时机由调用方决定。"""
    cache.add_faq(patterns=["临时问题"], answer="临时答案")

    assert "临时" not in faq_path.read_text(encoding="utf-8")


def test_add_faq_normalizes_patterns(cache: FAQCache) -> None:
    """入参里的空白项被丢掉，归一化后重复的 pattern 去重。"""
    entry = cache.add_faq(patterns=["  问法 A  ", "", "问法A"], answer="答案")

    assert entry.patterns == ["问法 A"]


@pytest.mark.parametrize(
    ("patterns", "answer"),
    [
        ([], "答案"),
        (["   "], "答案"),
        (["有效问法"], "   "),
    ],
)
def test_add_faq_rejects_invalid_input(cache: FAQCache, patterns: list[str], answer: str) -> None:
    """入参不合规时抛 :class:`FAQDataError`，而不是造出一条永远查不到的条目。"""
    with pytest.raises(FAQDataError):
        cache.add_faq(patterns=patterns, answer=answer)


def test_save_and_reload_round_trip(cache: FAQCache, faq_path: Path) -> None:
    """保存→重新加载，条目完全一致；且格式是规范的 ``{"questions": [...]}``。"""
    cache.add_faq(patterns=["通用问题"], answer="通用答案")
    cache.save()

    payload = json.loads(faq_path.read_text(encoding="utf-8"))
    assert list(payload) == ["questions"]
    assert len(payload["questions"]) == 3
    assert payload["questions"][-1]["course_id"] is None
    assert not (faq_path.parent / (faq_path.name + ".tmp")).exists()

    reloaded = FAQCache(path=faq_path)
    reloaded.load()

    assert [entry.model_dump() for entry in reloaded.entries] == [
        entry.model_dump() for entry in cache.entries
    ]
    assert reloaded.lookup("通用问题") is not None


def test_save_accepts_entries_without_course_id(tmp_path: Path) -> None:
    """``course_id`` 可以为空，保存后仍是合法文件。"""
    path = _write(
        tmp_path / "faq.json",
        {"questions": [{"id": "faq_001", "patterns": ["A"], "answer": "答案 A"}]},
    )
    instance = FAQCache(path=path)
    instance.load()
    instance.save()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["questions"][0]["course_id"] is None


# ---------------------------------------------------------------------------
# 验收：FAQ 命中 → 直接返回；FAQ 未命中 → 继续 RAG
# ---------------------------------------------------------------------------
class _CountingLLM:
    """假 LLM 分类器：只数调用次数。"""

    def __init__(self, intent: str = "course_query") -> None:
        self._intent = intent
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return f'{{"intent": "{self._intent}", "confidence": 0.9}}'


def _forbidden(what: str) -> Callable[..., object]:
    """造一个"被调用就失败"的替身，用来证明某条路径确实没碰某个组件。"""

    def _fail(*args: object, **kwargs: object) -> object:
        pytest.fail(f"FAQ 命中路径不应触碰 {what}")

    return _fail


@dataclass
class _MiniPipeline:
    """把 FAQ 与 RAG 串起来的迷你链路。

    阶段 6 的 ``services/pipeline.py`` 还没写，这里用最小实现把"命中即返回、
    未命中继续检索"这条验收要求固定下来：``handle()`` 里 **没有**任何拒答分支。
    """

    cache: FAQCache
    classifier: IntentClassifier
    llm: _CountingLLM
    rag_questions: list[str] = field(default_factory=list)

    def handle(self, question: str) -> dict[str, object]:
        match = self.cache.match_faq(question)
        if match is not None:
            return {
                "answer": match.entry.answer,
                "cached": True,
                "sources": [],
            }
        result = self.classifier.classify(question)
        self.rag_questions.append(question)
        return {
            "answer": "（检索 + 生成得到的答案）",
            "cached": False,
            "sources": [{"course_id": "CS302"}],
            "intent": result.intent,
        }


@pytest.fixture
def mini_pipeline(cache: FAQCache, monkeypatch: pytest.MonkeyPatch) -> _MiniPipeline:
    """迷你链路；同时把 Embedding 与向量库设成"被调用就失败"。"""
    monkeypatch.setattr(EmbeddingService, "get_embeddings", _forbidden("Embedding"))
    monkeypatch.setattr(VectorStore, "similarity_search_with_relevance", _forbidden("Chroma"))

    llm = _CountingLLM()
    return _MiniPipeline(cache=cache, classifier=IntentClassifier(llm=llm), llm=llm)


@pytest.mark.parametrize("question", ["CS101 几学分", "cs101 几学分？", "CS201的考核方式"])
def test_faq_hit_returns_answer_without_llm_embedding_or_chroma(
    mini_pipeline: _MiniPipeline, question: str
) -> None:
    """验收：FAQ 命中 → 直接返回答案，记录 cached=true，且零外部调用。"""
    response = mini_pipeline.handle(question)

    assert response["cached"] is True
    assert response["answer"] in {
        "CS101《程序设计基础》为 4 学分。",
        "期末考试 60%、平时作业 20%、实验项目 20%。",
    }
    assert mini_pipeline.llm.calls == 0
    assert mini_pipeline.rag_questions == []


def test_faq_miss_falls_through_to_rag(mini_pipeline: _MiniPipeline) -> None:
    """验收：FAQ 未命中 → 继续 RAG（并且经过意图分类），**绝不拒答**。"""
    question = "操作系统这门课的实验安排是什么"

    response = mini_pipeline.handle(question)

    assert response["cached"] is False
    assert response["answer"] == "（检索 + 生成得到的答案）"
    assert mini_pipeline.rag_questions == [question]
    assert mini_pipeline.llm.calls == 1
    assert response["intent"] is IntentType.COURSE_QUERY


def test_faq_entry_model_rejects_unknown_fields() -> None:
    """条目模型自己也守住字段名（写错了要在加载阶段就发现）。"""
    with pytest.raises(ValidationError):
        FAQEntry(id="faq_001", patterns=["A"], answer="B", cours_id="CS101")
