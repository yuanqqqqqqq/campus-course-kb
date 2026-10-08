"""FAQ 快路径：命中即直接返回答案，全程不调用 LLM / Embedding / Chroma。

**它为什么值得单独存在**

"CS101 几学分"这类问题的答案是一个**固定值**，而向量检索是一条概率链路：
切分、嵌入、相似度、阈值，每一步都可能让它擦着阈值掉下去。这类问题走检索既能
花钱又更容易答错，所以把它们预先写成问答对，命中就原样返回。

**命中判定分两步：精确 → 模糊**

.. code-block:: text

    问题 → 归一化（小写 / 去空白 / 去标点）
             ↓
          精确匹配（内存字典，O(1)）        命中 → 返回
             ↓ 未命中
          模糊匹配（difflib.SequenceMatcher，阈值 FAQ_MATCH_THRESHOLD）
             ↓ 达到阈值
          课号一致性校验（问句点了课号就必须对得上）  不一致 → 丢弃该候选
             ↓ 通过
          命中 → 返回

归一化（:func:`normalize_question`）是这一步的关键：``"CS101几学分？"``、
``"cs101 几学分！"``、``"CS101，几学分。"`` 在归一化后是同一个串，用户怎么打标点
都不影响命中。

**模糊匹配的已知局限（务必知道）**

``SequenceMatcher`` 看的是整串相似度，不是语义。比例阈值 0.85 挡不住"只差一个
字符"的问句——实测 ``"CS202的考核方式是什么"`` 对 ``"CS201的考核方式是什么"``
的比值是 0.9231，按相似度就该命中 CS201 那条。

**所以匹配之外还有一道课号一致性校验**（见 :func:`_is_consistent_with_question`）：
问句里点了名的课程编号，必须与条目的 ``course_id`` 一致，否则这条候选作废。
没有这道校验时，"问 CS202 却拿到 CS201 的学分/考核方式"会以 ``cached=true`` 返回，
既不检索也不生成，**整条链路上没有任何一环能拦住它**——这是最容易产生
"自信的错误答案"的路径。

仍然要注意：

- ``patterns`` 要写成**完整问句**，并把课程编号写进去（校验只在问句**含课号**时生效，
  不含课号时无从校验）；
- ``answer`` 必须与 ``patterns`` 严格对应，**不要指望模糊匹配去兜错别字**，
  它兜住的是标点、语序与轻微改写的差异。

**未命中不等于拒答。** :meth:`FAQCache.lookup` 返回 ``None`` 只是"这本册子里没有"，
调用方（阶段 6 的 Pipeline）必须继续走 RAG，不得据此返回拒答话术。

**不做自动落盘。** :meth:`add_faq` 只改内存，``save()`` 负责写文件——什么时候把
FAQ 写回磁盘是业务决策（要不要热更新、要不要人工审核），不该由缓存组件替调用方
决定。反过来，``load()`` 也不自动触发：忘了加载时宁可报错（:class:`FAQDataError`），
也不要静默地"永远不命中"。
"""

from __future__ import annotations

import json
import os
import string
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from src.config import settings
from src.routing.classifier import COURSE_CODE_PATTERN
from src.schemas.chat import MAX_PATTERN_CHARS
from src.utils.exceptions import ConfigurationError, FAQDataError
from src.utils.logging import get_logger, question_for_log
from src.utils.text import format_validation_error, simplify_validation_errors

logger = get_logger(__name__)

#: 相似度的合法下界。
MIN_MATCH_SCORE: Final[float] = 0.0

#: 相似度的合法上界。
MAX_MATCH_SCORE: Final[float] = 1.0

#: 精确匹配的得分（归一化后完全相同）。
EXACT_MATCH_SCORE: Final[float] = 1.0

#: 读取用的编码。utf-8-sig 能同时正确处理带 BOM 的 UTF-8（Windows 记事本另存为
#: UTF-8 时会加 BOM），写回时用不带 BOM 的 utf-8。
_READ_ENCODING: Final[str] = "utf-8-sig"
_WRITE_ENCODING: Final[str] = "utf-8"

#: 归一化时要剔除的标点：ASCII 标点 + 常见中文全角标点。
#:
#: 逐个列出来而不是用正则的 ``\W``：``\W`` 会把中文也当成"非单词字符"，那样会把
#: 整个问句删空。这里要的只是标点。
_PUNCTUATION: Final[frozenset[str]] = frozenset(
    string.punctuation
    + "，。、；：？！…—～·－＋＝＜＞（）〈〉《》「」『』【】〔〕〖〗“”‘’＂＇"
    + "％＆＊／＠＃＄＾＿｀｜＼　"
)


def normalize_question(text: str) -> str:
    """归一化问题文本，用于 FAQ 的精确匹配与模糊匹配。

    规则：``casefold()`` 转小写（``CS101`` 与 ``cs101`` 视为同一个）→ 去掉所有空白
    字符 → 去掉中英文标点。**不做**同义词替换、繁简转换这类语义处理——那属于改写
    而不是归一化，交给 patterns 多写几条更可控。

    :param text: 原始问题或 pattern。
    :return: 归一化后的字符串；全是标点/空白时返回空串。
    """
    lowered = text.casefold()
    return "".join(
        char for char in lowered if not char.isspace() and char not in _PUNCTUATION
    )


class MatchMethod(str, Enum):
    """FAQ 命中的方式。写进日志，用于判断阈值调得对不对。"""

    EXACT = "exact"
    FUZZY = "fuzzy"


class FAQEntry(BaseModel):
    """一条预置问答对。

    对应 ``data/faq.json`` 里 ``questions`` 数组的一个元素。校验偏严格：字段名写错、
    类型不对、缺字段都会在加载阶段报错——FAQ 数据量小，没必要容忍脏数据，而它一旦
    带着错内容被答给用户，是没有下游环节能拦住的。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        min_length=1,
        max_length=64,
        description="唯一标识，形如 faq_001。重复 id 会在加载时直接报错。",
    )
    patterns: list[str] = Field(
        min_length=1,
        description=(
            "唤起这条答案的问法，可以写多条。归一化后重复的 pattern 只保留第一条。"
            "建议写成完整问句并带上课程编号（例如「CS201 的考核方式是什么」）。"
        ),
    )
    answer: str = Field(
        min_length=1,
        description="命中后原样返回的答案文本。**不经过 LLM 改写**，所以必须自己写清楚。",
    )
    course_id: str | None = Field(
        default=None,
        description="关联的课程编号，便于过滤与回溯；通用 FAQ（不属于某门课）留空。",
    )

    @field_validator("id", "answer", mode="after")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        """去首尾空白，并拒绝"全是空格"的字符串（``min_length=1`` 拦不住）。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("不能为空字符串或纯空白字符")
        return stripped

    @field_validator("patterns", mode="after")
    @classmethod
    def _normalize_patterns(cls, values: list[str]) -> list[str]:
        """去空白 → 丢弃空项 → 长度上限 → 按归一化结果去重（保留原写法）。"""
        cleaned: list[str] = []
        seen: set[str] = set()
        for raw in values:
            pattern = raw.strip()
            if not pattern:
                continue
            if len(pattern) > MAX_PATTERN_CHARS:
                raise ValueError(f"单条问法不能超过 {MAX_PATTERN_CHARS} 个字符")
            key = normalize_question(pattern)
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned.append(pattern)
        if not cleaned:
            raise ValueError("patterns 至少要有一条非空问法")
        return cleaned

    @field_validator("course_id", mode="after")
    @classmethod
    def _blank_course_id_becomes_none(cls, value: str | None) -> str | None:
        """把空串 / 纯空白的 ``course_id`` 归一成 ``None``，避免出现"有值但是空"。"""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


@dataclass(frozen=True)
class FAQMatch:
    """一次 FAQ 匹配的完整结果。

    只返回 :class:`FAQEntry` 的话，线上就没法回答"这条是精确命中还是模糊擦边命中"
    ——而这两种情况的风险完全不同，是调阈值时唯一可用的证据。
    """

    entry: FAQEntry
    score: float
    """相似度，精确匹配恒为 1.0。"""

    method: MatchMethod
    """精确命中还是模糊命中。"""

    pattern: str
    """命中的那条 pattern（归一化后的形式）。"""


class FAQCache:
    """FAQ 的加载、匹配与保存。

    :param path: FAQ 文件路径。``None`` 时取 ``settings.faq_path``。
    :param threshold: 模糊匹配阈值。``None`` 时取 ``settings.faq_match_threshold``。

    **必须先 :meth:`load`。** 未加载就调用匹配/保存会抛 :class:`FAQDataError`：
    "配了 FAQ 却不生效"是一类几乎无法从日志上看出来的故障，宁可在这里报错。

    **写入是带锁的。** FastAPI 的同步接口跑在线程池里，``POST /api/faq`` 并发时就
    会有多个线程同时改同一个缓存：``_next_id`` 会算出同一个 id、两次 ``save`` 会
    交叉写文件。读路径不加锁——索引换的是整个字典引用（原子替换），读到的要么是
    旧版要么是新版，不会读到半个。
    """

    def __init__(self, path: str | Path | None = None, threshold: float | None = None) -> None:
        self._path = Path(path) if path is not None else settings.faq_path
        resolved_threshold = (
            threshold if threshold is not None else settings.faq_match_threshold
        )
        if not MIN_MATCH_SCORE <= resolved_threshold <= MAX_MATCH_SCORE:
            raise ConfigurationError(
                f"FAQ_MATCH_THRESHOLD 必须落在 "
                f"[{MIN_MATCH_SCORE}, {MAX_MATCH_SCORE}] 内，当前为 {resolved_threshold}。"
            )

        self._threshold = resolved_threshold
        self._entries: list[FAQEntry] = []
        self._exact_index: dict[str, FAQEntry] = {}
        self._fuzzy_index: list[tuple[str, FAQEntry]] = []
        self._loaded = False
        #: 只保护写路径（load / save / add_faq），见类文档。
        self._write_lock = threading.RLock()

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        """FAQ 文件路径。"""
        return self._path

    @property
    def threshold(self) -> float:
        """当前生效的模糊匹配阈值。"""
        return self._threshold

    @property
    def entries(self) -> tuple[FAQEntry, ...]:
        """全部条目（只读快照）。"""
        return tuple(self._entries)

    def __len__(self) -> int:
        """条目数量；未加载时为 0。"""
        return len(self._entries)

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    def load(self) -> None:
        """从磁盘加载 FAQ。

        失败策略分两种，是有意区分开的：

        - **文件不存在** —— 视为"还没建 FAQ"，清空内存并记 WARNING，不报错。开发
          早期没有这个文件很正常，为此让服务起不来没有意义。
        - **文件存在但内容不合规** —— 抛 :class:`FAQDataError`。文件在那里却读不了，
          说明数据坏了或写错了字段名，必须立刻暴露。

        :raises FAQDataError: 编码错误、JSON 非法、结构不符合预期、条目校验不过、id 重复。
        """
        with self._write_lock:
            self._load_locked()

    def _load_locked(self) -> None:
        """:meth:`load` 的实现（调用方需已持有写锁）。"""
        if not self._path.exists():
            self._replace_entries([])
            logger.warning(
                "FAQ 文件不存在，FAQ 快路径将始终未命中（问题会继续走 RAG） | path=%s",
                self._path,
            )
            return

        text = _read_text(self._path)
        if not text.strip():
            raise FAQDataError(
                f"{self._path.name} 是空文件，无法作为 FAQ 数据源。"
                f"空内容请写成 {{\"questions\": []}}。",
                details={"path": str(self._path)},
            )

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise FAQDataError(
                f"{self._path.name} 不是合法 JSON：第 {exc.lineno} 行第 {exc.colno} 列 {exc.msg}",
                details={
                    "path": str(self._path),
                    "line": exc.lineno,
                    "column": exc.colno,
                    "reason": exc.msg,
                },
            ) from exc

        entries = self._parse_entries(payload)
        self._replace_entries(entries)
        logger.info(
            "FAQ 加载完成 | path=%s entries=%d patterns=%d threshold=%.2f",
            self._path,
            len(self._entries),
            len(self._exact_index),
            self._threshold,
        )

    def save(self) -> None:
        """把当前内存中的条写回磁盘（``{"questions": [...]}`` 格式）。

        先写 ``*.tmp`` 再 ``os.replace`` 原子替换：直接覆写的话，进程在写一半时挂掉
        会留下一个半截的 JSON，下次启动直接读不出来。

        :raises FAQDataError: 未加载，或写入失败。
        """
        with self._write_lock:
            self._ensure_loaded()
            # 序列化在锁内完成：entries 列表在写盘过程中被追加的话，文件里会缺条目
            # （内容与 entry 数对不上）。
            payload: dict[str, Any] = {
                "questions": [entry.model_dump() for entry in self._entries]
            }
            text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

            tmp_path = self._path.with_name(self._path.name + ".tmp")
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_text(text, encoding=_WRITE_ENCODING)
                os.replace(tmp_path, self._path)
            except OSError as exc:
                tmp_path.unlink(missing_ok=True)
                raise FAQDataError(
                    f"FAQ 写入失败：{self._path}（{exc.strerror or exc}）",
                    details={"path": str(self._path)},
                ) from exc

            logger.info("FAQ 已保存 | path=%s entries=%d", self._path, len(self._entries))

    # ------------------------------------------------------------------
    # 匹配
    # ------------------------------------------------------------------
    def lookup(self, question: str) -> FAQEntry | None:
        """查 FAQ，命中返回条目，未命中返回 ``None``。

        :param question: 用户问题原文（标点、大小写、空格都不影响匹配）。

        **返回 ``None`` 不是拒答信号**：调用方必须继续走 RAG。
        需要命中细节（相似度、命中方式）时用 :meth:`match_faq`。
        """
        match = self.match_faq(question)
        return match.entry if match is not None else None

    def match_faq(self, question: str) -> FAQMatch | None:
        """查 FAQ 并返回命中细节；未命中返回 ``None``。

        先查精确索引，再线性扫一遍 pattern 算相似度取最高分。当前实现是逐个算
        ``SequenceMatcher``：条目上千以后可以考虑先用 ``real_quick_ratio()`` 粗筛，
        在 FAQ 只有几十条的规模下这属于过早优化。

        :raises FAQDataError: 未加载。
        """
        self._ensure_loaded()

        key = normalize_question(question)
        if not key:
            # 问题全是标点或空白，匹配任何 pattern 都会得到虚高的相似度（例如两个
            # 空串的比值是 1.0），直接判未命中。
            logger.debug("FAQ 匹配：问题归一化后为空 | question=%s", question_for_log(question))
            return None

        entry = self._exact_index.get(key)
        if entry is not None and _is_consistent_with_question(key, entry):
            logger.info("FAQ 精确命中 | id=%s question=%s", entry.id, question_for_log(question))
            return FAQMatch(
                entry=entry,
                score=EXACT_MATCH_SCORE,
                method=MatchMethod.EXACT,
                pattern=key,
            )

        best: tuple[float, str, FAQEntry] | None = None
        for pattern, candidate in self._fuzzy_index:
            # 课号一致性校验在算相似度**之前**：它是一票否决，与分数高低无关。
            if not _is_consistent_with_question(key, candidate):
                logger.info(
                    "FAQ 候选被课号一致性校验拒绝 | question=%s entry=%s entry_course=%s",
                    question_for_log(question),
                    candidate.id,
                    candidate.course_id,
                )
                continue
            score = SequenceMatcher(None, key, pattern).ratio()
            if best is None or score > best[0]:
                best = (score, pattern, candidate)

        if best is None or best[0] < self._threshold:
            logger.debug(
                "FAQ 未命中，继续走 RAG | question=%s best=%s threshold=%.2f",
                question_for_log(question),
                f"{best[0]:.4f}" if best is not None else "无 pattern",
                self._threshold,
            )
            return None

        score, pattern, entry = best
        logger.info(
            "FAQ 模糊命中 | id=%s score=%.4f threshold=%.2f question=%s",
            entry.id,
            score,
            self._threshold,
            question_for_log(question),
        )
        return FAQMatch(entry=entry, score=score, method=MatchMethod.FUZZY, pattern=pattern)

    # ------------------------------------------------------------------
    # 写入单条
    # ------------------------------------------------------------------
    def add_faq(
        self,
        patterns: Sequence[str],
        answer: str,
        course_id: str | None = None,
    ) -> FAQEntry:
        """新增一条 FAQ 并立即生效（改内存，不落盘）。

        :param patterns: 唤起这条答案的问法，至少一条非空。
        :param answer: 要原样返回的答案。
        :param course_id: 关联课程编号，可省略。
        :return: 新建的 :class:`FAQEntry`，``id`` 由本方法自动分配（``faq_NNN``）。
        :raises FAQDataError: 未加载，或入参不合规。

        需要持久化时由调用方显式调用 :meth:`save`。
        """
        with self._write_lock:
            return self._add_faq_locked(patterns, answer, course_id)

    def _add_faq_locked(
        self,
        patterns: Sequence[str],
        answer: str,
        course_id: str | None,
    ) -> FAQEntry:
        """:meth:`add_faq` 的实现（调用方需已持有写锁）。

        整个"算 id → 追加 → 重建索引"必须在同一把锁里：``_next_id`` 先看一遍现有
        id 再生成，两个线程交错就会造出两条同 id 的 FAQ。
        """
        self._ensure_loaded()

        try:
            entry = FAQEntry(
                id=self._next_id(),
                patterns=list(patterns),
                answer=answer,
                course_id=course_id,
            )
        except ValidationError as exc:
            raise FAQDataError(
                f"新增 FAQ 不符合 FAQEntry 模型：{format_validation_error(exc)}",
                details={"errors": simplify_validation_errors(exc)},
            ) from exc

        self._entries.append(entry)
        self._rebuild_index()
        logger.info(
            "FAQ 新增（仅内存，尚未落盘） | id=%s patterns=%d course_id=%s",
            entry.id,
            len(entry.patterns),
            entry.course_id or "-",
        )
        return entry

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _parse_entries(self, payload: object) -> list[FAQEntry]:
        """把 JSON 载荷解析成条目列表。"""
        records = _extract_records(payload, self._path)

        entries: list[FAQEntry] = []
        seen_ids: dict[str, int] = {}
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise FAQDataError(
                    f"{self._path.name} 第 {index} 条 FAQ 应当是 JSON 对象，"
                    f"实际是 {type(record).__name__}",
                    details={"path": str(self._path), "index": index},
                )
            try:
                entry = FAQEntry.model_validate(dict(record))
            except ValidationError as exc:
                raise FAQDataError(
                    f"{self._path.name} 第 {index} 条 FAQ 不符合 FAQEntry 模型："
                    f"{format_validation_error(exc)}",
                    details={
                        "path": str(self._path),
                        "index": index,
                        "errors": simplify_validation_errors(exc),
                    },
                ) from exc

            if entry.id in seen_ids:
                raise FAQDataError(
                    f"{self._path.name} 中 id 重复：{entry.id}"
                    f"（第 {seen_ids[entry.id]} 条与第 {index} 条）",
                    details={"path": str(self._path), "id": entry.id},
                )
            seen_ids[entry.id] = index
            entries.append(entry)

        return entries

    def _replace_entries(self, entries: Sequence[FAQEntry]) -> None:
        """整体替换内存中的条目并重建索引。"""
        self._entries = list(entries)
        self._rebuild_index()
        self._loaded = True

    def _rebuild_index(self) -> None:
        """重建精确索引与模糊匹配列表。

        同一归一化 pattern 出现在多条 FAQ 里时**保留先出现的**并记 WARNING：两条
        不同答案共用一条问法是数据错误，但为此拒绝加载整个文件不划算——记下
        被忽略的 id，好让人回去改数据。
        """
        exact: dict[str, FAQEntry] = {}
        fuzzy: list[tuple[str, FAQEntry]] = []

        for entry in self._entries:
            for pattern in entry.patterns:
                key = normalize_question(pattern)
                if not key:
                    logger.warning(
                        "FAQ 的 pattern 归一化后为空，已忽略 | id=%s pattern=%r",
                        entry.id,
                        pattern,
                    )
                    continue
                if key in exact:
                    logger.warning(
                        "FAQ pattern 重复，保留先出现的条目 | pattern=%r kept=%s ignored=%s",
                        pattern,
                        exact[key].id,
                        entry.id,
                    )
                    continue
                exact[key] = entry
                fuzzy.append((key, entry))

        self._exact_index = exact
        self._fuzzy_index = fuzzy

    def _next_id(self) -> str:
        """分配下一个 ``faq_NNN`` 形式的 id。

        从"已有条数 + 1"起步，撞上就往后找，因此混合了手工指定 id 的文件也不会
        产生冲突。
        """
        used = {entry.id for entry in self._entries}
        index = len(self._entries) + 1
        while f"faq_{index:03d}" in used:
            index += 1
        return f"faq_{index:03d}"

    def _ensure_loaded(self) -> None:
        """未加载就使用属于调用顺序错误，直接报错而不是静默返回空结果。"""
        if not self._loaded:
            raise FAQDataError(
                "FAQ 缓存尚未加载，请先调用 load()。"
                "（静默跳过会让 FAQ 无声失效，只表现为「问同样的问题却走了 RAG」）",
                details={"path": str(self._path)},
            )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _is_consistent_with_question(question_key: str, entry: FAQEntry) -> bool:
    """问句里点名的课程编号，必须与条目的 ``course_id`` 一致。

    **为什么需要这一道校验**

    问句只差一个字符时，相似度依然很高：实测 ``"CS202的考核方式是什么"`` 对
    ``"CS201的考核方式是什么"`` 是 **0.9231**，远高于 0.85 的阈值。没有这道校验时，
    问 CS202 会拿到 CS201 的学分/考核方式，而且是以 ``cached=true`` 直接返回的——
    不检索、不生成、不过相关性闸门，整条链路上没有任何一环能拦住它。这是本系统里
    最容易产出"自信的错误答案"的一条路径。

    规则很简单：

    - 条目没有 ``course_id``（通用 FAQ）→ 放行。它本来就与具体课程无关；
    - 问句里没有任何课号 → 放行。无从校验，交给相似度与阈值；
    - 问句含课号 → 条目的 ``course_id`` 必须出现在问句里。

    :param question_key: 已归一化的问题文本。
    :param entry: 候选条目。
    """
    if entry.course_id is None:
        return True

    codes = _course_codes(question_key)
    if not codes:
        return True

    return entry.course_id.casefold() in codes


def _course_codes(normalized_text: str) -> frozenset[str]:
    """从归一化文本里取出全部课程编号（统一小写）。

    与意图分类用的是同一个正则（:data:`~src.routing.classifier.COURSE_CODE_PATTERN`），
    避免"分类认得出 CS201、FAQ 却认不出"这种两套口径的问题。
    """
    return frozenset(match.casefold() for match in COURSE_CODE_PATTERN.findall(normalized_text))

def _extract_records(payload: object, path: Path) -> Sequence[object]:
    """从两种受支持的形态里取出条目序列。

    规范写法是 ``{"questions": [...]}``。同时也接受裸数组 ``[...]``：早期版本的
    ``data/faq.json`` 就是这个形态，兼容一行代码就能做到，比让人对着"结构无法识别"
    改数据划算。
    """
    if isinstance(payload, list):
        return payload

    if isinstance(payload, Mapping) and "questions" in payload:
        questions = payload["questions"]
        if not isinstance(questions, list):
            raise FAQDataError(
                f"{path.name} 的 'questions' 字段应当是数组，"
                f"实际是 {type(questions).__name__}",
                details={"path": str(path)},
            )
        return questions

    raise FAQDataError(
        f"{path.name} 的结构无法识别。支持两种写法："
        f'{{"questions": [...]}} 包装对象，或直接的问答数组 [...]。',
        details={"path": str(path), "top_level_type": type(payload).__name__},
    )


def _read_text(path: Path) -> str:
    """以 UTF-8 读取文本，失败时给出可操作的提示。"""
    try:
        return path.read_text(encoding=_READ_ENCODING)
    except UnicodeDecodeError as exc:
        raise FAQDataError(
            f"{path.name} 不是 UTF-8 编码，无法读取（{exc.reason}，位置 {exc.start}）。"
            f"请用编辑器另存为 UTF-8 后重试。",
            details={"path": str(path), "encoding": _READ_ENCODING},
        ) from exc
    except OSError as exc:
        raise FAQDataError(
            f"{path.name} 读取失败：{exc.strerror or exc}",
            details={"path": str(path)},
        ) from exc



