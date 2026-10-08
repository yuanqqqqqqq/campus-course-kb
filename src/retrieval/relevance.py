"""相关性判定：把向量分数换算成统一口径的 relevance_score，并据此决定是否放行。

**先说清楚"分数"到底是什么**

这是整个检索层最容易出错的地方，因此结论不是推测出来的，是在当前依赖版本上
实测得到的。构造三个已知余弦相似度的向量，观察 Chroma 返回什么::

    与查询的余弦相似度:  A = 1.0     B = 0.7071    C = 0.0
    --------------------------------------------------------
    集合不指定 hnsw:space（默认）:   0.0     0.5858     2.0
    hnsw:space = "cosine":          0.0     0.2929     1.0
    hnsw:space = "l2":              0.0     0.5858     2.0

两个结论：

1. **Chroma 的默认距离不是余弦，而是平方欧氏距离**。默认配置下
   ``0.5858 = |A - B|²``，与余弦相似度没有任何直接关系。
2. **``similarity_search_with_score`` 返回的是"距离"，不是"相似度"**。
   在 cosine 空间下它就是 ``1 - 余弦相似度``，取值范围 ``[0, 2]``。

所以"score > 0.6 就算相关"这个直觉在这里是**反的**——距离越小才越相关。
本模块的职责就是把这件事收敛成一条口径：

.. code-block:: text

    距离（越小越相关，[0,2]）
        ↓  to_relevance_score()
    relevance_score（越大越相关，[0,1]，即余弦相似度）
        ↓  与 settings.relevance_threshold 比较
    is_relevant

**为什么用余弦而不是默认的 L2**

文本 Embedding 关心的是方向而不是模长。用 L2 时，一篇长文档和一段短查询的
距离会被长度差异主导；余弦则只看向量夹角，这也是 BGE 等中文模型的推荐用法。
向量库侧由 :data:`~src.retrieval.vectorstore.DISTANCE_SPACE` 固定为 ``cosine``，
本地 Embedding 侧配合 ``normalize_embeddings=True``，两边必须同时成立。

**本模块不生成任何自然语言答案。** 检索结果为空或不达标时只返回"不相关"，
具体返回什么话术由上层 Pipeline 决定（见总控要求第七节）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from langchain_core.documents import Document

from src.config import settings
from src.utils.exceptions import AppError, ConfigurationError, RetrievalError
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: relevance_score 的合法下界（两个向量完全无关时的余弦相似度）。
MIN_RELEVANCE_SCORE: Final[float] = 0.0

#: relevance_score 的合法上界（两个向量完全相同时的余弦相似度）。
MAX_RELEVANCE_SCORE: Final[float] = 1.0

#: LLM 复核回调的类型。
#:
#: 入参是 ``(用户问题, 待判定的候选文档)``，返回这些文档是否足以回答该问题。
#: 用回调而不是直接依赖 LLM 客户端，是为了让相关性判定这层不反向依赖生成层
#: （阶段 5 才实现 LLM 调用），同时测试里可以注入一个假判定器。
RelevanceJudge = Callable[[str, Sequence[Document]], bool]


def to_relevance_score(distance: float) -> float:
    """把 Chroma 的余弦距离换算成 relevance_score。

    :param distance: ``VectorStore.similarity_search_with_score`` 返回的原始距离，
        在 cosine 空间下等于 ``1 - 余弦相似度``。
    :return: 归一化到 ``[0, 1]`` 的相关性分数，**越大越相关**。

    映射关系（``relevance = 1 - distance``）只对 cosine 空间成立。向量库的
    距离空间由 :data:`~src.retrieval.vectorstore.DISTANCE_SPACE` 固定并在建
    集合时写入 metadata，所以这里可以放心按余弦处理。

    最后截断到 ``[0, 1]``：反方向向量会给出 ``distance > 1``，即负的余弦
    相似度，而"负相关"在检索场景下没有意义，统一按 0 处理，避免出现负分数
    让阈值比较变得难以理解。
    """
    score = 1.0 - distance
    if score < MIN_RELEVANCE_SCORE:
        return MIN_RELEVANCE_SCORE
    if score > MAX_RELEVANCE_SCORE:
        return MAX_RELEVANCE_SCORE
    return score


@dataclass(frozen=True)
class RelevanceResult:
    """一次相关性判定的完整结果。

    带上这些字段而不只是返回 bool，是为了让上层（和日志）能回答"为什么拒答了"
    ——只返回 ``False`` 的话，线上排查时完全看不出是阈值太高、还是检索压根没
    命中、还是 LLM 复核否掉了。
    """

    is_relevant: bool
    """最终结论。"""

    best_score: float | None
    """最高的一条 relevance_score；候选为空时为 ``None``。"""

    threshold: float
    """本次判定使用的阈值。"""

    checked_by_llm: bool
    """是否真的调用过 LLM 复核。"""

    reason: str
    """人类可读的判定理由，直接写进日志。"""

    candidate_count: int
    """参与判定的候选文档数量。"""


class RelevanceChecker:
    """按阈值判定检索结果是否足以支撑回答。

    :param threshold: 相关性阈值。``None`` 时取 ``settings.relevance_threshold``。
    :param enable_llm_check: 是否开启 LLM 复核。``None`` 时取
        ``settings.enable_llm_relevance_check``。
    :param judge: LLM 复核回调。仅在 ``enable_llm_check=True`` 时会用到。

    **判定顺序是刻意的**：先用免费的向量分数判断，只有**不达标**时才可能触发
    LLM 复核。这样绝大多数"命中良好"的请求完全不产生额外 LLM 调用，避免出现
    "分类 LLM + 相关性 LLM + 生成 LLM 每次都全跑一遍"的开销堆积。
    """

    def __init__(
        self,
        threshold: float | None = None,
        enable_llm_check: bool | None = None,
        judge: RelevanceJudge | None = None,
    ) -> None:
        resolved_threshold = threshold if threshold is not None else settings.relevance_threshold
        if not MIN_RELEVANCE_SCORE <= resolved_threshold <= MAX_RELEVANCE_SCORE:
            raise ConfigurationError(
                f"RELEVANCE_THRESHOLD 必须落在 "
                f"[{MIN_RELEVANCE_SCORE}, {MAX_RELEVANCE_SCORE}] 内，当前为 {resolved_threshold}。"
            )

        self._threshold = resolved_threshold
        self._enable_llm_check = (
            enable_llm_check
            if enable_llm_check is not None
            else settings.enable_llm_relevance_check
        )
        self._judge = judge

    @property
    def threshold(self) -> float:
        """当前生效的阈值。"""
        return self._threshold

    @property
    def enable_llm_check(self) -> bool:
        """是否开启了 LLM 复核。"""
        return self._enable_llm_check

    def check(
        self,
        query: str,
        results: Sequence[tuple[Document, float]],
    ) -> RelevanceResult:
        """判定检索结果是否与问题相关。

        :param query: 用户问题（仅在触发 LLM 复核时用到）。
        :param results: ``(文档, relevance_score)`` 序列，score 必须已经过
            :func:`to_relevance_score` 归一化。直接传
            ``VectorStore.similarity_search_with_score`` 的原始距离会触发
            :class:`RetrievalError`——那两者方向相反，混用会导致完全错误的判定。
        :raises RetrievalError: 分数超出 ``[0, 1]``。
        :raises ConfigurationError: 开启了 LLM 复核但没有注入 judge。
        """
        _validate_scores(results)

        if not results:
            # 空结果不是"错误"，只是一条正常的业务分支。这里不抛异常、也不生成
            # 任何自然语言——上层 Pipeline 负责据此返回固定拒答话术。
            logger.debug("相关性判定：候选为空 | query=%r", query)
            return RelevanceResult(
                is_relevant=False,
                best_score=None,
                threshold=self._threshold,
                checked_by_llm=False,
                reason="检索未返回任何候选文档",
                candidate_count=0,
            )

        best_score = max(score for _document, score in results)
        if best_score >= self._threshold:
            return RelevanceResult(
                is_relevant=True,
                best_score=best_score,
                threshold=self._threshold,
                checked_by_llm=False,
                reason=f"最高向量分数 {best_score:.4f} 达到阈值 {self._threshold}",
                candidate_count=len(results),
            )

        # 到这里说明向量分数不达标。只有此时才考虑动用 LLM。
        if not self._enable_llm_check:
            return RelevanceResult(
                is_relevant=False,
                best_score=best_score,
                threshold=self._threshold,
                checked_by_llm=False,
                reason=(
                    f"最高向量分数 {best_score:.4f} 低于阈值 {self._threshold}，"
                    f"且未开启 LLM 复核"
                ),
                candidate_count=len(results),
            )

        if self._judge is None:
            raise ConfigurationError(
                "已开启 LLM 相关性复核（ENABLE_LLM_RELEVANCE_CHECK=true）但未注入 judge 回调。"
                "请向 RelevanceChecker 传入 judge，或关闭该开关。"
            )

        documents = [document for document, _score in results]
        try:
            verdict = bool(self._judge(query, documents))
        except AppError as exc:
            # 复核只是"救回低分命中"的额外尝试，失败不该让整条请求变成 502：
            # 向量分数本来就没达标，按不相关处理（拒答）才是安全方向。
            logger.warning(
                "LLM 复核失败，按不相关处理 | code=%s error=%s", exc.code, exc
            )
            return RelevanceResult(
                is_relevant=False,
                best_score=best_score,
                threshold=self._threshold,
                checked_by_llm=True,
                reason=(
                    f"最高向量分数 {best_score:.4f} 低于阈值 {self._threshold}，"
                    f"且 LLM 复核失败（{exc.code}），按不相关处理"
                ),
                candidate_count=len(results),
            )
        logger.info(
            "LLM 复核完成 | query=%r vector_score=%.4f threshold=%.2f verdict=%s",
            query,
            best_score,
            self._threshold,
            verdict,
        )
        return RelevanceResult(
            is_relevant=verdict,
            best_score=best_score,
            threshold=self._threshold,
            checked_by_llm=True,
            reason=(
                f"最高向量分数 {best_score:.4f} 低于阈值 {self._threshold}，"
                f"LLM 复核判定{'相关' if verdict else '不相关'}"
            ),
            candidate_count=len(results),
        )


def check_relevance(
    query: str,
    results: Sequence[tuple[Document, float]],
    *,
    threshold: float | None = None,
) -> bool:
    """便捷函数：用默认配置判定结果是否相关。

    等价于 ``RelevanceChecker(threshold=threshold).check(query, results).is_relevant``。
    需要拿到判定细节（分数、理由）时请直接用 :class:`RelevanceChecker`。

    :param query: 用户问题。
    :param results: ``(文档, relevance_score)`` 序列，score 由
        :func:`to_relevance_score` 归一化得到。空序列返回 ``False``。
    :param threshold: 覆盖默认阈值，主要供评测脚本与测试使用。
    """
    return RelevanceChecker(threshold=threshold).check(query, results).is_relevant


def _validate_scores(results: Sequence[tuple[Document, float]]) -> None:
    """校验分数落在 relevance_score 的合法区间内。

    这条校验专门用来拦住一类很难自查的错误：把
    ``similarity_search_with_score`` 的原始**距离**直接当成 relevance_score
    传进来。两者方向相反，误用之后"越不相关分数越高"，而阈值比较不会报错，
    只会让检索结果安静地全错。

    余弦距离的取值范围是 ``[0, 2]``，因此任何大于 1 的分数一定是距离。
    小于等于 1 的距离无法与合法分数区分——这一点只能靠调用方遵守接口约定，
    函数名与 docstring 已尽量写清楚。
    """
    for index, (_document, score) in enumerate(results):
        if not MIN_RELEVANCE_SCORE <= score <= MAX_RELEVANCE_SCORE:
            raise RetrievalError(
                f"第 {index} 条结果的分数 {score!r} 不在 "
                f"[{MIN_RELEVANCE_SCORE}, {MAX_RELEVANCE_SCORE}] 内。"
                f"relevance_score 应由 to_relevance_score() 从余弦距离换算得到；"
                f"若直接传入了 similarity_search_with_score() 的原始距离，"
                f"请改用 similarity_search_with_relevance()。",
                details={"index": index, "score": score},
            )
