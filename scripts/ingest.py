"""建库脚本：把 ``data/raw`` 的语料灌进 Chroma 向量库。

用法：

.. code-block:: bash

    python scripts/ingest.py                      # 全量重建（默认）
    python scripts/ingest.py --dry-run            # 只解析与切分，不写向量库
    python scripts/ingest.py --raw-dir data/raw   # 指定语料目录
    python scripts/ingest.py --batch-size 64      # 调整写库批次

**为什么是全量重建而不是增量追加**

``VectorStore.build_index`` 的语义是先删同名集合再重新写入。增量追加看起来更省时间，
但它会带来一类很难发现的事故：**从语料里删掉的课程仍然能被检索到**（旧文档还在库里）。
几十份文档的规模下，重建只要几秒，不值得拿正确性去换。需要增量追加时请直接用
``VectorStore`` 的底层接口，并在调用方自行负责一致性。

**运行前请注意**

- 首次运行会加载 Embedding 模型（本机实测约 5 秒）：``.env`` 里的
  ``EMBEDDING_MODEL`` 指向本地模型目录可以避免联网下载。
- 建库过程**不调用 LLM**，因此没有配置 ``DEEPSEEK_API_KEY`` 也能跑，只花钱不花的
  只有本地算力。
- 语料解析失败会直接报错退出（而不是写进半个索引）：解析结果可信是索引可信的前提。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 直接以脚本方式运行时（python scripts/ingest.py），sys.path[0] 是 scripts/ 而不是
# 项目根目录，`import src...` 会失败。这里补上项目根目录，让脚本在任意工作目录下都能跑。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.service import IngestionService  # noqa: E402
from src.retrieval.vectorstore import VectorStore  # noqa: E402
from src.utils.exceptions import AppError  # noqa: E402
from src.utils.logging import bind_request_id, get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

#: 默认语料目录。
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="把 data/raw 下的语料建进 Chroma 向量库（全量重建）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="语料目录")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="写库批次大小（文档很多时可以调大）"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只加载与切分，不写向量库；用于检查语料能不能解析通过",
    )
    parser.add_argument(
        "--collection", default=None, help="覆盖集合名（默认取 CHROMA_COLLECTION_NAME）"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行建库，返回进程退出码。"""
    args = _parse_args(argv)
    setup_logging()

    if args.batch_size < 1:
        print("参数不合法：--batch-size 必须为正整数", file=sys.stderr)
        return 2

    with bind_request_id() as request_id:
        logger.info(
            "建库开始 | request_id=%s raw_dir=%s collection=%s dry_run=%s",
            request_id,
            args.raw_dir,
            args.collection or settings.chroma_collection_name,
            args.dry_run,
        )

        started_at = time.perf_counter()
        try:
            documents = IngestionService().load_and_split(args.raw_dir)
        except AppError as exc:
            # 语料问题要给出可操作的提示，而不是一段堆栈——这类失败几乎都是数据写错了。
            print(f"\n建库失败：[{exc.code}] {exc.message}", file=sys.stderr)
            if exc.details:
                print(f"  上下文：{exc.details}", file=sys.stderr)
            return 2

        _report_documents(documents)

        if args.dry_run:
            logger.info("--dry-run：跳过写库 | 解析耗时 %.2fs", time.perf_counter() - started_at)
            print(f"\n试运行完成：共解析出 {len(documents)} 个文档切片，未写入向量库。")
            return 0

        store = VectorStore(collection_name=args.collection)
        try:
            store.build_index(documents)
        except AppError as exc:
            print(f"\n建库失败：[{exc.code}] {exc.message}", file=sys.stderr)
            return 2

        elapsed = time.perf_counter() - started_at
        print(
            f"\n建库完成：{store.count()} 个文档切片写入集合 "
            f"{store.collection_name}（{store.persist_dir}），耗时 {elapsed:.1f}s。"
        )
        return 0


def _report_documents(documents: list[object]) -> None:
    """打印语料构成，让人一眼看出"库里到底有什么"。"""
    by_type: dict[str, int] = {}
    courses: set[str] = set()
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        doc_type = str(metadata.get("type", "unknown"))
        by_type[doc_type] = by_type.get(doc_type, 0) + 1
        course_id = metadata.get("course_id")
        if course_id:
            courses.add(str(course_id))

    summary = "、".join(f"{doc_type}={count}" for doc_type, count in sorted(by_type.items()))
    print(f"\n语料解析结果：{len(documents)} 个切片（{summary}），涉及 {len(courses)} 门课程")
    print(f"  课程：{'、'.join(sorted(courses)) or '（无）'}")


if __name__ == "__main__":
    raise SystemExit(main())
