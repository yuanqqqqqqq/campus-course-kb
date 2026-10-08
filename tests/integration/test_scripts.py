"""命令行脚本的冒烟测试（``scripts/ingest.py`` / ``scripts/benchmark.py``）。

脚本是最容易腐坏的一类代码：它们不被业务代码引用，改配置、改接口时没人会想起来
去动它们，等到"要建库了"才发现早就跑不通。这里用 pytest 把它们拉回测试网里：

- 建库脚本：真实语料能不能解析通过（不写向量库、不联网）
- 压测脚本：参数校验、以及"不许悄悄烧钱"的那道闸

都是秒级、离线、不需要 API Key 的检查。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.benchmark import main as benchmark_main
from scripts.ingest import DEFAULT_RAW_DIR
from scripts.ingest import main as ingest_main
from src.config import settings


@pytest.fixture(autouse=True)
def _restore_settings() -> object:
    """这两个脚本会读全局 settings，跑完把改动还回去。"""
    original = settings.enable_real_llm_benchmark
    yield
    settings.enable_real_llm_benchmark = original


def test_ingest_dry_run_reports_the_corpus(capsys: pytest.CaptureFixture[str]) -> None:
    """建库脚本的试运行必须能在真实语料上跑通，并报出切片数。"""
    assert DEFAULT_RAW_DIR.exists(), "data/raw 不见了，建库脚本的默认值已失效"

    exit_code = ingest_main(["--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "试运行完成" in captured.out
    assert "个文档切片" in captured.out


def test_ingest_rejects_missing_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """语料目录不存在时给出可读的错误与退出码，而不是抛一堆堆栈。"""
    exit_code = ingest_main(["--dry-run", "--raw-dir", str(tmp_path / "not-there")])

    assert exit_code == 2
    assert "建库失败" in capsys.readouterr().err


def test_ingest_rejects_invalid_batch_size(capsys: pytest.CaptureFixture[str]) -> None:
    """批次大小必须是正整数。"""
    exit_code = ingest_main(["--dry-run", "--batch-size", "0"])

    assert exit_code == 2
    assert "参数不合法" in capsys.readouterr().err


def test_benchmark_refuses_real_llm_without_explicit_opt_in(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--real-llm`` 必须配合 ENABLE_REAL_LLM_BENCHMARK：几百次真实调用不能靠手滑触发。"""
    settings.enable_real_llm_benchmark = False

    exit_code = benchmark_main(["--real-llm", "--requests", "1"])

    assert exit_code == 2
    assert "ENABLE_REAL_LLM_BENCHMARK" in capsys.readouterr().err


def test_benchmark_rejects_invalid_parameters(capsys: pytest.CaptureFixture[str]) -> None:
    """参数校验：请求数 / 并发 / top_k 都必须为正整数。"""
    assert benchmark_main(["--requests", "0"]) == 2
    assert "参数不合法" in capsys.readouterr().err


def test_benchmark_help_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--help`` 正常退出（写成 -h 时 argparse 抛 SystemExit(0)）。"""
    monkeypatch.setattr(sys, "argv", ["benchmark.py", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        benchmark_main(["--help"])

    assert excinfo.value.code == 0
