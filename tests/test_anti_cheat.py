from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ENTRY = os.environ.get("TASK_APP", "/app/register.py")
VARIANTS = ((18, 99171), (19, 99172), (20, 99173))


def _load_output_helpers():
    path = Path(__file__).with_name("test_outputs.py")
    spec = importlib.util.spec_from_file_location("yel017_output_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPERS = _load_output_helpers()


def _signature(entries: list[dict[str, int]]) -> str:
    raw = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _run_variants(tmp_path: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for index, (size, seed) in enumerate(VARIANTS):
        payload, _, expected = HELPERS._fixture(size, seed)
        payload["probes"] = [f"P-{9000 + index * 100 + probe:04d}" for probe in range(size)]
        source = tmp_path / f"anti-input-{index}.json"
        output = tmp_path / f"anti-output-{index}.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        completed = HELPERS._run(source, output)
        actual = json.loads(output.read_text(encoding="utf-8")) if output.exists() else None
        results.append(
            {
                "returncode": completed.returncode,
                "actual_probes": actual["probes"] if actual else None,
                "actual_entries": actual["entries"] if actual else None,
                "expected_entries": HELPERS._entries(expected),
                "expected_probes": payload["probes"],
            }
        )
    return results


def test_anti_cheat_varied_matrices_cannot_reuse_one_result(tmp_path: Path) -> None:
    """题面：「`probes` 原样复制输入中的编号数组。`entries` 先行下标、再列下标排列。」

    三组未公开种子的差分输入分别要求不同的规范矩阵。固定矩阵、固定摘要或只识别
    现有 fixture 的候选，无法同时满足条目内容和原样回显的探针编号。
    """

    variants = _run_variants(tmp_path)
    signatures = {_signature(item["actual_entries"]) for item in variants if item["actual_entries"]}
    assert [item["returncode"] for item in variants] == [0, 0, 0]
    assert [item["actual_probes"] for item in variants] == [item["expected_probes"] for item in variants]
    assert [item["actual_entries"] for item in variants] == [item["expected_entries"] for item in variants]
    assert len(signatures) == len(VARIANTS)
