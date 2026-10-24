from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path


ENTRY = os.environ.get("TASK_APP", "/app/register.py")

SAFE_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}
FIELD = 1019


def _run(
    source: Path,
    output: Path,
    *extra: str,
    timeout: int = 20,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-S", "-P", "-B", ENTRY, str(source), str(output), *extra]
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        env=SAFE_ENV,
        timeout=timeout,
        check=False,
    )


def _run_without_paths() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-S", "-P", "-B", ENTRY],
        text=True,
        capture_output=True,
        env=SAFE_ENV,
        timeout=20,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _inverse(matrix: list[list[int]]) -> list[list[int]]:
    size = len(matrix)
    augmented = [
        [value % FIELD for value in row]
        + [1 if row_index == column else 0 for column in range(size)]
        for row_index, row in enumerate(matrix)
    ]
    pivot_row = 0
    for column in range(size):
        pivot = next(
            index
            for index in range(pivot_row, size)
            if augmented[index][column] % FIELD
        )
        augmented[pivot_row], augmented[pivot] = augmented[pivot], augmented[pivot_row]
        factor = pow(augmented[pivot_row][column], FIELD - 2, FIELD)
        augmented[pivot_row] = [value * factor % FIELD for value in augmented[pivot_row]]
        for index in range(size):
            if index == pivot_row:
                continue
            factor = augmented[index][column] % FIELD
            if factor:
                augmented[index] = [
                    (left - factor * right) % FIELD
                    for left, right in zip(augmented[index], augmented[pivot_row])
                ]
        pivot_row += 1
    return [row[size:] for row in augmented]


def _multiply(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    size = len(left)
    return [
        [
            sum(left[row][middle] * right[middle][column] for middle in range(size))
            % FIELD
            for column in range(size)
        ]
        for row in range(size)
    ]


def _transpose(matrix: list[list[int]]) -> list[list[int]]:
    return [list(column) for column in zip(*matrix)]


def _determinant(matrix: list[list[int]]) -> int:
    if len(matrix) == 1:
        return matrix[0][0] % FIELD
    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) % FIELD
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    ) % FIELD


def _characteristic(matrix: list[list[int]], active: tuple[int, ...]) -> list[int]:
    submatrix = [[matrix[row][column] for column in active] for row in active]
    trace = sum(submatrix[index][index] for index in range(len(active))) % FIELD
    if len(active) == 1:
        return [(-trace) % FIELD, 1]
    if len(active) == 2:
        return [_determinant(submatrix), (-trace) % FIELD, 1]
    second = sum(
        submatrix[left][left] * submatrix[right][right]
        - submatrix[left][right] * submatrix[right][left]
        for left, right in itertools.combinations(range(3), 2)
    ) % FIELD
    return [_determinant(submatrix), second, (-trace) % FIELD, 1]


def _canonical(matrix: list[list[int]]) -> list[list[int]]:
    size = len(matrix)
    signs = [1] + [
        1 if 1 <= matrix[0][column] <= FIELD // 2 else -1
        for column in range(1, size)
    ]
    return [
        [signs[row] * matrix[row][column] * signs[column] % FIELD for column in range(size)]
        for row in range(size)
    ]


def _fixture(size: int, seed: int) -> tuple[dict[str, object], list[list[int]], list[list[int]]]:
    rng = random.Random(seed)
    original = [[0] * size for _ in range(size)]
    for row in range(size):
        original[row][row] = rng.randrange(1, FIELD)
        for column in range(row + 1, size):
            value = rng.randrange(1, FIELD)
            original[row][column] = original[column][row] = value
    basis = [[0] * size for _ in range(size)]
    for row in range(size):
        basis[row][row] = 1
        for column in range(row):
            if rng.random() < 0.38:
                basis[row][column] = rng.randrange(1, FIELD)
    observed = _multiply(_multiply(basis, original), _transpose(basis))
    canonical = _canonical(observed)
    inverse = _inverse(basis)
    expected = _multiply(_multiply(inverse, canonical), _transpose(inverse))
    profile_rows = []
    for width in (1, 2, 3):
        for active in itertools.combinations(range(size), width):
            profile_rows.append(
                {
                    "active": list(active),
                    "spectrum": _characteristic(observed, active),
                }
            )
    rng.shuffle(profile_rows)
    payload: dict[str, object] = {
        "schema": "coupling-profile-v1",
        "field": FIELD,
        "probes": [f"P-{index + 1:04d}" for index in range(size)],
        "basis_change": basis,
        "profiles": profile_rows,
    }
    return payload, observed, expected


def _entries(matrix: list[list[int]]) -> list[dict[str, int]]:
    size = len(matrix)
    return [
        {"row": row, "column": column, "value": matrix[row][column] % FIELD}
        for row in range(size)
        for column in range(row, size)
    ]


def _matrix(size: int, entries: list[dict[str, int]]) -> list[list[int]]:
    matrix = [[0] * size for _ in range(size)]
    for entry in entries:
        matrix[entry["row"]][entry["column"]] = entry["value"]
        matrix[entry["column"]][entry["row"]] = entry["value"]
    return matrix


def _profiles(matrix: list[list[int]]) -> dict[tuple[int, ...], list[int]]:
    size = len(matrix)
    return {
        active: _characteristic(matrix, active)
        for width in (1, 2, 3)
        for active in itertools.combinations(range(size), width)
    }


def _digest(entries: list[dict[str, int]]) -> str:
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_inv_01_recovers_canonical_coupling_matrix(tmp_path: Path) -> None:
    """题面：
    「现场需要把原始对称耦合矩阵还原到可以直接复核的规范形式。」
    """

    payload, _, expected = _fixture(18, 8101)
    source = tmp_path / "archive-a.json"
    output = tmp_path / "board-a.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["entries"] == _entries(expected)


def test_inv_02_recovers_signed_triple_orientation(tmp_path: Path) -> None:
    """题面：
    「`spectrum` 是该子集对应 `B` 的主子阵特征多项式系数」
    """

    payload, observed, expected = _fixture(19, 8102)
    source = tmp_path / "archive-b.json"
    output = tmp_path / "board-b.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert sum(observed[0][index] >= 510 for index in range(1, 19)) >= 5
    assert result["entries"] == _entries(expected)


def test_inv_03_normalizes_row_zero_gauge(tmp_path: Path) -> None:
    """题面：
    「现场约定取第一行非对角元都落在 `1` 到 `509` 的那个 `B`。」
    """

    payload, _, expected = _fixture(20, 8103)
    source = tmp_path / "archive-c.json"
    output = tmp_path / "board-c.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    restored = _matrix(20, result["entries"])
    basis = payload["basis_change"]
    observed = _multiply(_multiply(basis, restored), _transpose(basis))
    assert completed.returncode == 0
    assert all(1 <= observed[0][index] <= 509 for index in range(1, 20))
    assert result["entries"] == _entries(expected)


def test_inv_04_reverses_congruence_basis_change(tmp_path: Path) -> None:
    """题面：
    「`A = L⁻¹ · B · (Lᵀ)⁻¹`」
    """

    payload, observed, expected = _fixture(18, 8104)
    source = tmp_path / "archive-d.json"
    output = tmp_path / "board-d.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["entries"] != _entries(observed)
    assert result["entries"] == _entries(expected)


def test_inv_05_profiles_survive_full_round_trip(tmp_path: Path) -> None:
    """题面：
    「若不存在符合全部记录的对称矩阵，整份输入无效」
    """

    payload, _, _ = _fixture(18, 8105)
    source = tmp_path / "archive-e.json"
    output = tmp_path / "board-e.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    restored = _matrix(18, result["entries"])
    basis = payload["basis_change"]
    observed = _multiply(_multiply(basis, restored), _transpose(basis))
    supplied = {
        tuple(profile["active"]): profile["spectrum"] for profile in payload["profiles"]
    }
    assert completed.returncode == 0
    assert _profiles(observed) == supplied


def test_range_01_minimum_probe_count_is_accepted(tmp_path: Path) -> None:
    """题面：
    「`probes` 有 18 到」
    """

    payload, _, expected = _fixture(18, 8106)
    source = tmp_path / "archive-min.json"
    output = tmp_path / "board-min.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["entries"] == _entries(expected)


def test_range_02_maximum_probe_count_finishes_before_deadline(tmp_path: Path) -> None:
    """题面：
    「公开最大一组记录要在 10 秒内算完。」
    """

    payload, _, expected = _fixture(22, 8107)
    source = tmp_path / "archive-max.json"
    output = tmp_path / "board-max.json"
    _write_json(source, payload)
    started = time.monotonic()
    completed = _run(source, output, timeout=20)
    elapsed = time.monotonic() - started
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert len(payload["profiles"]) == 1793
    assert result["entries"] == _entries(expected)
    assert elapsed < 10


def test_range_03_probe_count_just_outside_bounds_is_refused(tmp_path: Path) -> None:
    """题面：
    「22 个探针编号」
    """

    low, _, _ = _fixture(18, 8108)
    high, _, _ = _fixture(22, 8109)
    low["probes"].pop()
    high["probes"].append("P-9999")
    low_source = tmp_path / "archive-low.json"
    high_source = tmp_path / "archive-high.json"
    low_output = tmp_path / "board-low.json"
    high_output = tmp_path / "board-high.json"
    _write_json(low_source, low)
    _write_json(high_source, high)
    low_run = _run(low_source, low_output)
    high_run = _run(high_source, high_output)
    assert low_run.returncode == 2
    assert high_run.returncode == 2
    assert not low_output.exists()
    assert not high_output.exists()


def test_schema_01_top_level_and_field_are_exact(tmp_path: Path) -> None:
    """题面：
    「输入顶层键如下」
    """

    payload, _, _ = _fixture(18, 8110)
    extra = deepcopy(payload)
    extra["comment"] = "shift"
    wrong_schema = deepcopy(payload)
    wrong_schema["schema"] = "coupling-profile-v2"
    wrong_field = deepcopy(payload)
    wrong_field["field"] = 1013
    source = tmp_path / "archive-f.json"
    output = tmp_path / "board-f.json"
    _write_json(source, extra)
    extra_run = _run(source, output)
    _write_json(source, wrong_schema)
    schema_run = _run(source, output)
    _write_json(source, wrong_field)
    field_run = _run(source, output)
    assert extra_run.returncode == 2
    assert schema_run.returncode == 2
    assert field_run.returncode == 2
    assert field_run.stderr.startswith("INPUT_ERROR: ")
    assert not output.exists()


def test_schema_02_duplicate_top_level_key_is_refused(tmp_path: Path) -> None:
    """题面：
    「任何对象出现重复的 JSON 键都按结构错误处理。」
    """

    payload, _, _ = _fixture(18, 8111)
    text = json.dumps(payload)
    text = text[:-1] + ',"field":1019}'
    source = tmp_path / "archive-g.json"
    output = tmp_path / "board-g.json"
    source.write_text(text, encoding="utf-8")
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("INPUT_ERROR: ")
    assert not output.exists()


def test_probe_01_identifier_count_and_uniqueness_are_checked(tmp_path: Path) -> None:
    """题面：
    「编号以 `P-` 开头，再接四位大写字母或数字，数组内不得重复。」
    """

    payload, _, _ = _fixture(18, 8112)
    duplicate = deepcopy(payload)
    duplicate["probes"][1] = duplicate["probes"][0]
    malformed = deepcopy(payload)
    malformed["probes"][0] = "P-12"
    source = tmp_path / "archive-h.json"
    output = tmp_path / "board-h.json"
    _write_json(source, duplicate)
    duplicate_run = _run(source, output)
    _write_json(source, malformed)
    malformed_run = _run(source, output)
    assert duplicate_run.returncode == 2
    assert malformed_run.returncode == 2
    assert not output.exists()


def test_matrix_01_basis_shape_and_field_values_are_checked(tmp_path: Path) -> None:
    """题面：
    「`basis_change` 是 `n × n` 的方阵」
    """

    payload, _, _ = _fixture(18, 8113)
    short_row = deepcopy(payload)
    short_row["basis_change"][0].pop()
    outside = deepcopy(payload)
    outside["basis_change"][0][1] = FIELD
    source = tmp_path / "archive-i.json"
    output = tmp_path / "board-i.json"
    _write_json(source, short_row)
    shape_run = _run(source, output)
    _write_json(source, outside)
    range_run = _run(source, output)
    assert shape_run.returncode == 2
    assert range_run.returncode == 2
    assert not output.exists()


def test_matrix_02_singular_basis_change_is_refused(tmp_path: Path) -> None:
    """题面：
    「方阵在模 1019 下必须可逆。」
    """

    payload, _, _ = _fixture(18, 8114)
    payload["basis_change"][1] = list(payload["basis_change"][0])
    source = tmp_path / "archive-j.json"
    output = tmp_path / "board-j.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("INPUT_ERROR: ")
    assert not output.exists()


def test_profile_01_complete_subset_set_is_required(tmp_path: Path) -> None:
    """题面：
    「大小 1、2、3 的每个子集在 `profiles` 中对应一条记录」
    """

    payload, _, _ = _fixture(18, 8115)
    payload["profiles"].pop()
    source = tmp_path / "archive-k.json"
    output = tmp_path / "board-k.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("INPUT_ERROR: ")
    assert not output.exists()


def test_profile_02_active_order_and_bounds_are_checked(tmp_path: Path) -> None:
    """题面：
    「`active` 给出 1 到 3 个严格递增的」
    """

    payload, _, _ = _fixture(18, 8116)
    unsorted = deepcopy(payload)
    pair = next(profile for profile in unsorted["profiles"] if len(profile["active"]) == 2)
    pair["active"].reverse()
    outside = deepcopy(payload)
    next(profile for profile in outside["profiles"] if profile["active"])["active"][0] = 18
    source = tmp_path / "archive-l.json"
    output = tmp_path / "board-l.json"
    _write_json(source, unsorted)
    order_run = _run(source, output)
    _write_json(source, outside)
    bounds_run = _run(source, output)
    assert order_run.returncode == 2
    assert bounds_run.returncode == 2
    assert not output.exists()


def test_profile_03_spectrum_width_and_monicity_are_checked(tmp_path: Path) -> None:
    """题面：
    「最后一项等于 `1`。」
    """

    payload, _, _ = _fixture(18, 8117)
    short = deepcopy(payload)
    next(profile for profile in short["profiles"] if len(profile["active"]) == 3)[
        "spectrum"
    ].pop()
    non_monic = deepcopy(payload)
    next(profile for profile in non_monic["profiles"] if profile["active"])[
        "spectrum"
    ][-1] = 2
    source = tmp_path / "archive-m.json"
    output = tmp_path / "board-m.json"
    _write_json(source, short)
    width_run = _run(source, output)
    _write_json(source, non_monic)
    monic_run = _run(source, output)
    assert width_run.returncode == 2
    assert monic_run.returncode == 2
    assert not output.exists()


def test_profile_04_extra_nested_key_is_refused(tmp_path: Path) -> None:
    """题面：
    「每项恰有 `active` 和 `spectrum` 两个键。」
    """

    payload, _, _ = _fixture(18, 8118)
    payload["profiles"][0]["note"] = "shift"
    source = tmp_path / "archive-n.json"
    output = tmp_path / "board-n.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("INPUT_ERROR: ")
    assert not output.exists()


def test_type_01_boolean_values_are_refused(tmp_path: Path) -> None:
    """题面：
    「真假值不算整数」
    """

    payload, _, _ = _fixture(18, 8119)
    payload["field"] = True
    payload["basis_change"][0][0] = False
    payload["profiles"][0]["spectrum"][0] = True
    source = tmp_path / "archive-o.json"
    output = tmp_path / "board-o.json"
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("INPUT_ERROR: ")
    assert not output.exists()


def test_consist_01_inconsistent_triple_profile_is_refused_without_overwrite(
    tmp_path: Path,
) -> None:
    """题面：
    「已有文件保持原样」
    """

    payload, _, _ = _fixture(18, 8120)
    triple = next(profile for profile in payload["profiles"] if len(profile["active"]) == 3)
    triple["spectrum"][0] = (triple["spectrum"][0] + 1) % FIELD
    source = tmp_path / "archive-p.json"
    output = tmp_path / "board-p.json"
    old_bytes = b'{"kept":true}\n'
    output.write_bytes(old_bytes)
    _write_json(source, payload)
    completed = _run(source, output)
    assert completed.returncode == 2
    assert completed.stderr.startswith("INPUT_ERROR: ")
    assert output.read_bytes() == old_bytes


def test_fail_01_argument_and_parse_errors_use_public_channel(tmp_path: Path) -> None:
    """题面：
    「程序把单行说明写到 stderr，正文前冠 `INPUT_ERROR: `」
    """

    no_paths = _run_without_paths()
    source = tmp_path / "archive-q.json"
    output = tmp_path / "board-q.json"
    source.write_text("{broken", encoding="utf-8")
    parse_run = _run(source, output)
    assert no_paths.returncode == 2
    assert no_paths.stderr.startswith("INPUT_ERROR: ")
    assert parse_run.returncode == 2
    assert parse_run.stderr.startswith("INPUT_ERROR: ")
    assert not output.exists()


def test_out_01_success_shape_and_ordering_are_exact(tmp_path: Path) -> None:
    """题面：
    「成功结果采用下面的对象结构」
    """

    payload, _, expected = _fixture(18, 8121)
    source = tmp_path / "archive-r.json"
    output = tmp_path / "board-r.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    positions = [(entry["row"], entry["column"]) for entry in result["entries"]]
    assert completed.returncode == 0
    assert set(result) == {"status", "probes", "entries", "matrix_sha256"}
    assert result["status"] == "ok"
    assert result["probes"] == payload["probes"]
    assert len(result["entries"]) == 171
    assert positions == sorted(positions)
    assert result["entries"] == _entries(expected)


def test_hash_01_matrix_digest_matches_canonical_entries(tmp_path: Path) -> None:
    """题面：
    「`matrix_sha256` 是 `entries` 按紧凑、键排序 JSON 编码所得字节的 SHA-256」
    """

    payload, _, _ = _fixture(18, 8122)
    source = tmp_path / "archive-s.json"
    output = tmp_path / "board-s.json"
    _write_json(source, payload)
    completed = _run(source, output)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 0
    assert result["matrix_sha256"] == _digest(result["entries"])
    assert len(result["matrix_sha256"]) == 64
    assert result["matrix_sha256"] == result["matrix_sha256"].lower()


def test_serial_01_success_is_compact_silent_and_stable(tmp_path: Path) -> None:
    """题面：
    「成功时终端不输出字符」
    """

    payload, _, _ = _fixture(18, 8123)
    source = tmp_path / "archive-t.json"
    first = tmp_path / "board-t.json"
    second = tmp_path / "board-u.json"
    _write_json(source, payload)
    first_run = _run(source, first)
    second_run = _run(source, second)
    raw = first.read_bytes()
    parsed = json.loads(raw)
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert first_run.returncode == 0
    assert first_run.stdout == ""
    assert first_run.stderr == ""
    assert second_run.returncode == 0
    assert second.read_bytes() == raw
    assert raw == canonical
