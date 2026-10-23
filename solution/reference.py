#!/usr/bin/env python3
"""Reference inverse for canonical finite-field coupling profiles."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


FIELD = 1019
SCHEMA = "coupling-profile-v1"
INPUT_KEYS = {"schema", "field", "basis_change", "probes", "profiles"}
PROFILE_KEYS = {"active", "spectrum"}
PROBE_ID = re.compile(r"P-[A-Z0-9]{4}\Z")


class InputProblem(ValueError):
    pass


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise InputProblem("duplicate JSON key")
        value[key] = item
    return value


def _integer(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputProblem("expected integer")
    return value


def _field_integer(value: Any) -> int:
    item = _integer(value)
    if item < 0 or item >= FIELD:
        raise InputProblem("integer is outside the field")
    return item


def _matrix(value: Any, size: int) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != size:
        raise InputProblem("basis_change has the wrong height")
    result: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != size:
            raise InputProblem("basis_change has the wrong width")
        result.append([_field_integer(item) for item in row])
    return result


def parse_input(path: Path) -> tuple[list[str], list[list[int]], dict[tuple[int, ...], list[int]]]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object,
        )
    except OSError as exc:
        raise InputProblem("input cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise InputProblem("input is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != INPUT_KEYS:
        raise InputProblem("top-level key set is invalid")
    if payload.get("schema") != SCHEMA:
        raise InputProblem("unsupported schema")
    if _integer(payload.get("field")) != FIELD:
        raise InputProblem("field is outside the public contract")

    raw_probes = payload.get("probes")
    if not isinstance(raw_probes, list) or not 18 <= len(raw_probes) <= 22:
        raise InputProblem("probe count is outside the public range")
    probes: list[str] = []
    seen_probes: set[str] = set()
    for item in raw_probes:
        if not isinstance(item, str) or PROBE_ID.fullmatch(item) is None:
            raise InputProblem("probe identifier is invalid")
        if item in seen_probes:
            raise InputProblem("probe identifier is duplicated")
        seen_probes.add(item)
        probes.append(item)

    size = len(probes)
    basis_change = _matrix(payload.get("basis_change"), size)
    _inverse(basis_change)

    raw_profiles = payload.get("profiles")
    expected_count = size + size * (size - 1) // 2
    expected_count += size * (size - 1) * (size - 2) // 6
    if not isinstance(raw_profiles, list) or len(raw_profiles) != expected_count:
        raise InputProblem("profile set is incomplete")

    profiles: dict[tuple[int, ...], list[int]] = {}
    for item in raw_profiles:
        if not isinstance(item, dict) or set(item) != PROFILE_KEYS:
            raise InputProblem("profile key set is invalid")
        raw_active = item.get("active")
        if not isinstance(raw_active, list) or not 1 <= len(raw_active) <= 3:
            raise InputProblem("profile active set is invalid")
        active: list[int] = []
        for value in raw_active:
            index = _integer(value)
            if index < 0 or index >= size:
                raise InputProblem("profile index is outside the probe array")
            active.append(index)
        if active != sorted(set(active)):
            raise InputProblem("profile active set is not strictly increasing")
        key = tuple(active)
        if key in profiles:
            raise InputProblem("profile active set is duplicated")
        raw_spectrum = item.get("spectrum")
        if not isinstance(raw_spectrum, list) or len(raw_spectrum) != len(active) + 1:
            raise InputProblem("profile spectrum has the wrong width")
        spectrum = [_field_integer(value) for value in raw_spectrum]
        if spectrum[-1] != 1:
            raise InputProblem("profile spectrum is not monic")
        profiles[key] = spectrum

    expected_sets = {
        subset
        for width in (1, 2, 3)
        for subset in itertools.combinations(range(size), width)
    }
    if set(profiles) != expected_sets:
        raise InputProblem("profile subsets do not match the public range")
    return probes, basis_change, profiles


def _determinant(matrix: list[list[int]]) -> int:
    size = len(matrix)
    if size == 1:
        return matrix[0][0] % FIELD
    if size == 2:
        return (
            matrix[0][0] * matrix[1][1]
            - matrix[0][1] * matrix[1][0]
        ) % FIELD
    return (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    ) % FIELD


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
            (
                index
                for index in range(pivot_row, size)
                if augmented[index][column] % FIELD
            ),
            None,
        )
        if pivot is None:
            raise InputProblem("basis_change is singular")
        augmented[pivot_row], augmented[pivot] = augmented[pivot], augmented[pivot_row]
        inverse = pow(augmented[pivot_row][column], FIELD - 2, FIELD)
        augmented[pivot_row] = [
            value * inverse % FIELD for value in augmented[pivot_row]
        ]
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


def _characteristic(
    matrix: list[list[int]],
    active: tuple[int, ...],
) -> list[int]:
    submatrix = [[matrix[row][column] for column in active] for row in active]
    width = len(active)
    trace = sum(submatrix[index][index] for index in range(width)) % FIELD
    if width == 1:
        return [(-trace) % FIELD, 1]
    if width == 2:
        return [_determinant(submatrix), (-trace) % FIELD, 1]
    second = 0
    for left, right in itertools.combinations(range(width), 2):
        second += (
            submatrix[left][left] * submatrix[right][right]
            - submatrix[left][right] * submatrix[right][left]
        )
    return [_determinant(submatrix), second % FIELD, (-trace) % FIELD, 1]


def _square_root(value: int) -> int:
    if value == 0:
        return 0
    root = pow(value, (FIELD + 1) // 4, FIELD)
    if root * root % FIELD != value:
        raise InputProblem("pair profile has no square root")
    return min(root, FIELD - root)


def _inverse_product(
    basis: list[list[int]],
    observed: list[list[int]],
) -> list[list[int]]:
    inverse = _inverse(basis)
    return _multiply(_multiply(inverse, observed), _transpose(inverse))


def recover(
    probes: list[str],
    basis_change: list[list[int]],
    profiles: dict[tuple[int, ...], list[int]],
) -> dict[str, Any]:
    size = len(probes)
    diagonal = [0] * size
    for index in range(size):
        diagonal[index] = (-profiles[(index,)][0]) % FIELD

    roots = [[0] * size for _ in range(size)]
    for left in range(size):
        roots[left][left] = diagonal[left]
        for right in range(left + 1, size):
            spectrum = profiles[(left, right)]
            if (-spectrum[1]) % FIELD != (diagonal[left] + diagonal[right]) % FIELD:
                raise InputProblem("pair trace is inconsistent")
            square = (diagonal[left] * diagonal[right] - spectrum[0]) % FIELD
            roots[left][right] = roots[right][left] = _square_root(square)

    observed = [[0] * size for _ in range(size)]
    for index in range(size):
        observed[index][index] = diagonal[index]
    for column in range(1, size):
        observed[0][column] = observed[column][0] = roots[0][column]
        if observed[0][column] == 0:
            raise InputProblem("observed matrix has a zero first-row coupling")

    inverse_two = pow(2, FIELD - 2, FIELD)
    for left in range(1, size):
        for right in range(left + 1, size):
            if roots[left][right] == 0:
                observed[left][right] = observed[right][left] = 0
                continue
            determinant = profiles[(0, left, right)][0]
            diagonal_term = diagonal[0] * diagonal[left] * diagonal[right]
            edge_term = (
                diagonal[0] * roots[left][right] * roots[left][right]
                + diagonal[left] * roots[0][right] * roots[0][right]
                + diagonal[right] * roots[0][left] * roots[0][left]
            )
            product = (determinant - diagonal_term + edge_term) * inverse_two % FIELD
            divisor = (
                observed[0][left] * observed[0][right] * roots[left][right]
            ) % FIELD
            factor = product * pow(divisor, FIELD - 2, FIELD) % FIELD
            if factor == FIELD - 1:
                observed[left][right] = observed[right][left] = (
                    -roots[left][right]
                ) % FIELD
            elif factor == 1:
                observed[left][right] = observed[right][left] = roots[left][right]
            else:
                raise InputProblem("triple profile has an invalid orientation")

    for active, expected in profiles.items():
        if _characteristic(observed, active) != expected:
            raise InputProblem("profile set is inconsistent")

    original = _inverse_product(basis_change, observed)
    entries = [
        {"row": row, "column": column, "value": original[row][column] % FIELD}
        for row in range(size)
        for column in range(row, size)
    ]
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "status": "ok",
        "probes": probes,
        "entries": entries,
        "matrix_sha256": digest,
    }


def _publish(path: Path, payload: dict[str, Any]) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            )
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def main() -> int:
    if len(sys.argv) != 3:
        print("INPUT_ERROR: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    try:
        probes, basis_change, profiles = parse_input(source)
        result = recover(probes, basis_change, profiles)
        _publish(target, result)
    except (InputProblem, OSError, OverflowError) as exc:
        print(f"INPUT_ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
