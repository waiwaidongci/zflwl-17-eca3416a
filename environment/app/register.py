#!/usr/bin/env python3
"""Starter inverse for the coupling-profile archive."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


FIELD = 1019


def main() -> int:
    if len(sys.argv) != 3:
        print("INPUT_ERROR: expected input and output paths", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        probes = payload["probes"]
        size = len(probes)
        matrix = [[0] * size for _ in range(size)]
        for profile in payload["profiles"]:
            active = profile["active"]
            spectrum = profile["spectrum"]
            if len(active) == 1:
                matrix[active[0]][active[0]] = (-spectrum[0]) % FIELD
            elif len(active) == 2:
                left, right = active
                square = (
                    matrix[left][left] * matrix[right][right] - spectrum[0]
                ) % FIELD
                root = pow(square, (FIELD + 1) // 4, FIELD)
                matrix[left][right] = matrix[right][left] = min(root, FIELD - root)
        entries = [
            {"row": row, "column": column, "value": matrix[row][column]}
            for row in range(size)
            for column in range(row, size)
        ]
        digest = hashlib.sha256(
            json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        result = {
            "status": "ok",
            "probes": probes,
            "entries": entries,
            "matrix_sha256": digest,
        }
    except (KeyError, TypeError, ValueError, OSError):
        print("INPUT_ERROR: archive cannot be decoded", file=sys.stderr)
        return 2
    output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
