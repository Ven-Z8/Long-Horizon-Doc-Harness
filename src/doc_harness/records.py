"""Validated JSONL persistence for per-question run records."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import RunRecord


def write_run_record(record: RunRecord, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")


def read_run_records(path: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(RunRecord.model_validate(payload))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"invalid run record on line {line_number}") from exc
    return records
