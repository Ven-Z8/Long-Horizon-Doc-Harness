"""Crash-safe per-question checkpoints and prediction projections."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..core.answers import QuestionOutcome, export_outcome
from ..core.protocol import normalize_question


def sample_key(document_id: str, question: str) -> str:
    payload = json.dumps(
        {"document_id": str(document_id), "question": normalize_question(str(question))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def commit_question(
    run_dir: Path, key: str, payload: dict[str, Any], *, allow_replace: bool = False
) -> None:
    """Atomically commit one question state without overwriting its identity."""

    document_id = payload.get("document_id")
    question = payload.get("question")
    if not isinstance(document_id, str) or not isinstance(question, str):
        raise ValueError("checkpoint requires document_id and question")
    if sample_key(document_id, question) != key:
        raise ValueError("checkpoint key does not match document/question")
    outcome = payload.get("outcome")
    if not isinstance(outcome, dict):
        raise ValueError("checkpoint requires an outcome object")
    QuestionOutcome.model_validate(outcome)
    destination = Path(run_dir) / "questions" / f"{key}.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            if not allow_replace:
                raise FileExistsError(f"checkpoint already exists with different content: {key}")
            history = Path(run_dir) / "questions" / "history"
            history.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                history / f"{key}-{hashlib.sha256(json.dumps(existing, sort_keys=True).encode()).hexdigest()[:12]}.json",
                json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
        else:
            return
    _atomic_write(destination, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_committed(run_dir: Path) -> dict[str, dict[str, Any]]:
    directory = Path(run_dir) / "questions"
    if not directory.is_dir():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = path.stem
        document_id = payload.get("document_id")
        question = payload.get("question")
        if sample_key(str(document_id), str(question)) != key:
            raise ValueError(f"checkpoint filename does not match its question: {path}")
        QuestionOutcome.model_validate(payload.get("outcome"))
        result[key] = payload
    return result


def project_predictions(run_dir: Path) -> Path:
    rows: list[dict[str, str]] = []
    for payload in load_committed(run_dir).values():
        outcome = QuestionOutcome.model_validate(payload["outcome"])
        response = export_outcome(outcome)
        if response is None:
            continue
        rows.append(
            {
                "doc_id": payload["document_id"],
                "question": normalize_question(payload["question"]),
                "response": response,
            }
        )
    rows.sort(key=lambda row: (row["doc_id"], row["question"]))
    destination = Path(run_dir) / "predictions.json"
    _atomic_write(destination, json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    return destination
