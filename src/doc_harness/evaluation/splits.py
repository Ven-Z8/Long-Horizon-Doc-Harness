"""Deterministic document-level development and holdout partitions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from ..workflow.batch import load_v2_samples
from ..core.protocol import normalize_question


def _key(item: dict[str, object]) -> tuple[str, str]:
    return str(item["doc_id"]), normalize_question(str(item["question"]))


def create_document_split(
    samples_path: Path,
    exposed_keys: Iterable[tuple[str, str]],
    seed: str,
) -> dict[str, object]:
    """Create a stable document split while keeping exposed documents in development."""

    samples = load_v2_samples(samples_path)
    sample_keys = {
        (sample.doc_id, normalize_question(sample.question)) for sample in samples
    }
    exposed = {
        (str(doc_id), normalize_question(question)) for doc_id, question in exposed_keys
    }
    unknown = exposed - sample_keys
    if unknown:
        raise ValueError(f"exposed key not in samples: {sorted(unknown)[0]}")
    exposed_docs = {doc_id for doc_id, _ in exposed}
    all_docs = sorted({sample.doc_id for sample in samples})
    remaining = [doc_id for doc_id in all_docs if doc_id not in exposed_docs]
    ordered = sorted(
        remaining,
        key=lambda doc_id: hashlib.sha256(f"{seed}\0{doc_id}".encode()).hexdigest(),
    )
    development_documents = set(exposed_docs)
    holdout_documents: set[str] = set()
    for index, doc_id in enumerate(ordered):
        (development_documents if index % 2 == 0 else holdout_documents).add(doc_id)
    dev_keys = [
        {"doc_id": sample.doc_id, "question": normalize_question(sample.question)}
        for sample in samples
        if sample.doc_id in development_documents
    ]
    holdout_keys = [
        {"doc_id": sample.doc_id, "question": normalize_question(sample.question)}
        for sample in samples
        if sample.doc_id in holdout_documents
    ]
    return {
        "schema_version": 1,
        "seed": seed,
        "samples_path": str(Path(samples_path)),
        "development_documents": sorted(development_documents),
        "holdout_documents": sorted(holdout_documents),
        "development_keys": dev_keys,
        "holdout_keys": holdout_keys,
        "counts": {
            "documents": len(all_docs),
            "samples": len(samples),
            "development_documents": len(development_documents),
            "holdout_documents": len(holdout_documents),
            "development_samples": len(dev_keys),
            "holdout_samples": len(holdout_keys),
        },
    }


def prepare_phase2_selection(
    samples_path: Path, baseline_run: Path, output_dir: Path, *, limit: int = 100
) -> dict[str, Path]:
    """Freeze the baseline question order into development and smoke files."""

    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("samples JSON must be a list")
    by_key = {(str(row["doc_id"]), normalize_question(str(row["question"]))): row for row in samples}
    baseline_dir = Path(baseline_run)
    selected_path = baseline_dir / "selected-samples.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
    else:
        selected = json.loads((baseline_dir / "predictions.json").read_text(encoding="utf-8"))
    if not isinstance(selected, list) or len(selected) != limit:
        raise ValueError(f"baseline must contain exactly {limit} selected questions")
    keys = [(str(item.get("doc_id")), normalize_question(str(item.get("question", "")))) for item in selected]
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(f"unknown baseline question: {missing[0]}")
    if len(set(keys)) != len(keys):
        raise ValueError("baseline selected questions contain duplicates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    dev_rows = [by_key[key] for key in keys]
    smoke_rows = dev_rows[: min(20, len(dev_rows))]
    selection_rows = [{"doc_id": key[0], "question": key[1]} for key in keys]
    paths = {
        "dev": output_dir / "dev100.json",
        "smoke": output_dir / "smoke20.json",
        "selection": output_dir / "dev100-selection.json",
    }
    for path, payload in ((paths["dev"], dev_rows), (paths["smoke"], smoke_rows), (paths["selection"], selection_rows)):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths
