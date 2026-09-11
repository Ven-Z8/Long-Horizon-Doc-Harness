"""Benchmark-safe request construction and prediction validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from .contracts import BenchmarkSample, DraftAnswer, EvidenceSpan, ModelRequest, Page, Prediction


def normalize_question(question: str) -> str:
    """Normalize benchmark keys without changing the question's meaning."""

    return " ".join(str(question).split())


def build_request(
    sample: BenchmarkSample,
    pages: Sequence[Page],
    prompt: str,
    config_hash: str,
) -> ModelRequest:
    """Build a model-safe request from a sample and rendered pages."""

    if not pages:
        raise ValueError("at least one page is required")
    wrong_document = [page.document_id for page in pages if page.document_id != sample.doc_id]
    if wrong_document:
        raise ValueError(
            f"pages belong to a different document: {wrong_document[0]}"
        )
    return ModelRequest(
        document_id=sample.doc_id,
        question=normalize_question(sample.question),
        page_ids=[page.page_id for page in pages],
        image_paths=[page.image_path for page in pages],
        prompt=prompt,
        config_hash=config_hash,
    )


def _key(doc_id: str, question: str) -> tuple[str, str]:
    return doc_id, normalize_question(question)


def validate_prediction_coverage(
    samples: Sequence[BenchmarkSample], predictions: Sequence[Prediction]
) -> None:
    """Require a one-to-one prediction mapping for every expected sample."""

    expected = [_key(sample.doc_id, sample.question) for sample in samples]
    actual = [_key(prediction.doc_id, prediction.question) for prediction in predictions]
    if len(set(expected)) != len(expected):
        raise ValueError("duplicate benchmark sample keys")
    duplicate_keys = sorted({key for key in actual if actual.count(key) > 1})
    if duplicate_keys:
        raise ValueError(f"duplicate prediction key: {duplicate_keys[0]}")
    unknown = sorted(set(actual) - set(expected))
    if unknown:
        raise ValueError(f"unknown prediction key: {unknown[0]}")
    missing = sorted(set(expected) - set(actual))
    if missing:
        raise ValueError(f"missing prediction key: {missing[0]}")


def export_v2_predictions(predictions: Sequence[Prediction], path: Path) -> None:
    """Write the full response text in MMLongBench-Doc V2 format."""

    payload = [
        {"doc_id": prediction.doc_id, "question": prediction.question, "response": prediction.response}
        for prediction in predictions
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def validate_evidence_quotes(
    draft: DraftAnswer, page_text: Mapping[int, str]
) -> list[EvidenceSpan]:
    """Return evidence spans whose normalized quote occurs on the cited page."""

    valid: list[EvidenceSpan] = []
    for span in draft.evidence:
        source = page_text.get(span.page_id)
        if source is None:
            continue
        if normalize_question(span.quote) in normalize_question(source):
            valid.append(span)
    return valid
