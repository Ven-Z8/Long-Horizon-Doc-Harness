import json
from pathlib import Path

import pytest

from doc_harness.splits import create_document_split


def _samples(path: Path):
    path.write_text(
        json.dumps(
            [
                {"doc_id": "a.pdf", "question": "Q1", "answer": "A"},
                {"doc_id": "a.pdf", "question": "Q2", "answer": "A"},
                {"doc_id": "b.pdf", "question": "Q3", "answer": "A"},
                {"doc_id": "c.pdf", "question": "Q4", "answer": "A"},
                {"doc_id": "d.pdf", "question": "Q5", "answer": "A"},
            ]
        )
    )


def test_exposed_documents_and_all_questions_stay_in_development(tmp_path: Path):
    samples = tmp_path / "samples.json"
    _samples(samples)
    split = create_document_split(samples, {("a.pdf", "Q1")}, seed="seed")
    assert "a.pdf" in split["development_documents"]
    assert "a.pdf" not in split["holdout_documents"]
    dev_keys = {(x["doc_id"], x["question"]) for x in split["development_keys"]}
    assert ("a.pdf", "Q2") in dev_keys
    assert not set(split["development_documents"]) & set(split["holdout_documents"])


def test_split_is_stable_and_unknown_exposed_key_is_rejected(tmp_path: Path):
    samples = tmp_path / "samples.json"
    _samples(samples)
    first = create_document_split(samples, set(), seed="seed")
    second = create_document_split(samples, set(), seed="seed")
    assert first == second
    with pytest.raises(ValueError, match="not in samples"):
        create_document_split(samples, {("missing.pdf", "Q")}, seed="seed")
