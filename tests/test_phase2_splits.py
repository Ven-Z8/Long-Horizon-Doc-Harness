import json

import pytest

from doc_harness.evaluation.splits import prepare_phase2_selection


def test_prepare_phase2_selection_preserves_baseline_order(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps([
            {"doc_id": "d.pdf", "question": "Q1", "answer": "A1"},
            {"doc_id": "d.pdf", "question": "Q2", "answer": "A2"},
        ])
    )
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "selected-samples.json").write_text(
        json.dumps([
            {"doc_id": "d.pdf", "question": "Q2"},
            {"doc_id": "d.pdf", "question": "Q1"},
        ])
    )
    with pytest.raises(ValueError, match="exactly 100"):
        prepare_phase2_selection(samples, baseline, tmp_path / "out")


def test_prepare_phase2_selection_rejects_unknown_baseline_key(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"doc_id": "d.pdf", "question": "Q", "answer": "A"}]))
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "selected-samples.json").write_text(
        json.dumps([{"doc_id": "missing.pdf", "question": "Q"}] * 100)
    )
    with pytest.raises(ValueError, match="unknown baseline"):
        prepare_phase2_selection(samples, baseline, tmp_path / "out")
