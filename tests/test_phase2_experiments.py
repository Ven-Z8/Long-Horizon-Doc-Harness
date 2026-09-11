import json

import pytest

from doc_harness.evaluation.judge_runner import build_judge_rows


def test_judge_rows_require_complete_prediction_coverage(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps([
            {"doc_id": "d.pdf", "question": "Q", "answer": "A", "answer_format": "Str"},
            {"doc_id": "d.pdf", "question": "R", "answer": "B", "answer_format": "Str"},
        ])
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps([{"doc_id": "d.pdf", "question": "Q", "response": "A"}]))
    with pytest.raises(ValueError, match="missing predictions"):
        build_judge_rows(predictions, samples)


def test_judge_rows_do_not_include_gold_fields_in_model_response_payload(tmp_path):
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"doc_id": "d.pdf", "question": "Q", "answer": "A", "answer_format": "Str"}]))
    predictions = tmp_path / "predictions.json"
    predictions.write_text(json.dumps([{"doc_id": "d.pdf", "question": "Q", "response": "A"}]))
    rows = build_judge_rows(predictions, samples)
    assert rows[0]["response"] == "A"
    assert rows[0]["answer"] == "A"
    assert rows[0]["answer_format"] == "Str"
