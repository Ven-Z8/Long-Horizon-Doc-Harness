import json
from pathlib import Path

import pytest

from doc_harness.evaluation import (
    JudgeErrorType,
    JudgeEvaluationError,
    classify_judge_error,
    judge_cache_key,
    score_run,
    validate_judge_verdict,
)


def test_judge_errors_are_classified_without_becoming_negative_verdicts():
    assert classify_judge_error(TimeoutError("request timed out")) == JudgeErrorType.timeout
    assert classify_judge_error("401 invalid api key") == JudgeErrorType.authentication
    assert classify_judge_error(json.JSONDecodeError("bad json", "{", 1)) == JudgeErrorType.invalid_json
    assert classify_judge_error("judge failed: retries exhausted") == JudgeErrorType.exhausted_retries

    with pytest.raises(JudgeEvaluationError) as exc_info:
        validate_judge_verdict(
            {"equivalent": False, "abstained": False, "reason": "judge failed: timeout"}
        )
    assert exc_info.value.error_type == JudgeErrorType.timeout


def test_judge_verdict_schema_is_strict():
    assert validate_judge_verdict(
        {"equivalent": True, "abstained": False, "reason": "same answer"}
    ).equivalent is True
    with pytest.raises(JudgeEvaluationError):
        validate_judge_verdict({"equivalent": 1, "abstained": False, "reason": "bad"})
    with pytest.raises(JudgeEvaluationError):
        validate_judge_verdict(
            {
                "equivalent": True,
                "abstained": False,
                "reason": "same",
                "unexpected": "field",
            }
        )


def test_judge_cache_key_includes_all_semantic_inputs():
    values = {
        "question": "What is the total?",
        "reference_answer": "10",
        "response": "The total is 10.",
        "judge_model": "openai/gpt-5.6-luna",
        "prompt": "judge-v1",
    }
    original = judge_cache_key(**values)
    assert len(original) == 64
    for field in values:
        changed = dict(values)
        changed[field] = str(changed[field]) + " changed"
        assert judge_cache_key(**changed) != original


def test_score_run_blocks_comparable_score_when_judge_errors_remain(tmp_path: Path):
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            [
                {"doc_id": "doc.pdf", "question": "Q1", "answer": "A1"},
                {"doc_id": "doc.pdf", "question": "Q2", "answer": "A2"},
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "doc_id": "doc.pdf",
                    "question": "Q1",
                    "response": "A1",
                    "llm_judge": {
                        "equivalent": True,
                        "abstained": False,
                        "reason": "same",
                    },
                },
                {
                    "doc_id": "doc.pdf",
                    "question": "Q2",
                    "response": "A2",
                    "llm_judge": {
                        "equivalent": False,
                        "abstained": False,
                        "reason": "judge failed: timeout",
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    result = score_run(run_dir, samples)

    assert result["status"] == "incomplete"
    assert result["metrics"] is None
    assert result["judge_errors"][0]["error_type"] == "timeout"
    assert result["counts"]["expected"] == 2


def test_score_run_exports_complete_metrics_for_valid_verdicts(tmp_path: Path):
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            [
                {"doc_id": "doc.pdf", "question": "Q1", "answer": "A1"},
                {"doc_id": "doc.pdf", "question": "Q2", "answer": "Not answerable"},
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "doc_id": "doc.pdf",
                    "question": "Q1",
                    "response": "A1",
                    "llm_judge": {"equivalent": True, "abstained": False, "reason": "same"},
                },
                {
                    "doc_id": "doc.pdf",
                    "question": "Q2",
                    "response": "No answer",
                    "llm_judge": {"equivalent": True, "abstained": True, "reason": "unanswerable"},
                },
            ]
        ),
        encoding="utf-8",
    )
    result = score_run(run_dir, samples)
    assert result["status"] == "complete"
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["f1"] == 1.0
