import json
from pathlib import Path

import pytest

from doc_harness.experiments import compare_runs, write_report


def _run(path: Path, rows: list[dict]) -> None:
    path.mkdir()
    (path / "predictions.json").write_text(json.dumps(rows))


def test_compare_runs_requires_paired_keys(tmp_path: Path):
    _run(
        tmp_path / "a",
        [
            {"doc_id": "d", "question": "Q1", "answer": "A", "llm_judge": {"equivalent": True, "abstained": False}},
            {"doc_id": "d", "question": "Q2", "answer": "A", "llm_judge": {"equivalent": False, "abstained": False}},
        ],
    )
    _run(
        tmp_path / "b",
        [{"doc_id": "d", "question": "Q1", "answer": "A", "llm_judge": {"equivalent": True, "abstained": False}}],
    )
    with pytest.raises(ValueError, match="same sample keys"):
        compare_runs([tmp_path / "a", tmp_path / "b"])


def test_compare_runs_reports_paired_improvements_and_failures(tmp_path: Path):
    rows_a = [
        {"doc_id": "d", "question": "Q1", "answer": "A", "llm_judge": {"equivalent": False, "abstained": False}},
        {"doc_id": "d", "question": "Q2", "answer": "A", "llm_judge": {"equivalent": True, "abstained": False}},
    ]
    rows_b = [
        {"doc_id": "d", "question": "Q1", "answer": "A", "llm_judge": {"equivalent": True, "abstained": False}},
        {"doc_id": "d", "question": "Q2", "answer": "A", "llm_judge": {"equivalent": False, "abstained": False}},
    ]
    _run(tmp_path / "a", rows_a)
    _run(tmp_path / "b", rows_b)
    comparison = compare_runs([tmp_path / "a", tmp_path / "b"])
    assert comparison["variants"][0]["accuracy"] == 0.5
    assert comparison["paired"]["a_to_b"]["improved"] == 1
    assert comparison["paired"]["a_to_b"]["regressed"] == 1
    output = tmp_path / "report.json"
    write_report(comparison, output)
    assert json.loads(output.read_text())["variants"][1]["accuracy"] == 0.5
