import json
from pathlib import Path

from doc_harness.cli import _parser, main, run_smoke


def test_smoke_writes_record_and_prediction(tmp_path: Path):
    output = tmp_path / "run.jsonl"
    prediction = tmp_path / "prediction.json"
    run_smoke(
        question="What is shown?",
        document_id="doc.pdf",
        image_paths=[str(tmp_path / "page.png")],
        output_path=output,
        prediction_path=prediction,
    )
    assert output.exists()
    assert json.loads(prediction.read_text())[0]["response"] == "stub answer"


def test_audit_run_cli_writes_report(tmp_path: Path):
    records = tmp_path / "records.jsonl"
    prediction = tmp_path / "predictions.json"
    output = tmp_path / "audit.json"
    run_smoke(
        question="Q",
        document_id="doc.pdf",
        image_paths=[str(tmp_path / "page.png")],
        output_path=records,
        prediction_path=prediction,
    )
    assert main(
        [
            "audit-run",
            "--records",
            str(records),
            "--predictions",
            str(prediction),
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text())["records"]["total"] == 1


def test_create_split_cli_uses_exposed_record_documents(tmp_path: Path):
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            [
                {"doc_id": "doc.pdf", "question": "Q", "answer": "A"},
                {"doc_id": "other.pdf", "question": "Q2", "answer": "A"},
            ]
        )
    )
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "document_id": "doc.pdf",
                "question": "Q",
                "status": "ok",
                "config_hash": "h",
            }
        )
        + "\n"
    )
    output = tmp_path / "split.json"
    assert main(
        [
            "create-split",
            "--samples",
            str(samples),
            "--exposed-records",
            str(records),
            "--output",
            str(output),
            "--seed",
            "test",
        ]
    ) == 0
    split = json.loads(output.read_text())
    assert "doc.pdf" in split["development_documents"]


def test_compare_runs_cli_writes_report(tmp_path: Path):
    rows = [
        {
            "doc_id": "doc.pdf",
            "question": "Q",
            "answer": "A",
            "response": "A",
            "llm_judge": {"equivalent": True, "abstained": False, "reason": "same"},
        }
    ]
    run_a = tmp_path / "a"
    run_b = tmp_path / "b"
    run_a.mkdir()
    run_b.mkdir()
    (run_a / "predictions.json").write_text(json.dumps(rows))
    (run_b / "predictions.json").write_text(json.dumps(rows))
    output = tmp_path / "comparison.json"
    assert main(["compare-runs", "--runs", str(run_a), str(run_b), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["sample_count"] == 1


def test_build_graph_is_registered_with_required_graph_inputs():
    """Catches accidental removal of the offline graph-artifact CLI."""

    parser = _parser()
    help_text = parser.format_help()
    args = parser.parse_args(
        [
            "build-graph",
            "--config",
            "graph.toml",
            "--documents",
            "documents",
            "--output",
            "graphs",
            "--models-dir",
            "models",
            "--document-id",
            "one.pdf",
        ]
    )

    assert "build-graph" in help_text
    assert args.command == "build-graph"
    assert args.document_ids == ["one.pdf"]
