import hashlib
import json
from pathlib import Path

import pytest

from doc_harness.contracts import RunRecord
from doc_harness.manifests import (
    RunManifest,
    assert_resume_compatible,
    audit_records,
    audit_verdicts,
    freeze_run,
)
from doc_harness.records import write_run_record


def _record(*, run_id: str, question: str, response: str) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        document_id="doc.pdf",
        question=question,
        status="ok",
        response=response,
        config_hash="config",
    )


def test_freeze_preserves_bytes_and_refuses_overwrite(tmp_path: Path):
    first = tmp_path / "first.bin"
    second = tmp_path / "second.txt"
    first.write_bytes(b"\x00\x01 immutable bytes")
    second.write_text("a unicode value: λ\n", encoding="utf-8")
    destination = tmp_path / "frozen" / "run-001"

    inventory = freeze_run([first, second], destination)

    assert (destination / first.name).read_bytes() == first.read_bytes()
    assert (destination / second.name).read_bytes() == second.read_bytes()
    assert inventory[first.name] == hashlib.sha256(first.read_bytes()).hexdigest()
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"][first.name]["size_bytes"] == len(first.read_bytes())
    assert manifest["files"][first.name]["sha256"] == inventory[first.name]
    assert manifest["identity"]["code_hash"] == "unknown"

    with pytest.raises(FileExistsError):
        freeze_run([first, second], destination)


def test_audit_detects_duplicate_missing_and_conflicting_rows(tmp_path: Path):
    records_path = tmp_path / "runs.jsonl"
    write_run_record(_record(run_id="r1", question="  What?  ", response="record"), records_path)
    write_run_record(_record(run_id="r2", question="What?", response="duplicate"), records_path)
    write_run_record(_record(run_id="r3", question="Only record", response="one"), records_path)

    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {"doc_id": "doc.pdf", "question": "What?", "response": "prediction"},
                {"doc_id": "doc.pdf", "question": "Only prediction", "response": "two"},
            ]
        ),
        encoding="utf-8",
    )

    audit = audit_records(records_path, predictions_path)

    assert audit["duplicate_keys"]
    assert {item["question"] for item in audit["missing_records"]} == {"Only prediction"}
    assert {item["question"] for item in audit["missing_predictions"]} == {"Only record"}
    assert audit["response_mismatches"][0]["question"] == "What?"
    assert audit["records"]["total"] == 3
    assert audit["predictions"]["total"] == 2


def test_audit_separates_failed_attempts_from_response_artifacts(tmp_path: Path):
    records_path = tmp_path / "runs.jsonl"
    write_run_record(
        RunRecord(
            run_id="failed",
            document_id="doc.pdf",
            question="Q",
            status="failed",
            response=None,
            config_hash="config",
            failure={
                "stage": "answer",
                "error_type": "RuntimeError",
                "message": "crashed",
                "retryable": False,
            },
        ),
        records_path,
    )
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps([{"doc_id": "doc.pdf", "question": "Q", "response": ""}]),
        encoding="utf-8",
    )

    audit = audit_records(records_path, predictions_path)

    assert len(audit["runtime_failures"]) == 1
    assert audit["invalid_response_count"] == 0


def test_audit_uses_generation_metadata_when_available(tmp_path: Path):
    records_path = tmp_path / "runs.jsonl"
    write_run_record(
        RunRecord(
            run_id="invalid",
            document_id="doc.pdf",
            question="Q",
            status="ok",
            response="{incomplete",
            raw_response="{incomplete",
            parse_status="invalid",
            finish_reason="length",
            input_tokens=4,
            output_tokens=2,
            config_hash="config",
        ),
        records_path,
    )
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps([{"doc_id": "doc.pdf", "question": "Q", "response": "{incomplete"}]),
        encoding="utf-8",
    )

    audit = audit_records(records_path, predictions_path)

    assert audit["invalid_response_count"] == 1
    assert audit["termination_metadata_missing_count"] == 0


def test_audit_verdicts_counts_upstream_failure_markers(tmp_path: Path):
    path = tmp_path / "scored.json"
    path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "doc.pdf",
                    "question": "Q1",
                    "llm_judge": {
                        "equivalent": True,
                        "abstained": False,
                        "reason": "same",
                    },
                },
                {
                    "doc_id": "doc.pdf",
                    "question": "Q2",
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
    report = audit_verdicts(path)
    assert report["total"] == 2
    assert report["valid"] == 1
    assert report["failure_count"] == 1


def test_resume_rejects_changed_identity_fields():
    manifest = RunManifest(
        schema_version=1,
        run_id="run-1",
        config_hash="config-a",
        code_hash="code-a",
        dataset_hash="dataset-a",
        samples_hash="samples-a",
        models_hash="models-a",
        prompts_hash="prompts-a",
        dependencies_hash="deps-a",
    )
    changed = manifest.model_copy(update={"prompts_hash": "prompts-b"})

    assert_resume_compatible(manifest, manifest)
    with pytest.raises(ValueError, match="prompts_hash"):
        assert_resume_compatible(manifest, changed)
