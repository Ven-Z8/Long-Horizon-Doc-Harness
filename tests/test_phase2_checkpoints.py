import json

import pytest

from doc_harness.workflow.checkpoints import (
    commit_question,
    load_committed,
    project_predictions,
    sample_key,
)
from doc_harness.workflow.stages import outcome_for_generation


def outcome(document_id="d.pdf", question="Q"):
    return {
        "document_id": document_id,
        "question": question,
        "outcome": {"kind": "answered", "answer": "A", "reason": "supported"},
        "attempt_paths": [],
    }


def test_uncommitted_temporary_file_is_ignored(tmp_path):
    key = sample_key("d.pdf", "Q")
    commit_question(tmp_path, key, outcome())
    (tmp_path / "questions" / ".sample-b.tmp").write_text('{"partial":')
    assert set(load_committed(tmp_path)) == {key}


def test_committed_question_is_rejected_if_key_payload_disagrees(tmp_path):
    key = sample_key("d.pdf", "Q")
    with pytest.raises(ValueError, match="does not match"):
        commit_question(tmp_path, key, outcome(question="different"))


def test_replacing_a_failed_checkpoint_preserves_history(tmp_path):
    key = sample_key("d.pdf", "Q")
    failed = outcome()
    failed["outcome"] = {"kind": "failed", "answer": None, "reason": "worker"}
    commit_question(tmp_path, key, failed)
    commit_question(tmp_path, key, outcome(), allow_replace=True)
    assert (tmp_path / "questions" / "history").exists()
    assert load_committed(tmp_path)[key]["outcome"]["kind"] == "answered"


def test_projection_contains_only_terminal_predictions(tmp_path):
    commit_question(tmp_path, sample_key("d.pdf", "Q"), outcome())
    pending = outcome(document_id="d.pdf", question="P")
    pending["outcome"] = {"kind": "failed", "answer": None, "reason": "worker"}
    commit_question(tmp_path, sample_key("d.pdf", "P"), pending)
    path = project_predictions(tmp_path)
    rows = json.loads(path.read_text())
    assert rows == [{"doc_id": "d.pdf", "question": "Q", "response": "A"}]


def test_generation_failure_is_recorded_as_operational_failure():
    outcome = outcome_for_generation(
        response="Not answerable",
        generation_failed=True,
        verification_abstained=False,
    )
    assert outcome.kind == "failed"
    assert outcome.answer is None
