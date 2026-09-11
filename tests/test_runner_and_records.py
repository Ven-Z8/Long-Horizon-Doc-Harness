from pathlib import Path

from doc_harness.contracts import DraftAnswer, ModelRequest, RunRecord
from doc_harness.records import read_run_records, write_run_record
from doc_harness.runner import FakeRunner


def request():
    return ModelRequest(
        document_id="doc.pdf",
        question="Q",
        page_ids=[0],
        image_paths=["page.png"],
        prompt="P",
        config_hash="h",
    )


def test_fake_runner_returns_valid_draft():
    answer = FakeRunner(answer="A").run(request())
    assert isinstance(answer, DraftAnswer)
    assert answer.answer == "A"


def test_run_record_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    record = RunRecord(
        run_id="r1",
        document_id="doc.pdf",
        question="Q",
        status="ok",
        response="A",
        config_hash="h",
    )
    write_run_record(record, path)
    assert read_run_records(path) == [record]
