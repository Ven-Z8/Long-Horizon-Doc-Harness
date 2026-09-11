from pathlib import Path

from doc_harness.contracts import DraftAnswer, ModelRequest, RunRecord
from doc_harness.records import read_run_records, write_run_record
from doc_harness.runner import FakeRunner, parse_draft_answer


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


def test_parse_draft_answer_removes_thinking_and_json_fence():
    draft = parse_draft_answer(
        '<think>inspect the page</think>\n```json\n'
        '{"answer":"A","evidence":[],"insufficient_evidence":false}\n```'
    )
    assert draft.answer == "A"


def test_parse_draft_answer_extracts_json_from_surrounding_text():
    draft = parse_draft_answer(
        'Here is the answer. {"answer":"A","evidence":[],"insufficient_evidence":false}'
    )
    assert draft.answer == "A"


def test_parse_draft_answer_skips_evidence_only_object():
    draft = parse_draft_answer(
        '{"page_id":"1","quote":"Total debt"}\n'
        '{"answer":"100","evidence":[],"insufficient_evidence":false}'
    )
    assert draft.answer == "100"


def test_parse_draft_answer_preserves_non_json_prose():
    draft = parse_draft_answer("The answer is Less well-off.")
    assert draft.answer == "The answer is Less well-off."
