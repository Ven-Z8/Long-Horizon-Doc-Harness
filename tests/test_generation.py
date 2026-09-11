import pytest

from doc_harness.contracts import DraftAnswer, GenerationResult, RunRecord
from doc_harness.runner import parse_draft_answer


def test_non_json_generation_is_not_a_valid_draft():
    with pytest.raises(ValueError, match="valid DraftAnswer"):
        parse_draft_answer("The answer is 42.")


def test_generation_result_preserves_raw_text_and_parse_status():
    result = GenerationResult(
        raw_response='{"answer":"42","evidence":[],"insufficient_evidence":false}',
        draft=DraftAnswer(answer="42", evidence=[], insufficient_evidence=False),
        parse_status="valid",
        finish_reason="eos",
        input_tokens=12,
        output_tokens=7,
    )
    assert result.raw_response.startswith("{")
    assert result.draft.answer == "42"


def test_generation_result_rejects_negative_token_counts():
    with pytest.raises(ValueError):
        GenerationResult(
            raw_response="x",
            draft=None,
            parse_status="invalid",
            finish_reason="length",
            input_tokens=-1,
            output_tokens=2,
        )


def test_run_record_can_store_raw_generation_metadata():
    record = RunRecord(
        run_id="r1",
        document_id="doc.pdf",
        question="Q",
        status="ok",
        response="42",
        raw_response='{"answer":"42"}',
        parse_status="valid",
        finish_reason="eos",
        input_tokens=10,
        output_tokens=4,
        config_hash="h",
    )
    assert record.raw_response.startswith("{")
    assert record.output_tokens == 4
