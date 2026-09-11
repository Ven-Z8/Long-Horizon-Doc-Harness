from pydantic import ValidationError
import pytest

from doc_harness.contracts import DraftAnswer, EvidenceSpan, Page, StageFailure


def test_draft_answer_requires_insufficiency_for_missing_answer():
    with pytest.raises(ValidationError):
        DraftAnswer(answer=None, evidence=[], insufficient_evidence=False)


def test_page_rejects_negative_page_ids_and_zero_dimensions():
    with pytest.raises(ValidationError):
        Page(document_id="doc.pdf", page_id=-1, image_path="p.png", width=0,
             height=100, render_sha256="abc")


def test_evidence_span_and_failure_are_typed():
    span = EvidenceSpan(page_id=0, quote="Total debt")
    failure = StageFailure(stage="render", error_type="IOError",
                           message="missing PDF", retryable=False)
    assert span.page_id == 0
    assert failure.retryable is False


def test_visual_evidence_can_use_a_locator_without_an_ocr_quote():
    draft = DraftAnswer(
        answer="12%",
        evidence=[EvidenceSpan(page_id=2, kind="visual", locator="blue bar")],
        insufficient_evidence=False,
    )
    assert draft.evidence[0].quote is None
