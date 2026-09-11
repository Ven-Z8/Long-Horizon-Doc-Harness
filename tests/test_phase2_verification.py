from types import SimpleNamespace
import json

import pytest

from doc_harness.core.answers import AnswerDraftV2, EvidenceRefV2, VerificationReportV2
from doc_harness.core.config import HarnessConfig
from doc_harness.core.contracts import ModelRequest, SafeQuestion
from doc_harness.models.structured import StructuredCall
from doc_harness.workflow.verification import check_provenance, finalize_verified, verify_v2


def test_supported_count_cannot_override_missing_global_coverage():
    report = VerificationReportV2(
        verdict="supported",
        final_answer="4",
        reason="Four diagrams in the inspected pages",
        evidence=[EvidenceRefV2(page_id=0, kind="visual", locator="diagram panel")],
    )
    outcome = finalize_verified(report, exhaustive_required=True, coverage_complete=False)
    assert outcome.kind == "search_exhausted"
    assert outcome.answer is None


def test_visual_evidence_is_valid_without_ocr_text():
    draft = AnswerDraftV2(
        answer="12%",
        evidence=[EvidenceRefV2(page_id=2, kind="visual", locator="blue bar")],
        insufficient_evidence=False,
    )
    assert check_provenance(draft, {2}) == []


def test_unknown_page_is_rejected_even_when_quote_is_present():
    draft = AnswerDraftV2(
        answer="12%",
        evidence=[EvidenceRefV2(page_id=9, kind="text", quote="12%")],
        insufficient_evidence=False,
    )
    assert check_provenance(draft, {2}) == ["evidence references unavailable page 9"]


def test_supported_report_becomes_answered_when_coverage_is_complete():
    report = VerificationReportV2(
        verdict="supported",
        final_answer="12%",
        reason="The chart shows twelve percent.",
        evidence=[EvidenceRefV2(page_id=2, kind="visual", locator="blue bar")],
    )
    outcome = finalize_verified(report, exhaustive_required=False, coverage_complete=False)
    assert outcome.kind == "answered"
    assert outcome.answer == "12%"


def test_model_verifier_returns_a_typed_report():
    draft = AnswerDraftV2(
        answer="12%",
        evidence=[EvidenceRefV2(page_id=2, kind="visual", locator="blue bar")],
        insufficient_evidence=False,
    )
    bundle = SimpleNamespace(included_page_ids=[2], images=[], ocr_content={2: ""})
    config = HarnessConfig.model_validate({"model": {"model_id": "m", "revision": "r"}})
    def call(request: ModelRequest) -> StructuredCall:
        assert request.page_ids == [2]
        return StructuredCall(
            attempts=[],
            payload=VerificationReportV2(
                verdict="supported", final_answer="12%", evidence=draft.evidence,
                reason="supported by the chart",
            ).model_dump(mode="json"),
        )
    report = verify_v2(
        SafeQuestion(document_id="d.pdf", question="What is the value?"),
        draft,
        bundle,
        call=call,
        coverage={"complete": True},
        config=config,
    )
    assert report.verdict == "supported"
