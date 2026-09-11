from types import SimpleNamespace

import pytest

from doc_harness.contracts import DraftAnswer, EvidenceSpan, SafeQuestion, VerifyDecision
from doc_harness.verification import check_evidence, verify_answer


def bundle():
    return SimpleNamespace(
        included_page_ids=[0],
        ocr_content={0: "Total debt: 100 million"},
        regions=[],
    )


def test_check_evidence_accepts_normalized_text_quote():
    draft = DraftAnswer(
        answer="100 million",
        evidence=[EvidenceSpan(page_id=0, quote="Total  debt:\n100 million")],
        insufficient_evidence=False,
    )
    result = check_evidence(draft, bundle())
    assert result.complete is True
    assert result.unmatched_quotes == []


def test_check_evidence_reports_unknown_page_and_unmatched_quote():
    draft = DraftAnswer(
        answer="100 million",
        evidence=[
            EvidenceSpan(page_id=9, quote="Total debt"),
            EvidenceSpan(page_id=0, quote="Total debt: 200 million"),
        ],
        insufficient_evidence=False,
    )
    result = check_evidence(draft, bundle())
    assert result.complete is False
    assert result.unknown_page_ids == [9]
    assert result.unmatched_quotes == ["Total debt: 200 million"]


def test_verify_answer_abstains_when_evidence_is_incomplete():
    draft = DraftAnswer(answer="100 million", evidence=[], insufficient_evidence=False)
    decision = verify_answer(
        SafeQuestion(document_id="doc.pdf", question="What is total debt?"),
        draft,
        bundle(),
    )
    assert decision.decision is VerifyDecision.abstain
    assert decision.final_answer is None


def test_verify_answer_accepts_supported_answer():
    draft = DraftAnswer(
        answer="100 million",
        evidence=[EvidenceSpan(page_id=0, quote="Total debt: 100 million")],
        insufficient_evidence=False,
    )
    decision = verify_answer(
        SafeQuestion(document_id="doc.pdf", question="What is total debt?"),
        draft,
        bundle(),
    )
    assert decision.decision is VerifyDecision.accept
    assert decision.final_answer == "100 million"
