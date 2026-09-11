"""Deterministic grounding checks and conservative answer verification."""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol

from pydantic import Field

from ..core.contracts import (
    DraftAnswer,
    ModelRequest,
    SafeQuestion,
    StrictModel,
    Verification,
    VerifyDecision,
)
from ..core.answers import AnswerDraftV2, QuestionOutcome, VerificationReportV2
from ..core.config import HarnessConfig
from ..models.structured import StructuredCall


class EvidenceCheck(StrictModel):
    unknown_page_ids: list[int] = Field(default_factory=list)
    unmatched_quotes: list[str] = Field(default_factory=list)
    invalid_regions: list[int] = Field(default_factory=list)
    uncovered_claims: list[str] = Field(default_factory=list)
    complete: bool


class VerificationModel(Protocol):
    def verify(
        self, question: SafeQuestion, draft: DraftAnswer, bundle: Any
    ) -> Verification:
        """Return a validated model-based verification decision."""


def check_provenance(draft: AnswerDraftV2, allowed_page_ids: set[int]) -> list[str]:
    """Check that every model citation refers to evidence in the active bundle."""

    errors: list[str] = []
    for reference in draft.evidence:
        if reference.page_id not in allowed_page_ids:
            errors.append(f"evidence references unavailable page {reference.page_id}")
    for finding in draft.partial_findings:
        for reference in finding.evidence:
            if reference.page_id not in allowed_page_ids:
                errors.append(f"finding references unavailable page {reference.page_id}")
    return list(dict.fromkeys(errors))


def finalize_verified(
    report: VerificationReportV2,
    *,
    exhaustive_required: bool,
    coverage_complete: bool,
) -> QuestionOutcome:
    """Apply deterministic coverage gates to a semantic verification report."""

    if report.verdict == "supported":
        if exhaustive_required and not coverage_complete:
            return QuestionOutcome(
                kind="search_exhausted",
                answer=None,
                reason="supported evidence does not cover the required document scope",
            )
        return QuestionOutcome(kind="answered", answer=report.final_answer, reason=report.reason)
    if report.verdict == "unanswerable" and coverage_complete:
        return QuestionOutcome(kind="unanswerable", answer=None, reason=report.reason)
    return QuestionOutcome(
        kind="search_exhausted",
        answer=None,
        reason=report.reason,
    )


def verify_v2(
    question: SafeQuestion,
    draft: AnswerDraftV2,
    bundle: Any,
    *,
    call: Callable[[ModelRequest], StructuredCall],
    coverage: dict[str, Any],
    config: HarnessConfig,
) -> VerificationReportV2:
    """Run a semantic verifier and enforce provenance before accepting it."""

    del config
    allowed = set(getattr(bundle, "included_page_ids", []))
    provenance_errors = check_provenance(draft, allowed)
    if provenance_errors:
        raise ValueError("; ".join(provenance_errors))
    page_ids = list(getattr(bundle, "included_page_ids", []))
    image_paths = [image.image_path for image in getattr(bundle, "images", [])]
    request = ModelRequest(
        document_id=question.document_id,
        question=question.question,
        page_ids=page_ids,
        image_paths=image_paths,
        prompt=json.dumps(
            {
                "role": "verification",
                "draft": draft.model_dump(mode="json"),
                "coverage": coverage,
            },
            ensure_ascii=False,
        ),
        config_hash="verification-v2",
    )
    result = call(request)
    if result.payload is None:
        raise ValueError(result.failure.message if result.failure else "verification returned no payload")
    report = VerificationReportV2.model_validate(result.payload)
    invalid_pages = [
        reference.page_id
        for reference in report.evidence
        if reference.page_id not in allowed
    ]
    invalid_pages.extend(
        reference.page_id
        for finding in report.verified_findings
        for reference in finding.evidence
        if reference.page_id not in allowed
    )
    if invalid_pages:
        raise ValueError(f"verification references unavailable page {invalid_pages[0]}")
    return report


def _normalized(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def check_evidence(draft: DraftAnswer, bundle: Any) -> EvidenceCheck:
    """Check page identity and exact normalized OCR quote presence.

    A quote match only establishes that the text appears in the supplied OCR. It does
    not establish semantic truth, which is why this function never marks an answer
    supported merely because it has a citation.
    """

    included = set(getattr(bundle, "included_page_ids", []))
    ocr_content = dict(getattr(bundle, "ocr_content", {}) or {})
    unknown: list[int] = []
    unmatched: list[str] = []
    for span in draft.evidence:
        if span.page_id not in included and span.page_id not in ocr_content:
            unknown.append(span.page_id)
            continue
        if getattr(span, "kind", "text") == "visual":
            continue
        source = ocr_content.get(span.page_id)
        if source is None or not span.quote or _normalized(span.quote) not in _normalized(source):
            unmatched.append(span.quote or span.locator or "")
    unknown = list(dict.fromkeys(unknown))
    unmatched = list(dict.fromkeys(unmatched))
    uncovered: list[str] = []
    if draft.answer is not None and not draft.evidence:
        uncovered.append(draft.answer)
    return EvidenceCheck(
        unknown_page_ids=unknown,
        unmatched_quotes=unmatched,
        uncovered_claims=uncovered,
        complete=not (unknown or unmatched or uncovered),
    )


def verify_answer(
    question: SafeQuestion,
    draft: DraftAnswer,
    bundle: Any,
    verifier: VerificationModel | None = None,
) -> Verification:
    """Return a conservative verification decision for a draft answer."""

    if verifier is not None:
        decision = verifier.verify(question, draft, bundle)
        return Verification.model_validate(decision)
    check = check_evidence(draft, bundle)
    if draft.answer is None:
        return Verification(
            decision=VerifyDecision.abstain,
            final_answer=None,
            evidence=draft.evidence,
            reason="draft abstains",
        )
    if check.complete:
        return Verification(
            decision=VerifyDecision.accept,
            final_answer=draft.answer,
            evidence=draft.evidence,
            reason="all cited text occurs in the supplied evidence",
        )
    return Verification(
        decision=VerifyDecision.abstain,
        final_answer=None,
        evidence=[],
        reason="draft claims are not fully supported by supplied evidence",
    )
