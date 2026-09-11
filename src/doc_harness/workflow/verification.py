"""Deterministic grounding checks and conservative answer verification."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import Field

from ..core.contracts import (
    DraftAnswer,
    SafeQuestion,
    StrictModel,
    Verification,
    VerifyDecision,
)


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
        source = ocr_content.get(span.page_id)
        if source is None or _normalized(span.quote) not in _normalized(source):
            unmatched.append(span.quote)
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
