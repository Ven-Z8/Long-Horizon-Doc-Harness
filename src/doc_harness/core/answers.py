"""Versioned answer and evidence contracts for the repaired pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class V2Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceRefV2(V2Model):
    page_id: int = Field(ge=0)
    kind: Literal["text", "visual"]
    quote: str | None = Field(default=None, max_length=240)
    locator: str | None = Field(default=None, max_length=240)

    @model_validator(mode="after")
    def source_description_required(self) -> "EvidenceRefV2":
        if self.kind == "text" and not self.quote:
            raise ValueError("text evidence requires a quote")
        if self.kind == "visual" and not self.locator:
            raise ValueError("visual evidence requires a locator")
        return self


class EvidenceFactV2(V2Model):
    claim: str = Field(min_length=1, max_length=320)
    evidence: list[EvidenceRefV2] = Field(min_length=1, max_length=4)


class AnswerDraftV2(V2Model):
    answer: str | None = Field(default=None, max_length=1600)
    evidence: list[EvidenceRefV2] = Field(default_factory=list, max_length=8)
    insufficient_evidence: bool
    partial_findings: list[EvidenceFactV2] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def answer_and_evidence_agree(self) -> "AnswerDraftV2":
        if (self.answer is None) != self.insufficient_evidence:
            raise ValueError("answer must be null exactly when evidence is insufficient")
        if self.answer is not None and not self.answer.strip():
            raise ValueError("answer must not be blank")
        if self.answer is not None and not self.evidence:
            raise ValueError("answered drafts require evidence")
        return self


class VerificationReportV2(V2Model):
    verdict: Literal["supported", "contradicted", "needs_more_evidence", "unanswerable"]
    final_answer: str | None = Field(default=None, max_length=1600)
    evidence: list[EvidenceRefV2] = Field(default_factory=list, max_length=8)
    reason: str = Field(min_length=1, max_length=480)
    missing_evidence: list[str] = Field(default_factory=list, max_length=4)
    verified_findings: list[EvidenceFactV2] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def verdict_and_answer_agree(self) -> "VerificationReportV2":
        if self.verdict == "supported":
            if self.final_answer is None or not self.final_answer.strip():
                raise ValueError("supported verification requires a final answer")
            if not self.evidence:
                raise ValueError("supported verification requires evidence")
        elif self.final_answer is not None:
            raise ValueError("only supported verification may provide a final answer")
        return self


class QuestionOutcome(V2Model):
    kind: Literal["answered", "unanswerable", "search_exhausted", "failed"]
    answer: str | None = Field(default=None, max_length=1600)
    reason: str = Field(min_length=1, max_length=480)

    @model_validator(mode="after")
    def outcome_and_answer_agree(self) -> "QuestionOutcome":
        if self.kind == "answered" and (self.answer is None or not self.answer.strip()):
            raise ValueError("answered outcomes require a nonblank answer")
        if self.kind != "answered" and self.answer is not None:
            raise ValueError("only answered outcomes may contain an answer")
        return self


def export_outcome(outcome: QuestionOutcome) -> str | None:
    """Project an operational outcome into the benchmark response field."""

    if outcome.kind == "answered":
        return outcome.answer
    if outcome.kind in {"unanswerable", "search_exhausted"}:
        return "Not answerable"
    return None


__all__ = [
    "AnswerDraftV2",
    "EvidenceFactV2",
    "EvidenceRefV2",
    "QuestionOutcome",
    "VerificationReportV2",
    "export_outcome",
]
