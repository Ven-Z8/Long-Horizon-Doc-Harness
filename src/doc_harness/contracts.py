"""Pydantic contracts shared by every harness stage."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Status(str, Enum):
    ok = "ok"
    failed = "failed"


class Page(StrictModel):
    document_id: str
    page_id: int = Field(ge=0)
    image_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    render_sha256: str


class RankedPage(StrictModel):
    page_id: int = Field(ge=0)
    score: float
    rank: int = Field(ge=1)
    stage: str


class ParsedPage(StrictModel):
    page_id: int = Field(ge=0)
    markdown: str
    parser_model: str
    parser_revision: str
    prompt_version: str


class EvidenceSpan(StrictModel):
    page_id: int = Field(ge=0)
    quote: str = Field(min_length=1)


class DraftAnswer(StrictModel):
    answer: str | None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    insufficient_evidence: bool

    @model_validator(mode="after")
    def answer_and_insufficiency_agree(self) -> "DraftAnswer":
        if (self.answer is None) != self.insufficient_evidence:
            raise ValueError(
                "answer must be None exactly when insufficient_evidence is true"
            )
        return self


class VerifyDecision(str, Enum):
    accept = "accept"
    correct = "correct"
    abstain = "abstain"


class Verification(StrictModel):
    decision: VerifyDecision
    final_answer: str | None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    reason: str

    @model_validator(mode="after")
    def abstention_has_no_answer(self) -> "Verification":
        if (self.final_answer is None) != (self.decision == VerifyDecision.abstain):
            raise ValueError(
                "final_answer must be None exactly for an abstain decision"
            )
        return self


class StageFailure(StrictModel):
    stage: str
    error_type: str
    message: str
    retryable: bool


class BenchmarkSample(StrictModel):
    doc_id: str
    question: str
    answer: str
    evidence_pages: list[int] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    doc_type: str | None = None
    question_type: str | None = None
    answer_format: str | None = None


class ModelRequest(StrictModel):
    document_id: str
    question: str
    page_ids: list[int]
    image_paths: list[str]
    prompt: str
    config_hash: str


class Prediction(StrictModel):
    doc_id: str
    question: str
    response: str


class RunRecord(StrictModel):
    run_id: str
    document_id: str
    question: str
    status: Status
    response: str | None = None
    config_hash: str
    model_id: str | None = None
    model_revision: str | None = None
    page_ids: list[int] = Field(default_factory=list)
    latency_ms: float | None = Field(default=None, ge=0)
    peak_vram_mb: float | None = Field(default=None, ge=0)
    retry_count: int = Field(default=0, ge=0)
    failure: StageFailure | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
