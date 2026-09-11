"""Shared data contracts, configuration, wire protocol, and record storage."""

from .contracts import (
    BenchmarkSample,
    DraftAnswer,
    EvidenceSpan,
    GenerationResult,
    ModelRequest,
    Page,
    ParsedPage,
    Prediction,
    RankedPage,
    RunRecord,
    SafeQuestion,
    StageFailure,
    Verification,
)
from .answers import (
    AnswerDraftV2,
    EvidenceFactV2,
    EvidenceRefV2,
    QuestionOutcome,
    VerificationReportV2,
    export_outcome,
)

__all__ = [
    "BenchmarkSample",
    "DraftAnswer",
    "EvidenceSpan",
    "GenerationResult",
    "ModelRequest",
    "Page",
    "ParsedPage",
    "Prediction",
    "RankedPage",
    "RunRecord",
    "SafeQuestion",
    "StageFailure",
    "Verification",
    "AnswerDraftV2",
    "EvidenceFactV2",
    "EvidenceRefV2",
    "QuestionOutcome",
    "VerificationReportV2",
    "export_outcome",
]
