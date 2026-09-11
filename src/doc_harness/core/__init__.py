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
]
