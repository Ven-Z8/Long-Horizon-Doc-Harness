"""Typed, reproducible building blocks for long multimodal document QA."""

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
