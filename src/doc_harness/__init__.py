"""Typed, reproducible building blocks for long multimodal document QA."""

from .contracts import (
    BenchmarkSample,
    DraftAnswer,
    EvidenceSpan,
    ModelRequest,
    Page,
    ParsedPage,
    Prediction,
    RankedPage,
    RunRecord,
    StageFailure,
    Verification,
)

__all__ = [
    "BenchmarkSample",
    "DraftAnswer",
    "EvidenceSpan",
    "ModelRequest",
    "Page",
    "ParsedPage",
    "Prediction",
    "RankedPage",
    "RunRecord",
    "StageFailure",
    "Verification",
]
