"""Scoring, experiment comparison, run auditing, and split management."""

from .evaluation import (
    JudgeErrorKind,
    JudgeErrorType,
    JudgeEvaluationError,
    JudgeVerdict,
    classify_judge_error,
    judge_cache_identity,
    judge_cache_key,
    retrieval_metrics,
    score_run,
    validate_judge_verdict,
)
from .experiments import compare_runs, write_report
from .manifests import audit_records, audit_verdicts
from .splits import create_document_split

__all__ = [
    "JudgeErrorKind",
    "JudgeErrorType",
    "JudgeEvaluationError",
    "JudgeVerdict",
    "audit_records",
    "audit_verdicts",
    "classify_judge_error",
    "judge_cache_identity",
    "judge_cache_key",
    "compare_runs",
    "create_document_split",
    "retrieval_metrics",
    "score_run",
    "validate_judge_verdict",
    "write_report",
]
