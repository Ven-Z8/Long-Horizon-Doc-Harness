"""Typed, reproducible building blocks for long multimodal document QA.

Implementation modules are grouped by responsibility under ``core``,
``documents``, ``models``, ``workflow``, and ``evaluation``. The historical
``doc_harness.<module>`` imports remain available as compatibility aliases.
"""

from importlib import import_module
import sys

from .core.contracts import (
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

# Keep the original import surface stable while the implementation lives in
# responsibility-based subpackages. These aliases are intentionally explicit
# so older notebooks, tests, and benchmark scripts continue to work.
_LEGACY_MODULES = {
    "contracts": ".core.contracts",
    "config": ".core.config",
    "protocol": ".core.protocol",
    "records": ".core.records",
    "rendering": ".documents.rendering",
    "ocr": ".documents.ocr",
    "evidence": ".documents.evidence",
    "runner": ".models.runner",
    "retrieval": ".models.retrieval",
    "reranking": ".models.reranking",
    "batch": ".workflow.batch",
    "pipeline": ".workflow.pipeline",
    "planning": ".workflow.planning",
    "stages": ".workflow.stages",
    "verification": ".workflow.verification",
    "experiments": ".evaluation.experiments",
    "manifests": ".evaluation.manifests",
    "splits": ".evaluation.splits",
}

for _legacy_name, _canonical_name in _LEGACY_MODULES.items():
    _module = import_module(_canonical_name, __name__)
    sys.modules[f"{__name__}.{_legacy_name}"] = _module
    globals()[_legacy_name] = _module

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
