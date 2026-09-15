"""Configuration and reproducibility fingerprints for harness runs."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Iterable, Literal

from pydantic import Field, model_validator

from .contracts import StrictModel


class ModelConfig(StrictModel):
    model_id: str
    revision: str
    prompt_version: str = "baseline-v1"


class RenderConfig(StrictModel):
    dpi: int = Field(default=144, gt=0)
    page_policy: Literal["all", "explicit"] = "all"
    max_pixels: int = Field(default=1_840_000, gt=0)


class GenerationConfig(StrictModel):
    max_new_tokens: int = Field(default=256, gt=0)
    planning_max_new_tokens: int = Field(default=512, gt=0)
    verification_max_new_tokens: int = Field(default=1024, gt=0)
    schema_repair_attempts: Literal[0, 1] = 1
    do_sample: bool = False
    temperature: float | None = Field(default=None, ge=0)


class RetrievalConfig(StrictModel):
    enabled: bool = False
    rerank_enabled: bool = True
    candidate_k: int = Field(default=20, gt=0)
    selected_k: int = Field(default=6, gt=0)
    embedding_batch_size: int = Field(default=1, gt=0)
    embedding_model_id: str = "Qwen/Qwen3-VL-Embedding-2B"
    embedding_revision: str = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
    reranker_model_id: str = "Qwen/Qwen3-VL-Reranker-2B"
    reranker_revision: str = "4bd860ac4f15ad1897a214615cccc700f8f71818"


class GraphConfig(StrictModel):
    enabled: bool = False
    max_hops: int = Field(default=2, ge=0, le=4)
    max_candidates: int = Field(default=40, gt=0)
    require_connection: bool = False
    allowed_relations: list[str] = Field(
        default_factory=lambda: [
            "contains",
            "follows",
            "refers_to",
            "defines",
            "supports",
            "qualifies",
            "contradicts",
        ]
    )
    extraction_version: str = "graph-v1"


class OCRConfig(StrictModel):
    enabled: bool = False
    model_id: str = "baidu/Qianfan-OCR"
    revision: str = "623bf5d20d446abdb36606aa4547cd0c18886fe5"
    prompt_version: str = "qianfan-ocr-v1"
    max_text_tokens: int = Field(default=12_000, gt=0)
    max_new_tokens: int = Field(default=4096, gt=0)


class EvidenceConfig(StrictModel):
    max_pages: int = Field(default=6, gt=0)
    max_pixels_per_image: int = Field(default=1_048_576, gt=0)
    max_total_image_pixels: int = Field(default=6_291_456, gt=0)
    max_text_tokens: int = Field(default=12_000, gt=0)
    reserved_output_tokens: int = Field(default=1_024, ge=0)


class VerificationConfig(StrictModel):
    enabled: bool = False
    mode: Literal["deterministic_legacy", "model"] = "deterministic_legacy"
    max_expansion_rounds: int = Field(default=2, ge=0)
    max_pages: int = Field(default=24, gt=0)
    wall_time_seconds: int = Field(default=300, gt=0)
    retained_pages: int = Field(default=2, ge=0)
    max_synthesis_calls: int = Field(default=2, ge=0)
    max_source_recheck_windows: int = Field(default=2, ge=0)


class PromptConfig(StrictModel):
    set: Literal["control", "v2"] = "control"
    overrides: dict[str, Literal["control", "v2"]] = Field(default_factory=dict)


class PlanningConfig(StrictModel):
    mode: Literal["heuristic", "model"] = "heuristic"
    max_queries: int = Field(default=3, ge=1, le=3)


class PathConfig(StrictModel):
    artifact_dir: Path = Path("artifacts")
    cache_dir: Path = Path("cache")


class HarnessConfig(StrictModel):
    model: ModelConfig
    render: RenderConfig = Field(default_factory=RenderConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    planning: PlanningConfig = Field(default_factory=PlanningConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

    @model_validator(mode="after")
    def validate_phase2_budgets(self) -> "HarnessConfig":
        if self.retrieval.selected_k > self.evidence.max_pages:
            raise ValueError("retrieval.selected_k must not exceed evidence.max_pages")
        if self.verification.retained_pages >= self.evidence.max_pages:
            raise ValueError("verification.retained_pages must be less than evidence.max_pages")
        if self.verification.max_pages < self.evidence.max_pages:
            raise ValueError("verification.max_pages must cover evidence.max_pages")
        return self

    def effective_hash(self) -> str:
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_paths(config: HarnessConfig, base_dir: Path) -> HarnessConfig:
    values = config.model_dump()
    values["paths"]["artifact_dir"] = str(
        (base_dir / values["paths"]["artifact_dir"]).resolve()
        if not Path(values["paths"]["artifact_dir"]).is_absolute()
        else Path(values["paths"]["artifact_dir"])
    )
    values["paths"]["cache_dir"] = str(
        (base_dir / values["paths"]["cache_dir"]).resolve()
        if not Path(values["paths"]["cache_dir"]).is_absolute()
        else Path(values["paths"]["cache_dir"])
    )
    return HarnessConfig.model_validate(values)


def load_config(path: Path) -> HarnessConfig:
    """Load and validate TOML configuration, resolving paths relative to it."""

    path = Path(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return _resolve_paths(HarnessConfig.model_validate(raw), path.parent.resolve())


def lock_hash(paths: Iterable[Path]) -> str:
    """Hash lock-file names and bytes in deterministic path order."""

    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
