"""Configuration and reproducibility fingerprints for harness runs."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Iterable, Literal

from pydantic import Field

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
    do_sample: bool = False
    temperature: float | None = Field(default=None, ge=0)


class PathConfig(StrictModel):
    artifact_dir: Path = Path("artifacts")
    cache_dir: Path = Path("cache")


class HarnessConfig(StrictModel):
    model: ModelConfig
    render: RenderConfig = Field(default_factory=RenderConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    paths: PathConfig = Field(default_factory=PathConfig)

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
