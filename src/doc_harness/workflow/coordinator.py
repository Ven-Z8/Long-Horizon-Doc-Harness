"""Coordinator for reproducible staged runs and durable answer checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

from ..core.config import HarnessConfig
from ..evaluation.manifests import RunManifest, assert_resume_compatible
from .stages import (
    answer_questions,
    build_indexes,
    build_run_manifest,
    ocr_questions,
    rerank_questions,
    retrieve_questions,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_coordinator(
    config: HarnessConfig,
    samples_path: Path,
    documents_dir: Path,
    render_dir: Path,
    run_dir: Path,
    *,
    index_manifest: Path | None = None,
    graph_manifest: Path | None = None,
    models_dir: Path = Path("models"),
    limit: int | None = None,
    resume: bool = False,
    retry_failed: bool = False,
    stop_after: int | None = None,
) -> Path:
    """Run retrieval, reranking, OCR, and answer stages with a preflight identity."""

    if config.graph.enabled:
        if graph_manifest is None:
            raise ValueError("graph manifest is required when graph retrieval is enabled")
        graph_manifest = Path(graph_manifest)
        if not graph_manifest.is_file():
            raise ValueError(f"graph manifest does not exist: {graph_manifest}")
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if index_manifest is None:
        existing = run_dir / "index" / "index-manifest.json"
        if resume and existing.exists():
            index_manifest = existing
        else:
            index_manifest = build_indexes(
                config,
                documents_dir,
                render_dir,
                run_dir / "index",
                models_dir=models_dir,
            )
    manifest_kwargs = {
        "run_id": run_dir.name,
        "samples_path": samples_path,
        "index_manifest": index_manifest,
    }
    if config.graph.enabled:
        manifest_kwargs["graph_manifest"] = graph_manifest
    manifest = build_run_manifest(config, **manifest_kwargs)
    manifest_path = run_dir / "manifest.json"
    if resume and manifest_path.exists():
        existing_manifest = RunManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
        assert_resume_compatible(existing_manifest, manifest)
    else:
        _write_json(manifest_path, manifest.model_dump(mode="json"))

    stage_dir = run_dir / "stages"
    retrieval_kwargs = {"models_dir": models_dir, "limit": limit}
    if config.graph.enabled:
        retrieval_kwargs["graph_manifest"] = graph_manifest
    retrieval_path = retrieve_questions(
        config, samples_path, index_manifest, stage_dir / "retrieval.jsonl", **retrieval_kwargs
    )
    rerank_kwargs = {"models_dir": models_dir}
    if config.graph.enabled:
        rerank_kwargs["graph_manifest"] = graph_manifest
    reranked_path = rerank_questions(
        config, retrieval_path, index_manifest, stage_dir / "reranked", **rerank_kwargs
    )
    ocr_path = ocr_questions(
        config,
        reranked_path,
        index_manifest,
        stage_dir / "ocr",
        models_dir=models_dir,
    )
    answer_questions(
        config,
        samples_path,
        ocr_path,
        index_manifest,
        run_dir,
        models_dir=models_dir,
        resume=resume,
        retry_failed=retry_failed,
        stop_after=stop_after,
    )
    return run_dir
