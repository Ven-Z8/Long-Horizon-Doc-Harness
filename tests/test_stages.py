import json
from pathlib import Path

from doc_harness.config import load_config
import pytest

from doc_harness.stages import build_run_manifest, resolve_checkpoint
from doc_harness.workflow import coordinator


def test_resolve_checkpoint_maps_huggingface_id_to_local_directory(tmp_path: Path):
    checkpoint = tmp_path / "Qwen3-VL-Embedding-2B"
    checkpoint.mkdir()
    assert resolve_checkpoint("Qwen/Qwen3-VL-Embedding-2B", tmp_path) == checkpoint


def test_build_run_manifest_records_all_resume_identities(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[model]\nmodel_id = "test"\nrevision = "rev"\n')
    config = load_config(config_path)
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{ "doc_id": "d.pdf", "question": "Q", "answer": "A" }]))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"schema_version": 1, "stage": "embedding", "entries": []}))

    manifest = build_run_manifest(
        config,
        run_id="run-1",
        samples_path=samples,
        index_manifest=index,
        prompt="prompt-v1",
    )

    assert manifest.run_id == "run-1"
    assert manifest.config_hash == config.effective_hash()
    assert manifest.samples_hash
    assert manifest.dataset_hash
    assert manifest.prompts_hash


def test_build_run_manifest_includes_graph_manifest_only_when_enabled(tmp_path: Path):
    """Catches graph artifact changes being accepted during an enabled resume."""

    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{ "doc_id": "d.pdf", "question": "Q", "answer": "A" }]))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"schema_version": 1, "stage": "embedding", "entries": []}))
    graph = tmp_path / "graph-manifest.json"
    graph.write_text('{"graph":"one"}')
    disabled = load_config(_write_config(tmp_path / "disabled.toml", graph_enabled=False))
    enabled = load_config(_write_config(tmp_path / "enabled.toml", graph_enabled=True))

    baseline = build_run_manifest(
        disabled, run_id="run", samples_path=samples, index_manifest=index
    )
    graph_enabled = build_run_manifest(
        enabled,
        run_id="run",
        samples_path=samples,
        index_manifest=index,
        graph_manifest=graph,
    )
    graph.write_text('{"graph":"two"}')
    changed = build_run_manifest(
        enabled,
        run_id="run",
        samples_path=samples,
        index_manifest=index,
        graph_manifest=graph,
    )

    assert baseline.dataset_hash == __import__("hashlib").sha256(index.read_bytes()).hexdigest()
    assert graph_enabled.dataset_hash != changed.dataset_hash


def test_graph_enabled_coordinator_requires_graph_manifest(tmp_path: Path):
    """Catches graph runs silently starting without their immutable artifact."""

    config = load_config(_write_config(tmp_path / "enabled.toml", graph_enabled=True))
    samples = tmp_path / "samples.json"
    samples.write_text("[]")
    with pytest.raises(ValueError, match="graph manifest is required"):
        coordinator.run_coordinator(
            config,
            samples,
            tmp_path,
            tmp_path / "render",
            tmp_path / "run",
        )


def test_graph_disabled_coordinator_keeps_original_call_path(tmp_path: Path, monkeypatch):
    """Catches a new graph input becoming mandatory for the baseline."""

    config = load_config(_write_config(tmp_path / "disabled.toml", graph_enabled=False))
    samples = tmp_path / "samples.json"
    samples.write_text("[]")
    index = tmp_path / "index.json"
    index.write_text('{"schema_version": 1, "stage": "embedding", "entries": []}')
    calls = []
    monkeypatch.setattr(coordinator, "build_indexes", lambda *args, **kwargs: index)
    monkeypatch.setattr(coordinator, "retrieve_questions", lambda *args, **kwargs: calls.append(kwargs) or tmp_path / "retrieval.jsonl")
    monkeypatch.setattr(coordinator, "rerank_questions", lambda *args, **kwargs: tmp_path / "reranked.jsonl")
    monkeypatch.setattr(coordinator, "ocr_questions", lambda *args, **kwargs: tmp_path / "ocr.jsonl")
    monkeypatch.setattr(coordinator, "answer_questions", lambda *args, **kwargs: tmp_path / "predictions.json")

    coordinator.run_coordinator(config, samples, tmp_path, tmp_path / "render", tmp_path / "run")

    assert calls == [{"models_dir": Path("models"), "limit": None}]


def _write_config(path: Path, *, graph_enabled: bool) -> Path:
    path.write_text(
        '[model]\nmodel_id = "test"\nrevision = "rev"\n'
        + ("[graph]\nenabled = true\n" if graph_enabled else "")
    )
    return path
