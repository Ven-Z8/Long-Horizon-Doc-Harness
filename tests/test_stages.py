import json
from pathlib import Path

import numpy as np
from doc_harness.config import load_config
import pytest

from doc_harness.contracts import Page
from doc_harness.core.config import HarnessConfig
from doc_harness.documents.graph import (
    build_document_graph,
    graph_fingerprint,
    write_graph,
)
from doc_harness.models.reranking import (
    SelectionEntry,
    SelectionManifest,
    read_selection_manifest,
)
from doc_harness.models.retrieval import IndexIdentity, PageIndex
from doc_harness.stages import build_run_manifest, resolve_checkpoint
from doc_harness.workflow import coordinator
from doc_harness.workflow import stages


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


def test_ocr_pool_keeps_selected_page_when_it_falls_below_candidate_cutoff(tmp_path: Path):
    """Catches connected evidence being dropped before OCR by a candidate cap."""

    manifest = SelectionManifest(
        document_id="d.pdf",
        question="Q",
        index_fingerprint="index",
        candidate_k=6,
        selected_k=1,
        candidates=[
            SelectionEntry(
                page_id=page_id,
                retrieval_score=float(10 - page_id),
                retrieval_rank=page_id + 1,
                rerank_score=float(10 - page_id),
                rerank_rank=page_id + 1,
            )
            for page_id in [0, 1, 2, 9]
        ],
        selected=[
            SelectionEntry(
                page_id=9,
                retrieval_score=1.0,
                retrieval_rank=4,
                rerank_score=1.0,
                rerank_rank=4,
            )
        ],
    )

    assert stages._ocr_page_pool(manifest, max_pages=2) == [9, 0]


def test_disabled_global_ocr_pool_keeps_original_index_order(tmp_path: Path):
    """Catches graph-only OCR promotion leaking into a disabled baseline run."""

    selected = SelectionEntry(
        page_id=9,
        retrieval_score=1.0,
        retrieval_rank=4,
        rerank_score=1.0,
        rerank_rank=4,
    )
    manifest = SelectionManifest(
        document_id="d.pdf",
        question="How many?",
        index_fingerprint="index",
        candidate_k=4,
        selected_k=1,
        candidates=[selected],
        selected=[selected],
    )

    assert stages._verification_ocr_page_pool(
        manifest,
        _index(tmp_path, source_sha256="pdf-bytes", page_count=10),
        global_scope=True,
        graph_enabled=False,
        max_pages=2,
    ) == [0, 1]


def test_graph_eligibility_rejects_same_document_with_different_source_hash(tmp_path: Path):
    """Catches a stale graph that happens to keep the same document filename."""

    config = HarnessConfig.model_validate(
        {"model": {"model_id": "m", "revision": "r"}, "graph": {"enabled": True}}
    )
    index = _index(tmp_path, source_sha256="new-pdf-bytes")
    graph = build_document_graph(
        "d.pdf", ["first", "second"], source_sha256="old-pdf-bytes", extraction_version="graph-v1"
    )

    assert stages._graph_eligibility_error(config, index, graph) == "graph source hash does not match page index"


def test_graph_eligibility_rejects_extraction_fallback_before_connected_selection(tmp_path: Path):
    """Catches reranking reconnecting a graph retrieval already ruled ineligible."""

    config = HarnessConfig.model_validate(
        {
            "model": {"model_id": "m", "revision": "r"},
            "graph": {"enabled": True, "extraction_version": "graph-v2"},
        }
    )
    index = _index(tmp_path, source_sha256="pdf-bytes")
    graph = build_document_graph(
        "d.pdf", ["first", "second"], source_sha256="pdf-bytes", extraction_version="graph-v1"
    )

    assert stages._graph_eligibility_error(config, index, graph) == "graph extraction version does not match configuration"


def test_rerank_keeps_ordinary_selection_after_retrieval_graph_fallback(tmp_path: Path, monkeypatch):
    """Catches reranking from applying connectivity after retrieval rejected a graph."""

    config = HarnessConfig.model_validate(
        {
            "model": {"model_id": "m", "revision": "r"},
            "retrieval": {"enabled": True, "rerank_enabled": False, "selected_k": 2},
            "graph": {"enabled": True, "require_connection": True},
        }
    )
    index = _index(tmp_path, source_sha256="pdf-bytes", page_count=3)
    graph = build_document_graph(
        "d.pdf",
        ["See page 3.", "unrelated", "connected"],
        source_sha256="pdf-bytes",
        extraction_version="graph-v1",
    )
    retrieval_path = tmp_path / "retrieval.jsonl"
    retrieval_path.write_text(
        json.dumps(
            {
                "doc_id": "d.pdf",
                "question": "Q",
                "candidates": [
                    {"page_id": 0, "score": 0.9, "rank": 1, "stage": "screen"},
                    {"page_id": 1, "score": 0.8, "rank": 2, "stage": "screen"},
                    {"page_id": 2, "score": 0.7, "rank": 3, "stage": "screen"},
                ],
                "graph": {
                    "eligible": False,
                    "cache_status": "fallback",
                    "terminal_reason": "graph_failure",
                    "failure": "ValueError: graph extraction version does not match configuration",
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(stages, "_index_by_document", lambda _path: {"d.pdf": index})
    monkeypatch.setattr(
        stages,
        "_graph_manifest_for_enabled_run",
        lambda *_args: ({"d.pdf": graph}, "graph-manifest", None),
    )

    reranked = stages.rerank_questions(
        config,
        retrieval_path,
        tmp_path / "index-manifest.json",
        tmp_path / "reranked",
        graph_manifest=tmp_path / "graph-manifest.json",
    )

    row = json.loads(reranked.read_text())
    manifest = read_selection_manifest(Path(row["manifest_path"]))
    assert [entry.page_id for entry in manifest.selected] == [0, 1]
    assert row["graph"]["eligible"] is False
    assert row["graph"]["terminal_reason"] == "graph_failure"


def test_graph_retrieval_caps_seed_union_when_max_candidates_is_smaller_than_candidate_k(
    tmp_path: Path, monkeypatch
):
    """Catches embedding seeds bypassing the graph candidate-union cap."""

    config = HarnessConfig.model_validate(
        {
            "model": {"model_id": "m", "revision": "r"},
            "retrieval": {"enabled": True, "candidate_k": 4, "selected_k": 2},
            "graph": {"enabled": True, "max_candidates": 2},
        }
    )
    index = _index(tmp_path, source_sha256="pdf-bytes", page_count=4)
    graph = build_document_graph(
        "d.pdf",
        ["one", "two", "three", "four"],
        source_sha256="pdf-bytes",
        extraction_version="graph-v1",
    )
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"doc_id": "d.pdf", "question": "Q", "answer": "A"}]))

    class _Embedder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode_questions(self, _questions):
            return [np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)]

    monkeypatch.setattr(stages, "_index_by_document", lambda _path: {"d.pdf": index})
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda *_args: tmp_path)
    monkeypatch.setattr(stages, "Qwen3VLPageEmbedder", _Embedder)
    monkeypatch.setattr(
        stages,
        "_graph_manifest_for_enabled_run",
        lambda *_args: ({"d.pdf": graph}, "manifest-fingerprint", None),
    )

    output = stages.retrieve_questions(
        config,
        samples,
        tmp_path / "index-manifest.json",
        tmp_path / "retrieval.jsonl",
        graph_manifest=tmp_path / "graph-manifest.json",
    )

    row = json.loads(output.read_text())
    assert [item["page_id"] for item in row["candidates"]] == [0, 1]
    assert row["graph"]["seed_page_ids"] == [0, 1]


def test_persisted_graph_candidates_reject_lists_over_graph_cap():
    """Catches tampered retrieval artifacts bypassing the reranking cap."""

    row = {
        "doc_id": "d.pdf",
        "candidates": [
            {"page_id": page_id, "score": 1.0 - page_id / 10, "rank": page_id + 1, "stage": "screen"}
            for page_id in range(3)
        ],
    }

    with pytest.raises(ValueError, match="graph candidate cap"):
        stages._persisted_candidates(row, "d.pdf", max_candidates=2)


def test_rerank_rejects_oversized_graph_fallback_candidate_list(
    tmp_path: Path, monkeypatch
):
    """Catches fallback metadata being used to bypass the graph cap at reranking."""

    config = HarnessConfig.model_validate(
        {
            "model": {"model_id": "m", "revision": "r"},
            "retrieval": {
                "enabled": True,
                "rerank_enabled": False,
                "candidate_k": 3,
                "selected_k": 1,
            },
            "graph": {"enabled": True, "max_candidates": 2},
        }
    )
    index = _index(tmp_path, source_sha256="pdf-bytes", page_count=3)
    graph = build_document_graph(
        "d.pdf", ["one", "two", "three"], "pdf-bytes", "graph-v1"
    )
    retrieval_path = tmp_path / "retrieval.jsonl"
    retrieval_path.write_text(
        json.dumps(
            {
                "doc_id": "d.pdf",
                "question": "Q",
                "candidates": [
                    {
                        "page_id": page_id,
                        "score": 1.0 - page_id / 10,
                        "rank": page_id + 1,
                        "stage": "screen",
                    }
                    for page_id in range(3)
                ],
                "graph": {"eligible": False, "cache_status": "fallback"},
            }
        )
        + "\n"
    )
    monkeypatch.setattr(stages, "_index_by_document", lambda _path: {"d.pdf": index})
    monkeypatch.setattr(
        stages,
        "_graph_manifest_for_enabled_run",
        lambda *_args: ({"d.pdf": graph}, "manifest-fingerprint", None),
    )

    with pytest.raises(ValueError, match="graph candidate cap"):
        stages.rerank_questions(
            config,
            retrieval_path,
            tmp_path / "index-manifest.json",
            tmp_path / "reranked",
            graph_manifest=tmp_path / "graph-manifest.json",
        )


def test_graph_construction_identity_mismatch_falls_back_with_metadata(
    tmp_path: Path, monkeypatch
):
    """Catches incompatible graph construction being used instead of ordinary retrieval."""

    config = HarnessConfig.model_validate(
        {
            "model": {"model_id": "m", "revision": "r"},
            "retrieval": {"enabled": True, "candidate_k": 2, "selected_k": 1},
            "graph": {
                "enabled": True,
                "max_candidates": 1,
                "extraction_version": "graph-v1",
            },
        }
    )
    index = _index(tmp_path, source_sha256="pdf-bytes", page_count=2)
    graph = build_document_graph(
        "d.pdf", ["one", "two"], source_sha256="pdf-bytes", extraction_version="graph-v1"
    )
    graph_path = tmp_path / "d.pdf.graph.json"
    write_graph(graph, graph_path)
    graph_manifest = tmp_path / "graph-manifest.json"
    graph_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "graph",
                "config_hash": "full-build-config-hash",
                "construction_identity": "wrong-construction-identity",
                "entries": [
                    {
                        "document_id": graph.document_id,
                        "source_sha256": graph.source_sha256,
                        "schema_version": graph.schema_version,
                        "extraction_version": graph.extraction_version,
                        "graph_fingerprint": graph_fingerprint(graph),
                        "page_count": 2,
                        "graph_path": str(graph_path),
                    }
                ],
            }
        )
    )
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{"doc_id": "d.pdf", "question": "Q", "answer": "A"}]))

    class _Embedder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode_questions(self, _questions):
            return [np.array([1.0, 0.0], dtype=np.float32)]

    monkeypatch.setattr(stages, "_index_by_document", lambda _path: {"d.pdf": index})
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda *_args: tmp_path)
    monkeypatch.setattr(stages, "Qwen3VLPageEmbedder", _Embedder)

    output = stages.retrieve_questions(
        config,
        samples,
        tmp_path / "index-manifest.json",
        tmp_path / "retrieval.jsonl",
        graph_manifest=graph_manifest,
    )

    row = json.loads(output.read_text())
    assert [item["page_id"] for item in row["candidates"]] == [0]
    assert row["graph"]["eligible"] is False
    assert row["graph"]["cache_status"] == "fallback"
    assert row["graph"]["terminal_reason"] == "graph_failure"
    assert "construction identity" in row["graph"]["failure"]


def _write_config(path: Path, *, graph_enabled: bool) -> Path:
    path.write_text(
        '[model]\nmodel_id = "test"\nrevision = "rev"\n'
        + ("[graph]\nenabled = true\n" if graph_enabled else "")
    )
    return path


def _index(tmp_path: Path, *, source_sha256: str, page_count: int = 2) -> PageIndex:
    pages = [
        Page(
            document_id="d.pdf",
            page_id=page_id,
            image_path=str(tmp_path / f"page-{page_id}.png"),
            width=1,
            height=1,
            render_sha256=f"render-{page_id}",
        )
        for page_id in range(page_count)
    ]
    return PageIndex(
        pages,
        np.eye(page_count, dtype=np.float32),
        identity=IndexIdentity(document_id="d.pdf", pdf_sha256=source_sha256),
    )
