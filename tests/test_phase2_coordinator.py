from pathlib import Path

from doc_harness.core.config import HarnessConfig
from doc_harness.workflow import coordinator


def test_coordinator_runs_stages_in_order_and_freezes_manifest(tmp_path, monkeypatch):
    config = HarnessConfig.model_validate({"model": {"model_id": "m", "revision": "r"}})
    calls = []
    index = tmp_path / "index.json"
    index.write_text('{"schema_version":1,"stage":"embedding","entries":[]}')

    monkeypatch.setattr(coordinator, "build_indexes", lambda *args, **kwargs: calls.append("index") or index)
    monkeypatch.setattr(coordinator, "retrieve_questions", lambda *args, **kwargs: calls.append("retrieve") or tmp_path / "retrieve.jsonl")
    monkeypatch.setattr(coordinator, "rerank_questions", lambda *args, **kwargs: calls.append("rerank") or tmp_path / "rerank.jsonl")
    monkeypatch.setattr(coordinator, "ocr_questions", lambda *args, **kwargs: calls.append("ocr") or tmp_path / "ocr.jsonl")
    monkeypatch.setattr(coordinator, "answer_questions", lambda *args, **kwargs: calls.append("answer") or tmp_path / "predictions.json")
    monkeypatch.setattr(coordinator, "build_run_manifest", lambda *args, **kwargs: coordinator.RunManifest(
        schema_version=1, run_id="r", config_hash="c", code_hash="c", dataset_hash="d",
        samples_hash="s", models_hash="m", prompts_hash="p", dependencies_hash="dep"
    ))
    samples = tmp_path / "samples.json"
    samples.write_text("[]")
    run_dir = tmp_path / "run"
    result = coordinator.run_coordinator(config, samples, tmp_path, tmp_path / "render", run_dir)
    assert result == run_dir
    assert calls == ["index", "retrieve", "rerank", "ocr", "answer"]
    assert (run_dir / "manifest.json").exists()
