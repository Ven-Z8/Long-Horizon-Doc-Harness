import json
from pathlib import Path

from doc_harness.config import load_config
from doc_harness.stages import build_run_manifest, resolve_checkpoint


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
