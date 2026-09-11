from pathlib import Path

from doc_harness.config import load_config, lock_hash


def test_load_config_preserves_baseline_protocol(tmp_path: Path):
    config_path = tmp_path / "baseline.toml"
    config_path.write_text(
        '[model]\nmodel_id = "Qwen/Qwen3.5-4B"\nrevision = "rev"\n'
        '[render]\ndpi = 144\npage_policy = "explicit"\n'
        '[generation]\nmax_new_tokens = 128\ndo_sample = false\n'
    )
    config = load_config(config_path)
    assert config.model.model_id == "Qwen/Qwen3.5-4B"
    assert config.render.dpi == 144
    assert config.generation.do_sample is False
    assert len(config.effective_hash()) == 64


def test_lock_hash_changes_when_a_lock_changes(tmp_path: Path):
    lock = tmp_path / "lock"
    lock.write_text("one")
    first = lock_hash([lock])
    lock.write_text("two")
    assert lock_hash([lock]) != first
