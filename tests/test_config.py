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


def test_stage_config_records_four_model_roles_and_budgets(tmp_path: Path):
    config_path = tmp_path / "stages.toml"
    config_path.write_text(
        '[model]\nmodel_id = "reasoner"\nrevision = "r"\n'
        '[retrieval]\nenabled = true\ncandidate_k = 20\nselected_k = 6\n'
        '[ocr]\nenabled = true\nmax_text_tokens = 12000\n'
        '[evidence]\nmax_pages = 6\nmax_total_image_pixels = 6291456\n'
        '[verification]\nenabled = true\nmax_expansion_rounds = 2\n'
    )
    config = load_config(config_path)
    assert config.retrieval.candidate_k == 20
    assert config.ocr.max_text_tokens == 12000
    assert config.evidence.max_pages == 6
    assert config.verification.max_expansion_rounds == 2
