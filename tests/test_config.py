from pathlib import Path

import pytest

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


def test_graph_config_defaults_to_disabled_with_two_hops(tmp_path: Path):
    """Catches removal or regression of the opt-in graph baseline defaults."""

    config_path = tmp_path / "minimal.toml"
    config_path.write_text('[model]\nmodel_id = "reasoner"\nrevision = "r"\n')

    config = load_config(config_path)

    assert config.graph.enabled is False
    assert config.graph.max_hops == 2


@pytest.mark.parametrize(
    "graph_toml",
    [
        "[graph]\nmax_hops = 5\n",
        "[graph]\nnot_a_graph_option = true\n",
    ],
)
def test_graph_config_rejects_invalid_bounds_and_unknown_keys(tmp_path: Path, graph_toml: str):
    """Catches graph configurations that can exceed bounded retrieval or hide typos."""

    config_path = tmp_path / "invalid-graph.toml"
    config_path.write_text(
        '[model]\nmodel_id = "reasoner"\nrevision = "r"\n' + graph_toml
    )

    with pytest.raises(ValueError):
        load_config(config_path)
