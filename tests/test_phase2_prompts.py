import pytest

from doc_harness.prompts.registry import (
    PromptResourceError,
    load_prompt,
    prompt_inventory,
    render_prompt,
)
from doc_harness.config import HarnessConfig
from doc_harness.workflow.stages import prompt_set_for


def test_question_content_is_not_a_second_template():
    text = render_prompt(
        "planning",
        "v2",
        {"question": "What does $schema mean in this document?", "schema": '{"type":"object"}'},
    )
    assert "What does $schema mean in this document?" in text
    assert len(prompt_inventory("v2")["planning.txt"]) == 64


def test_missing_template_field_is_rejected():
    with pytest.raises(PromptResourceError, match="missing fields"):
        render_prompt("planning", "v2", {"question": "Q"})


def test_prompt_sets_have_all_stage_resources():
    expected = {
        "system.txt",
        "planning.txt",
        "retrieval.txt",
        "reranking.txt",
        "ocr.txt",
        "answer.txt",
        "verification.txt",
        "repair.txt",
        "synthesis.txt",
    }
    for prompt_set in ("control", "v2"):
        assert set(prompt_inventory(prompt_set)) == expected
        assert load_prompt("system", prompt_set)


def test_role_prompt_override_is_resolved_explicitly():
    config = HarnessConfig.model_validate({
        "model": {"model_id": "m", "revision": "r"},
        "prompts": {"set": "control", "overrides": {"answer": "v2"}},
    })
    assert prompt_set_for(config, "answer") == "v2"
    assert prompt_set_for(config, "ocr") == "control"
