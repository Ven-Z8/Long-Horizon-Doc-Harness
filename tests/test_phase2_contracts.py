import pytest
from pydantic import ValidationError

from doc_harness.core.answers import (
    AnswerDraftV2,
    EvidenceRefV2,
    QuestionOutcome,
    VerificationReportV2,
    export_outcome,
)
from doc_harness.config import load_config


def test_visual_answer_does_not_require_invented_ocr_quote():
    ref = EvidenceRefV2(page_id=17, kind="visual", locator="blue bar in the right chart")
    draft = AnswerDraftV2(answer="12%", evidence=[ref], insufficient_evidence=False)
    assert draft.evidence[0].quote is None


def test_text_evidence_requires_a_quote():
    with pytest.raises(ValidationError):
        EvidenceRefV2(page_id=1, kind="text")


def test_failed_sample_is_not_projected_as_semantic_abstention():
    failed = QuestionOutcome(kind="failed", answer=None, reason="invalid JSON after repair")
    assert export_outcome(failed) is None


def test_supported_verification_requires_answer_and_evidence():
    with pytest.raises(ValidationError):
        VerificationReportV2(
            verdict="supported",
            final_answer=None,
            evidence=[],
            reason="supported",
        )


def test_unanswerable_verification_has_no_final_answer():
    report = VerificationReportV2(
        verdict="unanswerable",
        final_answer=None,
        evidence=[],
        reason="The supplied document states that the value is unavailable.",
    )
    assert report.final_answer is None


def test_phase2_config_exposes_prompt_and_role_budgets(tmp_path):
    config_path = tmp_path / "phase2.toml"
    config_path.write_text(
        '[model]\nmodel_id = "reasoner"\nrevision = "r"\n'
        '[prompts]\nset = "v2"\n'
        '[generation]\nmax_new_tokens = 1024\nschema_repair_attempts = 1\n'
        'planning_max_new_tokens = 512\nverification_max_new_tokens = 1024\n'
        '[planning]\nmode = "model"\n'
        '[retrieval]\nselected_k = 6\nrerank_enabled = true\n'
        '[verification]\nenabled = true\nmode = "model"\nretained_pages = 2\n'
        'max_synthesis_calls = 2\nmax_source_recheck_windows = 2\n'
    )
    config = load_config(config_path)
    assert config.prompts.set == "v2"
    assert config.generation.schema_repair_attempts == 1
    assert config.planning.mode == "model"
    assert config.verification.mode == "model"


def test_phase2_config_rejects_retained_pages_that_fill_context(tmp_path):
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        '[model]\nmodel_id = "reasoner"\nrevision = "r"\n'
        '[evidence]\nmax_pages = 6\n'
        '[verification]\nretained_pages = 6\n'
    )
    with pytest.raises(ValueError, match="retained_pages"):
        load_config(config_path)
