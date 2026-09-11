from doc_harness.core.contracts import SafeQuestion
from doc_harness.config import HarnessConfig
from doc_harness.models.structured import StructuredCall
from doc_harness.workflow.planning import plan_with_model
from doc_harness.workflow.planning import fallback_plan


def test_local_numeric_question_does_not_trigger_document_wide_count():
    q = SafeQuestion(document_id="d.pdf", question="How much was revenue in 2023?")
    assert fallback_plan(q).scope == "local"


def test_explicit_whole_document_question_requires_exhaustive_scan():
    q = SafeQuestion(document_id="d.pdf", question="How many diagrams appear throughout the document?")
    assert fallback_plan(q).scope == "exhaustive"


def test_comparison_question_is_cross_page():
    q = SafeQuestion(document_id="d.pdf", question="What is the difference between 2022 and 2023?")
    assert fallback_plan(q).scope == "cross_page"


def test_model_plan_preserves_original_query_and_caps_queries():
    question = SafeQuestion(document_id="d.pdf", question="What changed between 2022 and 2023?")
    config = HarnessConfig.model_validate({
        "model": {"model_id": "m", "revision": "r"},
        "planning": {"mode": "model", "max_queries": 3},
    })
    def generate(request):
        assert "evidence_pages" not in request.prompt
        return StructuredCall(payload={
            "scope": "cross_page",
            "queries": ["2022", "2023", "extra", "ignored"],
            "required_operations": ["locate", "compare"],
            "missing_evidence": [],
        })
    plan = plan_with_model(question, generate, config)
    assert plan.queries[0] == question.question
    assert len(plan.queries) == 3
