import pytest

from doc_harness.contracts import SafeQuestion
from doc_harness.planning import (
    ExpansionDecision,
    QuestionPlan,
    SearchState,
    expand_evidence,
    plan_question,
)


def test_plan_detects_global_counting_questions():
    plan = plan_question(
        SafeQuestion(document_id="doc.pdf", question="How many charts are in this report?")
    )
    assert plan.scope == "global"
    assert "count" in plan.required_operations


def test_plan_keeps_named_page_lookup_local():
    plan = plan_question(
        SafeQuestion(document_id="doc.pdf", question="What color is the line in Figure 5?")
    )
    assert plan.scope == "local"


def test_expansion_is_bounded_and_does_not_repeat_pages():
    plan = QuestionPlan(scope="local", queries=["Q"], required_operations=["locate"])
    state = SearchState(
        selected_page_ids=[2],
        visited_page_ids=[1, 2],
        candidate_page_ids=[1, 2, 3, 4],
        expansion_round=0,
        max_rounds=2,
        max_pages=3,
        unresolved_evidence=True,
    )
    decision = expand_evidence(plan, state)
    assert decision.additional_page_ids == [3]
    assert decision.terminal is False


@pytest.mark.parametrize(
    "state",
    [
        SearchState(max_rounds=1, expansion_round=1, unresolved_evidence=True),
        SearchState(max_pages=1, visited_page_ids=[0], candidate_page_ids=[1], unresolved_evidence=True),
        SearchState(candidate_page_ids=[], unresolved_evidence=True),
        SearchState(candidate_page_ids=[1], unresolved_evidence=False),
    ],
)
def test_expansion_has_explicit_terminal_state(state):
    plan = QuestionPlan(scope="local", queries=["Q"], required_operations=["locate"])
    decision = expand_evidence(plan, state)
    assert isinstance(decision, ExpansionDecision)
    assert decision.terminal is True
    assert decision.additional_page_ids == []
