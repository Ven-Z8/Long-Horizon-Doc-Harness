from doc_harness.workflow.planning import WindowState, choose_window, mark_inspected


def test_expansion_introduces_new_pages_inside_the_six_page_window():
    state = WindowState(
        document_page_ids=list(range(12)),
        candidate_page_ids=list(range(12)),
        active_page_ids=[0, 1, 2, 3, 4, 5],
        inspected_page_ids=[0, 1, 2, 3, 4, 5],
        retained_page_ids=[2, 4],
        round_index=1,
        max_rounds=2,
        max_unique_pages=24,
        requires_exhaustive_coverage=False,
        deadline_exceeded=False,
    )
    decision = choose_window(state, active_limit=6, retained_limit=2)
    assert decision.page_ids == [2, 4, 6, 7, 8, 9]
    assert not decision.terminal


def test_failed_window_is_not_marked_as_inspected():
    state = WindowState(
        document_page_ids=[0, 1], candidate_page_ids=[0, 1], active_page_ids=[0],
        inspected_page_ids=[], retained_page_ids=[], round_index=0, max_rounds=1,
        max_unique_pages=2, requires_exhaustive_coverage=False, deadline_exceeded=False,
    )
    assert mark_inspected(state, [0], completed=False).inspected_page_ids == []
    assert mark_inspected(state, [0], completed=True).inspected_page_ids == [0]


def test_exhaustive_window_uses_retrieved_pages_before_document_order():
    state = WindowState(
        document_page_ids=list(range(10)), candidate_page_ids=[8, 9, 0, 1, 2],
        active_page_ids=[8, 9], inspected_page_ids=[8, 9], retained_page_ids=[],
        round_index=1, max_rounds=2, max_unique_pages=10,
        requires_exhaustive_coverage=True, deadline_exceeded=False,
    )
    assert choose_window(state, active_limit=3, retained_limit=0).page_ids == [0, 1, 2]


def test_window_stops_when_round_budget_is_exhausted():
    state = WindowState(
        document_page_ids=[0, 1], candidate_page_ids=[0, 1], active_page_ids=[0],
        inspected_page_ids=[0], retained_page_ids=[], round_index=2, max_rounds=2,
        max_unique_pages=2, requires_exhaustive_coverage=False, deadline_exceeded=False,
    )
    decision = choose_window(state, active_limit=1, retained_limit=0)
    assert decision.terminal
    assert decision.reason == "expansion_round_budget"
