"""Question planning and bounded evidence expansion."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from .contracts import SafeQuestion, StrictModel


class QuestionPlan(StrictModel):
    scope: Literal["local", "global"]
    queries: list[str] = Field(min_length=1)
    required_operations: list[str] = Field(default_factory=list)


class SearchState(StrictModel):
    selected_page_ids: list[int] = Field(default_factory=list)
    visited_page_ids: list[int] = Field(default_factory=list)
    candidate_page_ids: list[int] = Field(default_factory=list)
    expansion_round: int = Field(default=0, ge=0)
    max_rounds: int = Field(default=2, ge=0)
    max_pages: int = Field(default=24, ge=1)
    unresolved_evidence: bool = False
    timed_out: bool = False


class ExpansionDecision(StrictModel):
    additional_page_ids: list[int] = Field(default_factory=list)
    terminal: bool
    reason: str = Field(min_length=1)


_GLOBAL_PATTERNS = (
    r"\bhow many\b",
    r"\bhow much\b",
    r"\bhow often\b",
    r"\b(all|every|each|throughout|among all|across the report)\b",
    r"\bnumber of\b",
    r"\bcount\b",
)


def plan_question(question: SafeQuestion) -> QuestionPlan:
    """Create a conservative plan from question text alone."""

    text = " ".join(question.question.split())
    lowered = text.lower()
    scope: Literal["local", "global"] = (
        "global" if any(re.search(pattern, lowered) for pattern in _GLOBAL_PATTERNS) else "local"
    )
    operations = ["locate", "read"]
    if scope == "global":
        operations = ["enumerate", "deduplicate", "count"]
    elif any(word in lowered for word in ("compare", "difference", "between")):
        operations.append("compare")
    return QuestionPlan(scope=scope, queries=[text], required_operations=operations)


def expand_evidence(plan: QuestionPlan, state: SearchState) -> ExpansionDecision:
    """Choose new candidate pages, respecting rounds, page count, and timeout."""

    if state.timed_out:
        return ExpansionDecision(additional_page_ids=[], terminal=True, reason="wall_time_budget")
    if not state.unresolved_evidence:
        return ExpansionDecision(additional_page_ids=[], terminal=True, reason="evidence_resolved")
    if state.expansion_round >= state.max_rounds:
        return ExpansionDecision(additional_page_ids=[], terminal=True, reason="expansion_round_budget")
    remaining_slots = state.max_pages - len(set(state.visited_page_ids))
    if remaining_slots <= 0:
        return ExpansionDecision(additional_page_ids=[], terminal=True, reason="page_budget")

    visited = set(state.visited_page_ids)
    candidates = [page_id for page_id in state.candidate_page_ids if page_id not in visited]
    if not candidates:
        return ExpansionDecision(additional_page_ids=[], terminal=True, reason="no_unvisited_candidates")
    additional = candidates[:remaining_slots]
    return ExpansionDecision(
        additional_page_ids=additional,
        terminal=False,
        reason=("global_coverage_expansion" if plan.scope == "global" else "unresolved_evidence_expansion"),
    )
