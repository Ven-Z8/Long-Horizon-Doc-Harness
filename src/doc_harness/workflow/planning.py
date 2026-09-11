"""Question planning and bounded evidence expansion."""

from __future__ import annotations

import json
import re
from typing import Callable, Literal

from pydantic import Field, model_validator

from ..core.config import HarnessConfig
from ..core.contracts import SafeQuestion, StrictModel
from ..core.contracts import ModelRequest
from ..models.structured import StructuredCall
from ..prompts.registry import render_prompt


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


class WindowState(StrictModel):
    document_page_ids: list[int] = Field(default_factory=list)
    candidate_page_ids: list[int] = Field(default_factory=list)
    active_page_ids: list[int] = Field(default_factory=list)
    inspected_page_ids: list[int] = Field(default_factory=list)
    retained_page_ids: list[int] = Field(default_factory=list)
    round_index: int = Field(default=0, ge=0)
    max_rounds: int = Field(default=2, ge=0)
    max_unique_pages: int = Field(default=24, ge=1)
    requires_exhaustive_coverage: bool = False
    deadline_exceeded: bool = False

    @model_validator(mode="after")
    def validate_ids(self) -> "WindowState":
        document = set(self.document_page_ids)
        for name in ("candidate_page_ids", "active_page_ids", "inspected_page_ids", "retained_page_ids"):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique page IDs")
            if not set(values) <= document:
                raise ValueError(f"{name} contains a page outside document_page_ids")
        if self.round_index > self.max_rounds:
            raise ValueError("round_index cannot exceed max_rounds")
        return self


class WindowDecision(StrictModel):
    page_ids: list[int] = Field(default_factory=list)
    terminal: bool
    reason: str = Field(min_length=1)


def choose_window(
    state: WindowState, *, active_limit: int, retained_limit: int
) -> WindowDecision:
    """Choose a fresh bounded page window while retaining only trusted evidence."""

    if active_limit <= 0:
        raise ValueError("active_limit must be positive")
    if retained_limit < 0 or retained_limit > active_limit:
        raise ValueError("retained_limit must be between zero and active_limit")
    if state.deadline_exceeded:
        return WindowDecision(page_ids=[], terminal=True, reason="wall_time_budget")
    if state.round_index >= state.max_rounds:
        return WindowDecision(page_ids=[], terminal=True, reason="expansion_round_budget")

    inspected = set(state.inspected_page_ids)
    remaining = state.max_unique_pages - len(inspected)
    if remaining <= 0:
        return WindowDecision(page_ids=[], terminal=True, reason="unique_page_budget")

    retained = [page_id for page_id in state.retained_page_ids if page_id in set(state.document_page_ids)]
    retained = list(dict.fromkeys(retained))[:retained_limit]
    new_pages = [page_id for page_id in state.candidate_page_ids if page_id not in inspected]
    new_pages = new_pages[: max(0, min(active_limit - len(retained), remaining))]
    selected = list(dict.fromkeys(retained + new_pages))
    if not selected:
        return WindowDecision(page_ids=[], terminal=True, reason="no_unvisited_candidates")
    reason = "exhaustive_coverage_window" if state.requires_exhaustive_coverage else "unresolved_evidence_window"
    return WindowDecision(page_ids=selected, terminal=False, reason=reason)


def mark_inspected(
    state: WindowState, included_ids: list[int], *, completed: bool
) -> WindowState:
    """Count only pages that were successfully read by a completed window."""

    if not set(included_ids) <= set(state.document_page_ids):
        raise ValueError("included page is outside the document")
    if not completed:
        return state
    return state.model_copy(
        update={"inspected_page_ids": list(dict.fromkeys(state.inspected_page_ids + included_ids))}
    )


class QuestionPlanV2(StrictModel):
    scope: Literal["local", "cross_page", "exhaustive"]
    queries: list[str] = Field(min_length=1, max_length=3)
    required_operations: list[str] = Field(default_factory=list, max_length=6)
    missing_evidence: list[str] = Field(default_factory=list, max_length=4)


def fallback_plan(question: SafeQuestion) -> QuestionPlanV2:
    """Conservative safe-question-only fallback for a failed planner call."""

    text = " ".join(question.question.split())
    lowered = text.casefold()
    exhaustive_markers = (
        "throughout the document",
        "throughout the report",
        "entire document",
        "whole document",
        "all pages",
        "each page",
        "every page",
        "across the document",
    )
    cross_page = any(word in lowered for word in ("compare", "difference between", "respectively", "across"))
    if any(marker in lowered for marker in exhaustive_markers):
        scope: Literal["local", "cross_page", "exhaustive"] = "exhaustive"
        operations = ["enumerate", "deduplicate", "count"]
    elif cross_page:
        scope = "cross_page"
        operations = ["locate", "read", "compare"]
    else:
        scope = "local"
        operations = ["locate", "read"]
    return QuestionPlanV2(scope=scope, queries=[text], required_operations=operations)


def plan_with_model(
    question: SafeQuestion,
    generate: Callable[[ModelRequest], StructuredCall],
    config: HarnessConfig,
) -> QuestionPlanV2:
    """Generate a bounded safe-question-only plan, with a deterministic fallback."""

    fallback = fallback_plan(question)
    if config.planning.mode != "model":
        return fallback
    prompt = render_prompt(
        "planning",
        config.prompts.set,
        {
            "question": question.question,
            "schema": json.dumps(QuestionPlanV2.model_json_schema(), ensure_ascii=False),
        },
    )
    request = ModelRequest(
        document_id=question.document_id,
        question=question.question,
        page_ids=[],
        image_paths=[],
        prompt=prompt,
        config_hash=config.effective_hash(),
    )
    try:
        result = generate(request)
        if result.payload is None:
            return fallback
        payload = dict(result.payload)
        raw_queries = payload.get("queries", [])
        queries = [str(item).strip() for item in raw_queries if str(item).strip()]
        payload["queries"] = list(dict.fromkeys(queries))[: config.planning.max_queries]
        if not payload["queries"]:
            return fallback
        payload["queries"] = [question.question] + [item for item in payload["queries"] if item != question.question]
        payload["queries"] = payload["queries"][: config.planning.max_queries]
        payload["required_operations"] = [str(item).strip() for item in payload.get("required_operations", []) if str(item).strip()][:6]
        payload["missing_evidence"] = [str(item).strip() for item in payload.get("missing_evidence", []) if str(item).strip()][:4]
        return QuestionPlanV2.model_validate(payload)
    except Exception:
        return fallback


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
