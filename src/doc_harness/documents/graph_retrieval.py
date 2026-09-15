"""Deterministic, bounded page retrieval over document graph projections."""

from __future__ import annotations

import math
from collections import deque
from typing import Sequence

from ..core.contracts import RankedPage
from .graph import DocumentGraph


def _check_limit(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _page_adjacency(
    graph: DocumentGraph, allowed_relations: set[str], *, directed: bool
) -> dict[int, list[int]]:
    """Project only edges whose two endpoint nodes carry page provenance."""

    nodes = {node.node_id: node for node in graph.nodes}
    adjacency: dict[int, list[int]] = {}
    for edge in graph.edges:
        if edge.relation not in allowed_relations:
            continue
        source_pages = nodes[edge.source_id].page_ids
        target_pages = nodes[edge.target_id].page_ids
        if not source_pages or not target_pages:
            continue
        for source_page in source_pages:
            neighbors = adjacency.setdefault(source_page, [])
            for target_page in target_pages:
                if target_page != source_page and target_page not in neighbors:
                    neighbors.append(target_page)
                if not directed and target_page != source_page:
                    reverse = adjacency.setdefault(target_page, [])
                    if source_page not in reverse:
                        reverse.append(source_page)
    return adjacency


def expand_graph_candidates(
    seed_page_ids: Sequence[int],
    graph: DocumentGraph,
    *,
    max_hops: int,
    max_candidates: int,
    allowed_relations: Sequence[str],
) -> list[int]:
    """Return seed-first, breadth-first page candidates under explicit caps."""

    hop_limit = _check_limit(max_hops, "max_hops", allow_zero=True)
    candidate_limit = _check_limit(max_candidates, "max_candidates")
    adjacency = _page_adjacency(graph, set(allowed_relations), directed=True)

    candidates: list[int] = []
    seen_pages: set[int] = set()
    queue: deque[tuple[int, int]] = deque()
    seen_states: set[tuple[int, int]] = set()
    for page_id in seed_page_ids:
        if isinstance(page_id, bool) or not isinstance(page_id, int) or page_id < 0:
            raise ValueError("seed_page_ids must contain non-negative integers")
        if page_id not in seen_pages:
            seen_pages.add(page_id)
            candidates.append(page_id)
            state = (page_id, 0)
            queue.append(state)
            seen_states.add(state)
            if len(candidates) == candidate_limit:
                return candidates

    while queue and len(candidates) < candidate_limit:
        page_id, hops = queue.popleft()
        if hops >= hop_limit:
            continue
        for neighbor in adjacency.get(page_id, []):
            next_state = (neighbor, hops + 1)
            if next_state not in seen_states:
                seen_states.add(next_state)
                queue.append(next_state)
            if neighbor in seen_pages:
                continue
            seen_pages.add(neighbor)
            candidates.append(neighbor)
            if len(candidates) == candidate_limit:
                break
    return candidates


def _ranked_order(ranked_pages: Sequence[RankedPage]) -> list[RankedPage]:
    """Sort scores reproducibly while retaining original model objects."""

    unique: dict[int, RankedPage] = {}
    for page in ranked_pages:
        if not isinstance(page, RankedPage):
            raise ValueError("ranked_pages must contain RankedPage values")
        if not math.isfinite(page.score):
            raise ValueError("ranked page scores must be finite")
        previous = unique.get(page.page_id)
        if previous is None or (-page.score, page.rank, page.page_id) < (
            -previous.score,
            previous.rank,
            previous.page_id,
        ):
            unique[page.page_id] = page
    return sorted(unique.values(), key=lambda page: (-page.score, page.rank, page.page_id))


def select_connected_pages(
    ranked_pages: Sequence[RankedPage],
    graph: DocumentGraph,
    *,
    selected_k: int,
    require_connection: bool,
) -> list[RankedPage]:
    """Select a connected evidence bundle, retaining score order when disabled."""

    limit = _check_limit(selected_k, "selected_k")
    ordered = _ranked_order(ranked_pages)
    if not require_connection:
        return ordered[:limit]
    if not ordered:
        return []

    adjacency = _page_adjacency(
        graph, {edge.relation for edge in graph.edges}, directed=False
    )
    selected = [ordered[0]]
    selected_ids = {ordered[0].page_id}
    while len(selected) < limit:
        connected = next(
            (
                page
                for page in ordered
                if page.page_id not in selected_ids
                and any(neighbor in selected_ids for neighbor in adjacency.get(page.page_id, []))
            ),
            None,
        )
        if connected is None:
            break
        selected.append(connected)
        selected_ids.add(connected.page_id)
    for page in ordered:
        if len(selected) == limit:
            break
        if page.page_id not in selected_ids:
            selected.append(page)
            selected_ids.add(page.page_id)
    return selected


__all__ = ["expand_graph_candidates", "select_connected_pages"]
