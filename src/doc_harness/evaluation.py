"""Offline metrics for retrieval evidence coverage.

This module intentionally accepts benchmark labels only at evaluation time.
Retrieval and reranking code consume ``SafeQuestion`` values and never call
these helpers while selecting pages.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field

from .contracts import BenchmarkSample, StrictModel
from .protocol import normalize_question


_RECALL_KS = (4, 8, 16, 20)


def _as_page_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    if not isinstance(value, Iterable) or isinstance(value, (bytes, Mapping)):
        return []
    result: list[int] = []
    for page_id in value:
        if isinstance(page_id, bool):
            continue
        try:
            result.append(int(page_id))
        except (TypeError, ValueError):
            continue
    return result


def convert_evidence_pages(evidence_pages: Any, *, evidence_page_base: int = 0) -> list[int]:
    """Convert reviewed benchmark page labels to zero-based internal IDs.

    The base is explicit because MMLongBench annotations contain both PDF
    page labels and internal-style zero values in different reviewed cases.
    This function does not infer a base from the data.
    """

    if evidence_page_base not in (0, 1):
        raise ValueError("evidence_page_base must be 0 or 1")
    converted = [page_id - evidence_page_base for page_id in _as_page_list(evidence_pages)]
    if any(page_id < 0 for page_id in converted):
        raise ValueError("evidence page labels cannot convert to negative page IDs")
    return list(dict.fromkeys(converted))


def _sample_key(sample: BenchmarkSample) -> tuple[str, str]:
    return sample.doc_id, normalize_question(sample.question)


def _selected_pages(
    selected: Mapping[tuple[str, str], Sequence[int]], sample: BenchmarkSample
) -> list[int]:
    raw = selected.get(_sample_key(sample), ())
    seen: set[int] = set()
    result: list[int] = []
    for page_id in raw:
        if isinstance(page_id, bool):
            continue
        try:
            page_id = int(page_id)
        except (TypeError, ValueError):
            continue
        if page_id not in seen:
            seen.add(page_id)
            result.append(page_id)
    return result


def _coverage_metrics(
    rows: Sequence[tuple[set[int], list[int]]],
    *,
    include_hit_rates: bool = True,
) -> dict[str, Any]:
    nonempty = [(evidence, pages) for evidence, pages in rows if evidence]
    empty_count = len(rows) - len(nonempty)
    recalls = [len(evidence.intersection(pages)) / len(evidence) for evidence, pages in nonempty]
    complete = [evidence.issubset(pages) for evidence, pages in nonempty]
    result: dict[str, Any] = {
        "num_samples": len(rows),
        "num_with_evidence": len(nonempty),
        "empty_evidence": empty_count,
        "mean_evidence_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "complete_evidence_recall": sum(complete) / len(complete) if complete else 0.0,
    }
    # Short aliases make the report convenient to consume while retaining the
    # explicit names used in the plan.
    result["evidence_recall"] = result["mean_evidence_recall"]
    result["complete_recall"] = result["complete_evidence_recall"]
    if include_hit_rates:
        hit_rates: dict[str, float] = {}
        for k in _RECALL_KS:
            hits = [bool(evidence.intersection(pages[:k])) for evidence, pages in nonempty]
            hit_rates[str(k)] = sum(hits) / len(hits) if hits else 0.0
        result["hit_rate_at_k"] = hit_rates
    return result


def retrieval_metrics(
    selected: Mapping[tuple[str, str], Sequence[int]],
    labelled_samples: Sequence[BenchmarkSample],
    *,
    evidence_page_base: int = 0,
    page_base: int | None = None,
) -> dict[str, Any]:
    """Measure evidence recall without using labels during page selection.

    ``selected`` should contain internal zero-based IDs.  Set
    ``evidence_page_base=1`` for a reviewed fixture whose labels are printed
    one-based PDF pages.  Missing selection keys are scored as empty sets.
    Empty-evidence samples are excluded from recall denominators and reported
    via ``empty_evidence``.
    """

    if page_base is not None:
        if evidence_page_base != 0 and evidence_page_base != page_base:
            raise ValueError("page_base and evidence_page_base disagree")
        evidence_page_base = page_base
    rows: list[tuple[set[int], list[int]]] = []
    single_rows: list[tuple[set[int], list[int]]] = []
    cross_rows: list[tuple[set[int], list[int]]] = []
    for sample in labelled_samples:
        evidence = set(
            convert_evidence_pages(
                sample.evidence_pages, evidence_page_base=evidence_page_base
            )
        )
        pages = _selected_pages(selected, sample)
        row = (evidence, pages)
        rows.append(row)
        if len(evidence) == 1:
            single_rows.append(row)
        elif len(evidence) > 1:
            cross_rows.append(row)
    result = _coverage_metrics(rows)
    result["single_page"] = _coverage_metrics(single_rows)
    result["cross_page"] = _coverage_metrics(cross_rows)
    result["missing_selection"] = sum(
        1 for evidence, pages in rows if evidence and not pages
    )
    return result


__all__ = ["convert_evidence_pages", "retrieval_metrics"]
