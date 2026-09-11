"""Composable staged pipeline for retrieval, OCR, answering, and verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from pydantic import Field

from .config import HarnessConfig
from .contracts import (
    DraftAnswer,
    Page,
    RankedPage,
    SafeQuestion,
    StageFailure,
    StrictModel,
    Verification,
    VerifyDecision,
)


@dataclass(frozen=True)
class PipelineServices:
    pages_for: Callable[[str], Sequence[Page]]
    select_pages: Callable[[SafeQuestion, Sequence[Page]], Sequence[Page | RankedPage | int]]
    parse_page: Callable[[Page], Any]
    build_bundle: Callable[[SafeQuestion, Sequence[Page], Sequence[Any], HarnessConfig], Any]
    answer: Callable[[SafeQuestion, Any], DraftAnswer]
    verify: Callable[[SafeQuestion, DraftAnswer, Any], Verification]


class PipelineResult(StrictModel):
    question: SafeQuestion
    selected_page_ids: list[int]
    draft: DraftAnswer | None = None
    verification: Verification | None = None
    final_response: str
    bundle_hash: str | None = None
    stage_attempts: dict[str, int]
    failures: list[StageFailure]
    metadata: dict[str, Any] = Field(default_factory=dict)


def _failure(stage: str, exc: Exception, retryable: bool = False) -> StageFailure:
    return StageFailure(
        stage=stage,
        error_type=type(exc).__name__,
        message=str(exc),
        retryable=retryable,
    )


def _selected_pages(selection: Sequence[Page | RankedPage | int], pages: Sequence[Page]) -> list[Page]:
    by_id = {page.page_id: page for page in pages}
    selected: list[Page] = []
    seen: set[int] = set()
    for item in selection:
        page_id = item.page_id if isinstance(item, (Page, RankedPage)) else int(item)
        if page_id in seen:
            continue
        page = by_id.get(page_id)
        if page is None:
            raise ValueError(f"selected page is unavailable: {page_id}")
        selected.append(page)
        seen.add(page_id)
    return selected


def _result(
    question: SafeQuestion,
    failures: list[StageFailure],
    attempts: dict[str, int],
    selected: Sequence[Page] = (),
    **kwargs: Any,
) -> PipelineResult:
    return PipelineResult(
        question=question,
        selected_page_ids=[page.page_id for page in selected],
        stage_attempts=attempts,
        failures=failures,
        final_response=kwargs.pop("final_response", ""),
        **kwargs,
    )


def run_pipeline(
    question: SafeQuestion,
    services: PipelineServices,
    config: HarnessConfig,
) -> PipelineResult:
    """Execute each stage once and return a typed trace on any failure."""

    failures: list[StageFailure] = []
    attempts = {"retrieve": 0, "ocr": 0, "bundle": 0, "answer": 0, "verify": 0}
    attempts["retrieve"] += 1
    try:
        pages = list(services.pages_for(question.document_id))
        if not pages:
            raise ValueError("document has no pages")
        selected = _selected_pages(services.select_pages(question, pages), pages)
        if not selected:
            raise ValueError("page selector returned no pages")
    except Exception as exc:
        failures.append(_failure("retrieve", exc))
        return _result(question, failures, attempts)

    parsed: list[Any] = []
    for page in selected:
        attempts["ocr"] += 1
        try:
            parsed.append(services.parse_page(page))
        except Exception as exc:
            failures.append(_failure("ocr", exc, retryable=True))
    if len(parsed) != len(selected):
        return _result(question, failures, attempts, selected)

    attempts["bundle"] += 1
    try:
        bundle = services.build_bundle(question, selected, parsed, config)
    except Exception as exc:
        failures.append(_failure("bundle", exc))
        return _result(question, failures, attempts, selected)

    attempts["answer"] += 1
    try:
        draft = services.answer(question, bundle)
    except Exception as exc:
        failures.append(_failure("answer", exc, retryable=True))
        return _result(
            question,
            failures,
            attempts,
            selected,
            bundle_hash=getattr(bundle, "content_hash", None),
        )

    attempts["verify"] += 1
    try:
        verification = services.verify(question, draft, bundle)
    except Exception as exc:
        failures.append(_failure("verify", exc, retryable=True))
        return _result(
            question,
            failures,
            attempts,
            selected,
            draft=draft,
            bundle_hash=getattr(bundle, "content_hash", None),
        )
    final_response = (
        verification.final_answer
        if verification.decision is not VerifyDecision.abstain
        and verification.final_answer is not None
        else "Not answerable"
    )
    return _result(
        question,
        failures,
        attempts,
        selected,
        draft=draft,
        verification=verification,
        final_response=final_response,
        bundle_hash=getattr(bundle, "content_hash", None),
    )
