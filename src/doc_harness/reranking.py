"""Page reranking and deterministic selection manifests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from pydantic import Field

from .contracts import Page, RankedPage, SafeQuestion, StrictModel
from .retrieval import PageIndex, _check_top_k


_SELECTION_SCHEMA_VERSION = 1


class SelectionEntry(StrictModel):
    """One candidate's independent screening and reranking scores."""

    page_id: int = Field(ge=0)
    retrieval_score: float
    retrieval_rank: int = Field(ge=1)
    rerank_score: float
    rerank_rank: int = Field(ge=1)


class SelectionManifest(StrictModel):
    """Serializable page selection for the answer worker."""

    schema_version: int = _SELECTION_SCHEMA_VERSION
    document_id: str
    question: str
    index_fingerprint: str
    candidate_k: int = Field(gt=0)
    selected_k: int = Field(gt=0)
    candidates: list[SelectionEntry] = Field(default_factory=list)
    selected: list[SelectionEntry] = Field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@runtime_checkable
class PageReranker(Protocol):
    """Protocol implemented by cross-encoder rerankers and test fakes."""

    def rank(self, question: SafeQuestion, pages: list[Page]) -> list[RankedPage]:
        ...


def _finite_score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("reranker scores must be finite numbers")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("reranker scores must be finite numbers") from exc
    if not math.isfinite(score):
        raise ValueError("reranker scores must be finite numbers")
    return score


class QwenPageReranker:
    """Adapter for the bundled Qwen3-VL yes/no relevance scorer."""

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        instruction: str = "Retrieve images or text relevant to the user's query.",
        model_revision: str = "",
        model_id: str | None = None,
        backend: Any | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.instruction = str(instruction)
        self.model_revision = str(model_revision)
        self.model_id = str(model_id if model_id is not None else model_name_or_path)
        self._backend = backend
        if self._backend is None:
            self._backend = self._load_backend(model_kwargs)

    def _load_backend(self, model_kwargs: dict[str, Any]) -> Any:
        model_path = Path(self.model_name_or_path)
        script_path = model_path / "scripts" / "qwen3_vl_reranker.py"
        if not script_path.is_file():
            raise RuntimeError(
                "the Qwen3-VL reranker adapter requires a local checkpoint with "
                "scripts/qwen3_vl_reranker.py"
            )
        module_name = "doc_harness_qwen3_vl_reranker"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load reranker adapter from {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Qwen3VLReranker(
            model_name_or_path=self.model_name_or_path,
            default_instruction=self.instruction,
            **model_kwargs,
        )

    def rank(self, question: SafeQuestion, pages: list[Page]) -> list[RankedPage]:
        if not pages:
            return []
        if any(page.document_id != question.document_id for page in pages):
            raise ValueError("reranker pages belong to a different document")
        missing = next(
            (page for page in pages if not Path(page.image_path).is_file()), None
        )
        if missing is not None:
            raise FileNotFoundError(f"page asset does not exist: {missing.image_path}")
        raw_scores = self._backend.process(
            {
                "instruction": self.instruction,
                "query": {"text": question.question},
                "documents": [{"image": page.image_path} for page in pages],
            }
        )
        if isinstance(raw_scores, (str, bytes)):
            raise ValueError("reranker returned a non-sequence of scores")
        try:
            raw_scores = list(raw_scores)
        except TypeError as exc:
            raise ValueError("reranker returned a non-sequence of scores") from exc
        if len(raw_scores) != len(pages):
            raise ValueError(
                f"reranker score count mismatch: expected {len(pages)}, got {len(raw_scores)}"
            )
        scored = [
            (page.page_id, _finite_score(score)) for page, score in zip(pages, raw_scores)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            RankedPage(page_id=page_id, score=score, rank=rank, stage="rerank")
            for rank, (page_id, score) in enumerate(scored, start=1)
        ]


def _validate_candidate_scores(
    ranked: Sequence[Any],
    candidate_pages: dict[int, Page],
    retrieval_by_id: dict[int, RankedPage],
) -> dict[int, float]:
    """Normalize fake and model outputs while rejecting unknown/invalid pages."""

    rerank_scores: dict[int, float] = {}
    for entry in ranked:
        if isinstance(entry, RankedPage):
            page_id = entry.page_id
            score = entry.score
        elif isinstance(entry, dict):
            try:
                page_id = int(entry["page_id"])
                score = entry["score"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("reranker returned an invalid ranked page") from exc
        else:
            raise ValueError("reranker must return RankedPage values")
        if page_id not in candidate_pages:
            raise ValueError(f"reranker returned unknown candidate page: {page_id}")
        # A duplicate candidate can occur when an adapter de-duplicates its
        # own output.  Retain the highest score and rank it once.
        score_value = _finite_score(score)
        previous = rerank_scores.get(page_id)
        if previous is None or score_value > previous:
            rerank_scores[page_id] = score_value
    if not rerank_scores:
        raise ValueError("reranker returned no candidate scores")
    missing = sorted(set(candidate_pages) - set(rerank_scores))
    if missing:
        raise ValueError(f"reranker omitted candidate page: {missing[0]}")
    return rerank_scores


def select_page_manifest(
    question: SafeQuestion,
    index: PageIndex,
    reranker: PageReranker,
    candidate_k: int = 20,
    selected_k: int = 6,
    *,
    query_vector: Any | None = None,
) -> SelectionManifest:
    """Retrieve candidates, rerank them, and persist both score spaces.

    ``query_vector`` is optional for compatibility with callers that already
    materialized a candidate list.  Supplying it enables exact cosine
    screening; when omitted, the index returns document-scoped candidates with
    neutral screening scores so a reranker can still be exercised in a
    separate phase.
    """

    candidate_limit = _check_top_k(candidate_k)
    selected_limit = _check_top_k(selected_k)
    if not isinstance(question, SafeQuestion):
        try:
            question = SafeQuestion.model_validate(question)
        except (TypeError, ValueError) as exc:
            raise ValueError("question must contain only document_id and question") from exc

    retrieved = index.search(
        query_vector,
        candidate_limit,
        document_id=question.document_id,
    )
    retrieval_by_id: dict[int, RankedPage] = {}
    for item in retrieved:
        if item.page_id in retrieval_by_id:
            continue
        score = _finite_score(item.score)
        retrieval_by_id[item.page_id] = RankedPage(
            page_id=item.page_id,
            score=score,
            rank=item.rank,
            stage="screen",
        )
    page_by_id = {
        page.page_id: page
        for page in index.pages
        if page.document_id == question.document_id
    }
    candidate_pages = {
        page_id: page_by_id[page_id]
        for page_id in retrieval_by_id
        if page_id in page_by_id
    }
    if len(candidate_pages) != len(retrieval_by_id):
        missing = sorted(set(retrieval_by_id) - set(candidate_pages))[0]
        raise ValueError(f"retrieval returned unknown page: {missing}")
    if not candidate_pages:
        return SelectionManifest(
            document_id=question.document_id,
            question=question.question,
            index_fingerprint=index.fingerprint,
            candidate_k=candidate_limit,
            selected_k=selected_limit,
        )
    missing_asset = next(
        (page for page in candidate_pages.values() if not Path(page.image_path).is_file()),
        None,
    )
    if missing_asset is not None:
        raise FileNotFoundError(f"page asset does not exist: {missing_asset.image_path}")

    ranked_output = reranker.rank(
        question,
        [candidate_pages[page_id] for page_id in sorted(candidate_pages)],
    )
    if isinstance(ranked_output, (str, bytes)):
        raise ValueError("reranker returned a non-sequence of ranked pages")
    if isinstance(ranked_output, Mapping):
        ranked_output = [
            {"page_id": page_id, "score": score}
            for page_id, score in ranked_output.items()
        ]
    else:
        try:
            ranked_output = list(ranked_output)
        except TypeError as exc:
            raise ValueError("reranker returned a non-sequence of ranked pages") from exc
    if ranked_output and all(
        not isinstance(item, (RankedPage, dict)) for item in ranked_output
    ):
        if len(ranked_output) != len(candidate_pages):
            raise ValueError(
                "reranker score count mismatch: "
                f"expected {len(candidate_pages)}, got {len(ranked_output)}"
            )
        ranked_output = [
            {"page_id": page_id, "score": score}
            for page_id, score in zip(sorted(candidate_pages), ranked_output)
        ]
    rerank_by_id = _validate_candidate_scores(
        ranked_output, candidate_pages, retrieval_by_id
    )
    ordered = sorted(
        rerank_by_id.items(),
        key=lambda item: (-item[1], item[0]),
    )
    entries: list[SelectionEntry] = []
    rerank_rank_by_id = {page_id: rank for rank, (page_id, _) in enumerate(ordered, 1)}
    for page_id, rerank_score in ordered:
        retrieval = retrieval_by_id[page_id]
        entries.append(
            SelectionEntry(
                page_id=page_id,
                retrieval_score=retrieval.score,
                retrieval_rank=retrieval.rank,
                rerank_score=rerank_score,
                rerank_rank=rerank_rank_by_id[page_id],
            )
        )
    return SelectionManifest(
        document_id=question.document_id,
        question=question.question,
        index_fingerprint=index.fingerprint,
        candidate_k=candidate_limit,
        selected_k=selected_limit,
        candidates=entries,
        selected=entries[: min(selected_limit, len(entries))],
    )


def select_pages(
    question: SafeQuestion,
    index: PageIndex,
    reranker: PageReranker,
    candidate_k: int = 20,
    selected_k: int = 6,
    *,
    query_vector: Any | None = None,
) -> list[RankedPage]:
    """Return selected pages as rerank-stage ``RankedPage`` values."""

    manifest = select_page_manifest(
        question,
        index,
        reranker,
        candidate_k,
        selected_k,
        query_vector=query_vector,
    )
    return [
        RankedPage(
            page_id=entry.page_id,
            score=entry.rerank_score,
            rank=entry.rerank_rank,
            stage="rerank",
        )
        for entry in manifest.selected
    ]


def write_selection_manifest(manifest: SelectionManifest, path: Path) -> None:
    """Write a validated selection manifest as deterministic JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def read_selection_manifest(path: Path) -> SelectionManifest:
    """Read and validate a selection manifest produced by this module."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SelectionManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid selection manifest: {path}") from exc


__all__ = [
    "PageReranker",
    "QwenPageReranker",
    "SelectionEntry",
    "SelectionManifest",
    "read_selection_manifest",
    "select_page_manifest",
    "select_pages",
    "write_selection_manifest",
]
