"""Phased execution helpers for the four-model document harness.

The phase boundaries in this module are deliberately file based.  Embedding,
reranking, OCR, and answer generation each load one heavyweight checkpoint,
write a validated artifact, and release the model before the next phase starts.
That keeps the 24 GB GPU requirement explicit and makes every intermediate
decision inspectable or resumable.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .batch import BASELINE_PROMPT, load_v2_samples
from ..core.config import HarnessConfig
from ..core.contracts import (
    DraftAnswer,
    ModelRequest,
    Prediction,
    RankedPage,
    RunRecord,
    SafeQuestion,
    StageFailure,
    Status,
)
from ..core.answers import QuestionOutcome
from ..core.answers import AnswerDraftV2, VerificationReportV2
from ..documents.evidence import EvidenceBudget, EvidenceBundle, build_bundle
from ..documents.graph import (
    DocumentGraph,
    build_document_graph,
    graph_fingerprint,
    read_graph,
    write_graph,
)
from ..documents.graph_retrieval import expand_graph_candidates, select_connected_pages
from ..evaluation.manifests import RunManifest, sha256_file
from ..documents.ocr import OCRParsedPage, QianfanOCRParser, parse_cached
from .planning import SearchState, expand_evidence, plan_question
from ..core.protocol import export_v2_predictions, normalize_question
from ..documents.rendering import render_pdf, resize_page_for_budget
from ..models.reranking import (
    SelectionEntry,
    SelectionManifest,
    QwenPageReranker,
    read_selection_manifest,
    select_page_manifest,
    write_selection_manifest,
)
from ..models.retrieval import (
    IndexIdentity,
    PageIndex,
    Qwen3VLPageEmbedder,
)
from ..models.runner import QwenTransformersRunner
from ..models.structured import RawAttempt, StructuredCall
from ..prompts.registry import prompt_inventory, render_prompt
from .verification import verify_answer, verify_v2
from .checkpoints import commit_question, load_committed, sample_key


SCHEMA_VERSION = 1


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def required_ocr_pages(windows: Mapping[str, Sequence[Any]]) -> list[Any]:
    """Return unique document/page assets requested by current evidence windows."""

    result: dict[tuple[str, int], Any] = {}
    for window in windows.values():
        for page in window:
            key = (str(page.document_id), int(page.page_id))
            existing = result.get(key)
            if existing is not None and getattr(existing, "render_sha256", None) != getattr(page, "render_sha256", None):
                raise ValueError(f"page identity has conflicting renders: {key}")
            result[key] = page
    return [result[key] for key in sorted(result)]


def outcome_for_generation(
    *,
    response: str | None,
    generation_failed: bool,
    verification_abstained: bool,
    reason: str = "",
) -> QuestionOutcome:
    """Map a final stage state to an operationally explicit outcome."""

    reason = reason[:480]
    if generation_failed:
        return QuestionOutcome(kind="failed", answer=None, reason=reason or "structured generation failed")
    if verification_abstained:
        return QuestionOutcome(kind="search_exhausted", answer=None, reason=reason or "evidence search exhausted")
    if response is None or response == "Not answerable":
        return QuestionOutcome(kind="unanswerable", answer=None, reason=reason or "document evidence is insufficient")
    return QuestionOutcome(kind="answered", answer=response, reason=reason or "answer generated")


def prompt_set_for(config: HarnessConfig, role: str) -> str:
    """Resolve a role-specific prompt override and keep it in run identity."""

    return config.prompts.overrides.get(role, config.prompts.set)


def _legacy_verification_from_v2(report: VerificationReportV2) -> Any:
    """Adapt the semantic verifier result to the historical stage contract."""

    from ..core.contracts import EvidenceSpan, Verification, VerifyDecision

    decision = VerifyDecision.accept if report.verdict == "supported" else VerifyDecision.abstain
    evidence = [
        EvidenceSpan(
            page_id=item.page_id,
            kind=item.kind,
            quote=item.quote,
            locator=item.locator,
        )
        for item in report.evidence
    ]
    return Verification(
        decision=decision,
        final_answer=report.final_answer if decision is VerifyDecision.accept else None,
        evidence=evidence if decision is VerifyDecision.accept else [],
        reason=report.reason,
    )


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def resolve_checkpoint(model_id: str, models_dir: Path = Path("models")) -> Path:
    """Resolve a pinned Hugging Face ID to a downloaded local checkpoint."""

    candidate = Path(model_id)
    if candidate.is_dir():
        return candidate
    local = Path(models_dir) / model_id.rsplit("/", 1)[-1]
    if local.is_dir():
        return local
    raise FileNotFoundError(
        f"checkpoint {model_id!r} is not present at {local}; download the pinned model first"
    )


def _release_model(model: Any) -> None:
    """Best-effort release between phase workers."""

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def _render_document(pdf_path: Path, render_dir: Path, config: HarnessConfig) -> list[Any]:
    """Render one PDF with a content-addressed, settings-aware manifest."""

    document_dir = Path(render_dir) / pdf_path.name
    manifest_path = document_dir / "manifest.json"
    source = {
        "sha256": sha256_file(pdf_path),
        "dpi": config.render.dpi,
        "max_pixels": config.render.max_pixels,
    }
    if manifest_path.is_file():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached.get("source") == source:
                from ..core.contracts import Page

                pages = [Page.model_validate(item) for item in cached.get("pages", [])]
                if pages and all(Path(page.image_path).is_file() for page in pages):
                    return pages
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    pages = render_pdf(pdf_path, document_dir, dpi=config.render.dpi)
    _write_json(manifest_path, {"source": source, "pages": [page.model_dump(mode="json") for page in pages]})
    return pages


def _document_paths(documents_dir: Path, document_ids: Sequence[str] | None = None) -> list[Path]:
    if document_ids is None:
        paths = sorted(Path(documents_dir).glob("*.pdf"))
    else:
        paths = [Path(documents_dir) / document_id for document_id in document_ids]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    if not paths:
        raise ValueError(f"no PDF documents found in {documents_dir}")
    return paths


def build_graphs(
    config: HarnessConfig,
    documents_dir: Path,
    output_dir: Path,
    document_ids: Sequence[str] | None = None,
) -> Path:
    """Extract PDF text locally and persist one deterministic graph per document."""

    try:
        import pymupdf as fitz
    except ImportError as exc:
        raise RuntimeError("graph construction requires PyMuPDF; install the project's pdf extra") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for pdf_path in _document_paths(documents_dir, document_ids):
        document = fitz.open(pdf_path)
        try:
            page_texts = [page.get_text("text") for page in document]
        finally:
            document.close()
        source_sha256 = sha256_file(pdf_path)
        graph = build_document_graph(
            pdf_path.name,
            page_texts,
            source_sha256=source_sha256,
            extraction_version=config.graph.extraction_version,
        )
        graph_path = output_dir / f"{pdf_path.name}.graph.json"
        write_graph(graph, graph_path)
        entries.append(
            {
                "document_id": graph.document_id,
                "source_sha256": graph.source_sha256,
                "schema_version": graph.schema_version,
                "extraction_version": graph.extraction_version,
                "graph_fingerprint": graph_fingerprint(graph),
                "page_count": len(page_texts),
                "graph_path": str(graph_path.resolve()),
            }
        )
    manifest_path = output_dir / "graph-manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "stage": "graph",
            "config_hash": config.effective_hash(),
            "entries": entries,
        },
    )
    return manifest_path


def load_graph_manifest(path: Path) -> dict[str, DocumentGraph]:
    """Load graphs only when every manifest and artifact identity agrees."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid graph manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported graph manifest schema")
    if payload.get("stage") != "graph":
        raise ValueError("unsupported graph manifest")
    if not isinstance(payload.get("config_hash"), str) or not payload["config_hash"]:
        raise ValueError("graph manifest config hash is required")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("graph manifest entries must be a list")

    graphs: dict[str, DocumentGraph] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid graph manifest entry")
        document_id = entry.get("document_id")
        source_sha256 = entry.get("source_sha256")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("graph manifest document ID is required")
        if document_id in graphs:
            raise ValueError(f"duplicate graph document: {document_id}")
        if not isinstance(source_sha256, str) or not source_sha256:
            raise ValueError("graph manifest source hash is required")
        if entry.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("graph manifest graph schema identity does not match")
        graph_path = entry.get("graph_path")
        if not isinstance(graph_path, str) or not graph_path:
            raise ValueError("graph manifest graph path is required")
        resolved_graph_path = Path(graph_path)
        if not resolved_graph_path.is_absolute():
            resolved_graph_path = Path(path).parent / resolved_graph_path
        graph = read_graph(resolved_graph_path, expected_source_sha256=source_sha256)
        if graph.document_id != document_id:
            raise ValueError("graph manifest document identity does not match graph")
        if graph.extraction_version != entry.get("extraction_version"):
            raise ValueError("graph manifest extraction identity does not match graph")
        if graph_fingerprint(graph) != entry.get("graph_fingerprint"):
            raise ValueError("graph manifest fingerprint does not match graph")
        if len([node for node in graph.nodes if node.kind == "page"]) != entry.get("page_count"):
            raise ValueError("graph manifest page count does not match graph")
        graphs[document_id] = graph
    return graphs


def build_indexes(
    config: HarnessConfig,
    documents_dir: Path,
    render_dir: Path,
    output_dir: Path,
    *,
    models_dir: Path = Path("models"),
    document_ids: Sequence[str] | None = None,
) -> Path:
    """Render and embed each document, persisting one exact page index per PDF."""

    if not config.retrieval.enabled:
        raise ValueError("retrieval must be enabled to build page indexes")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _document_paths(documents_dir, document_ids)
    checkpoint = resolve_checkpoint(config.retrieval.embedding_model_id, models_dir)
    retrieval_instruction = render_prompt("retrieval", prompt_set_for(config, "retrieval"), {})
    embedder = Qwen3VLPageEmbedder(
        checkpoint,
        model_revision=config.retrieval.embedding_revision,
        model_id=config.retrieval.embedding_model_id,
        batch_size=config.retrieval.embedding_batch_size,
        max_pixels=config.render.max_pixels,
        instruction=retrieval_instruction,
    )
    entries: list[dict[str, Any]] = []
    try:
        for pdf_path in paths:
            pages = _render_document(pdf_path, render_dir, config)
            vectors = embedder.encode_pages(list(pages))
            identity = IndexIdentity.from_inputs(
                pdf_sha256=sha256_file(pdf_path),
                document_id=pdf_path.name,
                render_settings=config.render.model_dump(mode="json"),
                model_id=config.retrieval.embedding_model_id,
                model_revision=config.retrieval.embedding_revision,
                embedding_instruction=embedder.instruction,
            )
            index_path = output_dir / f"{pdf_path.name}.npz"
            PageIndex(pages, vectors, identity=identity).save(index_path)
            entries.append(
                {
                    "document_id": pdf_path.name,
                    "pdf_sha256": identity.pdf_sha256,
                    "index_path": str(index_path),
                    "page_count": len(pages),
                }
            )
    finally:
        _release_model(embedder)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "embedding",
        "config_hash": config.effective_hash(),
        "model_id": config.retrieval.embedding_model_id,
        "model_revision": config.retrieval.embedding_revision,
        "entries": entries,
    }
    manifest_path = output_dir / "index-manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _load_index_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid index manifest: {path}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("stage") != "embedding":
        raise ValueError("unsupported index manifest")
    if not isinstance(payload.get("entries"), list):
        raise ValueError("index manifest entries must be a list")
    return payload


def _index_by_document(path: Path) -> dict[str, PageIndex]:
    manifest = _load_index_manifest(path)
    result: dict[str, PageIndex] = {}
    for entry in manifest["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("document_id"), str):
            raise ValueError("invalid index manifest entry")
        document_id = entry["document_id"]
        if document_id in result:
            raise ValueError(f"duplicate index document: {document_id}")
        result[document_id] = PageIndex.load(Path(entry["index_path"]))
    return result


def _graph_manifest_for_enabled_run(
    config: HarnessConfig, graph_manifest: Path | None
) -> tuple[dict[str, DocumentGraph] | None, str | None, str | None]:
    """Load a graph artifact while retaining the documented fallback behavior."""

    if not config.graph.enabled:
        return None, None, None
    if graph_manifest is None:
        raise ValueError("graph manifest is required when graph retrieval is enabled")
    graph_manifest = Path(graph_manifest)
    if not graph_manifest.is_file():
        raise ValueError(f"graph manifest does not exist: {graph_manifest}")
    fingerprint = sha256_file(graph_manifest)
    try:
        return load_graph_manifest(graph_manifest), fingerprint, None
    except Exception as exc:
        return None, fingerprint, f"{type(exc).__name__}: {exc}"


def _graph_eligibility_error(
    config: HarnessConfig, index: PageIndex, graph: DocumentGraph
) -> str | None:
    """Return the reason a graph cannot safely serve a page index."""

    if graph.document_id != index.identity.document_id:
        return "graph document ID does not match page index"
    if graph.source_sha256 != index.identity.pdf_sha256:
        return "graph source hash does not match page index"
    if graph.extraction_version != config.graph.extraction_version:
        return "graph extraction version does not match configuration"
    return None


def _graph_relation_types(
    graph: DocumentGraph,
    candidate_page_ids: Sequence[int],
    allowed_relations: Sequence[str],
) -> list[str]:
    """Return the deterministic relation labels available inside a page route."""

    candidate_ids = set(candidate_page_ids)
    allowed = set(allowed_relations)
    nodes = {node.node_id: node for node in graph.nodes}
    relations = {
        edge.relation
        for edge in graph.edges
        if edge.relation in allowed
        and set(nodes[edge.source_id].page_ids).intersection(candidate_ids)
        and set(nodes[edge.target_id].page_ids).intersection(candidate_ids)
    }
    return sorted(relations)


def _selected_graph_path(graph: DocumentGraph, selected_page_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Serialize only provenance-bearing edges between the selected pages."""

    selected_ids = set(selected_page_ids)
    nodes = {node.node_id: node for node in graph.nodes}
    path: list[dict[str, Any]] = []
    for edge in graph.edges:
        source = nodes[edge.source_id]
        target = nodes[edge.target_id]
        if not set(source.page_ids).intersection(selected_ids):
            continue
        if not set(target.page_ids).intersection(selected_ids):
            continue
        path.append(
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation,
                "page_ids": edge.page_ids,
                "quote": edge.quote,
                "source_locator": edge.source_locator,
            }
        )
    return path


def _persisted_candidates(row: Mapping[str, Any], document_id: str) -> list[RankedPage]:
    """Validate stored retrieval candidates before handing them to reranking."""

    raw_candidates = row.get("candidates")
    if not isinstance(raw_candidates, list):
        raise ValueError("retrieval artifact candidates must be a list")
    candidates: list[RankedPage] = []
    seen: set[int] = set()
    for raw in raw_candidates:
        try:
            candidate = RankedPage.model_validate(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("retrieval artifact contains an invalid candidate") from exc
        if candidate.page_id in seen:
            raise ValueError(f"retrieval artifact contains duplicate page: {candidate.page_id}")
        if not math.isfinite(candidate.score):
            raise ValueError("retrieval artifact candidate scores must be finite")
        seen.add(candidate.page_id)
        candidates.append(candidate)
    if str(row.get("doc_id")) != document_id:
        raise ValueError("retrieval artifact document identity does not match row")
    return candidates


def _ocr_page_pool(manifest: SelectionManifest, *, max_pages: int) -> list[int]:
    """Keep selected evidence first before filling a bounded OCR pool."""

    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    ordered = [entry.page_id for entry in manifest.selected]
    ordered.extend(entry.page_id for entry in manifest.candidates)
    return list(dict.fromkeys(ordered))[:max_pages]


def _verification_ocr_page_pool(
    manifest: SelectionManifest,
    index: PageIndex,
    *,
    global_scope: bool,
    graph_enabled: bool,
    max_pages: int,
) -> list[int]:
    """Choose OCR pages without changing the graph-disabled baseline order."""

    if not graph_enabled:
        if global_scope:
            return [page.page_id for page in index.pages[:max_pages]]
        return [entry.page_id for entry in manifest.candidates[:max_pages]]
    if global_scope:
        return list(
            dict.fromkeys(
                [entry.page_id for entry in manifest.selected]
                + [page.page_id for page in index.pages]
            )
        )[:max_pages]
    return _ocr_page_pool(manifest, max_pages=max_pages)


def retrieve_questions(
    config: HarnessConfig,
    samples_path: Path,
    index_manifest: Path,
    output_path: Path,
    *,
    models_dir: Path = Path("models"),
    limit: int | None = None,
    graph_manifest: Path | None = None,
) -> Path:
    """Encode safe questions and persist document-scoped candidate pages."""

    if not config.retrieval.enabled:
        raise ValueError("retrieval must be enabled for the retrieval phase")
    samples = load_v2_samples(samples_path)
    selected = samples if limit is None else samples[:limit]
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    indexes = _index_by_document(index_manifest)
    graphs, graph_manifest_fingerprint, graph_load_failure = _graph_manifest_for_enabled_run(
        config, graph_manifest
    )
    checkpoint = resolve_checkpoint(config.retrieval.embedding_model_id, models_dir)
    retrieval_instruction = render_prompt("retrieval", prompt_set_for(config, "retrieval"), {})
    embedder = Qwen3VLPageEmbedder(
        checkpoint,
        model_revision=config.retrieval.embedding_revision,
        model_id=config.retrieval.embedding_model_id,
        batch_size=config.retrieval.embedding_batch_size,
        max_pixels=config.render.max_pixels,
        instruction=retrieval_instruction,
    )
    rows: list[dict[str, Any]] = []
    try:
        for sample in selected:
            index = indexes.get(sample.doc_id)
            if index is None:
                raise FileNotFoundError(f"no page index for {sample.doc_id}")
            question = SafeQuestion(document_id=sample.doc_id, question=normalize_question(sample.question))
            vector = embedder.encode_questions([question])[0]
            candidates = index.search(
                vector,
                config.retrieval.candidate_k,
                document_id=sample.doc_id,
            )
            row: dict[str, Any] = {
                "doc_id": sample.doc_id,
                "question": question.question,
                "query_vector": vector.tolist(),
                "index_fingerprint": index.fingerprint,
                "candidates": [item.model_dump(mode="json") for item in candidates],
            }
            if config.graph.enabled:
                seed_ids = [item.page_id for item in candidates]
                graph_metadata: dict[str, Any] = {
                    "manifest_fingerprint": graph_manifest_fingerprint,
                    "seed_page_ids": seed_ids,
                    "expanded_page_ids": [],
                    "relation_types": [],
                    "cache_status": "loaded",
                    "eligible": True,
                    "terminal_reason": "no_new_graph_candidates",
                }
                graph = graphs.get(sample.doc_id) if graphs is not None else None
                try:
                    if graph_load_failure is not None:
                        raise ValueError(graph_load_failure)
                    if graph is None:
                        raise ValueError(f"no graph for document: {sample.doc_id}")
                    eligibility_error = _graph_eligibility_error(config, index, graph)
                    if eligibility_error is not None:
                        raise ValueError(eligibility_error)
                    expanded_ids = expand_graph_candidates(
                        seed_ids,
                        graph,
                        max_hops=config.graph.max_hops,
                        max_candidates=config.graph.max_candidates,
                        allowed_relations=config.graph.allowed_relations,
                    )
                    seed_id_set = set(seed_ids)
                    indexed_page_ids = {page.page_id for page in index.pages}
                    additions = [
                        page_id
                        for page_id in expanded_ids
                        if page_id not in seed_id_set and page_id in indexed_page_ids
                    ]
                    lowest_seed_score = min((item.score for item in candidates), default=0.0)
                    expanded_candidates = [
                        RankedPage(
                            page_id=page_id,
                            score=lowest_seed_score - float(offset),
                            rank=len(candidates) + offset,
                            stage="graph_expand",
                        )
                        for offset, page_id in enumerate(additions, start=1)
                    ]
                    candidates = [*candidates, *expanded_candidates]
                    row["candidates"] = [item.model_dump(mode="json") for item in candidates]
                    graph_metadata.update(
                        {
                            "graph_fingerprint": graph_fingerprint(graph),
                            "expanded_page_ids": additions,
                            "relation_types": _graph_relation_types(
                                graph,
                                [*seed_ids, *additions],
                                config.graph.allowed_relations,
                            ),
                            "terminal_reason": (
                                "expanded_graph_candidates"
                                if additions
                                else "no_new_graph_candidates"
                            ),
                        }
                    )
                except Exception as exc:
                    graph_metadata.update(
                        {
                            "cache_status": "fallback",
                            "eligible": False,
                            "terminal_reason": "graph_failure",
                            "failure": f"{type(exc).__name__}: {exc}",
                        }
                    )
                row["graph"] = graph_metadata
            rows.append(row)
    finally:
        _release_model(embedder)
    _write_jsonl(output_path, rows)
    return Path(output_path)


def rerank_questions(
    config: HarnessConfig,
    retrieval_path: Path,
    index_manifest: Path,
    output_dir: Path,
    *,
    models_dir: Path = Path("models"),
    graph_manifest: Path | None = None,
) -> Path:
    """Rerank persisted candidates and write one selection manifest per question."""

    if not config.retrieval.enabled:
        raise ValueError("retrieval must be enabled for the reranking phase")
    indexes = _index_by_document(index_manifest)
    graphs, graph_manifest_fingerprint, graph_load_failure = _graph_manifest_for_enabled_run(
        config, graph_manifest
    )
    reranker = None
    if config.retrieval.rerank_enabled:
        checkpoint = resolve_checkpoint(config.retrieval.reranker_model_id, models_dir)
        reranker = QwenPageReranker(
            checkpoint,
            model_revision=config.retrieval.reranker_revision,
            model_id=config.retrieval.reranker_model_id,
            instruction=render_prompt("reranking", prompt_set_for(config, "reranking"), {}),
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    try:
        for row in _read_jsonl(retrieval_path):
            doc_id = str(row["doc_id"])
            question = SafeQuestion(document_id=doc_id, question=str(row["question"]))
            index = indexes.get(doc_id)
            if index is None:
                raise FileNotFoundError(f"no page index for {doc_id}")
            persisted = _persisted_candidates(row, doc_id) if config.graph.enabled else []
            if reranker is not None:
                if config.graph.enabled:
                    manifest = select_page_manifest(
                        question,
                        index,
                        reranker,
                        candidate_k=config.graph.max_candidates,
                        selected_k=config.retrieval.selected_k,
                        retrieved=persisted,
                    )
                else:
                    manifest = select_page_manifest(
                        question,
                        index,
                        reranker,
                        candidate_k=config.retrieval.candidate_k,
                        selected_k=config.retrieval.selected_k,
                        query_vector=row.get("query_vector"),
                    )
            else:
                candidates = (
                    persisted
                    if config.graph.enabled
                    else index.search(
                        row.get("query_vector"),
                        config.retrieval.candidate_k,
                        document_id=doc_id,
                    )
                )
                entries = [
                    SelectionEntry(
                        page_id=item.page_id,
                        retrieval_score=item.score,
                        retrieval_rank=item.rank,
                        rerank_score=item.score,
                        rerank_rank=item.rank,
                    )
                    for item in candidates
                ]
                manifest = SelectionManifest(
                    document_id=doc_id,
                    question=question.question,
                    index_fingerprint=index.fingerprint,
                    candidate_k=(
                        config.graph.max_candidates
                        if config.graph.enabled
                        else config.retrieval.candidate_k
                    ),
                    selected_k=config.retrieval.selected_k,
                    candidates=entries,
                    selected=entries[: config.retrieval.selected_k],
                )
            graph_metadata: dict[str, Any] | None = None
            if config.graph.enabled:
                raw_graph_metadata = row.get("graph")
                graph_metadata = (
                    dict(raw_graph_metadata)
                    if isinstance(raw_graph_metadata, dict)
                    else {"eligible": True, "cache_status": "loaded"}
                )
                graph_metadata["manifest_fingerprint"] = graph_manifest_fingerprint
                if graph_metadata.get("eligible") is False:
                    graph_metadata.update(
                        {
                            "selected_page_ids": [item.page_id for item in manifest.selected],
                            "selected_path": graph_metadata.get("selected_path", []),
                        }
                    )
                else:
                    graph = graphs.get(doc_id) if graphs is not None else None
                    try:
                        if graph_load_failure is not None:
                            raise ValueError(graph_load_failure)
                        if graph is None:
                            raise ValueError(f"no graph for document: {doc_id}")
                        eligibility_error = _graph_eligibility_error(config, index, graph)
                        if eligibility_error is not None:
                            raise ValueError(eligibility_error)
                        connected = select_connected_pages(
                            [
                                RankedPage(
                                    page_id=item.page_id,
                                    score=item.rerank_score,
                                    rank=item.rerank_rank,
                                    stage="rerank",
                                )
                                for item in manifest.candidates
                            ],
                            graph,
                            selected_k=config.retrieval.selected_k,
                            require_connection=config.graph.require_connection,
                        )
                        entries_by_id = {entry.page_id: entry for entry in manifest.candidates}
                        manifest = manifest.model_copy(
                            update={"selected": [entries_by_id[item.page_id] for item in connected]}
                        )
                        selected_ids = [item.page_id for item in manifest.selected]
                        graph_metadata.update(
                            {
                                "eligible": True,
                                "selected_page_ids": selected_ids,
                                "selected_path": _selected_graph_path(graph, selected_ids),
                            }
                        )
                    except Exception as exc:
                        graph_metadata.update(
                            {
                                "cache_status": "fallback",
                                "eligible": False,
                                "terminal_reason": "graph_selection_failure",
                                "failure": f"{type(exc).__name__}: {exc}",
                                "selected_page_ids": [item.page_id for item in manifest.selected],
                                "selected_path": [],
                            }
                        )
            manifest_path = output_dir / f"{_canonical_hash([doc_id, question.question])}.json"
            write_selection_manifest(manifest, manifest_path)
            output_row: dict[str, Any] = {
                "doc_id": doc_id,
                "question": question.question,
                "manifest_path": str(manifest_path),
                "manifest_fingerprint": manifest.fingerprint,
            }
            if graph_metadata is not None:
                output_row["graph"] = graph_metadata
            rows.append(output_row)
    finally:
        _release_model(reranker)
    output_path = output_dir / "reranked.jsonl"
    _write_jsonl(output_path, rows)
    return output_path


def ocr_questions(
    config: HarnessConfig,
    reranked_path: Path,
    index_manifest: Path,
    output_dir: Path,
    *,
    models_dir: Path = Path("models"),
) -> Path:
    """OCR the union of selected pages once and attach validated page records."""

    if not config.ocr.enabled:
        raise ValueError("OCR must be enabled for the OCR phase")
    indexes = _index_by_document(index_manifest)
    checkpoint = resolve_checkpoint(config.ocr.model_id, models_dir)
    ocr_prompt = render_prompt("ocr", prompt_set_for(config, "ocr"), {})
    parser = QianfanOCRParser.from_pretrained(
        model_id=str(checkpoint),
        revision=config.ocr.revision,
        prompt_version=config.ocr.prompt_version,
        max_new_tokens=config.ocr.max_new_tokens,
        ocr_config_hash=_canonical_hash({"config": config.ocr.model_dump(mode="json"), "prompt": ocr_prompt}),
        prompt_text=ocr_prompt,
    )
    cache_dir = Path(config.paths.cache_dir) / "ocr"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = _read_jsonl(reranked_path)
    needed: dict[tuple[str, int], Any] = {}
    requested_by_row: list[tuple[dict[str, Any], Any, list[int]]] = []
    for row in source_rows:
        manifest = read_selection_manifest(Path(row["manifest_path"]))
        index = indexes.get(manifest.document_id)
        if index is None:
            raise FileNotFoundError(f"no page index for {manifest.document_id}")
        pages = {page.page_id: page for page in index.pages}
        plan = plan_question(SafeQuestion(document_id=manifest.document_id, question=manifest.question))
        if config.verification.enabled:
            page_ids = _verification_ocr_page_pool(
                manifest,
                index,
                global_scope=plan.scope == "global",
                graph_enabled=config.graph.enabled,
                max_pages=config.verification.max_pages,
            )
        else:
            page_ids = [entry.page_id for entry in manifest.selected]
        requested_by_row.append((row, manifest, page_ids))
        for page_id in page_ids:
            page = pages.get(page_id)
            if page is None:
                raise ValueError(f"selection references unknown page {page_id}")
            needed[(manifest.document_id, page.page_id)] = page
    parsed: dict[tuple[str, int], OCRParsedPage] = {}
    try:
        for key, page in needed.items():
            parsed[key] = parse_cached(
                page,
                parser,
                cache_dir,
                max_text_tokens=config.ocr.max_text_tokens,
            )
    finally:
        _release_model(parser)
    rows: list[dict[str, Any]] = []
    for row, manifest, page_ids in requested_by_row:
        page_rows = [
            parsed[(manifest.document_id, page_id)].model_dump(mode="json")
            for page_id in page_ids
            if (manifest.document_id, page_id) in parsed
        ]
        rows.append({**row, "parsed": page_rows, "available_page_ids": page_ids})
    output_path = output_dir / "ocr.jsonl"
    _write_jsonl(output_path, rows)
    return output_path


def _answer_prompt(bundle: EvidenceBundle, *, plan_scope: str, prompt_set: str = "control") -> str:
    evidence = []
    for page_id in bundle.included_page_ids:
        text = bundle.ocr_content.get(page_id, "")
        evidence.append(f"[page_id={page_id}]\n{text}")
    schema = json.dumps(DraftAnswer.model_json_schema(), ensure_ascii=False, sort_keys=True)
    rendered = render_prompt(
        "answer",
        prompt_set,
        {
            "question": bundle.question,
            "plan": plan_scope,
            "coverage": json.dumps(
                {"included_page_ids": bundle.included_page_ids, "omitted": [item.model_dump(mode="json") for item in bundle.omitted_items]},
                ensure_ascii=False,
            ),
            "evidence": "\n\n".join(evidence),
            "schema": schema,
        },
    )
    return render_prompt("system", prompt_set, {}) + "\n\n" + rendered


def answer_questions(
    config: HarnessConfig,
    samples_path: Path,
    ocr_path: Path,
    index_manifest: Path,
    run_dir: Path,
    *,
    models_dir: Path = Path("models"),
    resume: bool = False,
    retry_failed: bool = False,
    stop_after: int | None = None,
) -> Path:
    """Answer OCR-plus-visual bundles and persist predictions, records, and manifest."""

    indexes = _index_by_document(index_manifest)
    samples = load_v2_samples(samples_path)
    sample_by_key = {
        (sample.doc_id, normalize_question(sample.question)): sample for sample in samples
    }
    rows = _read_jsonl(ocr_path)
    selected_keys = [(str(row["doc_id"]), normalize_question(str(row["question"]))) for row in rows]
    unknown = [key for key in selected_keys if key not in sample_by_key]
    if unknown:
        raise ValueError(f"OCR artifact contains an unknown sample: {unknown[0]}")
    if stop_after is not None and stop_after <= 0:
        raise ValueError("stop_after must be positive")
    existing_predictions: dict[tuple[str, str], Prediction] = {}
    existing_records: list[RunRecord] = []
    committed = load_committed(run_dir) if resume else {}
    if resume and (Path(run_dir) / "predictions.json").exists():
        existing_payload = json.loads((Path(run_dir) / "predictions.json").read_text(encoding="utf-8"))
        for item in existing_payload:
            prediction = Prediction.model_validate(item)
            existing_predictions[(prediction.doc_id, normalize_question(prediction.question))] = prediction
    if resume and (Path(run_dir) / "records.jsonl").exists():
        existing_records = [RunRecord.model_validate(item) for item in _read_jsonl(Path(run_dir) / "records.jsonl")]
    pending_rows = []
    for row in rows:
        key = sample_key(str(row["doc_id"]), str(row["question"]))
        saved = committed.get(key)
        if saved is None:
            pending_rows.append(row)
            continue
        kind = str(saved.get("outcome", {}).get("kind", ""))
        if kind == "failed" and retry_failed:
            pending_rows.append(row)
    if resume and not pending_rows:
        destination = Path(run_dir) / "predictions.json"
        if existing_predictions:
            _write_json(destination, [item.model_dump(mode="json") for item in existing_predictions.values()])
        return destination
    rows = pending_rows if resume else rows
    checkpoint = resolve_checkpoint(config.model.model_id, models_dir)
    runner = QwenTransformersRunner.from_pretrained(
        str(checkpoint),
        config.model.revision,
        max_new_tokens=config.generation.max_new_tokens,
        do_sample=config.generation.do_sample,
        max_pixels=config.render.max_pixels,
    )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_key = dict(existing_predictions)
    records: list[RunRecord] = list(existing_records)
    finalized_this_call = 0
    try:
        for row in rows:
            if stop_after is not None and finalized_this_call >= stop_after:
                break
            started = time.perf_counter()
            doc_id = str(row["doc_id"])
            question_text = normalize_question(str(row["question"]))
            safe_question = SafeQuestion(document_id=doc_id, question=question_text)
            index = indexes[doc_id]
            pages_by_id = {page.page_id: page for page in index.pages}
            selection = read_selection_manifest(Path(row["manifest_path"]))
            selected_ids = [item.page_id for item in selection.selected]
            parsed = [OCRParsedPage.model_validate(item) for item in row.get("parsed", [])]
            plan = plan_question(safe_question)
            available_ids = [int(page_id) for page_id in row.get("available_page_ids", selected_ids)]
            parsed_by_id = {item.page_id: item for item in parsed}
            evidence_budget = EvidenceBudget(
                max_pages=config.evidence.max_pages,
                max_pixels_per_image=config.evidence.max_pixels_per_image,
                max_total_image_pixels=config.evidence.max_total_image_pixels,
                max_text_tokens=config.evidence.max_text_tokens,
                reserved_output_tokens=config.evidence.reserved_output_tokens,
            )
            focused_dir = (
                Path(config.paths.cache_dir)
                / "focused-pages"
                / str(evidence_budget.max_pixels_per_image)
            )
            focused_pages_by_id = {
                page_id: resize_page_for_budget(
                    pages_by_id[page_id],
                    evidence_budget.max_pixels_per_image,
                    focused_dir,
                )
                for page_id in available_ids
                if page_id in pages_by_id
            }

            def make_bundle(page_ids: Sequence[int]) -> EvidenceBundle:
                return build_bundle(
                    safe_question,
                    [
                        focused_pages_by_id[page_id]
                        for page_id in page_ids
                        if page_id in focused_pages_by_id
                    ],
                    [parsed_by_id[page_id] for page_id in page_ids if page_id in parsed_by_id],
                    evidence_budget,
                )

            def generate_for(bundle: EvidenceBundle, prompt_override: str | None = None):
                request = ModelRequest(
                    document_id=doc_id,
                    question=question_text,
                    page_ids=bundle.included_page_ids,
                    image_paths=[image.image_path for image in bundle.images],
                    prompt=prompt_override
                    or _answer_prompt(bundle, plan_scope=plan.scope, prompt_set=prompt_set_for(config, "answer")),
                    config_hash=config.effective_hash(),
                )
                return runner.generate(request)

            def verify_bundle(draft: Any, bundle: EvidenceBundle):
                if config.verification.mode != "model":
                    return verify_answer(safe_question, draft, bundle)
                from ..core.answers import EvidenceRefV2

                draft_v2 = AnswerDraftV2(
                    answer=draft.answer,
                    evidence=[
                        EvidenceRefV2(page_id=item.page_id, kind="text", quote=item.quote)
                        for item in draft.evidence
                    ],
                    insufficient_evidence=draft.insufficient_evidence,
                )

                def call_verifier(request: ModelRequest) -> StructuredCall:
                    try:
                        request_data = json.loads(request.prompt)
                    except json.JSONDecodeError as exc:
                        return StructuredCall(
                            attempts=[],
                            failure=StageFailure(
                                stage="verification",
                                error_type="JSONDecodeError",
                                message=str(exc),
                                retryable=False,
                            ),
                        )
                    evidence = "\n\n".join(
                        f"[page_id={page_id}]\n{bundle.ocr_content.get(page_id, '')}"
                        for page_id in bundle.included_page_ids
                    )
                    prompt = render_prompt(
                        "verification",
                        prompt_set_for(config, "verification"),
                        {
                            "question": request.question,
                            "draft": json.dumps(request_data.get("draft", {}), ensure_ascii=False),
                            "coverage": json.dumps(request_data.get("coverage", {}), ensure_ascii=False),
                            "evidence": evidence,
                            "schema": json.dumps(VerificationReportV2.model_json_schema(), ensure_ascii=False),
                        },
                    )
                    raw_result = runner.generate(
                        request.model_copy(
                            update={
                                "prompt": render_prompt("system", prompt_set_for(config, "verification"), {})
                                + "\n\n"
                                + prompt
                            }
                        )
                    )
                    attempt = RawAttempt(
                        raw_response=raw_result.raw_response,
                        finish_reason=raw_result.finish_reason,
                        input_tokens=raw_result.input_tokens,
                        output_tokens=raw_result.output_tokens,
                        failure=raw_result.failure,
                    )
                    if raw_result.finish_reason != "eos":
                        return StructuredCall(attempts=[attempt], failure=StageFailure(
                            stage="verification", error_type="IncompleteGeneration",
                            message=f"generation ended with {raw_result.finish_reason}", retryable=True,
                        ))
                    try:
                        text = raw_result.raw_response.strip()
                        if text.startswith("```"):
                            text = "\n".join(text.splitlines()[1:-1]).strip()
                        payload = json.loads(text)
                        report = VerificationReportV2.model_validate(payload)
                    except Exception as exc:
                        return StructuredCall(attempts=[attempt], failure=StageFailure(
                            stage="verification", error_type=type(exc).__name__,
                            message=str(exc), retryable=True,
                        ))
                    return StructuredCall(attempts=[attempt], payload=report.model_dump(mode="json"))

                report = verify_v2(
                    safe_question,
                    draft_v2,
                    bundle,
                    call=call_verifier,
                    coverage={
                        "included_page_ids": bundle.included_page_ids,
                        "complete": plan.scope != "global" or len(available_ids) >= len(index.pages),
                    },
                    config=config,
                )
                return _legacy_verification_from_v2(report)

            bundle = make_bundle(selected_ids)
            base_prompt = _answer_prompt(bundle, plan_scope=plan.scope, prompt_set=prompt_set_for(config, "answer"))
            generation = generate_for(bundle, base_prompt)
            generation_attempts = [generation]
            if (
                (generation.draft is None or generation.finish_reason != "eos")
                and config.generation.schema_repair_attempts
            ):
                repair_prompt = render_prompt(
                    "repair",
                    prompt_set_for(config, "repair"),
                    {
                        "validation_error": generation.failure.message
                        if generation.failure is not None
                        else f"generation ended with {generation.finish_reason}",
                        "draft": generation.raw_response[:4000],
                        "schema": json.dumps(DraftAnswer.model_json_schema(), ensure_ascii=False, sort_keys=True),
                    },
                )
                generation = generate_for(bundle, base_prompt + "\n\n" + repair_prompt)
                generation_attempts.append(generation)
            verification = None
            expansion_trace: list[dict[str, Any]] = []
            generation_usable = generation.draft is not None and generation.finish_reason == "eos"
            if generation_usable and config.verification.enabled:
                verification = verify_bundle(generation.draft, bundle)
                state = SearchState(
                    selected_page_ids=list(bundle.included_page_ids),
                    visited_page_ids=list(selected_ids),
                    candidate_page_ids=available_ids,
                    max_rounds=config.verification.max_expansion_rounds,
                    max_pages=config.verification.max_pages,
                    unresolved_evidence=verification.decision.value == "abstain",
                )
                while verification.decision.value == "abstain":
                    decision = expand_evidence(plan, state)
                    if decision.terminal:
                        expansion_trace.append(decision.model_dump(mode="json"))
                        break
                    retained = state.selected_page_ids[: config.verification.retained_pages]
                    capacity = max(0, evidence_budget.max_pages - len(retained))
                    additional = decision.additional_page_ids[:capacity]
                    if not additional:
                        expansion_trace.append(
                            {
                                **decision.model_dump(mode="json"),
                                "additional_page_ids": [],
                                "terminal": True,
                                "reason": "active_window_budget",
                            }
                        )
                        break
                    expansion_trace.append(
                        decision.model_copy(update={"additional_page_ids": additional}).model_dump(mode="json")
                    )
                    state = state.model_copy(
                        update={
                            "selected_page_ids": list(dict.fromkeys(retained + additional)),
                            "visited_page_ids": list(
                                dict.fromkeys(state.visited_page_ids + additional)
                            ),
                            "expansion_round": state.expansion_round + 1,
                            "unresolved_evidence": True,
                        }
                    )
                    bundle = make_bundle(state.selected_page_ids)
                    base_prompt = _answer_prompt(bundle, plan_scope=plan.scope, prompt_set=prompt_set_for(config, "answer"))
                    generation = generate_for(bundle, base_prompt)
                    generation_attempts.append(generation)
                    if (
                        (generation.draft is None or generation.finish_reason != "eos")
                        and config.generation.schema_repair_attempts
                    ):
                        repair_prompt = render_prompt(
                            "repair",
                            prompt_set_for(config, "repair"),
                            {
                                "validation_error": generation.failure.message
                                if generation.failure is not None
                                else f"generation ended with {generation.finish_reason}",
                                "draft": generation.raw_response[:4000],
                                "schema": json.dumps(DraftAnswer.model_json_schema(), ensure_ascii=False, sort_keys=True),
                            },
                        )
                        generation = generate_for(bundle, base_prompt + "\n\n" + repair_prompt)
                        generation_attempts.append(generation)
                    if generation.draft is None:
                        verification = None
                        break
                    verification = verify_bundle(generation.draft, bundle)
            if (
                verification is not None
                and plan.scope == "global"
                and len(available_ids) < len(index.pages)
                and verification.decision.value != "abstain"
            ):
                # A top-k candidate set cannot support an exhaustive count.  Keep
                # the draft in metadata but require a coverage-aware abstention.
                from ..core.contracts import Verification, VerifyDecision

                verification = Verification(
                    decision=VerifyDecision.abstain,
                    final_answer=None,
                    evidence=[],
                    reason="global coverage is incomplete within the page budget",
                )
            if not generation_usable:
                response = "Not answerable"
            elif verification is not None:
                response = verification.final_answer or "Not answerable"
            else:
                response = generation.draft.answer or "Not answerable"
            prediction = Prediction(doc_id=doc_id, question=question_text, response=response)
            predictions_by_key[(doc_id, question_text)] = prediction
            graph_metadata = row.get("graph") if config.graph.enabled else None
            record = RunRecord(
                    run_id=uuid.uuid4().hex,
                    document_id=doc_id,
                    question=question_text,
                    status=Status.failed if not generation_usable else Status.ok,
                    response=response,
                    raw_response=generation.raw_response,
                    parse_status=generation.parse_status,
                    finish_reason=generation.finish_reason,
                    input_tokens=generation.input_tokens,
                    output_tokens=generation.output_tokens,
                    config_hash=config.effective_hash(),
                    model_id=config.model.model_id,
                    model_revision=config.model.revision,
                    page_ids=bundle.included_page_ids,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    failure=generation.failure,
                    metadata={
                        "bundle_hash": bundle.content_hash,
                        "selection_manifest": selection.fingerprint,
                        "plan": plan.model_dump(mode="json"),
                        "verification": verification.model_dump(mode="json") if verification else None,
                        "expansion": expansion_trace,
                        "coverage_complete": (
                            plan.scope != "global" or len(available_ids) >= len(index.pages)
                        ),
                        "parsed_answer": generation.draft.answer if generation_usable and generation.draft else None,
                        "generation_attempts": [
                            {
                                "parse_status": attempt.parse_status,
                                "finish_reason": attempt.finish_reason,
                                "input_tokens": attempt.input_tokens,
                                "output_tokens": attempt.output_tokens,
                                "raw_response": attempt.raw_response,
                                "failure": attempt.failure.model_dump(mode="json") if attempt.failure else None,
                            }
                            for attempt in generation_attempts
                        ],
                        **(
                            {
                                "graph": graph_metadata,
                                "selected_graph_path": graph_metadata.get("selected_path", []),
                            }
                            if isinstance(graph_metadata, dict)
                            else {}
                        ),
                    },
                )
            records.append(record)
            outcome = outcome_for_generation(
                response=response,
                generation_failed=not generation_usable,
                verification_abstained=(verification is not None and verification.decision.value == "abstain"),
                reason=(
                    generation.failure.message
                    if generation.failure is not None
                    else (verification.reason if verification is not None else "")
                ),
            )
            commit_question(
                run_dir,
                sample_key(doc_id, question_text),
                {
                    "document_id": doc_id,
                    "question": question_text,
                    "outcome": outcome.model_dump(mode="json"),
                    "attempt_paths": [],
                    "record_run_id": record.run_id,
                },
                allow_replace=retry_failed,
            )
            finalized_this_call += 1
    except Exception as exc:
        # A failure is attached to the question that was being processed.  The
        # caller can rerun after fixing the phase artifact without fabricating an
        # answer for the failed sample.
        raise RuntimeError(f"answer phase failed: {type(exc).__name__}: {exc}") from exc
    finally:
        _release_model(runner)

    predictions = list(predictions_by_key.values())
    _write_json(run_dir / "predictions.json", [item.model_dump(mode="json") for item in predictions])
    _write_jsonl(run_dir / "records.jsonl", [item.model_dump(mode="json") for item in records])
    _write_json(
        run_dir / "selected-samples.json",
        [{"doc_id": item.doc_id, "question": item.question} for item in predictions],
    )
    return run_dir / "predictions.json"


def build_run_manifest(
    config: HarnessConfig,
    *,
    run_id: str,
    samples_path: Path,
    index_manifest: Path,
    prompt: str = BASELINE_PROMPT,
    graph_manifest: Path | None = None,
) -> RunManifest:
    """Create a resumable identity for a staged run."""

    code_files = sorted(Path("src/doc_harness").rglob("*.py"))
    code_hash = _canonical_hash({str(path): sha256_file(path) for path in code_files})
    models_file = Path("models.lock")
    dependencies = [
        path
        for path in (Path("pyproject.toml"), Path("uv.lock"), Path("environment.lock"))
        if path.is_file()
    ]
    prompt_identity = {
        "legacy_prompt": prompt,
        "selected_set": config.prompts.set,
        "overrides": config.prompts.overrides,
        "inventory": {
            role: prompt_inventory(prompt_set_for(config, role))
            for role in ("system", "planning", "retrieval", "reranking", "ocr", "answer", "verification", "repair", "synthesis")
        },
    }
    selected_keys_hash = _canonical_hash(
        [
            {"doc_id": sample.doc_id, "question": normalize_question(sample.question)}
            for sample in load_v2_samples(samples_path)
        ]
    )
    if config.graph.enabled:
        if graph_manifest is None:
            raise ValueError("graph manifest is required when graph retrieval is enabled")
        graph_manifest = Path(graph_manifest)
        if not graph_manifest.is_file():
            raise ValueError(f"graph manifest does not exist: {graph_manifest}")
        dataset_hash = _canonical_hash(
            {
                "index_manifest": sha256_file(index_manifest),
                "graph_manifest": sha256_file(graph_manifest),
            }
        )
    else:
        dataset_hash = sha256_file(index_manifest)
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        config_hash=config.effective_hash(),
        code_hash=code_hash,
        dataset_hash=dataset_hash,
        samples_hash=sha256_file(samples_path),
        models_hash=sha256_file(models_file) if models_file.is_file() else "unknown",
        prompts_hash=_canonical_hash(prompt_identity),
        dependencies_hash=_canonical_hash({str(path): sha256_file(path) for path in dependencies}),
        selected_keys_hash=selected_keys_hash,
    )


__all__ = [
    "answer_questions",
    "build_graphs",
    "build_indexes",
    "build_run_manifest",
    "ocr_questions",
    "load_graph_manifest",
    "resolve_checkpoint",
    "rerank_questions",
    "retrieve_questions",
]
