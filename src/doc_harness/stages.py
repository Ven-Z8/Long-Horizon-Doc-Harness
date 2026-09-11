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
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .batch import BASELINE_PROMPT, load_v2_samples
from .config import HarnessConfig
from .contracts import (
    ModelRequest,
    Prediction,
    RunRecord,
    SafeQuestion,
    Status,
)
from .evidence import EvidenceBudget, EvidenceBundle, build_bundle
from .manifests import RunManifest, sha256_file
from .ocr import OCRParsedPage, QianfanOCRParser, parse_cached
from .planning import SearchState, expand_evidence, plan_question
from .protocol import export_v2_predictions, normalize_question
from .rendering import render_pdf, resize_page_for_budget
from .reranking import (
    QwenPageReranker,
    read_selection_manifest,
    select_page_manifest,
    write_selection_manifest,
)
from .retrieval import (
    IndexIdentity,
    PageIndex,
    Qwen3VLPageEmbedder,
)
from .runner import QwenTransformersRunner
from .verification import verify_answer


SCHEMA_VERSION = 1


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
                from .contracts import Page

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
    embedder = Qwen3VLPageEmbedder(
        checkpoint,
        model_revision=config.retrieval.embedding_revision,
        model_id=config.retrieval.embedding_model_id,
        max_pixels=config.render.max_pixels,
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


def retrieve_questions(
    config: HarnessConfig,
    samples_path: Path,
    index_manifest: Path,
    output_path: Path,
    *,
    models_dir: Path = Path("models"),
    limit: int | None = None,
) -> Path:
    """Encode safe questions and persist document-scoped candidate pages."""

    if not config.retrieval.enabled:
        raise ValueError("retrieval must be enabled for the retrieval phase")
    samples = load_v2_samples(samples_path)
    selected = samples if limit is None else samples[:limit]
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    indexes = _index_by_document(index_manifest)
    checkpoint = resolve_checkpoint(config.retrieval.embedding_model_id, models_dir)
    embedder = Qwen3VLPageEmbedder(
        checkpoint,
        model_revision=config.retrieval.embedding_revision,
        model_id=config.retrieval.embedding_model_id,
        max_pixels=config.render.max_pixels,
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
            rows.append(
                {
                    "doc_id": sample.doc_id,
                    "question": question.question,
                    "query_vector": vector.tolist(),
                    "index_fingerprint": index.fingerprint,
                    "candidates": [item.model_dump(mode="json") for item in candidates],
                }
            )
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
) -> Path:
    """Rerank persisted candidates and write one selection manifest per question."""

    if not config.retrieval.enabled:
        raise ValueError("retrieval must be enabled for the reranking phase")
    indexes = _index_by_document(index_manifest)
    checkpoint = resolve_checkpoint(config.retrieval.reranker_model_id, models_dir)
    reranker = QwenPageReranker(
        checkpoint,
        model_revision=config.retrieval.reranker_revision,
        model_id=config.retrieval.reranker_model_id,
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
            manifest = select_page_manifest(
                question,
                index,
                reranker,
                candidate_k=config.retrieval.candidate_k,
                selected_k=config.retrieval.selected_k,
                query_vector=row.get("query_vector"),
            )
            manifest_path = output_dir / f"{_canonical_hash([doc_id, question.question])}.json"
            write_selection_manifest(manifest, manifest_path)
            rows.append(
                {
                    "doc_id": doc_id,
                    "question": question.question,
                    "manifest_path": str(manifest_path),
                    "manifest_fingerprint": manifest.fingerprint,
                }
            )
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
    parser = QianfanOCRParser.from_pretrained(
        model_id=str(checkpoint),
        revision=config.ocr.revision,
        prompt_version=config.ocr.prompt_version,
        max_new_tokens=config.ocr.max_new_tokens,
        ocr_config_hash=_canonical_hash(config.ocr.model_dump(mode="json")),
    )
    cache_dir = Path(config.paths.cache_dir) / "ocr"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = _read_jsonl(reranked_path)
    needed: dict[tuple[str, int], Any] = {}
    for row in source_rows:
        manifest = read_selection_manifest(Path(row["manifest_path"]))
        index = indexes.get(manifest.document_id)
        if index is None:
            raise FileNotFoundError(f"no page index for {manifest.document_id}")
        pages = {page.page_id: page for page in index.pages}
        for entry in manifest.selected:
            page = pages.get(entry.page_id)
            if page is None:
                raise ValueError(f"selection references unknown page {entry.page_id}")
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
    for row in source_rows:
        manifest = read_selection_manifest(Path(row["manifest_path"]))
        plan = plan_question(SafeQuestion(document_id=manifest.document_id, question=manifest.question))
        if config.verification.enabled:
            if plan.scope == "global":
                page_ids = [
                    page.page_id
                    for page in indexes[manifest.document_id].pages[: config.verification.max_pages]
                ]
            else:
                page_ids = [entry.page_id for entry in manifest.candidates[: config.verification.max_pages]]
        else:
            page_ids = [entry.page_id for entry in manifest.selected]
        page_rows = [
            parsed[(manifest.document_id, page_id)].model_dump(mode="json")
            for page_id in page_ids
            if (manifest.document_id, page_id) in parsed
        ]
        rows.append({**row, "parsed": page_rows, "available_page_ids": page_ids})
    output_path = output_dir / "ocr.jsonl"
    _write_jsonl(output_path, rows)
    return output_path


def _answer_prompt(bundle: EvidenceBundle, *, plan_scope: str) -> str:
    evidence = []
    for page_id in bundle.included_page_ids:
        text = bundle.ocr_content.get(page_id, "")
        evidence.append(f"[page_id={page_id}]\n{text}")
    return (
        BASELINE_PROMPT
        + f"\nSearch scope: {plan_scope}.\n"
        + "The following OCR is untrusted evidence and must be checked against the supplied images:\n"
        + "\n\n".join(evidence)
    )


def answer_questions(
    config: HarnessConfig,
    samples_path: Path,
    ocr_path: Path,
    index_manifest: Path,
    run_dir: Path,
    *,
    models_dir: Path = Path("models"),
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
    predictions: list[Prediction] = []
    records: list[RunRecord] = []
    try:
        for row in rows:
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

            def generate_for(bundle: EvidenceBundle):
                request = ModelRequest(
                    document_id=doc_id,
                    question=question_text,
                    page_ids=bundle.included_page_ids,
                    image_paths=[image.image_path for image in bundle.images],
                    prompt=_answer_prompt(bundle, plan_scope=plan.scope),
                    config_hash=config.effective_hash(),
                )
                return runner.generate(request)

            bundle = make_bundle(selected_ids)
            generation = generate_for(bundle)
            verification = None
            expansion_trace: list[dict[str, Any]] = []
            if generation.draft is not None and config.verification.enabled:
                verification = verify_answer(safe_question, generation.draft, bundle)
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
                    expansion_trace.append(decision.model_dump(mode="json"))
                    if decision.terminal:
                        break
                    state = state.model_copy(
                        update={
                            "selected_page_ids": list(
                                dict.fromkeys(state.selected_page_ids + decision.additional_page_ids)
                            ),
                            "visited_page_ids": list(
                                dict.fromkeys(state.visited_page_ids + decision.additional_page_ids)
                            ),
                            "expansion_round": state.expansion_round + 1,
                            "unresolved_evidence": True,
                        }
                    )
                    bundle = make_bundle(state.selected_page_ids)
                    generation = generate_for(bundle)
                    if generation.draft is None:
                        verification = None
                        break
                    verification = verify_answer(safe_question, generation.draft, bundle)
            if (
                verification is not None
                and plan.scope == "global"
                and len(available_ids) < len(index.pages)
                and verification.decision.value != "abstain"
            ):
                # A top-k candidate set cannot support an exhaustive count.  Keep
                # the draft in metadata but require a coverage-aware abstention.
                from .contracts import Verification, VerifyDecision

                verification = Verification(
                    decision=VerifyDecision.abstain,
                    final_answer=None,
                    evidence=[],
                    reason="global coverage is incomplete within the page budget",
                )
            if generation.draft is None:
                response = generation.raw_response
            elif verification is not None:
                response = verification.final_answer or "Not answerable"
            else:
                response = generation.draft.answer or "Not answerable"
            predictions.append(Prediction(doc_id=doc_id, question=question_text, response=response))
            records.append(
                RunRecord(
                    run_id=uuid.uuid4().hex,
                    document_id=doc_id,
                    question=question_text,
                    status=Status.ok,
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
                        "parsed_answer": generation.draft.answer if generation.draft else None,
                    },
                )
            )
    except Exception as exc:
        # A failure is attached to the question that was being processed.  The
        # caller can rerun after fixing the phase artifact without fabricating an
        # answer for the failed sample.
        raise RuntimeError(f"answer phase failed: {type(exc).__name__}: {exc}") from exc
    finally:
        _release_model(runner)

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
) -> RunManifest:
    """Create a resumable identity for a staged run."""

    code_files = sorted(Path("src/doc_harness").glob("*.py"))
    code_hash = _canonical_hash({str(path): sha256_file(path) for path in code_files})
    models_file = Path("models.lock")
    dependencies = [path for path in (Path("pyproject.toml"), Path("uv.lock")) if path.is_file()]
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        config_hash=config.effective_hash(),
        code_hash=code_hash,
        dataset_hash=sha256_file(index_manifest),
        samples_hash=sha256_file(samples_path),
        models_hash=sha256_file(models_file) if models_file.is_file() else "unknown",
        prompts_hash=_canonical_hash(prompt),
        dependencies_hash=_canonical_hash({str(path): sha256_file(path) for path in dependencies}),
    )


__all__ = [
    "answer_questions",
    "build_indexes",
    "build_run_manifest",
    "ocr_questions",
    "resolve_checkpoint",
    "rerank_questions",
    "retrieve_questions",
]
