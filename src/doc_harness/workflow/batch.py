"""Resumable, document-grouped execution for MMLongBench-Doc V2."""

from __future__ import annotations

import ast
import json
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from ..core.config import HarnessConfig
from ..core.contracts import (
    BenchmarkSample,
    Prediction,
    RunRecord,
    StageFailure,
    Status,
)
from ..core.protocol import build_request, export_v2_predictions, normalize_question
from ..core.records import read_run_records, write_run_record
from ..documents.rendering import render_pdf
from ..models.runner import ModelRunner


BASELINE_PROMPT = (
    "Answer only from the supplied document pages. Return JSON exactly with keys "
    "answer (string or null), evidence (list of {page_id, quote}), and "
    "insufficient_evidence (boolean). The top-level object must contain all three "
    "keys and no evidence-only object. Use zero-based page_id values exactly as "
    "supplied. Use null and true when the pages do not support an answer."
)


def _list_field(value: Any, field_name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"invalid {field_name}: {value!r}") from exc
        if isinstance(parsed, list):
            return parsed
    raise ValueError(f"invalid {field_name}: expected a list")


def load_v2_samples(path: Path) -> list[BenchmarkSample]:
    """Load V2 samples and normalize its string-encoded list columns."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("samples JSON must contain a list")
    samples: list[BenchmarkSample] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"sample {index} must be an object")
        item = dict(raw)
        item["evidence_pages"] = _list_field(item.get("evidence_pages"), "evidence_pages")
        item["evidence_sources"] = _list_field(
            item.get("evidence_sources"), "evidence_sources"
        )
        try:
            samples.append(BenchmarkSample.model_validate(item))
        except ValueError as exc:
            raise ValueError(f"invalid sample {index}") from exc
    keys = [(sample.doc_id, normalize_question(sample.question)) for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("samples contain duplicate (doc_id, question) keys")
    return samples


def _prediction_key(prediction: Prediction) -> tuple[str, str]:
    return prediction.doc_id, normalize_question(prediction.question)


def _load_predictions(path: Path) -> OrderedDict[tuple[str, str], Prediction]:
    if not Path(path).exists():
        return OrderedDict()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("predictions JSON must contain a list")
    result: OrderedDict[tuple[str, str], Prediction] = OrderedDict()
    for item in payload:
        prediction = Prediction.model_validate(item)
        key = _prediction_key(prediction)
        if key in result:
            raise ValueError(f"duplicate prediction key: {key}")
        result[key] = prediction
    return result


def _render_document(
    pdf_path: Path,
    render_dir: Path,
    config: HarnessConfig,
):
    """Render a document once and reuse a manifest while its source is unchanged."""

    document_dir = Path(render_dir) / pdf_path.name
    manifest_path = document_dir / "manifest.json"
    source_stat = pdf_path.stat()
    source = {
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "dpi": config.render.dpi,
    }
    if manifest_path.exists():
        try:
            cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached.get("source") == source:
                pages = cached.get("pages", [])
                if pages and all(Path(page["image_path"]).is_file() for page in pages):
                    from ..core.contracts import Page

                    return [Page.model_validate(page) for page in pages]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    pages = render_pdf(pdf_path, document_dir, dpi=config.render.dpi)
    manifest_path.write_text(
        json.dumps(
            {"source": source, "pages": [page.model_dump(mode="json") for page in pages]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return pages


def _failure_record(
    sample: BenchmarkSample,
    config: HarnessConfig,
    started: float,
    exc: Exception,
    stage: str,
) -> RunRecord:
    return RunRecord(
        run_id=uuid.uuid4().hex,
        document_id=sample.doc_id,
        question=normalize_question(sample.question),
        status=Status.failed,
        response=None,
        config_hash=config.effective_hash(),
        model_id=config.model.model_id,
        model_revision=config.model.revision,
        latency_ms=(time.perf_counter() - started) * 1000,
        failure=StageFailure(
            stage=stage,
            error_type=type(exc).__name__,
            message=str(exc),
            retryable=False,
        ),
    )


def _ordered_predictions(
    samples: Iterable[BenchmarkSample],
    predictions: dict[tuple[str, str], Prediction],
) -> list[Prediction]:
    return [
        predictions[(sample.doc_id, normalize_question(sample.question))]
        for sample in samples
        if (sample.doc_id, normalize_question(sample.question)) in predictions
    ]


def run_v2_batch(
    samples_path: Path,
    documents_dir: Path,
    output_path: Path,
    records_path: Path,
    render_dir: Path,
    config: HarnessConfig,
    runner: ModelRunner,
    limit: int | None = None,
    prompt: str = BASELINE_PROMPT,
) -> list[Prediction]:
    """Run V2 questions grouped by PDF, resuming existing prediction rows."""

    samples = load_v2_samples(samples_path)
    selected = samples if limit is None else samples[:limit]
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    predictions = _load_predictions(output_path)
    sample_keys = {
        (sample.doc_id, normalize_question(sample.question)) for sample in samples
    }
    unknown = set(predictions) - sample_keys
    if unknown:
        raise ValueError(f"prediction is not in samples: {sorted(unknown)[0]}")

    existing_records = []
    if Path(records_path).exists():
        existing_records = read_run_records(records_path)
    record_by_key = {
        (record.document_id, normalize_question(record.question)): record
        for record in existing_records
    }
    groups: OrderedDict[str, list[BenchmarkSample]] = OrderedDict()
    for sample in selected:
        key = (sample.doc_id, normalize_question(sample.question))
        prior = record_by_key.get(key)
        if (
            key not in predictions
            or prior is None
            or prior.status != Status.ok
            or prior.parse_status == "invalid"
        ):
            groups.setdefault(sample.doc_id, []).append(sample)

    for doc_id, doc_samples in groups.items():
        pdf_path = Path(documents_dir) / doc_id
        pages = None
        render_error: Exception | None = None
        if not pdf_path.is_file():
            render_error = FileNotFoundError(pdf_path)
        elif config.render.page_policy != "all":
            render_error = ValueError("V2 batch execution requires render.page_policy = 'all'")
        else:
            try:
                pages = _render_document(pdf_path, render_dir, config)
            except Exception as exc:  # preserve one explicit failure row per sample
                render_error = exc

        for sample in doc_samples:
            started = time.perf_counter()
            key = (sample.doc_id, normalize_question(sample.question))
            if render_error is not None or pages is None:
                error = render_error or RuntimeError("document pages were not rendered")
                record = _failure_record(sample, config, started, error, "render")
                prediction = Prediction(
                    doc_id=sample.doc_id,
                    question=normalize_question(sample.question),
                    response="",
                )
            else:
                try:
                    request = build_request(
                        sample, pages, prompt, config.effective_hash()
                    )
                    generation = None
                    generate = getattr(runner, "generate", None)
                    if callable(generate):
                        generation = generate(request)
                        if generation.raw_response:
                            response = generation.raw_response
                        elif generation.draft is None:
                            response = ""
                        else:
                            response = generation.draft.answer or ""
                        draft = generation.draft
                    else:
                        draft = runner.run(request)
                        response = "" if draft.answer is None else draft.answer
                    prediction = Prediction(
                        doc_id=sample.doc_id,
                        question=normalize_question(sample.question),
                        response=response,
                    )
                    record = RunRecord(
                        run_id=uuid.uuid4().hex,
                        document_id=sample.doc_id,
                        question=normalize_question(sample.question),
                        status=Status.ok,
                        response=response,
                        config_hash=config.effective_hash(),
                        model_id=config.model.model_id,
                        model_revision=config.model.revision,
                        page_ids=[page.page_id for page in pages],
                        latency_ms=(time.perf_counter() - started) * 1000,
                        raw_response=(generation.raw_response if generation else None),
                        parse_status=(generation.parse_status if generation else None),
                        finish_reason=(generation.finish_reason if generation else None),
                        input_tokens=(generation.input_tokens if generation else None),
                        output_tokens=(generation.output_tokens if generation else None),
                        failure=(generation.failure if generation else None),
                        metadata=(
                            {
                                "parsed_answer": draft.answer,
                                "evidence": [
                                    span.model_dump(mode="json") for span in draft.evidence
                                ],
                                "insufficient_evidence": draft.insufficient_evidence,
                            }
                            if draft is not None
                            else {}
                        ),
                    )
                except Exception as exc:  # preserve one explicit failure row per sample
                    record = _failure_record(sample, config, started, exc, "answer")
                    prediction = Prediction(
                        doc_id=sample.doc_id,
                        question=normalize_question(sample.question),
                        response="",
                    )
            predictions[key] = prediction
            write_run_record(record, records_path)
            record_by_key[key] = record
            export_v2_predictions(
                _ordered_predictions(samples, predictions), output_path
            )

    result = _ordered_predictions(samples, predictions)
    export_v2_predictions(result, output_path)
    return result
