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

from ..core.contracts import BenchmarkSample, StrictModel
from ..core.protocol import normalize_question


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


class JudgeErrorType(str, Enum):
    timeout = "timeout"
    invalid_json = "invalid_json"
    authentication = "authentication"
    exhausted_retries = "exhausted_retries"
    unknown = "unknown"


JudgeErrorKind = JudgeErrorType


class JudgeEvaluationError(ValueError):
    """An invalid or operationally failed judge response."""

    def __init__(self, message: str, error_type: JudgeErrorType = JudgeErrorType.unknown):
        super().__init__(message)
        self.error_type = error_type


class JudgeVerdict(StrictModel):
    """The exact three-field response schema used by the V2 evaluator."""

    equivalent: bool
    abstained: bool
    reason: str = Field(min_length=1)


def classify_judge_error(error: BaseException | str) -> JudgeErrorType:
    """Classify judge failure text without converting it into a negative verdict."""

    if isinstance(error, JudgeEvaluationError):
        return error.error_type
    if isinstance(error, TimeoutError):
        return JudgeErrorType.timeout
    if isinstance(error, json.JSONDecodeError):
        return JudgeErrorType.invalid_json
    text = str(error).strip().lower()
    if text.startswith("judge failed:"):
        text = text[len("judge failed:") :].strip()
    if any(marker in text for marker in ("authentication", "unauthorized", "invalid api key", "incorrect api key", "api key", "401", "403", "permission denied")):
        return JudgeErrorType.authentication
    if any(marker in text for marker in ("exhausted retries", "retries exhausted", "retry exhausted", "attempts exhausted", "max retries", "maximum retries", "after 3 attempts", "after three attempts")):
        return JudgeErrorType.exhausted_retries
    if any(marker in text for marker in ("timeout", "timed out", "time-out")):
        return JudgeErrorType.timeout
    if any(marker in text for marker in ("jsondecodeerror", "json decode", "invalid json", "malformed json", "schema validation", "invalid verdict")):
        return JudgeErrorType.invalid_json
    return JudgeErrorType.unknown


def validate_judge_verdict(payload: Mapping[str, Any] | str) -> JudgeVerdict:
    """Validate one judge output and reject upstream ``judge failed:`` markers."""

    raw: Any = payload
    if isinstance(payload, str):
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise JudgeEvaluationError(f"judge output is not valid JSON: {exc}", JudgeErrorType.invalid_json) from exc
    if not isinstance(raw, Mapping):
        raise JudgeEvaluationError("judge output must be a JSON object", JudgeErrorType.invalid_json)
    if set(raw) != {"equivalent", "abstained", "reason"}:
        raise JudgeEvaluationError("judge verdict must contain exactly equivalent, abstained, and reason", JudgeErrorType.invalid_json)
    if type(raw.get("equivalent")) is not bool or type(raw.get("abstained")) is not bool:
        raise JudgeEvaluationError("judge verdict equivalent and abstained must be booleans", JudgeErrorType.invalid_json)
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise JudgeEvaluationError("judge verdict reason must be a nonempty string", JudgeErrorType.invalid_json)
    if reason.startswith("judge failed:"):
        raise JudgeEvaluationError(reason, classify_judge_error(reason))
    try:
        return JudgeVerdict.model_validate(dict(raw))
    except ValueError as exc:
        raise JudgeEvaluationError(f"invalid judge verdict: {exc}", JudgeErrorType.invalid_json) from exc


def _hash_judge_text(value: Any, *, normalize: bool = False) -> str:
    text = str(value)
    if normalize:
        text = normalize_question(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def judge_cache_identity(*, question: str, reference_answer: str, response: str, judge_model: str, prompt: str) -> dict[str, str]:
    """Return component hashes and a combined identity for one judge call."""

    components = {
        "question_hash": _hash_judge_text(question, normalize=True),
        "reference_hash": _hash_judge_text(reference_answer),
        "response_hash": _hash_judge_text(response),
        "judge_hash": _hash_judge_text(judge_model),
        "prompt_hash": _hash_judge_text(prompt),
    }
    canonical = json.dumps({"schema_version": 1, **components}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**components, "cache_key": hashlib.sha256(canonical).hexdigest()}


def judge_cache_key(question: str, reference_answer: str, response: str, judge_model: str, prompt: str) -> str:
    """Return a digest that changes with every judge input identity component."""

    return judge_cache_identity(question=question, reference_answer=reference_answer, response=response, judge_model=judge_model, prompt=prompt)["cache_key"]


make_judge_cache_key = judge_cache_key
cache_key = judge_cache_key


def _score_list_field(value: Any, field_name: str) -> list[Any]:
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


def _load_score_samples(path: Path) -> list[BenchmarkSample]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("samples JSON must contain a list")
    samples: list[BenchmarkSample] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"sample {index} must be an object")
        item = dict(raw)
        item["evidence_pages"] = _score_list_field(item.get("evidence_pages"), "evidence_pages")
        item["evidence_sources"] = _score_list_field(item.get("evidence_sources"), "evidence_sources")
        try:
            samples.append(BenchmarkSample.model_validate(item))
        except ValueError as exc:
            raise ValueError(f"invalid sample {index}") from exc
    return samples


def _score_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("doc_id", "")), normalize_question(str(row.get("question", "")))


def _score_key_payload(key: tuple[str, str]) -> dict[str, str]:
    return {"doc_id": key[0], "question": key[1]}


def _score_manifest_identity(run_dir: Path) -> tuple[str, str]:
    path = Path(run_dir) / "manifest.json"
    if not path.exists():
        return "unknown", "unknown"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown", "unknown"
    if not isinstance(payload, dict):
        return "unknown", "unknown"
    return str(payload.get("judge_model") or payload.get("evaluator_model") or "unknown"), str(payload.get("judge_prompt") or payload.get("prompts_hash") or "unknown")


def _score_attempt_failures(run_dir: Path) -> list[dict[str, Any]]:
    for name in ("attempts.jsonl", "runs.jsonl", "records.jsonl"):
        path = Path(run_dir) / name
        if not path.exists():
            continue
        failures: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                failures.append({"line": line_number, "error_type": "invalid_record"})
                continue
            if isinstance(raw, dict) and str(raw.get("status", "")) in {"failed", "error"}:
                failure = raw.get("failure")
                failures.append({
                    "doc_id": str(raw.get("document_id", raw.get("doc_id", ""))),
                    "question": normalize_question(str(raw.get("question", ""))),
                    "error_type": failure.get("error_type", "unknown") if isinstance(failure, dict) else "unknown",
                    "message": failure.get("message", "") if isinstance(failure, dict) else "",
                })
        return failures
    return []


def _score_metrics(rows: list[tuple[BenchmarkSample, JudgeVerdict]]) -> dict[str, Any]:
    scores = [int(verdict.equivalent) for _, verdict in rows]
    answerable = [(sample, verdict) for sample, verdict in rows if not str(sample.answer).startswith("Not answerable")]
    hits = sum(int(verdict.equivalent) for _, verdict in answerable)
    answered = sum(not verdict.abstained for _, verdict in rows)
    recall = hits / len(answerable) if answerable else 0.0
    precision = hits / answered if answered else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"accuracy": sum(scores) / len(rows) if rows else 0.0, "f1": f1, "denominator": len(rows), "correct": sum(scores), "answerable_denominator": len(answerable), "answered_count": answered, "answerable_recall": recall, "answer_precision": precision}


def score_run(run_dir: Path, samples_path: Path) -> dict[str, Any]:
    """Validate a pre-judged predictions projection and export comparable metrics only when complete."""

    run_dir = Path(run_dir)
    predictions_path = run_dir / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("predictions JSON must contain a list")
    samples = _load_score_samples(samples_path)
    expected: dict[tuple[str, str], BenchmarkSample] = {}
    duplicate_expected: list[tuple[str, str]] = []
    for sample in samples:
        key = (sample.doc_id, normalize_question(sample.question))
        if key in expected:
            duplicate_expected.append(key)
        expected[key] = sample
    rows_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    invalid_predictions: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping) or not isinstance(row.get("doc_id"), str) or not isinstance(row.get("question"), str):
            invalid_predictions.append({"index": index, "error_type": "invalid_prediction"})
            continue
        rows_by_key.setdefault(_score_key(row), []).append(row)
    duplicate_predictions = [key for key, rows in rows_by_key.items() if len(rows) > 1]
    unknown_predictions = sorted(set(rows_by_key) - set(expected))
    missing_predictions = sorted(set(expected) - set(rows_by_key))
    judge_errors: list[dict[str, Any]] = []
    valid_rows: list[tuple[BenchmarkSample, JudgeVerdict]] = []
    judge_model, judge_prompt = _score_manifest_identity(run_dir)
    cache_identities: list[dict[str, str]] = []
    for key, sample in expected.items():
        rows = rows_by_key.get(key, [])
        if len(rows) != 1:
            if not rows:
                judge_errors.append({**_score_key_payload(key), "error_type": "missing_verdict", "reason": "missing prediction"})
            continue
        row = rows[0]
        response = row.get("response", "")
        if not isinstance(response, str):
            judge_errors.append({**_score_key_payload(key), "error_type": "invalid_prediction", "reason": "response must be a string"})
            continue
        cache_identities.append(judge_cache_identity(question=sample.question, reference_answer=sample.answer, response=response, judge_model=str(row.get("judge_model", judge_model)), prompt=str(row.get("judge_prompt", judge_prompt))))
        if "llm_judge" not in row:
            judge_errors.append({**_score_key_payload(key), "error_type": "missing_verdict", "reason": "prediction has no llm_judge"})
            continue
        try:
            verdict = validate_judge_verdict(row["llm_judge"])
        except JudgeEvaluationError as exc:
            judge_errors.append({**_score_key_payload(key), "error_type": exc.error_type.value, "reason": str(exc)})
            continue
        valid_rows.append((sample, verdict))
    operational_failures = _score_attempt_failures(run_dir)
    errors_exist = bool(duplicate_expected or duplicate_predictions or unknown_predictions or missing_predictions or invalid_predictions or judge_errors)
    metrics = _score_metrics(valid_rows) if not errors_exist and len(valid_rows) == len(expected) else None
    return {
        "status": "complete" if metrics is not None else "incomplete",
        "metrics": metrics,
        "counts": {"expected": len(expected), "predictions": len(payload), "valid_verdicts": len(valid_rows), "operational_failures": len(operational_failures)},
        "duplicate_expected": [_score_key_payload(key) for key in duplicate_expected],
        "duplicate_predictions": [_score_key_payload(key) for key in duplicate_predictions],
        "unknown_predictions": [_score_key_payload(key) for key in unknown_predictions],
        "missing_predictions": [_score_key_payload(key) for key in missing_predictions],
        "invalid_predictions": invalid_predictions,
        "judge_errors": judge_errors,
        "operational_failures_detail": operational_failures,
        "cache_identities": cache_identities,
    }


__all__ = [
    "JudgeErrorKind", "JudgeErrorType", "JudgeEvaluationError", "JudgeVerdict",
    "cache_key", "classify_judge_error", "convert_evidence_pages",
    "judge_cache_identity", "judge_cache_key", "make_judge_cache_key",
    "retrieval_metrics", "score_run", "validate_judge_verdict",
]
