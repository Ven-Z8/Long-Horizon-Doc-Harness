"""Run manifests, immutable artifact snapshots, and baseline audits.

The baseline predates run manifests, so this module is deliberately tolerant when
reading historical JSONL and prediction files.  It reports inconsistencies instead
of turning an incomplete old run into a successful one.  New runs can use
``RunManifest`` and ``assert_resume_compatible`` to make the same identity checks
explicit before resuming.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from pydantic import Field

from ..core.contracts import RunRecord, StrictModel, Status
from ..core.protocol import normalize_question


class RunManifest(StrictModel):
    """Identity of the inputs that can affect a run's outputs.

    ``run_id`` names an artifact directory.  It is intentionally excluded from
    compatibility comparison: a caller may copy a run directory to a new location
    while retaining the exact computational identity.
    """

    schema_version: int = Field(ge=1)
    run_id: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    code_hash: str = Field(min_length=1)
    dataset_hash: str = Field(min_length=1)
    samples_hash: str = Field(min_length=1)
    models_hash: str = Field(min_length=1)
    prompts_hash: str = Field(min_length=1)
    dependencies_hash: str = Field(min_length=1)


RESUME_IDENTITY_FIELDS: tuple[str, ...] = (
    "schema_version",
    "config_hash",
    "code_hash",
    "dataset_hash",
    "samples_hash",
    "models_hash",
    "prompts_hash",
    "dependencies_hash",
)


class ResumeCompatibilityError(ValueError):
    """Raised when a run's inputs differ from the manifest used to start it."""


def assert_resume_compatible(expected: RunManifest, actual: RunManifest) -> None:
    """Raise if ``actual`` cannot safely resume ``expected``.

    Every field that influences inference is compared, including the schema version.
    The error names all changed fields so an operator can create a fresh run rather
    than guessing which cache or artifact is stale.
    """

    if not isinstance(expected, RunManifest) or not isinstance(actual, RunManifest):
        raise TypeError("expected and actual must be RunManifest instances")
    mismatches = [
        f"{field}: expected {getattr(expected, field)!r}, got {getattr(actual, field)!r}"
        for field in RESUME_IDENTITY_FIELDS
        if getattr(expected, field) != getattr(actual, field)
    ]
    if mismatches:
        raise ResumeCompatibilityError(
            "run manifest is not resume-compatible (" + "; ".join(mismatches) + ")"
        )


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_run(paths: list[Path], destination: Path) -> dict[str, str]:
    """Copy files byte-for-byte into a new, immutable run directory.

    The return value maps each copied relative filename to its SHA-256 digest.  A
    ``manifest.json`` beside the files records source paths, byte counts, and the
    same digests.  ``destination`` is created with ``exist_ok=False`` so a freeze
    can never silently replace an earlier snapshot.
    """

    if not paths:
        raise ValueError("at least one file is required to freeze a run")
    destination = Path(destination)
    sources = [Path(path) for path in paths]
    if len({path.resolve() for path in sources}) != len(sources):
        raise ValueError("freeze paths contain duplicates")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if any(source.name == "manifest.json" for source in sources):
        raise ValueError("manifest.json is reserved for the freeze inventory")

    # mkdir(exist_ok=False) is the no-overwrite guard.  It also avoids a race where
    # a separate process creates the same destination after an existence check.
    destination.mkdir(parents=True, exist_ok=False)
    inventory: dict[str, str] = {}
    manifest_files: dict[str, dict[str, Any]] = {}
    try:
        for source in sources:
            relative_name = source.name
            if relative_name in inventory:
                raise ValueError(
                    f"source filenames collide in freeze destination: {relative_name}"
                )
            target = destination / relative_name
            shutil.copyfile(source, target)
            digest = sha256_file(target)
            inventory[relative_name] = digest
            manifest_files[relative_name] = {
                "source": str(source),
                "size_bytes": target.stat().st_size,
                "sha256": digest,
            }

        manifest = {
            "schema_version": 1,
            "files": manifest_files,
            "file_count": len(manifest_files),
            "immutable": True,
            # A generic freeze cannot infer the code/configuration that produced a
            # historical artifact.  Keep the explicit unknown value rather than
            # reconstructing an identity from today's checkout.
            "identity": {
                "code_hash": "unknown",
                "config_hash": "unknown",
                "dataset_hash": "unknown",
                "samples_hash": "unknown",
                "models_hash": "unknown",
                "prompts_hash": "unknown",
                "dependencies_hash": "unknown",
            },
        }
        manifest_path = destination / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Mark the snapshot read-only after the manifest has been written.  This is
        # a useful guard against accidental edits; the checksum inventory remains the
        # source of truth even on filesystems where permissions are advisory.
        for child in destination.iterdir():
            child.chmod(0o444)
        destination.chmod(0o555)
    except Exception:
        # The caller gets the original error.  A partially written destination is
        # retained for forensic inspection and, crucially, can never be overwritten.
        raise
    return inventory


def _key(document_id: Any, question: Any) -> tuple[str, str]:
    return str(document_id), normalize_question(str(question))


def _key_payload(key: tuple[str, str]) -> dict[str, str]:
    return {"document_id": key[0], "question": key[1]}


def _read_records_tolerantly(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                item = RunRecord.model_validate(raw)
                records.append({"raw": raw, "record": item})
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                invalid.append({"line": line_number, "error": f"{type(exc).__name__}: {exc}"})
    return records, invalid


def _read_predictions_tolerantly(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("predictions JSON must contain a list")
    predictions: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        try:
            if not isinstance(raw, dict):
                raise TypeError("prediction must be an object")
            if not isinstance(raw.get("doc_id"), str):
                raise TypeError("doc_id must be a string")
            if not isinstance(raw.get("question"), str):
                raise TypeError("question must be a string")
            if not isinstance(raw.get("response"), str):
                raise TypeError("response must be a string")
            predictions.append({"raw": raw, "key": _key(raw["doc_id"], raw["question"])})
        except (TypeError, ValueError, KeyError) as exc:
            invalid.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
    return predictions, invalid


def _response_looks_truncated(response: Any) -> bool:
    if not isinstance(response, str) or not response.strip():
        return False
    candidate = response.rstrip()
    # These checks are deliberately conservative: historical records do not carry
    # tokenizer termination metadata, so the audit says “suspected” rather than
    # claiming that an output definitely was truncated.
    if candidate.endswith(("...", "…", ",", ":", "{", "[", "(")):
        return True
    if "<think>" in candidate and "</think>" not in candidate:
        return True
    pairs = {"{": "}", "[": "]", "(": ")"}
    for opening, closing in pairs.items():
        if candidate.count(opening) != candidate.count(closing):
            return True
    return False


def _has_termination_metadata(record: RunRecord) -> bool:
    if getattr(record, "finish_reason", None) is not None:
        return True
    metadata = record.metadata if isinstance(record.metadata, dict) else {}
    return any(
        name in metadata
        for name in ("finish_reason", "termination_reason", "eos_token_id", "terminated")
    )


def _judge_summary(predictions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(predictions)
    verdict_rows = [row for row in rows if "llm_judge" in row["raw"]]
    failures: list[dict[str, Any]] = []
    valid = 0
    for row in verdict_rows:
        verdict = row["raw"].get("llm_judge")
        if not isinstance(verdict, dict):
            failures.append({**_key_payload(row["key"]), "error_type": "invalid_json", "reason": "llm_judge is not an object"})
            continue
        reason = verdict.get("reason")
        if isinstance(reason, str) and reason.startswith("judge failed:"):
            failures.append({**_key_payload(row["key"]), "error_type": "judge_failed_marker", "reason": reason})
            continue
        if (
            set(verdict) != {"equivalent", "abstained", "reason"}
            or type(verdict.get("equivalent")) is not bool
            or type(verdict.get("abstained")) is not bool
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            failures.append({**_key_payload(row["key"]), "error_type": "invalid_json", "reason": "invalid judge verdict schema"})
            continue
        valid += 1
    return {
        "total": len(rows),
        "with_verdict": len(verdict_rows),
        "valid": valid,
        "failure_count": len(failures),
        "failures": failures,
        "unresolved": len(verdict_rows) != len(rows) or bool(failures),
    }


def audit_records(records_path: Path, predictions_path: Path) -> dict[str, Any]:
    """Reconcile records and predictions, retaining every discrepancy.

    Questions are joined on ``(document_id, normalized_question)``.  Duplicate,
    missing, conflicting, malformed, runtime-failed, and judge-failed rows are
    separate diagnostics so an old ``status=ok`` count cannot hide an incomplete
    generation or scoring run.
    """

    record_entries, invalid_record_rows = _read_records_tolerantly(records_path)
    prediction_entries, invalid_prediction_rows = _read_predictions_tolerantly(predictions_path)

    record_keys = [
        _key(entry["record"].document_id, entry["record"].question)
        for entry in record_entries
    ]
    prediction_keys = [entry["key"] for entry in prediction_entries]
    record_counts = Counter(record_keys)
    prediction_counts = Counter(prediction_keys)
    duplicate_record_keys = [key for key, count in record_counts.items() if count > 1]
    duplicate_prediction_keys = [key for key, count in prediction_counts.items() if count > 1]
    duplicate_keys = list(dict.fromkeys(duplicate_record_keys + duplicate_prediction_keys))

    record_key_set = set(record_keys)
    prediction_key_set = set(prediction_keys)
    missing_records = sorted(prediction_key_set - record_key_set)
    missing_predictions = sorted(record_key_set - prediction_key_set)

    record_responses: defaultdict[tuple[str, str], list[Any]] = defaultdict(list)
    for entry in record_entries:
        record = entry["record"]
        key = _key(record.document_id, record.question)
        record_responses[key].append(record.response)
    prediction_responses: defaultdict[tuple[str, str], list[Any]] = defaultdict(list)
    for entry in prediction_entries:
        prediction_responses[entry["key"]].append(entry["raw"].get("response"))

    response_mismatches: list[dict[str, Any]] = []
    for key in sorted(record_key_set & prediction_key_set):
        pred_values = prediction_responses[key]
        rec_values = record_responses[key]
        if any(pred_value != record_value for pred_value in pred_values for record_value in rec_values):
            response_mismatches.append(
                {
                    **_key_payload(key),
                    "record_responses": rec_values,
                    "prediction_responses": pred_values,
                }
            )

    runtime_failures: list[dict[str, Any]] = []
    missing_termination_metadata: list[dict[str, Any]] = []
    suspected_truncation: list[dict[str, Any]] = []
    invalid_responses: list[dict[str, Any]] = []
    for entry in record_entries:
        record = entry["record"]
        key = _key(record.document_id, record.question)
        if record.status == Status.failed:
            runtime_failures.append(
                {
                    **_key_payload(key),
                    "error_type": record.failure.error_type if record.failure else "unknown",
                    "message": record.failure.message if record.failure else "",
                }
            )
        if not _has_termination_metadata(record):
            missing_termination_metadata.append(_key_payload(key))
        # A failed attempt has no response by definition.  Keep that operational
        # failure in its own denominator rather than calling it malformed output.
        if record.status == Status.ok:
            if getattr(record, "parse_status", None) == "invalid":
                invalid_responses.append({**_key_payload(key), "reason": "invalid structured response"})
            elif record.response is None or (isinstance(record.response, str) and not record.response.strip()):
                invalid_responses.append({**_key_payload(key), "reason": "empty response"})
            elif _response_looks_truncated(record.response):
                suspected_truncation.append({**_key_payload(key), "reason": "suspected truncation"})

    return {
        "records": {
            "total": len(record_entries),
            "ok": sum(record.status == Status.ok for entry in record_entries for record in [entry["record"]]),
            "failed": sum(record.status == Status.failed for entry in record_entries for record in [entry["record"]]),
            "invalid": len(invalid_record_rows),
        },
        "predictions": {"total": len(prediction_entries), "invalid": len(invalid_prediction_rows)},
        "duplicate_keys": [_key_payload(key) for key in duplicate_keys],
        "duplicate_record_keys": [_key_payload(key) for key in duplicate_record_keys],
        "duplicate_prediction_keys": [_key_payload(key) for key in duplicate_prediction_keys],
        "missing_records": [_key_payload(key) for key in missing_records],
        "missing_predictions": [_key_payload(key) for key in missing_predictions],
        "response_mismatches": response_mismatches,
        "runtime_failures": runtime_failures,
        "invalid_response_artifacts": invalid_responses,
        "invalid_response_count": len(invalid_responses),
        "suspected_truncation": suspected_truncation,
        "suspected_truncation_count": len(suspected_truncation),
        "termination_metadata_missing": missing_termination_metadata,
        "termination_metadata_missing_count": len(missing_termination_metadata),
        "invalid_record_rows": invalid_record_rows,
        "invalid_prediction_rows": invalid_prediction_rows,
        "judge": _judge_summary(prediction_entries),
        "complete_key_coverage": not missing_records and not missing_predictions and not duplicate_keys,
    }


def audit_verdicts(scored_path: Path) -> dict[str, Any]:
    """Audit the optional ``llm_judge`` projection produced by the V2 evaluator."""

    payload = json.loads(Path(scored_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("scored JSON must contain a list")
    rows = []
    missing = 0
    failures = []
    valid = 0
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            missing += 1
            continue
        key = _key(row.get("doc_id", ""), row.get("question", ""))
        verdict = row.get("llm_judge")
        if verdict is None:
            missing += 1
        elif isinstance(verdict, dict) and isinstance(verdict.get("reason"), str) and verdict["reason"].startswith("judge failed:"):
            failures.append({"index": index, **_key_payload(key), "reason": verdict["reason"]})
        elif isinstance(verdict, dict) and set(verdict) == {"equivalent", "abstained", "reason"} and type(verdict.get("equivalent")) is bool and type(verdict.get("abstained")) is bool and isinstance(verdict.get("reason"), str):
            valid += 1
        else:
            failures.append({"index": index, **_key_payload(key), "reason": "invalid judge verdict schema"})
        rows.append(row)
    return {
        "total": len(payload),
        "valid": valid,
        "missing": missing,
        "failure_count": len(failures),
        "failures": failures,
        "equivalent": sum(row.get("llm_judge", {}).get("equivalent") is True for row in rows if isinstance(row.get("llm_judge"), dict)),
        "abstained": sum(row.get("llm_judge", {}).get("abstained") is True for row in rows if isinstance(row.get("llm_judge"), dict)),
    }


__all__ = [
    "RESUME_IDENTITY_FIELDS",
    "ResumeCompatibilityError",
    "RunManifest",
    "assert_resume_compatible",
    "audit_records",
    "audit_verdicts",
    "freeze_run",
    "sha256_file",
]
