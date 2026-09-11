"""Backend-independent schema validation and one bounded repair attempt."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from ..core.contracts import ModelRequest, StageFailure


class RawAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    raw_response: str = ""
    finish_reason: str = "unknown"
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    failure: StageFailure | None = None


class StructuredCall(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    attempts: list[RawAttempt] = Field(default_factory=list)
    payload: dict[str, Any] | None = None
    failure: StageFailure | None = None


def _failure(message: str, error_type: str = "ValidationError") -> StageFailure:
    return StageFailure(stage="structured_output", error_type=error_type, message=message, retryable=True)


def _validate_attempt(raw: RawAttempt, schema: type[BaseModel]) -> tuple[dict[str, Any] | None, StageFailure | None]:
    if raw.failure is not None:
        return None, raw.failure
    if raw.finish_reason != "eos":
        return None, _failure(f"generation ended with {raw.finish_reason}", "IncompleteGeneration")
    try:
        value = json.loads(raw.raw_response)
    except json.JSONDecodeError as exc:
        return None, _failure(f"response is not valid JSON: {exc}", "JSONDecodeError")
    if not isinstance(value, dict):
        return None, _failure("structured response must be a JSON object")
    try:
        parsed = schema.model_validate(value)
    except Exception as exc:
        return None, _failure(str(exc))
    return parsed.model_dump(mode="json"), None


def generate_structured(
    request: ModelRequest,
    schema: type[BaseModel],
    generate_raw: Callable[[ModelRequest], RawAttempt],
    repair_prompt: Callable[[str, str], str],
    *,
    max_repairs: int = 1,
) -> StructuredCall:
    """Generate one validated object and optionally perform one recorded repair."""

    if max_repairs not in (0, 1):
        raise ValueError("max_repairs must be 0 or 1")
    attempts: list[RawAttempt] = []
    current_request = request
    for attempt_number in range(max_repairs + 1):
        started = time.perf_counter()
        try:
            raw = generate_raw(current_request)
        except Exception as exc:
            raw = RawAttempt(
                raw_response="",
                finish_reason="error",
                latency_ms=(time.perf_counter() - started) * 1000,
                failure=_failure(str(exc), type(exc).__name__),
            )
        if raw.latency_ms == 0:
            raw = raw.model_copy(update={"latency_ms": (time.perf_counter() - started) * 1000})
        attempts.append(raw)
        payload, failure = _validate_attempt(raw, schema)
        if payload is not None:
            return StructuredCall(attempts=attempts, payload=payload)
        if failure is not None and raw.failure is None:
            attempts[-1] = raw.model_copy(update={"failure": failure})
        if attempt_number >= max_repairs:
            return StructuredCall(attempts=attempts, failure=failure)
        error_text = failure.message if failure is not None else "invalid structured response"
        current_request = current_request.model_copy(
            update={"prompt": repair_prompt(error_text[:1000], raw.raw_response)}
        )
    raise AssertionError("unreachable")
