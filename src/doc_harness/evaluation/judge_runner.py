"""Explicit OpenRouter runner for the pinned MMLongBench-Doc V2 judge."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

from ..core.protocol import normalize_question
from .evaluation import JudgeEvaluationError, judge_cache_key, validate_judge_verdict


DEFAULT_OPENROUTER_MODEL = "openai/gpt-5.6-luna"
_JUDGE_PATH = Path(__file__).resolve().parents[3] / "benchmark" / "mmlongbench-doc-v2" / "eval" / "judge.py"


def _load_judge_module() -> Any:
    spec = importlib.util.spec_from_file_location("mmlongbench_doc_v2_judge", _JUDGE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load judge module: {_JUDGE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_openrouter_key(env_path: Path = Path(".env")) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                value = line.partition("=")[2].strip().strip('"').strip("'")
                if value:
                    return value
    raise RuntimeError("OPENROUTER_API_KEY is required for judge-run")


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("doc_id", "")), normalize_question(str(row.get("question", "")))


def build_judge_rows(predictions_path: Path, samples_path: Path) -> list[dict[str, Any]]:
    predictions = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))
    if not isinstance(predictions, list) or not isinstance(samples, list):
        raise ValueError("predictions and samples must be JSON lists")
    sample_by_key = {_key(item): item for item in samples}
    prediction_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in predictions:
        if not isinstance(row, dict) or not isinstance(row.get("doc_id"), str) or not isinstance(row.get("question"), str):
            raise ValueError("prediction rows require doc_id and question")
        key = _key(row)
        if key in prediction_by_key:
            raise ValueError(f"duplicate prediction: {key}")
        if key not in sample_by_key:
            raise ValueError(f"unknown prediction: {key}")
        if not isinstance(row.get("response"), str):
            raise ValueError(f"prediction response must be a string: {key}")
        prediction_by_key[key] = row
    missing = sorted(set(sample_by_key) - set(prediction_by_key))
    if missing:
        raise ValueError(f"missing predictions: {missing[0]}")
    rows: list[dict[str, Any]] = []
    for sample in samples:
        key = _key(sample)
        prediction = prediction_by_key[key]
        rows.append(
            {
                "doc_id": sample["doc_id"],
                "question": sample["question"],
                "answer": sample.get("answer", ""),
                "answer_format": sample.get("answer_format", "Str"),
                "response": prediction["response"],
            }
        )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


async def _judge_rows(rows: list[dict[str, Any]], model: str, concurrency: int) -> list[dict[str, Any]]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - optional evaluation dependency
        raise RuntimeError("judge-run requires the eval extra with openai") from exc
    judge = _load_judge_module()
    client = AsyncOpenAI(api_key=_load_openrouter_key(), base_url=judge.OPENROUTER_BASE)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    state = {"done": 0, "total": len(rows), "t0": time.time()}

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        cache_key = judge_cache_key(
            question=row["question"],
            reference_answer=row["answer"],
            response=row["response"],
            judge_model=model,
            prompt=judge.PROMPT,
        )
        row["judge_cache_key"] = cache_key
        result = await judge.judge_one(client, semaphore, row, model, state, backend="openrouter")
        try:
            row["llm_judge"] = validate_judge_verdict(result.get("llm_judge", {})).model_dump(mode="json")
        except JudgeEvaluationError:
            raise
        return row

    return list(await asyncio.gather(*(one(row) for row in rows)))


def judge_run(
    run_dir: Path,
    samples_path: Path,
    *,
    model: str = DEFAULT_OPENROUTER_MODEL,
    concurrency: int = 4,
) -> Path:
    """Judge a complete prediction projection through OpenRouter only."""

    run_dir = Path(run_dir)
    predictions_path = run_dir / "predictions.json"
    rows = build_judge_rows(predictions_path, samples_path)
    output = run_dir / "evaluation" / "judged.json"
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        cached = {_key(row): row for row in previous if isinstance(row, dict) and "llm_judge" in row}
        for row in rows:
            prior = cached.get(_key(row))
            if prior and prior.get("judge_cache_key") == judge_cache_key(
                question=row["question"], reference_answer=row["answer"], response=row["response"],
                judge_model=model, prompt=_load_judge_module().PROMPT,
            ):
                row.update({"llm_judge": prior["llm_judge"], "judge_cache_key": prior["judge_cache_key"]})
    unresolved = [row for row in rows if "llm_judge" not in row]
    if unresolved:
        judged = asyncio.run(_judge_rows(unresolved, model, concurrency))
        by_key = {_key(row): row for row in judged}
        for row in rows:
            if _key(row) in by_key:
                row.update(by_key[_key(row)])
    _write_json(output, rows)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"judge_model": model, "judge_backend": "openrouter", "judge_judged_path": str(output)})
        _write_json(manifest_path, manifest)
    return output
