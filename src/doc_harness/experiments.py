"""Comparable run summaries and paired ablation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import normalize_question


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["doc_id"]), normalize_question(str(row["question"]))


def _rows(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "evaluation" / "scored.json",
        run_dir / "scored.json",
        run_dir / "predictions_scored.json",
        run_dir / "predictions.json",
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise FileNotFoundError(f"no scored predictions in {run_dir}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"run rows must be a list: {source}")
    return payload


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy": 0.0, "f1": 0.0, "judge_errors": 0}
    errors = []
    for row in rows:
        verdict = row.get("llm_judge")
        if not isinstance(verdict, dict) or not isinstance(verdict.get("equivalent"), bool):
            errors.append({"key": _key(row), "reason": "missing or invalid judge verdict"})
            continue
        if str(verdict.get("reason", "")).lower().startswith("judge failed:"):
            errors.append({"key": _key(row), "reason": verdict["reason"]})
    if errors:
        raise ValueError(f"unresolved judge errors: {errors[0]}")
    score = [1.0 if row["llm_judge"]["equivalent"] else 0.0 for row in rows]
    answerable = [row for row in rows if not str(row.get("answer", "")).startswith("Not answerable")]
    hits = sum(
        1.0
        for row in answerable
        if row["llm_judge"]["equivalent"]
    )
    answered = sum(1 for row in rows if not row["llm_judge"].get("abstained", False))
    recall = hits / len(answerable) if answerable else 0.0
    precision = hits / answered if answered else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {
        "n": len(rows),
        "accuracy": sum(score) / len(score),
        "f1": f1,
        "answerable_n": len(answerable),
        "answered_n": answered,
        "judge_errors": 0,
    }


def compare_runs(run_dirs: list[Path], split_path: Path | None = None) -> dict[str, Any]:
    """Compare runs with identical judged sample keys and paired transitions."""

    if len(run_dirs) < 2:
        raise ValueError("at least two run directories are required")
    loaded = [(Path(path), _rows(Path(path))) for path in run_dirs]
    key_sets = [{_key(row) for row in rows} for _, rows in loaded]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("runs must contain the same sample keys")
    variants = []
    by_run: list[dict[tuple[str, str], dict[str, Any]]] = []
    for path, rows in loaded:
        metrics = _metrics(rows)
        variants.append({"name": path.name, "run_dir": str(path), **metrics})
        by_run.append({_key(row): row for row in rows})
    paired: dict[str, Any] = {}
    for index in range(len(loaded) - 1):
        left_name = loaded[index][0].name
        right_name = loaded[index + 1][0].name
        improved = regressed = unchanged = 0
        for key in key_sets[0]:
            left = bool(by_run[index][key]["llm_judge"]["equivalent"])
            right = bool(by_run[index + 1][key]["llm_judge"]["equivalent"])
            if right and not left:
                improved += 1
            elif left and not right:
                regressed += 1
            else:
                unchanged += 1
        paired[f"{left_name}_to_{right_name}"] = {
            "improved": improved,
            "regressed": regressed,
            "unchanged": unchanged,
        }
    result: dict[str, Any] = {
        "variants": variants,
        "paired": paired,
        "sample_count": len(key_sets[0]),
    }
    if split_path is not None:
        result["split_path"] = str(Path(split_path))
    return result


def write_report(comparison: dict[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
