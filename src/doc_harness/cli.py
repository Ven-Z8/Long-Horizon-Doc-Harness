"""Command-line entry points for offline smoke runs and V2 export."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Sequence

from .workflow.batch import run_v2_batch
from .core.config import load_config
from .core.contracts import BenchmarkSample, ModelRequest, Page, Prediction, RunRecord, StageFailure, Status
from .evaluation.evaluation import score_run
from .evaluation.judge_runner import judge_run
from .evaluation.experiments import compare_runs, write_report
from .evaluation.manifests import audit_records
from .core.protocol import build_request, export_v2_predictions, validate_prediction_coverage
from .core.records import read_run_records, write_run_record
from .models.runner import FakeRunner, ModelRunner, QwenTransformersRunner
from .evaluation.splits import create_document_split, prepare_phase2_selection
from .workflow.stages import (
    answer_questions,
    build_indexes,
    build_run_manifest,
    ocr_questions,
    rerank_questions,
    retrieve_questions,
)
from .workflow.coordinator import run_coordinator


def _write_cli_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _synthetic_pages(document_id: str, image_paths: Sequence[str]) -> list[Page]:
    return [
        Page(
            document_id=document_id,
            page_id=page_id,
            image_path=str(image_path),
            width=1,
            height=1,
            render_sha256="unrendered-smoke-input",
        )
        for page_id, image_path in enumerate(image_paths)
    ]


def run_smoke(
    question: str,
    document_id: str,
    image_paths: Sequence[str],
    output_path: Path,
    prediction_path: Path,
    runner: ModelRunner | None = None,
    config_hash: str = "smoke",
) -> RunRecord:
    """Run one safe request and persist a record plus V2-shaped prediction."""

    if not image_paths:
        raise ValueError("at least one image is required")
    pages = _synthetic_pages(document_id, image_paths)
    sample = BenchmarkSample(doc_id=document_id, question=question, answer="")
    request = build_request(
        sample,
        pages,
        (
            "Answer only from the supplied pages. Return JSON exactly with keys "
            "answer (string or null), evidence (list of {page_id, quote}), and "
            "insufficient_evidence (boolean). The top-level object must contain "
            "all three keys and no evidence-only object. Use zero-based page_id "
            "values exactly as supplied. Use null and true when the pages do not "
            "support an answer."
        ),
        config_hash,
    )
    selected_runner = runner or FakeRunner()
    run_id = uuid.uuid4().hex
    started = time.perf_counter()
    try:
        draft = selected_runner.run(request)
    except Exception as exc:
        record = RunRecord(
            run_id=run_id,
            document_id=document_id,
            question=request.question,
            status=Status.failed,
            response=None,
            config_hash=config_hash,
            page_ids=request.page_ids,
            latency_ms=(time.perf_counter() - started) * 1000,
            failure=StageFailure(
                stage="answer",
                error_type=type(exc).__name__,
                message=str(exc),
                retryable=False,
            ),
        )
        write_run_record(record, output_path)
        raise

    response = "Not answerable" if draft.answer is None else draft.answer
    record = RunRecord(
        run_id=run_id,
        document_id=document_id,
        question=request.question,
        status=Status.ok,
        response=response,
        config_hash=config_hash,
        page_ids=request.page_ids,
        latency_ms=(time.perf_counter() - started) * 1000,
    )
    write_run_record(record, output_path)
    export_v2_predictions(
        [Prediction(doc_id=document_id, question=request.question, response=response)],
        prediction_path,
    )
    return record


def export_v2_from_files(samples_path: Path, predictions_path: Path, output_path: Path) -> None:
    samples = [BenchmarkSample.model_validate(item) for item in json.loads(Path(samples_path).read_text())]
    predictions = [Prediction.model_validate(item) for item in json.loads(Path(predictions_path).read_text())]
    validate_prediction_coverage(samples, predictions)
    export_v2_predictions(predictions, output_path)


def _read_exposed_keys(path: Path) -> list[tuple[str, str]]:
    """Read the minimal document/question fields needed for split creation.

    Run records are normally validated through ``read_run_records``.  Split
    creation also accepts audit exports and lightweight exposure logs, so it
    intentionally falls back to the two identifying fields when a line is not
    a complete ``RunRecord``.
    """

    try:
        records = read_run_records(path)
    except ValueError:
        records = []
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid exposure record on line {line_number}") from exc
                document_id = payload.get("document_id", payload.get("doc_id"))
                question = payload.get("question")
                if document_id is None or question is None:
                    raise ValueError(
                        f"exposure record on line {line_number} needs document_id/doc_id and question"
                    )
                records.append((str(document_id), str(question)))
        return records
    return [(record.document_id, record.question) for record in records]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doc-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run one offline fake-runner request")
    smoke.add_argument("--question", required=True)
    smoke.add_argument("--document-id", default="smoke.pdf")
    smoke.add_argument("--image", action="append", required=True, dest="images")
    smoke.add_argument("--output", required=True, type=Path)
    smoke.add_argument("--prediction", required=True, type=Path)

    export = subparsers.add_parser("export-v2", help="validate and copy predictions to V2 format")
    export.add_argument("--samples", required=True, type=Path)
    export.add_argument("--predictions", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)

    gpu = subparsers.add_parser("gpu-smoke", help="run one Qwen3.5-4B request")
    gpu.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    gpu.add_argument("--question", required=True)
    gpu.add_argument("--document-id", default="gpu-smoke.pdf")
    gpu.add_argument("--image", action="append", required=True, dest="images")
    gpu.add_argument("--output", required=True, type=Path)
    gpu.add_argument("--prediction", required=True, type=Path)

    batch = subparsers.add_parser("run-v2", help="run the Qwen baseline over V2 samples")
    batch.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    batch.add_argument(
        "--samples",
        type=Path,
        default=Path("benchmark/mmlongbench-doc-v2/data/samples.json"),
    )
    batch.add_argument("--documents", type=Path, default=Path("data/documents"))
    batch.add_argument("--output", type=Path, default=Path("artifacts/v2-predictions.json"))
    batch.add_argument("--records", type=Path, default=Path("artifacts/v2-runs.jsonl"))
    batch.add_argument("--render-dir", type=Path, default=Path("cache/v2-pages"))
    batch.add_argument("--limit", type=int, default=None)

    audit = subparsers.add_parser("audit-run", help="audit records and prediction coverage")
    audit.add_argument("--records", required=True, type=Path)
    audit.add_argument("--predictions", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)

    split = subparsers.add_parser("create-split", help="create a document-level dev/holdout split")
    split.add_argument("--samples", required=True, type=Path)
    split.add_argument("--exposed-records", type=Path, default=None)
    split.add_argument("--output", required=True, type=Path)
    split.add_argument("--seed", default="mmlongbench-doc-v2-harness-v1")

    prepare = subparsers.add_parser("prepare-phase2", help="freeze the phase-two development selection")
    prepare.add_argument("--samples", required=True, type=Path)
    prepare.add_argument("--baseline-run", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)

    compare = subparsers.add_parser("compare-runs", help="compare paired scored runs")
    compare.add_argument("--runs", required=True, nargs="+", type=Path)
    compare.add_argument("--output", required=True, type=Path)

    score = subparsers.add_parser("score-run", help="validate a pre-judged run")
    score.add_argument("--run-dir", required=True, type=Path)
    score.add_argument("--samples", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)

    judge = subparsers.add_parser("judge-run", help="judge a complete run through OpenRouter")
    judge.add_argument("--run-dir", required=True, type=Path)
    judge.add_argument("--samples", required=True, type=Path)
    judge.add_argument("--model", default="openai/gpt-5.6-luna")
    judge.add_argument("--concurrency", type=int, default=4)

    index = subparsers.add_parser(
        "build-index", help="render documents and build Qwen3-VL page indexes"
    )
    index.add_argument("--config", type=Path, default=Path("configs/experiments/retrieval.toml"))
    index.add_argument("--documents", type=Path, default=Path("data/documents"))
    index.add_argument("--render-dir", type=Path, default=Path("cache/v2-pages"))
    index.add_argument("--output", type=Path, default=Path("artifacts/index"))
    index.add_argument("--models-dir", type=Path, default=Path("models"))
    index.add_argument("--document-id", action="append", dest="document_ids")

    staged = subparsers.add_parser(
        "run-pipeline", help="run the phased embedding/reranking/OCR/answer harness"
    )
    staged.add_argument("--config", type=Path, default=Path("configs/experiments/expanded.toml"))
    staged.add_argument(
        "--samples", type=Path, default=Path("benchmark/mmlongbench-doc-v2/data/samples.json")
    )
    staged.add_argument("--documents", type=Path, default=Path("data/documents"))
    staged.add_argument("--render-dir", type=Path, default=Path("cache/v2-pages"))
    staged.add_argument("--index-manifest", type=Path, default=None)
    staged.add_argument("--run-dir", type=Path, required=True)
    staged.add_argument("--models-dir", type=Path, default=Path("models"))
    staged.add_argument("--limit", type=int, default=None)
    staged.add_argument("--resume", action="store_true", help="resume committed question checkpoints")
    staged.add_argument("--retry-failed", action="store_true", help="retry explicitly failed questions")
    staged.add_argument("--stop-after", type=int, default=None, help="stop after N newly finalized questions")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "smoke":
        missing = [path for path in args.images if not Path(path).is_file()]
        if missing:
            raise SystemExit(f"image does not exist: {missing[0]}")
        run_smoke(args.question, args.document_id, args.images, args.output, args.prediction)
        return 0
    if args.command == "export-v2":
        export_v2_from_files(args.samples, args.predictions, args.output)
        return 0
    if args.command == "audit-run":
        result = audit_records(args.records, args.predictions)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote audit to {args.output}")
        return 0
    if args.command == "create-split":
        exposed = set()
        if args.exposed_records is not None:
            exposed.update(_read_exposed_keys(args.exposed_records))
        result = create_document_split(args.samples, exposed, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote split to {args.output}")
        return 0
    if args.command == "prepare-phase2":
        paths = prepare_phase2_selection(args.samples, args.baseline_run, args.output_dir)
        print("wrote phase-two selections: " + ", ".join(str(path) for path in paths.values()))
        return 0
    if args.command == "compare-runs":
        write_report(compare_runs(args.runs), args.output)
        print(f"wrote comparison to {args.output}")
        return 0
    if args.command == "score-run":
        result = score_run(args.run_dir, args.samples)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"{result['status']}: wrote score report to {args.output}")
        return 0
    if args.command == "judge-run":
        output = judge_run(
            args.run_dir,
            args.samples,
            model=args.model,
            concurrency=args.concurrency,
        )
        print(f"wrote judged rows to {output}")
        return 0
    if args.command == "build-index":
        config = load_config(args.config)
        manifest = build_indexes(
            config,
            args.documents,
            args.render_dir,
            args.output,
            models_dir=args.models_dir,
            document_ids=args.document_ids,
        )
        print(f"wrote index manifest to {manifest}")
        return 0
    if args.command == "run-pipeline":
        config = load_config(args.config)
        run_dir = Path(args.run_dir)
        run_coordinator(
            config,
            args.samples,
            args.documents,
            args.render_dir,
            run_dir,
            index_manifest=args.index_manifest,
            models_dir=args.models_dir,
            limit=args.limit,
            resume=args.resume,
            retry_failed=args.retry_failed,
            stop_after=args.stop_after,
        )
        print(f"wrote staged run to {run_dir}")
        return 0
    if args.command == "gpu-smoke":
        config = load_config(args.config)
        missing = [path for path in args.images if not Path(path).is_file()]
        if missing:
            raise SystemExit(f"image does not exist: {missing[0]}")
        runner = QwenTransformersRunner.from_pretrained(
            config.model.model_id,
            config.model.revision,
            max_new_tokens=config.generation.max_new_tokens,
            do_sample=config.generation.do_sample,
            max_pixels=config.render.max_pixels,
        )
        run_smoke(
            args.question,
            args.document_id,
            args.images,
            args.output,
            args.prediction,
            runner=runner,
            config_hash=config.effective_hash(),
        )
        return 0
    config = load_config(args.config)
    if not args.samples.is_file():
        raise SystemExit(f"samples file does not exist: {args.samples}")
    if not args.documents.is_dir():
        raise SystemExit(f"documents directory does not exist: {args.documents}")
    runner = QwenTransformersRunner.from_pretrained(
        config.model.model_id,
        config.model.revision,
        max_new_tokens=config.generation.max_new_tokens,
        do_sample=config.generation.do_sample,
        max_pixels=config.render.max_pixels,
    )
    predictions = run_v2_batch(
        samples_path=args.samples,
        documents_dir=args.documents,
        output_path=args.output,
        records_path=args.records,
        render_dir=args.render_dir,
        config=config,
        runner=runner,
        limit=args.limit,
    )
    print(f"wrote {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
