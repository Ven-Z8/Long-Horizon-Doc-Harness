"""Command-line entry points for offline smoke runs and V2 export."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Sequence

from .config import load_config
from .contracts import BenchmarkSample, ModelRequest, Page, Prediction, RunRecord, StageFailure, Status
from .protocol import build_request, export_v2_predictions, validate_prediction_coverage
from .records import write_run_record
from .runner import FakeRunner, ModelRunner, QwenTransformersRunner


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
    config = load_config(args.config)
    missing = [path for path in args.images if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"image does not exist: {missing[0]}")
    runner = QwenTransformersRunner.from_pretrained(
        config.model.model_id,
        config.model.revision,
        max_new_tokens=config.generation.max_new_tokens,
        do_sample=config.generation.do_sample,
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


if __name__ == "__main__":
    raise SystemExit(main())
