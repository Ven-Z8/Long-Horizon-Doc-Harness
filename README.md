# Long-Horizon Doc Harness

This repository is the first implementation slice of a long-document, multimodal QA harness. The initial milestone is a reproducible direct-input Qwen3.5-4B baseline for MMLongBench-Doc V2. Retrieval, OCR, reranking, verification, and training are planned extensions and remain behind measured baseline results.

## Local setup

```bash
uv sync --extra dev --extra pdf
uv pip install -e .
```

Run the offline tests:

```bash
uv run pytest -q
```

## Offline smoke run

The smoke command uses a deterministic fake runner and checks that input images exist:

```bash
uv run python -m doc_harness.cli smoke \
  --question 'What is shown?' \
  --image /path/to/page.png \
  --output artifacts/smoke.jsonl \
  --prediction artifacts/smoke-predictions.json
```

The JSONL record contains the effective request metadata, response, latency, and status. The prediction file contains the complete response in the three-field shape accepted by the MMLongBench-Doc V2 evaluator.

## PDF rendering

Render zero-based page IDs with PyMuPDF through the package API:

```python
from pathlib import Path
from doc_harness.rendering import render_pdf

pages = render_pdf(Path('data/documents/example.pdf'), Path('artifacts/example-pages'))
```

## Qwen GPU smoke run

The optional GPU path loads the pinned Qwen3.5-4B revision from `configs/baseline.toml` and requires the `gpu` extra plus a CUDA environment:

```bash
uv sync --extra dev --extra pdf --extra gpu
uv pip install -e .
uv run python -m doc_harness.cli gpu-smoke \
  --question 'What is shown?' \
  --image /path/to/page.png \
  --output artifacts/qwen.jsonl \
  --prediction artifacts/qwen-predictions.json
```

If the checkpoint or CUDA runtime cannot be loaded, the command exits with the underlying dependency or model error and the run record identifies the failed stage.

## V2 scoring

First produce a complete prediction list with one row per `(doc_id, question)`. Validate and export it with:

```bash
uv run python -m doc_harness.cli export-v2 \
  --samples data/samples.json \
  --predictions artifacts/predictions.json \
  --output artifacts/predictions-v2.json
```

Then run the pinned MMLongBench-Doc V2 evaluator from its checkout. The evaluator requires a judge credential supplied by the user through `OPENAI_API_KEY` or `OPENROUTER_API_KEY`; the harness never stores or logs that credential.

The design and implementation plan are in `docs/superpowers/specs/` and `docs/superpowers/plans/`. Model revisions are in `models.lock`; the CUDA environment is intentionally finalized after the first RunPod feasibility probe.
