# Long-Horizon Doc Harness

This repository implements a reproducible long-document, multimodal QA harness. The first benchmark target is MMLongBench-Doc V2. The historical 103-question baseline is preserved, while the experimental path adds document-scoped retrieval, reranking, Qianfan OCR, bounded evidence bundles, verification, and controlled comparisons.

The four downloaded checkpoints have fixed roles:

| Checkpoint | Harness role |
| --- | --- |
| `Qwen/Qwen3.5-4B` | answer generation and final response verification |
| `Qwen/Qwen3-VL-Embedding-2B` | page and question embeddings |
| `Qwen/Qwen3-VL-Reranker-2B` | question/page relevance scoring |
| `baidu/Qianfan-OCR` | page-to-Markdown transcription |

The staged runner loads one heavyweight checkpoint at a time. Each phase writes an
artifact that the next phase validates, so an RTX 4090 does not need all four
models resident together.


## Source layout

The Python package is organized by responsibility:

```text
src/doc_harness/
├── core/       contracts, configuration, protocol, and run records
├── documents/  PDF rendering, OCR, and evidence bundles
├── models/     generation, retrieval, and reranking adapters
├── workflow/   batch and staged execution, planning, and verification
├── evaluation/ scoring, audits, comparisons, and split creation
└── cli.py      command-line entry point
```

The original flat imports such as `doc_harness.rendering` remain supported for
existing notebooks and scripts; new code can use the grouped paths such as
`doc_harness.documents.rendering`.

## Local setup

```bash
uv sync --extra dev --extra pdf
uv pip install -e .
```

Run the offline tests:

```bash
uv run pytest -q
```

## RunPod setup

For the persistent-volume layout, pinned model and benchmark downloads, CUDA
verification, safe OpenRouter key handling, and restart/resume procedure, follow
[`docs/experiments/runpod-setup.md`](docs/experiments/runpod-setup.md). The
idempotent scripts are `scripts/runpod/bootstrap.sh` and
`scripts/runpod/verify.sh`; they never delete models, PDFs, caches, or run
artifacts.

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

## Full V2 baseline run

The batch command groups questions by source PDF, caches deterministic page renders, and resumes rows already present in its prediction file. Start with a small limit before launching the full 1,071-question run:

```bash
uv run python -m doc_harness.cli run-v2 \
  --config configs/baseline.toml \
  --limit 1 \
  --output artifacts/v2-predictions.json \
  --records artifacts/v2-runs.jsonl \
  --render-dir cache/v2-pages
```

Remove `--limit` for the complete baseline. The model processes every page of each PDF; the processor downsizes each page to the configured `max_pixels` budget and the render cache is reused on resume.

## Four-model staged run

Build the document indexes once. This renders pages and runs the embedding model;
the index stores page IDs, render hashes, model revision, and normalized vectors:

```bash
uv run python -m doc_harness.cli build-index \
  --config configs/experiments/retrieval.toml \
  --documents data/documents \
  --render-dir cache/v2-pages \
  --output artifacts/index
```

Run the focused pipeline on a development subset. `--limit 100` selects exactly
the first 100 benchmark keys in the samples file; omit it only after the staged
smoke run and resource checks succeed:

```bash
uv run python -m doc_harness.cli run-pipeline \
  --config configs/experiments/expanded.toml \
  --samples benchmark/mmlongbench-doc-v2/data/samples.json \
  --documents data/documents \
  --render-dir cache/v2-pages \
  --index-manifest artifacts/index/index-manifest.json \
  --limit 100 \
  --run-dir artifacts/runs/B5-dev-100
```

The run directory contains `manifest.json`, safe selected sample keys,
`stages/retrieval.jsonl`, reranking manifests, OCR outputs, `records.jsonl`, and
`predictions.json`. Raw generations, parse status, termination state, selected
page IDs, OCR cache identities, verification decisions, and failures stay in the
records; runtime failures are never replaced with an abstention string.

The explicit ablation configurations are in `configs/experiments/`:
`baseline-repaired`, `retrieval`, `reranked`, `ocr-visual`, `verified`, and
`expanded`. Compare only runs with identical sample keys and valid judge verdicts:

```bash
uv run python -m doc_harness.cli compare-runs \
  --runs artifacts/runs/B0-dev artifacts/runs/B5-dev \
  --output artifacts/reports/development.json
```

The reproducibility and promotion rules are documented in
`docs/experiments/runbook.md` and the five-stage plan in
`docs/superpowers/plans/2026-09-11-five-stage-harness-implementation.md`.

Graph retrieval is an opt-in experiment and does not alter the baseline. Follow
the [graph retrieval runbook](docs/experiments/graph-retrieval-runbook.md) for
CPU validation, graph construction, and the bounded smoke run.

## V2 scoring

First produce a complete prediction list with one row per `(doc_id, question)`. Validate and export it with:

```bash
uv run python -m doc_harness.cli export-v2 \
  --samples data/samples.json \
  --predictions artifacts/predictions.json \
  --output artifacts/predictions-v2.json
```

Then run the pinned MMLongBench-Doc V2 evaluator from its checkout. The evaluator requires a judge credential supplied by the user through `OPENAI_API_KEY` or `OPENROUTER_API_KEY`; the harness never stores or logs that credential.

Install the evaluator client extra and score the complete prediction file with OpenRouter:

```bash
uv sync --extra eval
export OPENROUTER_API_KEY='your-key'
uv run python benchmark/mmlongbench-doc-v2/eval/evaluate.py \
  artifacts/v2-predictions.json \
  --out artifacts/v2-scored.json
```

The design and implementation plan are in `docs/superpowers/specs/` and `docs/superpowers/plans/`. Model revisions are in `models.lock`; the CUDA environment is intentionally finalized after the first RunPod feasibility probe.
