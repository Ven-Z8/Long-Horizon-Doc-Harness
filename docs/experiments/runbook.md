# MMLongBench-Doc V2 harness runbook

This runbook is for measured experiments on the downloaded V2 PDFs. The
historical 103-question artifacts stay immutable and are development data. Never
mix V1 and V2 samples or reuse a scored projection after changing a prompt,
model revision, render budget, or code revision.

## 1. Validate the environment

```bash
uv sync --extra dev --extra pdf --extra gpu --extra eval
uv run pytest -q
```

The model revisions are pinned in `models.lock`. The local checkpoint directories
must be present under `models/`. Keep `OPENROUTER_API_KEY` in the ignored `.env`
or process environment; it is used only by the external evaluator and is never
read by the inference phases.

## 2. Preserve the historical baseline

The saved 103-question run is not rerun or rewritten. Use the audit command to
reconcile its records and predictions:

```bash
uv run python -m doc_harness.cli audit-run \
  --records artifacts/v2-runs.jsonl \
  --predictions artifacts/v2-predictions.json \
  --output artifacts/reports/historical-103-audit.json
```

Its 39.81% accuracy and 39.24% F1 are historical development observations. The
audit separately reports suspected truncation, missing termination metadata,
runtime failures, malformed responses, and judge transport errors.

## 3. Freeze a document split

Keep every question from a document represented in the historical run in the
development partition. Exposed records can be complete JSONL run records or a
minimal JSONL exposure log containing `document_id`/`doc_id` and `question`:

```bash
uv run python -m doc_harness.cli create-split \
  --samples benchmark/mmlongbench-doc-v2/data/samples.json \
  --exposed-records artifacts/v2-runs.jsonl \
  --output artifacts/splits/v1.json \
  --seed mmlongbench-doc-v2-harness-v1
```

Do not inspect holdout answers while choosing prompts, page budgets, or routing.

## 4. Run the four-model phases

The `run-pipeline` command uses these persisted boundaries:

1. render PDFs and embed every page with Qwen3-VL-Embedding-2B;
2. encode safe questions, screen candidates, and save query vectors;
3. load Qwen3-VL-Reranker-2B and write per-question selection manifests;
4. load Qianfan-OCR once, parse the selected-page union, and use identity-keyed OCR caches;
5. load Qwen3.5-4B, answer bounded visual/OCR bundles, and apply deterministic evidence verification.

For a first probe, use `--limit 1` and a fresh run directory. Then use the
development limit (the current stopping point is 100) before removing the limit.
The index can be reused across variants when its manifest identity matches the
source PDF bytes, render settings, embedding revision, and instruction.

```bash
uv run python -m doc_harness.cli run-pipeline \
  --config configs/experiments/expanded.toml \
  --samples benchmark/mmlongbench-doc-v2/data/samples.json \
  --documents data/documents \
  --render-dir cache/v2-pages \
  --index-manifest artifacts/index/index-manifest.json \
  --limit 1 \
  --run-dir artifacts/runs/B5-dev-1
```

Do not assume that the proposed visual pixel budgets fit every document. The
records expose omitted pages, incomplete OCR, generation length, latency, and
verification decisions so a smaller named experiment can be measured if needed.

## 5. Score and compare

Export a complete prediction projection and run the pinned evaluator only after
all selected keys have committed records. Validate judge verdicts before using a
score in a comparison. A timeout, authentication failure, invalid JSON, or
exhausted retry sequence remains an evaluation error; it is never an incorrect
model answer.

```bash
uv run python -m doc_harness.cli compare-runs \
  --runs artifacts/runs/B0-dev artifacts/runs/B5-dev \
  --output artifacts/reports/dev-B0-to-B5.json
```

Promotion to holdout requires a frozen code/config/model identity and a fresh run
directory. Report accuracy, the evaluator-defined F1, evidence recall, complete
evidence coverage, operational failure rate, malformed/truncated output rate,
abstention behavior, latency, peak VRAM, preprocessing cost, and judge cost.
Choose the simplest candidate when measured differences are inconclusive.
