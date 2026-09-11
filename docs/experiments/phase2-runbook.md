# Phase-two runbook

This runbook executes the reliability and prompt phase described in the [implementation plan](../superpowers/plans/2026-09-11-harness-reliability-and-prompts-implementation.md). Run commands from `/workspace/Long-Horizon-Doc-Harness`.

The new configs use the pinned four local checkpoints. `phase2-control.toml` selects repaired control prompts; `phase2-prompts.toml` selects the v2 prompt resources. Both use the same page, OCR, generation, expansion, and judge budgets. Their output directories must remain separate.

Before a real run:

```bash
uv run pytest -q
uv run python -m doc_harness.cli prepare-phase2 \
  --samples benchmark/mmlongbench-doc-v2/data/samples.json \
  --baseline-run artifacts/runs/B5-dev-100 \
  --output-dir artifacts/splits/phase2
```

The preparation command requires exactly 100 unique baseline keys. It preserves their order and writes `dev100.json`, `smoke20.json`, and `dev100-selection.json`. It refuses to overwrite an existing selection directory.

Run the smoke pair first:

```bash
uv run python -m doc_harness.cli run-pipeline \
  --config configs/experiments/phase2-control.toml \
  --samples artifacts/splits/phase2/smoke20.json \
  --documents data/documents --models-dir models \
  --run-dir artifacts/runs/P2-control-smoke20 --limit 20

uv run python -m doc_harness.cli run-pipeline \
  --config configs/experiments/phase2-prompts.toml \
  --samples artifacts/splits/phase2/smoke20.json \
  --documents data/documents --models-dir models \
  --run-dir artifacts/runs/P2-prompts-smoke20 --limit 20
```

Inspect `manifest.json`, `records.jsonl`, `questions/`, and the stage files before judging. Every question must have a checkpoint. A record with `status=failed` is an operational failure, not a semantic unanswerable answer; it remains visible in the run audit.

If a run is interrupted, repeat the exact command with `--resume`. Use `--retry-failed` only for failed questions under the same manifest identity. A prompt, code, model, dataset, or dependency change requires a fresh run directory. `--stop-after N` is useful for a controlled interruption test; it leaves the run incomplete by design.

Then run the matched 100-question pair:

```bash
uv run python -m doc_harness.cli run-pipeline \
  --config configs/experiments/phase2-control.toml \
  --samples artifacts/splits/phase2/dev100.json \
  --documents data/documents --models-dir models \
  --run-dir artifacts/runs/P2-control-dev100 --limit 100

uv run python -m doc_harness.cli run-pipeline \
  --config configs/experiments/phase2-prompts.toml \
  --samples artifacts/splits/phase2/dev100.json \
  --documents data/documents --models-dir models \
  --run-dir artifacts/runs/P2-prompts-dev100 --limit 100
```

The current phase-two verifier is configured to use the local Qwen model for a separate verification call. This call is independent of the external benchmark judge and sees only the question, draft, supplied images, OCR, and coverage record.

Judge each run explicitly through OpenRouter. The wrapper ignores `OPENAI_API_KEY`, reads only `OPENROUTER_API_KEY`, and records the requested judge slug in the run manifest:

```bash
uv run python -m doc_harness.cli judge-run \
  --run-dir artifacts/runs/P2-control-dev100 \
  --samples artifacts/splits/phase2/dev100.json \
  --model openai/gpt-5.6-luna --concurrency 4

uv run python -m doc_harness.cli judge-run \
  --run-dir artifacts/runs/P2-prompts-dev100 \
  --samples artifacts/splits/phase2/dev100.json \
  --model openai/gpt-5.6-luna --concurrency 4
```

The judge wrapper writes `evaluation/judged.json`. It rejects missing predictions, duplicate keys, invalid verdicts, and `judge failed:` markers instead of treating them as wrong answers. If the configured slug is unavailable, stop and choose one replacement for both matched runs; do not silently score one variant with a different model.

Validate and compare:

```bash
uv run python -m doc_harness.cli score-run \
  --run-dir artifacts/runs/P2-control-dev100 \
  --samples artifacts/splits/phase2/dev100.json \
  --output artifacts/runs/P2-control-dev100/evaluation/validated-score.json

uv run python -m doc_harness.cli score-run \
  --run-dir artifacts/runs/P2-prompts-dev100 \
  --samples artifacts/splits/phase2/dev100.json \
  --output artifacts/runs/P2-prompts-dev100/evaluation/validated-score.json

uv run python -m doc_harness.cli compare-runs \
  --runs artifacts/runs/P2-control-dev100 artifacts/runs/P2-prompts-dev100 \
  --output artifacts/reports/phase2-comparison.json
```

Report the official accuracy/F1, answerable accuracy, false abstentions, schema validity before and after repair, output-limit terminations, operational failures, inspected evidence coverage, expansion page changes, model calls/tokens, p50/p95 latency, peak VRAM, OCR cache hits, judge failures, and cost. Include per-question transitions and a document-cluster bootstrap interval for the paired score difference. The old B5 score is historical context; it is not the matched control for this phase.
