# Opt-in graph retrieval runbook

This runbook describes the graph-retrieval experiment. Run these commands from
the checkout at `/workspace/Long-Horizon-Doc-Harness`:

```bash
cd /workspace/Long-Horizon-Doc-Harness
```

## CPU validation

Graph construction is question-independent and uses the local PDF text
extractor; it does not load a model or require CUDA. Validate the graph
contracts, configuration, and CLI on CPU before allocating GPU time:

```bash
cd /workspace/Long-Horizon-Doc-Harness
uv run pytest tests/test_graph.py tests/test_graph_retrieval.py tests/test_config.py tests/test_cli.py -q
```

## Build graph artifacts

Build one deterministic, provenance-preserving graph per document. The output
manifest is an input to an enabled pipeline run and must be retained with the
run artifacts:

```bash
cd /workspace/Long-Horizon-Doc-Harness
uv run python -m doc_harness.cli build-graph \
  --config configs/experiments/graph-retrieval.toml \
  --documents data/documents \
  --output artifacts/graphs
```

The graph artifacts are question-independent: they contain document/page
structure and source-grounded references, not benchmark questions, answers,
labels, or evaluator fields. Rebuild them when the source PDF bytes or graph
extraction identity changes. Keep `artifacts/graphs/graph-manifest.json` with
the experiment outputs.

## Twenty-question smoke run

The graph route keeps the current embedding retrieval and reranking behavior as
the control. The embedding screen supplies seed pages, bounded graph expansion
adds candidates, and the existing reranker selects the final evidence bundle.
Run a matched 20-question smoke before any larger ablation:

Gate: the existing one-page GPU feasibility probe must pass before starting
this 20-question smoke. Both the one-page probe and this smoke must pass before
launching any full benchmark.

```bash
cd /workspace/Long-Horizon-Doc-Harness
uv run python -m doc_harness.cli run-pipeline \
  --config configs/experiments/graph-retrieval.toml \
  --samples benchmark/mmlongbench-doc-v2/data/samples.json \
  --documents data/documents \
  --render-dir cache/v2-pages \
  --index-manifest artifacts/index/index-manifest.json \
  --graph-manifest artifacts/graphs/graph-manifest.json \
  --limit 20 \
  --run-dir artifacts/runs/graph-dev-20
```

Inspect the run manifest and records before increasing the limit. In
particular, check cache identities, selected page IDs, graph expansion and
fallback metadata, runtime failures, and peak VRAM. The full benchmark waits
until those checks pass; a smoke completion alone is not evidence that the
full run is safe or comparable.

Graph retrieval is enabled only by
`configs/experiments/graph-retrieval.toml`. Baseline and existing phase-two
configs remain graph-disabled, so their runs continue through the original
retrieval/reranking path and do not require a graph manifest. Do not interpret
an accuracy difference as an improvement until runs have identical sample keys
and judge settings and have been scored together.

GPU work for this experiment is limited to the graph construction smoke and a
matched development ablation after the existing one-page feasibility probe.
No benchmark-only field is added to graph JSON, model requests, or prompts.
