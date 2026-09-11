# MMLongBench-Doc V2 Baseline Design

## Goal

Create the first runnable slice of the Long-Horizon Doc Harness: a reproducible, direct-input Qwen3.5-4B baseline evaluated on MMLongBench-Doc V2. The slice must preserve enough typed and operational structure for later retrieval, parsing, verification, training, and benchmark adapters without implementing those optional stages yet.

## Scope and non-goals

The first slice includes PDF page rendering, benchmark sample loading, a model-runner interface, a Qwen3.5-4B Transformers runner, response validation, evidence-quote checks, prediction export, run records, and a smoke/evaluation CLI. It does not include retrieval, reranking, OCR, verification calls, fine-tuning, agent loops, benchmark-specific routing, or a hosted service.

The baseline protocol is direct input: the runner receives the question and a documented set of rendered pages according to configuration. The exact page-selection and image-budget policy is recorded in the run manifest and frozen before development results are used. The initial implementation supports all pages for small smoke documents and explicit page IDs for bounded experiments; it must fail loudly when a requested page is unavailable.

## Evaluation contract

MMLongBench-Doc V2 is the primary target. Its current corpus contains 1,071 questions over 134 documents and uses a semantics-aware LLM judge. V1 and V2 scores must never be mixed. The harness exports the full response text in the evaluator's required `doc_id`, `question`, and `response` shape.

The local validator requires exactly one prediction for every expected `(doc_id, normalized_question)` key, rejects duplicates and unknown keys, and treats operational failures as explicit failed rows rather than silently dropping them. Reference answers, evidence-page labels, question-type labels, and evaluator-only metadata are available to scoring code but never enter model request payloads.

## Architecture

```text
PDF -> PageRenderer -> Page records -> InputBuilder -> ModelRunner
                                              |             |
                                              v             v
                                      request manifest   DraftAnswer
                                                            |
                                                            v
                                                    PredictionExporter
                                                            |
                                                            v
                                                    V2 evaluator JSON
```

`contracts.py` owns Pydantic models and enums. `rendering.py` owns deterministic PDF-to-image conversion. `protocol.py` normalizes benchmark keys, builds model-safe requests, validates prediction coverage, and exports evaluator input. `runner.py` defines the runner protocol and a lazy Transformers implementation so importing the package does not require CUDA or model packages. `records.py` writes validated JSONL run records. `cli.py` provides smoke and export commands while keeping model execution injectable for tests.

Every boundary returns a typed value or a `StageFailure`; exceptions are captured with stage, error type, message, and retryability. Model calls are deterministic by default (`do_sample=False`) and record the exact model revision, prompt version, decoding, image budget, and environment lock hash. No request payload may contain benchmark answer fields or evidence labels.

## Data model

The implementation uses the contracts from the research plan: `Page`, `RankedPage`, `ParsedPage`, `EvidenceSpan`, `DraftAnswer`, `Verification`, and `StageFailure`. The baseline additionally defines `BenchmarkSample`, `ModelRequest`, `Prediction`, `RunRecord`, and `RunStatus`. Internal page IDs are zero-based; displayed page labels are never used as join keys.

`DraftAnswer` permits `answer=None` only when `insufficient_evidence=True`. Evidence spans carry page IDs and exact quotes. Quote validation is a deterministic check against normalized page text supplied by the caller; it reports invalid spans and never silently repairs them. Visual-only evidence remains representable through page IDs in this slice and is expanded to regions in the retrieval milestone.

## Configuration and reproducibility

`configs/baseline.toml` contains the initial prompt version, model ID and revision, page policy, render DPI, maximum image pixels, maximum new tokens, and deterministic decoding settings. `models.lock` records the pinned model revisions from the research plan. `environment.lock` records Python, CUDA, PyTorch, Transformers, and evaluator revisions once the CUDA environment is created. A run manifest includes a SHA-256 hash of these lock files and the complete effective configuration.

The Mac is the source-control and editing environment. CUDA execution is expected on the RunPod RTX 4090. Model imports and GPU allocation happen only inside the runner process. Caches and artifacts use explicit paths from configuration, making the pod filesystem disposable except for the persistent volume.

## Testing

Unit tests run without model weights, CUDA, PDFs, or network access. They cover key normalization, request redaction, duplicate/missing/unknown prediction detection, abstention invariants, quote presence, page rendering metadata validation, JSONL round trips, and a fake runner through the end-to-end smoke command. A separate optional integration command loads the pinned Qwen checkpoint, renders one page, validates one response, and records VRAM and latency; it is skipped when model dependencies or CUDA are unavailable.

## Acceptance criteria

The first slice is complete when:

1. `pytest` passes with no network or GPU dependency.
2. A fake-runner smoke command produces one validated JSONL record and one V2-shaped prediction.
3. The validator rejects duplicate, missing, and unknown benchmark rows.
4. The request builder cannot serialize reference answers, evidence-page labels, or question-type metadata.
5. The optional GPU command can be run with the pinned revision and records its exact configuration, latency, and peak VRAM, or emits a clear dependency/feasibility failure.
6. The project has a documented command for scoring predictions with the pinned V2 evaluator after the user supplies the evaluator's judge credentials.

## Later extension points

Retrieval will implement a `PageSelector` that produces `RankedPage` values and feeds the same `ModelRequest`. OCR will populate `ParsedPage` cache entries. Verification will consume `DraftAnswer` plus the same evidence bundle and produce `Verification`. Additional benchmark adapters will implement the sample and prediction protocols without changing core contracts. Training is blocked until an untrained harness and error taxonomy exist.

