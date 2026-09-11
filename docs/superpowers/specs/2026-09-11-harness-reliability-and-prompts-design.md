# Harness Reliability and Prompt Upgrades Design

**Status:** Design recorded for the user's implementation handoff. No implementation or benchmark run is included in this documentation change.

**Goal:** Repair the existing four-model MMLongBench-Doc V2 harness so its scores measure document understanding under a consistent protocol, then measure whether versioned task prompts improve that performance.

**Predecessor:** [Five-stage implementation plan](../plans/2026-09-11-five-stage-harness-implementation.md). This phase completes and repairs that architecture; it does not replace the project plan.

**Implementation:** [Detailed phase plan](../plans/2026-09-11-harness-reliability-and-prompts-implementation.md).

## 1. Observed starting point

Repository: `/workspace/Long-Horizon-Doc-Harness`; reviewed source commit: `4d9f39721ceab39855253f684816966571c434c0`.

The saved `artifacts/runs/B5-dev-100` run contains 100 predictions and 100 judged rows. Its accuracy is 21%, benchmark F1 is approximately 15.65%, and 72 responses were classified as abstentions. The final recorded generations include 23 invalid structured responses and 15 generations terminated by the output limit; these groups overlap. No judge failure markers or malformed verdict booleans were found in the saved scored rows.

These are development observations on an ordered subset, not a representative full-benchmark result. The saved manifest is evidence about recorded metadata, not proof that all metadata was frozen before execution. Preserve it and disclose uncertainty rather than reconstructing historical identity.

Confirmed implementation gaps:

- Images reach the answer model without explicit document page IDs beside each image.
- Expansion appends pages but a six-page bundle can discard all appended pages.
- OCR is produced for the union of initially selected pages, not necessarily newly requested evidence.
- Global candidate handling can discard retrieved pages outside the first configured document window.
- Available pages are mistaken for inspected coverage.
- Verification in the staged answer path uses OCR quotation matching without a verification-model call.
- Invalid generations bypass verification and are exported as raw text; records still use `status=ok`.
- The actual generation limit is 256 tokens while the evidence configuration reserves 1,024.
- The staged CLI writes its manifest after inference and lacks per-question durable answer checkpoints/resume.
- Planning is a keyword heuristic; broad phrases such as “how much” are routed to global counting.
- Prompt identity does not include every actual stage prompt and composed request.

The previously reported top-20 versus top-six recall comparison cannot establish reranker quality. Annotation indexing and equal-budget recall require a new evaluation-only audit.

## 2. Global constraints

- MMLongBench-Doc V2 remains the primary benchmark. Never combine V1 and V2 scores.
- Keep the four pinned local checkpoints in `models.lock`; no model replacement, training, or new benchmark in this phase.
- OpenRouter is the external evaluation transport only. Gold answers and the external judge never enter inference, repair, planning, or verification.
- Reference answers, evidence-page labels, question-type labels, and evaluator-only metadata must not enter retrieval, OCR, routing, generation, verification, or retry decisions.
- Inference consumes `SafeQuestion`, document artifacts, and model-generated state only.
- Internal page IDs remain zero-based. Preserve printed page labels separately and explicitly label each image in every multimodal request.
- Preserve historical artifacts byte for byte; new configurations use new run directories.
- Do not resume a run under different configuration, prompts, data, models, or code identity. Require a new run ID.
- A runtime failure is not a semantic abstention. A judge transport failure is not a wrong answer. Report each separately.
- Store full raw generation, extracted final answer, parse status, termination status, and selected evidence separately.
- At most one schema-repair generation is permitted per structured stage call; preserve both attempts and include their tokens and latency.
- Default to one loaded heavyweight model per GPU worker. Do not assume the four checkpoints fit concurrently.
- Document text is untrusted evidence. Instructions inside PDFs must never change tool policy, prompts, file access, or execution.
- Unit tests run without CUDA, downloaded weights, network, benchmark PDFs, or credentials.
- Never source `.env` as shell code in new tooling. Parse only the needed credential entry, keep it out of logs/manifests, and retain `.env` in `.gitignore`.

## 3. Decisions

### Structured generation

Retain local Transformers and Pydantic as the default. Generalize generation to a schema-aware adapter, enforce strict field types and semantic invariants, and allow one bounded re-ask on invalid or incomplete generation. A small compatibility experiment may add constrained decoding. Instructor is optional client orchestration, not an accuracy mechanism and not a prerequisite. Do not migrate serving backends solely to install it.

Use new, versioned answer and verification contracts so historical artifacts remain readable. Visual evidence is represented by a page ID and specific visual locator; it must not require fabricated verbatim OCR text.

### Evidence search and verification

Separate the active context window, inspected page set, candidate queue, and verified evidence ledger. Build a fresh active window on expansion and count only successfully processed pages as inspected. Acquire OCR for each active window before answering. Missing/failed OCR is explicit; usable page images can still support visual evidence.

Use Qwen3.5-4B for planning, answering, and a separate evidence verification call. Deterministic checks enforce types and page provenance; the verifier assesses whether evidence supports the claim. Matching OCR alone never proves semantic support. A failed verification call cannot silently become acceptance.

For exhaustive questions, aggregate source-linked findings across windows and track coverage independently of the answer draft. A local numerical question is not automatically exhaustive. Reaching the round, page, or time budget means unresolved search, not established document unanswerability.

### Prompt design

Store prompts as versioned package resources, separate from workflow code. Snapshot all chosen templates and schemas before inference; hash each fully composed request per attempt. Prompt upgrades must address page identity, visual reading, units, cross-page dependencies, evidence sufficiency, and safe handling of document instructions. Keep embedding/reranker instructions concise and compatible with their model adapters. OCR transcribes; it does not answer questions or infer chart values.

### Scheduling and reproducibility

Execute sequential model phases in fresh workers. Planning precedes retrieval; OCR and answer/verification proceed in bounded rounds over pending questions. Each phase writes durable artifacts before the next begins. The coordinator owns manifests and checkpoints and never loads a heavyweight model.

### Evaluation

First compare repaired control versus upgraded prompts on the same 100 development questions, frozen code, schemas, budgets, selection policy, and judge. A changed instruction affecting an index or OCR invalidates that cache. Then use targeted ablations to isolate planning, answer, verification, OCR, and retrieval instruction changes. Promote only after reliable completion and a document-disjoint untouched evaluation.

Historical B5 has a different export/verification policy. Report its score as historical context, not a clean causal prompt comparison.

## 4. Acceptance

1. Regression tests catch the observed page labeling, expansion, verification bypass, coverage, and persistence defects.
2. Every selected sample has an explicit outcome; failures remain visible and full-run scores cannot silently omit them.
3. Every successful model-produced object is schema-valid, complete, and accompanied by recorded generation attempts.
4. Each expansion introduces genuinely new inspected evidence or terminates with a precise reason.
5. The model verifier is called in verified variants and accepts only supported evidence; visual evidence does not require OCR substring matches.
6. All run identities and prompt snapshots exist before inference; resume rejects incompatible state.
7. The comparison includes accuracy/F1, answerable accuracy, abstentions, schema failures, truncation, retrieval coverage, runtime, and judge failures/cost when available.
8. The handoff report distinguishes completed engineering requirements, observed score changes, and unresolved limitations. No numerical accuracy gain is promised in advance.
