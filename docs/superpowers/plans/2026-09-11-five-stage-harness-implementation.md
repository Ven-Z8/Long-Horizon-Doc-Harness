# Five-Stage Document Harness Implementation Plan

> **Execution status:** This plan is being executed in the repository. The historical baseline remains preserved; new staged artifacts use fresh run directories and the pinned model identities below.

**Goal:** Build and measure a reproducible four-model document harness that improves MMLongBench-Doc V2 performance through trustworthy measurement, retrieval and reranking, OCR with focused visual reading, evidence verification, and controlled evaluation.

**Architecture:** Render and index each document once; retrieve and rerank pages for each question; extract relevant content and assemble a bounded evidence bundle; generate and verify an answer with explicit, bounded expansion when needed. Keep inference isolated from reference answers and evaluator metadata. Schedule models in separate phases on the available 24 GB GPU and retain caches, raw outputs, and manifests for reproducibility.

**Tech stack:** Existing Python 3.12 package, Pydantic 2, PyMuPDF, Pillow, Transformers, PyTorch, local pinned Hugging Face checkpoints, NumPy exact vector search initially, JSON/JSONL artifacts, pytest, and the pinned V2 evaluator through OpenRouter.

**Spec:** [Existing baseline design](../specs/2026-09-11-mmlongbench-v2-baseline-design.md). The user-approved five-stage sequence in this document extends that design's “Later extension points”; its original baseline-only exclusions remain applicable to the historical baseline, not the new experimental variants. No training, new benchmark, model replacement, or hosted product is added.

## 1. Starting point and interpretation

The existing package lives at `/workspace/Long-Horizon-Doc-Harness`. All paths below are relative to that root unless explicitly absolute. The implementation branch at planning time is `feature/mmlongbench-v2-baseline`.

The interrupted baseline produced 103 predictions. The saved evaluator report records 41/103 correct, 39.81% accuracy, and 39.24% F1. Median recorded answer latency is approximately 8.1 seconds. These are historical observations on an ordered, document-grouped subset, not a representative full-benchmark estimate or a guaranteed improvement target.

Historical artifacts to preserve:

- `artifacts/v2-predictions.json`
- `artifacts/v2-runs.jsonl`
- `artifacts/v2-scored-103.json`
- `artifacts/v2-scored-103.txt`

The reported 103 `ok` records mean no captured execution exception. They do not establish that every output was complete, schema-valid, or correctly grounded. The runner currently accepts arbitrary nonempty prose when JSON parsing fails. The upstream judge also records repeated API failures as negative verdicts; audit those before treating a saved score as valid.

Current baseline settings: all pages, 144 DPI source rendering, processor limit of 262,144 pixels per page, 256 output tokens, deterministic decoding, and thinking disabled in the latest runner. The recorded config does not yet fully identify every behavior affecting generation. High-resolution all-page input previously exhausted GPU memory on a 23-page PDF; focused reading must be benchmarked rather than assumed to fit.

### Four-model ownership

| Model | Pinned revision | Planned responsibility |
|---|---|---|
| Qwen/Qwen3.5-4B | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | Answer generation, query planning, bounded verification |
| Qwen/Qwen3-VL-Embedding-2B | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` | Page embeddings and question embeddings |
| Qwen/Qwen3-VL-Reranker-2B | `4bd860ac4f15ad1897a214615cccc700f8f71818` | Question/page relevance ranking |
| baidu/Qianfan-OCR | `623bf5d20d446abdb36606aa4547cd0c18886fe5` | Page transcription and structured document extraction |

OpenRouter is only the external evaluation transport. Its judge is not a fifth harness model and never supplies inference evidence. Use the pinned evaluator's model configuration only after confirming it resolves and produces valid verdicts; record the actual provider/model returned by the API when available. Never silently substitute a judge model.

## 2. Global constraints

- MMLongBench-Doc V2 remains the primary benchmark. Never combine V1 and V2 scores.
- Reference answers, evidence-page labels, question-type labels, and evaluator-only metadata must not enter retrieval, OCR, routing, generation, verification, or retry decisions.
- Inference consumes a safe question object with only document identity and question text. Evaluation labels live in a separate adapter.
- Internal page IDs remain zero-based. Preserve printed page labels separately and explicitly label each image in every multimodal request.
- Preserve historical artifacts byte for byte; new configurations use new run directories.
- Store full raw generation, extracted final answer, parse status, termination status, and selected evidence separately. Do not present extraction or repair as the original raw response.
- A runtime failure is not a semantic abstention. A judge transport failure is not a wrong answer. Report each separately.
- Use exact pinned model revisions and record code revision, effective settings, prompt content hashes, dataset hashes, and dependency versions in every run manifest.
- Do not resume a run under different configuration, prompts, data, models, or code identity. Require a new run ID.
- Unit tests run without CUDA, downloaded weights, network, benchmark PDFs, or credentials. Generated-PDF and real-model checks belong to explicitly marked integration tests.
- Default to one loaded heavyweight model per GPU worker. Do not assume the four checkpoints fit concurrently.
- Document text is untrusted evidence. Instructions inside PDFs must never change tool policy, prompts, file access, or execution.
- Never source `.env` as shell code in new tooling. Parse only the needed credential entry, keep it out of logs/manifests, and retain `.env` in `.gitignore`.
- Fine-tuning and additional benchmarks wait until the untrained harness has a frozen, valid evaluation and error taxonomy.
- No score is promised. Promotion requires measured improvement under the same evaluation conditions.

## 3. Delivery order and dependency gates

| Stage | Deliverable | Dependency | Exit gate |
|---|---|---|---|
| 1 | Audited baseline and durable execution | Existing package | Complete provenance, strict status separation, reproducible subset |
| 2 | Retrieval and reranking | Stage 1 | Deterministic page selection and measured evidence recall |
| 3 | OCR and focused visual answering | Stage 2 | Provenance-preserving evidence bundles and measured GPU fit |
| 4 | Verification and bounded expansion | Stage 3 | Grounding checks, coverage-aware abstention, bounded execution |
| 5 | Ablations, held-out evaluation, report | Instrumentation starts in Stage 1; final comparisons need Stage 4 | Valid paired results and a frozen final configuration |

Implement baseline repairs first, then the four-model pipeline. Establish the evaluation split before optimizing stages 2–4. Do not finish all model integrations before testing their individual value.

## 4. File and interface map

| File | Change | Responsibility |
|---|---|---|
| `src/doc_harness/contracts.py` | Modify | Safe question, generation result, evidence regions, execution statuses |
| `src/doc_harness/config.py` | Modify | Strict nested stage settings and budget validation |
| `src/doc_harness/runner.py` | Modify | Raw generation and deterministic decoding; no permissive success fallback |
| `src/doc_harness/batch.py` | Modify | Manifest-checked resume, explicit failures, graceful stopping |
| `src/doc_harness/protocol.py` | Modify | Safe input boundary and strict prediction export |
| `src/doc_harness/records.py` | Modify | Durable records and integrity checks |
| `src/doc_harness/manifests.py` | Create | Run identity, artifact checksums, resume compatibility |
| `src/doc_harness/evaluation.py` | Create | Judge validation, error classification, score export |
| `src/doc_harness/splits.py` | Create | Deterministic document-level development/holdout manifests |
| `src/doc_harness/retrieval.py` | Create | Embedding adapter, index persistence, exact search |
| `src/doc_harness/reranking.py` | Create | Reranker adapter and stable ranking |
| `src/doc_harness/ocr.py` | Create | OCR adapter, extraction validation, cache identity |
| `src/doc_harness/evidence.py` | Create | Budgeted bundles, crop coordinates, page provenance |
| `src/doc_harness/verification.py` | Create | Deterministic checks and model verdict validation |
| `src/doc_harness/planning.py` | Create | Safe question plans and coverage strategy |
| `src/doc_harness/pipeline.py` | Create | Stage orchestration and bounded expansion |
| `src/doc_harness/experiments.py` | Create | Paired comparisons, resource statistics, reports |
| `src/doc_harness/cli.py` | Modify | Thin entry points for the new components |
| `configs/experiments/*.toml` | Create | Frozen, explicit experimental variants |
| `tests/test_*.py` | Create/modify | Component contracts, failure handling, leakage regressions |
| `tests/integration/` | Create | Generated PDF, adapter, and GPU feasibility checks |
| `docs/experiments/` | Create | Runbook and measured reports during implementation |

Keep model imports lazy. `ModelRunner` remains defined in `runner.py`; do not import it from `contracts.py`. Existing `DraftAnswer`, `RankedPage`, `ParsedPage`, and `Verification` should be extended compatibly or migrated with explicit schema versions.

### Shared contracts to introduce in Stage 1

```python
class SafeQuestion(StrictModel):
    document_id: str
    question: str

class GenerationResult(StrictModel):
    raw_response: str
    draft: DraftAnswer | None
    parse_status: Literal['valid', 'invalid', 'empty']
    finish_reason: Literal['eos', 'length', 'error', 'unknown']
    input_tokens: int
    output_tokens: int
    failure: StageFailure | None = None

class RunManifest(StrictModel):
    schema_version: int
    run_id: str
    config_hash: str
    code_hash: str
    dataset_hash: str
    samples_hash: str
    models_hash: str
    prompts_hash: str
    dependencies_hash: str
```

Implementations must import `Literal` and the existing strict base. Do not infer `finish_reason='eos'` just because text is nonempty. Record actual generated-token termination, including the case where EOS is emitted exactly at the limit.

## 5. Stage 1 — Make baseline measurement trustworthy

### Task 1.1 — Freeze and audit the historical run

**Files:** Create `manifests.py`, `tests/test_manifests.py`; modify `records.py`; create `docs/experiments/baseline-audit.md` during execution.

**Interfaces:** `freeze_run(paths: list[Path], destination: Path) -> dict[str, str]`; `audit_records(records_path: Path, predictions_path: Path) -> dict`.

- [ ] Write `test_freeze_preserves_bytes_and_refuses_overwrite`: freeze two synthetic files, compare SHA-256 values, then assert a second write to the same destination raises `FileExistsError`.
- [ ] Write `test_audit_detects_duplicate_missing_and_conflicting_rows`: synthetic normalized duplicate keys, missing records, and a response mismatch must each appear in the audit output.
- [ ] Run `uv run pytest -q tests/test_manifests.py`; confirm the new interfaces are absent before implementation.
- [ ] Implement copying to a new immutable run directory with an inventory of byte counts and checksums. Record original code/config identity when known and mark missing historical metadata as unknown, never reconstructed fact.
- [ ] Audit all 103 predictions, records, and verdicts. Count invalid/truncated-looking response artifacts separately from known runtime failures; use “suspected truncation” for historical outputs without token termination metadata.
- [ ] Run the targeted tests and write the audit, including whether any verdict reason begins with the upstream `judge failed:` marker.
- [ ] Commit the audit tooling and documentation; keep benchmark artifacts ignored.

**Acceptance:** Historical outputs are recoverable unchanged; the audit reconciles all keys and qualifies the original `ok` count.

### Task 1.2 — Separate generation, parsing, and answer validity

**Files:** Modify `contracts.py`, `runner.py`, `protocol.py`, `tests/test_runner_and_records.py`; create `tests/test_generation.py`.

**Interfaces:** `QwenTransformersRunner.generate(request: ModelRequest) -> GenerationResult`; `parse_draft_answer(text: str) -> DraftAnswer` remains strict; migrate batch callers to `generate`.

- [ ] Add tests asserting truncated JSON yields `parse_status='invalid'`, nonempty prose does not become a valid `DraftAnswer`, and an explicit null answer with `insufficient_evidence=True` remains valid.
- [ ] Test that raw text survives unchanged even when a structured answer is extracted, and that actual input page IDs are included beside each image rather than assumed from position.
- [ ] Run `uv run pytest -q tests/test_generation.py tests/test_runner_and_records.py` and confirm the permissive fallback fails these assertions.
- [ ] Implement token counts, termination detection, prompt hashing, explicit thinking configuration, and separate raw/parsed output persistence. Treat an unfinished generation as incomplete even if a partial object can be parsed.
- [ ] Permit at most one schema-repair attempt with its own recorded prompt and output; no gold data and no silent mutation of the first response. A failed repair remains a structured-generation failure.
- [ ] Test repair bounds, empty responses, unsupported evidence pages, contradictory abstention, and lists/numbers where the contract requires a string. Do not silently coerce malformed structures into success.
- [ ] Run targeted tests and a synthetic one-page GPU integration check; commit the contract migration.

**Acceptance:** Every generation is diagnosable. Full response export follows one explicit policy for each variant; never compare raw-prose and extracted-answer scores as if they were identical protocols.

### Task 1.3 — Durable resume, exact stopping, and judge reliability

**Files:** Modify `batch.py`, `records.py`, `cli.py`, `protocol.py`; create `evaluation.py`, `tests/test_resume.py`, `tests/test_evaluation.py`.

**Interfaces:** `assert_resume_compatible(expected: RunManifest, actual: RunManifest) -> None`; `score_run(run_dir: Path, samples_path: Path) -> dict`; CLI `run-v2 --limit 100` defines exactly 100 selected keys, with `--stop-after N` limiting new attempts in an invocation.

- [ ] Test that changed prompt/config/data/code identity rejects resume, successful rows resume without extra inference, and failed attempts remain in history when retried.
- [ ] Test an interrupted prediction-file write using a synthetic failure before atomic rename; the previous JSON file must remain readable. Recover committed records when interruption occurs between JSONL and prediction export.
- [ ] Test exactly 100 selected samples, stop requests between questions, and a crash during inference; no fabricated completion row is allowed.
- [ ] Test judge timeout, invalid JSON, authentication error, and exhausted retries. Each must produce an evaluation error status, never a negative equivalence verdict.
- [ ] Implement atomic projection writes and record-first recovery; define a single writer per run directory. Derive predictions from committed records rather than trusting an independently edited projection.
- [ ] Wrap the pinned evaluator without editing its ignored checkout. Validate verdict schemas and failure markers; cache verdicts by question/reference/response/judge/prompt hashes, not question key alone.
- [ ] Export complete comparable scores only when every expected row has a valid verdict. For operationally failed samples report a conservative full-denominator score with failures counted as incorrect and a separate answer-only diagnostic. Do not manufacture “Not answerable” predictions for runtime failures.
- [ ] Run `uv run pytest -q tests/test_resume.py tests/test_evaluation.py`; validate authentication with one synthetic judge pair; commit.

**Acceptance:** Exact subset stopping works without manual polling; stale verdicts cannot leak across variants; no API failure is mistaken for model performance.

## 6. Stage 2 — Retrieval and reranking

### Task 2.1 — Page embeddings and exact search

**Files:** Create `retrieval.py`, `tests/test_retrieval.py`, `tests/integration/test_embedding_adapter.py`; modify `config.py`, `pyproject.toml` only if direct dependencies are needed.

**Interfaces:** `PageEmbedder.encode_pages(pages: list[Page]) -> numpy.ndarray`; `PageEmbedder.encode_questions(questions: list[SafeQuestion]) -> numpy.ndarray`; `PageIndex.search(vector: numpy.ndarray, top_k: int) -> list[RankedPage]`.

- [ ] Write tests with hand-constructed normalized vectors: nearest-page order is exact, ties use ascending page ID, and a question cannot retrieve pages from another document.
- [ ] Test that changing PDF bytes, render settings, model revision, or embedding instruction invalidates an index; reject nonfinite vectors and dimensional mismatch.
- [ ] Run `uv run pytest -q tests/test_retrieval.py` and confirm missing implementation failures.
- [ ] Inspect the pinned embedding model's bundled documentation and processor implementation before writing the adapter. Use its documented pooling/instruction conventions; never substitute generic mean pooling without evidence it is supported.
- [ ] Implement normalized float32 persistence with document/page IDs and model/render fingerprints. Start with exact cosine search within each question's document; the per-document corpus does not require approximate search infrastructure.
- [ ] Run a real checkpoint smoke with one question and a few synthetic pages; record embedding dimension, finite values, latency, peak VRAM, and reproducibility tolerance.
- [ ] Run tests and commit the adapter plus cache schema.

**Acceptance:** Safe questions produce stable document-scoped candidate rankings; benchmark labels never enter the embedding path.

### Task 2.2 — Reranker and selection manifest

**Files:** Create `reranking.py`, `tests/test_reranking.py`, `tests/integration/test_reranker_adapter.py`; modify `config.py`, `cli.py`.

**Interfaces:** `PageReranker.rank(question: SafeQuestion, pages: list[Page]) -> list[RankedPage]`; `select_pages(question: SafeQuestion, index: PageIndex, reranker: PageReranker, candidate_k: int, selected_k: int) -> list[RankedPage]`.

- [ ] Test candidate deduplication, stable ties, missing page assets, invalid scores, and top-k bounds using a fake scoring adapter.
- [ ] Run `uv run pytest -q tests/test_reranking.py` before adding the module.
- [ ] Inspect the pinned reranker instructions and scoring method. Implement the model's actual relevance scoring rather than treating embedding similarity as reranking.
- [ ] Begin with 20 retrieved candidates and 6 selected pages, each capped by document page count. Persist retrieval scores and reranker scores separately; they are not numerically interchangeable.
- [ ] Serialize question selection manifests so the reranker worker can exit before the answer worker starts. Preserve page IDs and fingerprints across phases.
- [ ] Measure a real reranking batch of 1, then increase only while recorded GPU memory allows. On OOM retry once at a smaller batch; record the retry and do not silently truncate candidates.
- [ ] Run tests and commit.

**Acceptance:** Retrieval and reranking can run independently and feed a deterministic page set into the existing answering path.

### Task 2.3 — Retrieval-only evaluation gate

**Files:** Extend `evaluation.py`; create `tests/test_retrieval_metrics.py`; create `docs/experiments/retrieval-report.md` during execution.

**Interfaces:** `retrieval_metrics(selected: dict[tuple[str, str], list[int]], labelled_samples: list[BenchmarkSample]) -> dict`.

- [ ] Test label-page conversion against a reviewed fixture from the pinned dataset; verify whether its evidence page labels are one-based before any conversion. Never assume that internal zero-based IDs match raw labels.
- [ ] Test two-evidence-page examples: one selected page yields 0.5 evidence recall and incomplete evidence coverage. Empty-evidence samples are excluded from recall denominators and counted separately.
- [ ] Run `uv run pytest -q tests/test_retrieval_metrics.py` before implementation.
- [ ] Implement mean evidence-page recall, fraction with all evidence pages retrieved, and hit rate at k=4, 8, 16, and 20. Report cross-page and single-page strata offline only.
- [ ] Compare embedding-only selection against reranked selection on the frozen development documents. Inspect misses before increasing answer prompt complexity.
- [ ] Document OCR/visual input limits and retrieval costs; commit metrics code and report.

**Acceptance:** Reranking is retained only if its quality or downstream answer benefit justifies its cost. Report negative results; do not force every model into the final variant merely because it was downloaded.

## 7. Stage 3 — OCR and focused visual reading

### Task 3.1 — OCR adapter and extraction cache

**Files:** Create `ocr.py`, `tests/test_ocr.py`, `tests/integration/test_ocr_adapter.py`; extend `ParsedPage` in `contracts.py`.

**Interfaces:** `PageParser.parse(page: Page) -> ParsedPage`; `parse_cached(page: Page, parser: PageParser, cache_dir: Path) -> ParsedPage`.

- [ ] Test cache hits for identical render/model/prompt identity and misses for each changed fingerprint.
- [ ] Test malformed extraction, empty output, multi-page cache collisions, and OCR worker exceptions. Failed extraction must not create a successful cache entry.
- [ ] Run `uv run pytest -q tests/test_ocr.py` before implementation.
- [ ] Inspect the pinned Qianfan-OCR model's documented loading, prompts, and output schema. Record any required remote code hash and dependency additions. If the checkpoint cannot run on the available hardware, record the blocker and resource requirement rather than substituting a model.
- [ ] Extend parsed output with document ID, render hash, extraction status, raw-output path, and OCR configuration hash. Retain table/layout structure that the model actually provides; do not invent coordinates or confidence scores.
- [ ] Parse the union of selected pages for all questions in a batch while the OCR model is loaded. Use explicit token limits and record truncated pages as incomplete extraction.
- [ ] Run real synthetic text/table checks, measure peak VRAM and per-page cost, then commit.

**Acceptance:** OCR outputs can be traced to exact source page bytes and cannot masquerade as verified truth.

### Task 3.2 — Evidence bundles and high-resolution budgets

**Files:** Create `evidence.py`, `tests/test_evidence.py`; modify `contracts.py`, `rendering.py`, `runner.py`, `config.py`.

**Interfaces:** `build_bundle(question: SafeQuestion, pages: list[Page], parsed: list[ParsedPage], budget: EvidenceBudget) -> EvidenceBundle`.

Define `EvidenceBudget` with maximum pages, per-image pixels, total image pixels, text tokens, and reserved output tokens. Define `EvidenceBundle` with document ID, question, included page IDs, images/crops, OCR content, omitted items with reasons, and a content hash. Define `EvidenceRegion` with page ID and normalized `(x0, y0, x1, y1)` coordinates satisfying `0 <= x0 < x1 <= 1` and `0 <= y0 < y1 <= 1`.

- [ ] Test page-ID preservation across reordered pages and crops, crop bounds, OCR from another document, and token-budget overflow.
- [ ] Test that the packer reports exclusions explicitly and never deletes a table's middle rows without an omission marker.
- [ ] Run `uv run pytest -q tests/test_evidence.py` before implementation.
- [ ] Start focused visual experiments at 6 pages, at most 1,048,576 pixels per page, at most 6,291,456 total image pixels, 12,000 OCR text tokens, and 1,024 output tokens. These are proposed probe limits, not verified GPU fit. Use the actual processor/tokenizer to validate context usage.
- [ ] Keep baseline source renders intact; produce separately fingerprinted focused renders/crops. If no reliable region coordinates exist, supply the full selected page rather than a guessed crop.
- [ ] Put page IDs next to image content, delimit OCR as evidence, and ask the reasoner to resolve OCR/visual disagreements from the original image. Preserve disagreement in the trace.
- [ ] Probe maximum configured bundle size. If it cannot fit, lower a named experiment's budget and create a new manifest before scoring; no unrecorded per-sample downscaling.
- [ ] Run tests and commit.

**Acceptance:** Every quoted passage and visual region has source provenance; the actual GPU feasibility envelope is documented.

### Task 3.3 — Phased four-model execution

**Files:** Create `pipeline.py`, `tests/test_pipeline.py`; modify `batch.py`, `cli.py`, `records.py`.

**Interfaces:** `run_pipeline(question: SafeQuestion, services: PipelineServices, config: HarnessConfig) -> PipelineResult`. `PipelineServices` owns the embedder, index, reranker, parser, answer runner, and verification adapter handles; `PipelineResult` contains answer generation, final response, stage attempts, provenance hashes, and failures.

- [ ] Test with fake adapters that phases consume prior manifests, labels are absent, failures propagate, and OCR results survive worker restarts.
- [ ] Run `uv run pytest -q tests/test_pipeline.py` before implementation.
- [ ] Execute page/question embedding in one worker, reranking in a second, OCR for the selected-page union in a third, and answering in a fourth. Each worker exits before loading the next model so GPU memory is released reliably.
- [ ] Implement bounded queues and atomic stage manifests; reuse computation shared by questions in the same document. Do not reuse answers across different questions.
- [ ] Record cold-start, cache-hit, and end-to-end latency separately, including preprocessing amortized across questions.
- [ ] Run a small synthetic multi-document integration job followed by a development subset; commit.

**Acceptance:** All four downloaded models have exercised their intended roles in a traceable run without simultaneous residency assumptions.

## 8. Stage 4 — Verification and broader search

### Task 4.1 — Deterministic and model-based evidence verification

**Files:** Create `verification.py`, `tests/test_verification.py`; extend `Verification` and `EvidenceSpan` compatibly.

**Interfaces:** `check_evidence(draft: DraftAnswer, bundle: EvidenceBundle) -> EvidenceCheck`; `verify_answer(question: SafeQuestion, draft: DraftAnswer, bundle: EvidenceBundle) -> Verification`.

Define `EvidenceCheck` with unknown-page citations, unmatched OCR quotes, invalid regions, uncovered answer claims, and a completeness flag. Extend evidence to distinguish quoted text and visual regions; a visual claim must not need a fabricated textual quote.

- [ ] Test unknown page IDs, empty quotes, valid normalized text matches, altered numeric values, and valid visual regions without OCR text.
- [ ] Test that an exact quote match proves transcription consistency only, not semantic correctness; require source-image checking when the claim depends on OCR accuracy.
- [ ] Run `uv run pytest -q tests/test_verification.py` before implementation.
- [ ] Implement deterministic checks first. Then use Qwen3.5 in a separate verifier call with question, draft, and actual evidence. Never present reference answers or external judge verdicts.
- [ ] Constrain verifier output to accept, correct, or abstain plus supported evidence and a short reason. Unsupported correction returns insufficient evidence for expansion; do not accept new facts from the verifier alone.
- [ ] Record draft and final answers so the evaluation can count wrong-to-right and right-to-wrong changes. Treat same-model verification as correlated with generation, not independent proof.
- [ ] Run fake-adapter tests and a few visual/text development cases; commit.

**Acceptance:** Verification cannot invent citations or silently erase the original answer, and its added value can be measured.

### Task 4.2 — Bounded planning, expansion, and global coverage

**Files:** Create `planning.py`, `tests/test_planning.py`, `tests/test_expansion.py`; extend `pipeline.py`, `config.py`.

**Interfaces:** `plan_question(question: SafeQuestion) -> QuestionPlan`; `expand_evidence(plan: QuestionPlan, state: SearchState) -> ExpansionDecision`.

Define `QuestionPlan` with search queries, local/global scope, and required operations. Define `SearchState` with selected/visited pages, page count, attempts, remaining budgets, and unresolved evidence checks. Define `ExpansionDecision` with additional page IDs and a reason, or an explicit terminal state.

- [ ] Test that a query asking “how many throughout the document” triggers coverage-aware handling, while a named page lookup can remain local. Use question text only, never benchmark question labels.
- [ ] Test no repeated page work, two-round expansion limits, timeout, exhausted context, and terminal incomplete coverage.
- [ ] Run `uv run pytest -q tests/test_planning.py tests/test_expansion.py` before implementation.
- [ ] Start with two expansion rounds and at most 24 unique focused pages for local questions. Retrieve with reformulated queries and add adjacent pages only for unresolved continuity. Each round reuses cached embeddings/OCR and records why pages were added.
- [ ] For global questions inspect every page in bounded batches. Create a page ledger of counted items and source regions, deduplicate repeated items, and aggregate with deterministic code where possible. A top-k selection is not evidence of exhaustive coverage.
- [ ] Set an initial per-question wall-time ceiling of 300 seconds, including expansions. If global coverage cannot finish within the ceiling, mark incomplete coverage; do not produce a confident exhaustive count.
- [ ] Enforce process-level deadlines for model calls, saving state before worker termination. Retry OOM only under an explicitly recorded fallback profile, preserving the original failure.
- [ ] Run an integration case requiring a second page and a synthetic global counting case; commit.

**Acceptance:** Search is finite, resumes safely, and distinguishes unsupported answers from uninspected evidence. No open-ended agent loop is introduced.

## 9. Stage 5 — Controlled measurement and final evaluation

### Task 5.1 — Freeze development and holdout documents before tuning

**Files:** Create `splits.py`, `tests/test_splits.py`; create ignored `artifacts/splits/v1.json` during execution.

**Interfaces:** `create_document_split(samples_path: Path, exposed_keys: set[tuple[str, str]], seed: str) -> dict`.

- [ ] Test that all questions from an exposed document stay in development, including unanswered questions from that document. No document may occur in both partitions.
- [ ] Test stable output under sample reordering and unchanged data, and different dataset hashes producing incompatible split manifests.
- [ ] Run `uv run pytest -q tests/test_splits.py` before implementation.
- [ ] Mark every document represented in the 103 inspected rows as development. Assign remaining documents by sorted SHA-256 of `seed + document_id`, using seed `mmlongbench-doc-v2-harness-v1` and alternating assignments to development and holdout.
- [ ] Persist exact keys, document IDs, hashes, and counts. Do not inspect holdout answers to rebalance or select difficult examples. Report document types later as descriptive metadata.
- [ ] Label all subsequent analysis on the 103 rows as development analysis. Historical access to other labels must be disclosed in the final report.
- [ ] Run tests and commit split tooling.

**Acceptance:** Held-out documents are excluded from prompt, budget, and routing optimization.

### Task 5.2 — Ablation matrix and comparable metrics

**Files:** Create `experiments.py`, `tests/test_experiments.py`, six explicit files in `configs/experiments/`; modify `evaluation.py`.

**Interfaces:** `compare_runs(run_dirs: list[Path], split_path: Path) -> dict`; `write_report(comparison: dict, output: Path) -> None`.

| Variant | Configuration file | Behavior |
|---|---|---|
| H0 | Historical manifest only | Original 103-row run, preserved and audited |
| B0 | `baseline-repaired.toml` | All-page baseline with repaired instrumentation and explicit decoding |
| B1 | `retrieval.toml` | Embedding retrieval with focused page input |
| B2 | `reranked.toml` | B1 plus reranking |
| B3 | `ocr-visual.toml` | B2 plus OCR alongside the same visual budget |
| B4 | `verified.toml` | B3 plus verification |
| B5 | `expanded.toml` | B4 plus bounded expansion and global coverage |

- [ ] Test rejection of mismatched sample sets, judge models/prompts, unresolved judge failures, and stale verdict-cache entries.
- [ ] Test a paired fixture with known corrected and regressed questions. Both transitions must appear in the report.
- [ ] Run `uv run pytest -q tests/test_experiments.py` before implementation.
- [ ] Freeze shared generation and focused-input settings across B1–B5 where possible. Record B0→B1 as a combined selection/resolution change. Run a matched-resolution comparison on a bounded development subset if attributing improvement specifically to retrieval.
- [ ] Report accuracy and upstream-defined F1 with their denominators; evidence recall; complete-evidence coverage; operational failure rate; malformed/truncated response rate; abstention precision/recall; median/p95 latency; peak VRAM; preprocessing costs; and judge cost when returned.
- [ ] Read the pinned `eval/metrics.py` implementation to reproduce its F1 definition exactly. Never substitute token-overlap F1 under the same label.
- [ ] Compute uncertainty by resampling documents, keeping paired variant results together, using 2,000 seeded bootstrap replicates. Mark subgroup estimates with small document counts as exploratory.
- [ ] Keep failure rates on the full selected denominator. Expose partial judge coverage explicitly; block any headline comparable score until judge errors are resolved.
- [ ] Run all variants on identical development keys and produce a report with examples linked to saved page evidence; commit tooling and report.

**Acceptance:** Every claimed improvement is traceable to a controlled comparison; a negative or negligible addition can be disabled in the final configuration.

### Task 5.3 — Freeze candidate and evaluate once on holdout

**Files:** Extend `experiments.py`, `cli.py`; create `docs/experiments/final-harness-report.md` and `docs/experiments/runbook.md` during execution.

- [ ] Select a candidate on development results using accuracy as the primary quality criterion, with F1, failure rate, and resource costs reported alongside it. Prefer the simpler configuration when differences are inconclusive.
- [ ] Freeze exact code, prompts, model revisions, settings, and sample manifests before holdout inference. Use fresh output directories and log the freeze timestamp.
- [ ] Run the repaired baseline and selected candidate on the same holdout documents, then judge with the same validated model/protocol.
- [ ] Verify complete sample-key coverage, no runtime-to-abstention conversion, and zero unresolved judge errors before publication of the comparison.
- [ ] Report paired accuracy/F1 changes, document-level uncertainty, resource costs, and failures. If the candidate fails to improve, record that outcome and return to development work; the inspected holdout cannot remain an untouched test set for repeated tuning.
- [ ] Run the frozen candidate on the full benchmark only after the holdout report. Label the full-corpus result as including development documents; do not claim it is entirely unseen.
- [ ] Document reproduction commands, artifact hashes, known limitations, and the next resource need, if any. Commit documentation and code after relevant tests pass.

**Acceptance:** A reader can reproduce and distinguish the historical subset score, development improvements, held-out comparison, and final full-corpus score.

## 10. Planned CLI and artifact contracts

The following commands are implementation targets; they are not all available today. New CLI tests must exercise argument validation with fake services before real model execution.

```bash
uv run python -m doc_harness.cli audit-run --run-dir artifacts/runs/historical-103
uv run python -m doc_harness.cli create-split --samples benchmark/mmlongbench-doc-v2/data/samples.json --exposed-run artifacts/runs/historical-103 --output artifacts/splits/v1.json
uv run python -m doc_harness.cli build-index --config configs/experiments/retrieval.toml --documents data/documents
uv run python -m doc_harness.cli run-pipeline --config configs/experiments/expanded.toml --split artifacts/splits/v1.json --partition development --limit 100 --run-dir artifacts/runs/B5-dev
uv run python -m doc_harness.cli score-run --run-dir artifacts/runs/B5-dev
uv run python -m doc_harness.cli compare-runs --runs artifacts/runs/B0-dev artifacts/runs/B5-dev --output artifacts/reports/development.json
```

Each new run directory contains `manifest.json`, `selected-samples.json`, `attempts.jsonl`, `predictions.json`, `stage-manifests/`, `raw/`, `evaluation/`, and `report.json`. All successful output projections must be rebuildable from committed records. Caches live outside run directories but every cache hit records its immutable content identity.

## 11. Resource plan and input requests

Proceed during implementation with the existing 4090, downloaded checkpoints, PDF corpus, and configured OpenRouter credential. Do not request the same resources again.

| Trigger | Action before asking the user | Input needed only if unresolved |
|---|---|---|
| OCR checkpoint cannot load | Record dependency and batch-1 VRAM probe | Additional GPU capacity or supported environment |
| Focused bundles exceed GPU budget | Measure smaller explicit experiment profiles | Larger GPU if acceptable quality requires it |
| Full-run judge cost cannot be bounded | Measure a small valid judge batch and estimate remaining calls | Spending limit if account/session limits are unknown |
| Cache storage exceeds available disk | Inventory projected render/OCR sizes and reusable artifacts | Additional persistent storage |
| Model requires additional gated access | Verify official checkpoint requirements without exposing credentials | User-provided access through a private environment entry |

Ask resource questions with measured evidence and a concrete affected milestone. Do not change model identities, benchmark protocol, or the project goal to bypass a resource constraint silently.

## 12. Review checklist and completion criteria

- [ ] Stage 1 separates raw output, schema validity, runtime success, and judge success.
- [ ] Resume identity includes prompts, code, dataset, dependencies, and model revisions.
- [ ] Retrieval metrics correctly map benchmark page labels and stay outside inference.
- [ ] All four adapters have passed real-checkpoint feasibility probes, or blockers are explicitly documented.
- [ ] OCR and crops preserve source provenance; unsupported coordinates are never fabricated.
- [ ] Global questions use coverage-aware processing and bounded resource policies.
- [ ] Verification records both improvements and harmful corrections.
- [ ] The 103 previously inspected questions and their documents remain development data.
- [ ] Ablations use the same sample keys and judge configuration, with full failure accounting.
- [ ] Final claims distinguish development, holdout, and full-corpus results.

**Implementation status:** The runtime now includes the typed contracts, strict generation provenance, audited baseline tooling, retrieval/reranking/OCR/evidence adapters, bounded verification and expansion, deterministic splits, ablation comparison, and the `build-index`/`run-pipeline` CLI phases. Real one-page probes have passed for all four downloaded checkpoints. Full benchmark execution remains a measured next run after a small staged smoke and resource check.
