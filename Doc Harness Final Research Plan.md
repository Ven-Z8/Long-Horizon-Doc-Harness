# Doc Harness: final build and research plan

**Working target:** build a minimal, benchmark-blind system around `Qwen/Qwen3.5-4B` that improves long multimodal-document QA through evidence retrieval, faithful parsing, grounded answering, and calibrated abstention.

**Available hardware:** one RunPod RTX 4090 with 24 GB VRAM. The Mac remains the editor and Git source of truth; the pod is the CUDA execution environment.

## Honest status

The architecture is credible, but no benchmark result has been earned yet.

- The 4090 is enough for inference with each chosen 2B/4B model, one stage at a time. It is likely enough for LoRA or QLoRA on the 4B reasoner after a memory probe. It is not a promise that full-parameter training, long-context multimodal RL, or all four models resident together will fit.

- Qwen reports **54.2 on MMLongBench-Doc** for Qwen3.5-4B. DocAtlas reports **54.4 direct and 63.7 after end-to-end RL** with the same-size model. Those are original-benchmark results, not automatically MMLongBench-Doc V2 results.

- V2 changed 106 annotations, removed 11 rows in total, and replaced string scoring with a semantics-aware judge. Its repository explicitly says V1 and V2 numbers are not comparable. Therefore, **63.7 is a research reference, not the V2 score to beat**.

- The first valid target is: reproduce a direct Qwen3.5-4B baseline under one frozen V2 protocol, then beat that baseline on a held-out test split. If a public comparison to DocAtlas is desired, run a second, separately reported experiment under the original protocol.

## Success definition

The project succeeds only when all three claims hold:

1. **Benchmark gain:** the complete harness beats its own reproduced direct-input Qwen3.5-4B baseline under the same V2 evaluator, model revision, rendering, prompt, and decoding settings.

2. **Grounding gain:** unsupported-answer behavior improves, measured with abstention precision and recall, without hiding a large loss on answerable questions.

3. **Generalization:** a materially different document set can use the same Python pipeline by adding configuration and prompts only.

A leaderboard number obtained by tuning on test answers, using evidence-page labels during inference, or changing the evaluator is not success.

---

## Frozen V1 architecture

```text
PDF + question
      |
      v
[1. INGEST] render ordered page images
      |
      v
[2. SCREEN] Qwen3-VL-Embedding-2B -> broad candidate set
      |
      v
[3. RERANK] Qwen3-VL-Reranker-2B -> compact evidence set
      |
      v
[4. PARSE] Qianfan-OCR -> page Markdown
      |
      v
[5. ANSWER] Qwen3.5-4B + images + Markdown -> draft + citations
      |
      v
[6. VERIFY] Qwen3.5-4B + same evidence -> accept, correct, or abstain
```

Initial values are `candidate_k=15` and `evidence_k=5`. They are hypotheses, not truths. Both stay in configuration and are tuned only on the development split.

### Deliberate exclusions

- No LangChain or LangGraph.
- No vector database; page embeddings are ordinary cached arrays.
- No open-ended agent loop in V1.
- No benchmark metadata in inference.
- No custom trainer.
- No simultaneous residency requirement for all models.
- No fine-tuning before the inference error taxonomy exists.

---

## Typed contracts: required before pipeline code

Pydantic owns every stage boundary. Instructor is optional glue for generative calls, not the architecture. If Instructor does not work reliably with the selected local server, use the server's JSON-schema decoding or strict JSON parsing with one repair attempt.

```python
from enum import Enum
from pydantic import BaseModel, Field

class Status(str, Enum):
    ok = "ok"
    failed = "failed"

class Page(BaseModel):
    document_id: str
    page_id: int = Field(ge=0)       # zero-based internal index
    image_path: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    render_sha256: str

class RankedPage(BaseModel):
    page_id: int = Field(ge=0)
    score: float
    rank: int = Field(ge=1)
    stage: str                       # screen | rerank

class ParsedPage(BaseModel):
    page_id: int = Field(ge=0)
    markdown: str
    parser_model: str
    parser_revision: str
    prompt_version: str

class EvidenceSpan(BaseModel):
    page_id: int = Field(ge=0)
    quote: str

class DraftAnswer(BaseModel):
    answer: str | None
    evidence: list[EvidenceSpan]
    insufficient_evidence: bool

class VerifyDecision(str, Enum):
    accept = "accept"
    correct = "correct"
    abstain = "abstain"

class Verification(BaseModel):
    decision: VerifyDecision
    final_answer: str | None
    evidence: list[EvidenceSpan]
    reason: str

class StageFailure(BaseModel):
    stage: str
    error_type: str
    message: str
    retryable: bool
```

Rules that types cannot enforce must be checked separately: cited quotes must occur on the cited page; page IDs must belong to the selected evidence bundle; an abstention must have `final_answer=None`; and benchmark labels must never appear in the request payload.

### One run record

Every question writes one validated JSONL record containing:

- document and question IDs from the caller;
- model IDs, exact revisions, prompts, decoding, image budgets, and environment lock hash;
- screen and rerank scores plus selected pages;
- parser cache references;
- draft, verification, and final response;
- per-stage latency, peak VRAM, retry count, and failure;
- no reference answer or test-only evidence labels.

This record is the basis for evaluation, error analysis, and later training-data selection.

---

## RunPod 4090 execution plan

### What the 4090 changes

The Mac feasibility risk is no longer central. CUDA, BF16, and FlashAttention make the pod the better place for model testing and evaluation. The new risk is environment compatibility and careless GPU-hour use.

Use a persistent network volume for Hugging Face caches, datasets, OCR outputs, and experiment records. Keep the repository in Git. Treat the pod filesystem outside the persistent volume as disposable.

### Runtime policy

- Start with **Transformers in BF16** for correctness and the least custom serving logic.
- Load one model per process. Exit the process between stages so CUDA memory is actually released.
- Cache page renders, embeddings, reranker scores, and OCR Markdown by content hash.
- Add vLLM only when baseline correctness is established and throughput is a measured bottleneck.
- Do not allocate a 262K context by default. Selected pages should determine the actual budget.
- Record `nvidia-smi` peak usage and wall time for every feasibility probe.
- Enable RunPod's idle timeout or stop the pod manually after every session.

### Starting model pins

These revisions were resolved from the Hugging Face model API on 2026-09-11. Preserve them in a lock manifest; change a pin only through an explicit experiment.

- `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
- `baidu/Qianfan-OCR` at `623bf5d20d446abdb36606aa4547cd0c18886fe5`
- `Qwen/Qwen3-VL-Embedding-2B` at `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`
- `Qwen/Qwen3-VL-Reranker-2B` at `4bd860ac4f15ad1897a214615cccc700f8f71818`

The lock must also include Python, CUDA, PyTorch, Transformers, vLLM if used, FlashAttention, `qwen-vl-utils`, and the evaluator commit.

### Feasibility gate

Before repository expansion, each model must pass one isolated smoke test on the 4090:

1. Load the exact revision in BF16.
2. Process one real benchmark page.
3. Produce the expected output type.
4. Run a batch at the intended image pixel budget.
5. Record cold-load time, steady-state VRAM, peak VRAM, and latency.
6. Restart and reproduce the result from the lock file.

If one environment cannot satisfy all four model dependencies, use two small locked environments: retrieval and generation. Do not create microservices merely to hide a dependency conflict.

---

## Evaluation protocol that must be fixed first

### Two score tracks, never mixed

**Track A: MMLongBench-Doc V2.** This is the primary scientific result. It uses 1,071 questions over 134 documents and the V2 semantics-aware judge. Report the judge model, provider, reasoning effort, evaluator commit, and abstention policy.

**Track B: original MMLongBench-Doc.** Run this only if comparing directly with published 54.4 and 63.7 DocAtlas numbers. Reproduce the original prompt, document representation, scorer, and split as closely as the public code permits. Label all unresolved protocol differences.

### Split policy

Create splits by **document**, not by question. Questions from the same PDF must not appear across development and held-out sets. Otherwise page content, terminology, and layouts leak across the boundary.

- **Smoke set:** 10–20 questions covering answerable, unanswerable, single-page, cross-page, table, chart, and image cases. Used only to catch broken code.
- **Development set:** fixed document-level subset for prompts, page limits, thresholds, and ablations.
- **Held-out test:** touched only after choices are frozen.
- **External set:** a different corpus for the domain-pack claim.

V2's evidence pages and question-type labels are evaluation-only. They may compute diagnostics after inference, but they may not select pages, route images, alter prompts, set thresholds per question, or enter training examples.

### Metrics

Report end-task results and stage diagnostics together:

- V2 accuracy, precision, recall, and F1;
- answerable and unanswerable results separately;
- single-page and cross-page results separately;
- complete-evidence recall@k and any-evidence recall@k;
- answer accuracy before and after verification;
- abstention precision and recall;
- latency and peak VRAM by stage;
- failures by type, never silently dropped.

Use paired bootstrap confidence intervals on question-level score differences. A small movement smaller than run-to-run or judge variance is not a win.

---

## Research ledger

Each item below is an unresolved question. Resolve it with the named experiment rather than design discussion.

### R0. What exactly are we trying to beat?

**Risk:** treating V1's 63.7 as a V2 target creates a false comparison.

**Experiment:** reproduce Qwen3.5-4B direct input on V2, then run the same predictions twice through the judge to measure judge stability. Separately reproduce the original protocol only if a DocAtlas comparison will be published.

**Decision:** freeze the primary target as the reproduced V2 direct score plus a statistically credible improvement. Keep original-benchmark numbers in a separate table.

### R1. Do the exact checkpoints work together on the 4090?

**Risk:** Qwen3.5 currently asks for recent framework builds; Qianfan uses its own vision stack; Qwen's retriever examples require newer vLLM for serving. A single dependency set may be fragile.

**Experiment:** run the feasibility gate above with one page and one batch per model. Begin with Transformers; test vLLM only after correctness. Save environment locks and minimal logs.

**Decision:** use one environment if reproducible. Otherwise split retrieval from generation. Do not continue to full evaluation until all four stages pass.

### R2. What is the true direct baseline?

**Risk:** “direct input” can mean text extraction, all page images, a PDF-native API, or chunked inputs. These are not equivalent.

**Experiment:** define and compare at least two reproducible baselines:

- all available text layer, truncated only by a documented context policy;
- rendered pages through the official Qwen processor, under an explicit pixel and context budget.

**Decision:** call only a fully specified run the baseline. The cheaper baseline remains useful even if it is not the published-card protocol.

### R3. Can screening recover all required pages?

**Risk:** average page recall can look strong while missing one necessary page on cross-page questions.

**Experiment:** measure both any-evidence and complete-evidence recall at 5, 10, 15, 20, and 30. Evaluate retrieval instructions, image pixel budgets, text-only page embeddings, image-only embeddings, and combined text-image embeddings.

**Decision:** choose the smallest candidate count whose complete-evidence recall is close to its plateau. The initial 98% recall@15 target is aspirational, not guaranteed.

### R4. Does reranking earn its cost?

**Risk:** a 2B cross-encoder adds latency but may reorder already-good candidates or lower complete-evidence recall.

**Experiment:** compare embedding top-5 against reranked top-5 from candidate sets of 10, 15, 20, and 30. Measure end-task score, complete-evidence recall, latency, and VRAM.

**Decision:** keep reranking only if it improves held-out evidence coverage or end-task score beyond noise. A prettier ranking is not enough.

### R5. Is a fixed top five too narrow?

**Risk:** many questions require dispersed evidence, adjacent pages, table continuations, or a page that defines a term used elsewhere.

**Experiment:** compare fixed `k`, confidence-based `k`, adjacency expansion, and diversity-aware selection. Classify misses as retrieval failure or reasoning failure using evidence annotations only after the answer is produced.

**Decision:** prefer the simplest rule that closes a meaningful number of complete-evidence failures. If failures remain common, add one bounded `request_more_pages` action in V2 of the system; do not add a general agent loop.

### R6. Does Qianfan-OCR improve the answer?

**Risk:** excellent OCR benchmark scores do not guarantee end-task gain. Markdown may duplicate visual inputs, lose chart structure, or consume context.

**Experiment:** run three matched conditions: images only, Markdown only, and images plus Markdown. Slice results by tables, charts, text, and cross-page questions. Also manually audit a small set of parser failures.

**Decision:** retain Qianfan only if it improves end-task results or permits a meaningful reduction in image/context cost. Do not fine-tune the parser until a repeated parser-specific bottleneck is shown.

### R7. Does verification correct errors or repeat them?

**Risk:** the same model can endorse its first-pass mistake. Pydantic guarantees a valid shape, not factual independence.

**Experiment:** compare:

- one strict answer call;
- answer plus same-model verification;
- answer plus verification with the draft hidden until evidence is extracted;
- if necessary, an independent verifier checkpoint.

The verifier must return `accept`, `correct`, or `abstain`, cite exact page quotes, and use only supplied evidence. Programmatically check quote presence.

**Decision:** ship the second pass only if final F1 improves and the reduction in false answers exceeds the correct answers it rejects. Otherwise merge the rubric into one call.

### R8. How should abstention be calibrated?

**Risk:** maximizing raw accuracy can encourage guessing, while aggressive abstention can inflate precision and destroy recall.

**Experiment:** on development data, measure the verifier decision, evidence-quote validity, answer consistency across deterministic reruns, and any available score margin. Plot precision-recall for abstention. Do not trust verbal self-confidence alone.

**Decision:** freeze one global policy before held-out evaluation. Domain packs may define the output phrase, but not test-specific thresholds.

### R9. Does thinking mode help?

**Risk:** extra reasoning can help calculations but increase latency, context use, and answer-format drift.

**Experiment:** compare thinking on/off for answer and verify separately with deterministic decoding. Slice by lookup, derive, count, compare, and visual questions.

**Decision:** choose one default per stage. Add routing only if a generic, input-derived rule beats the single default on held-out documents.

### R10. What image and context budgets are best?

**Risk:** high resolution helps small text but increases visual tokens and reduces the number of pages that fit. Long advertised context does not imply useful attention across all pages.

**Experiment:** sweep the processor's supported pixel budget and evidence-page count. Record accuracy, visual tokens, prompt tokens, latency, and peak VRAM. Include small-font tables and charts.

**Decision:** freeze separate screening and reasoning pixel budgets at the accuracy-cost knee, not at the maximum supported resolution.

### R11. Are answer format and judge behavior stable?

**Risk:** verbose evidence can accidentally introduce contradictory values; judge stochasticity can hide small gains.

**Experiment:** compare concise answer-only output against answer plus citations while still passing the full response to V2. Re-score a fixed sample multiple times and manually review disagreements.

**Decision:** use the shortest output that preserves grounding. Report judge/provider settings and uncertainty around close comparisons.

### R12. What training data is legitimate?

**Risk:** training on benchmark questions, answers, corrections, or test evidence invalidates the claim. Self-generated traces can also reinforce model errors.

**Experiment:** create SFT data from non-test documents or synthetic transformations, then filter it with deterministic checks and evidence validation. Track source-document hashes. Run contamination checks against benchmark questions and PDFs.

**Decision:** no test question, reference answer, evidence label, or correction note enters training. If contamination cannot be ruled out, label the result as benchmark-adapted rather than held-out.

### R13. What training fits a single 4090?

**Risk:** 24 GB is comfortable for inference but not automatically for multimodal training with long sequences. NeMo's existence is not proof that a chosen recipe fits this card.

**Experiment:** start with a 16-example LoRA/QLoRA over short sequences, batch size one, gradient accumulation, activation checkpointing, and conservative image tokens. Measure peak VRAM and tokens per second before expanding sequence length or trainable modules.

**Decision:** train only the answer behavior first. Freeze the vision encoder initially. Use a larger rented GPU if the required sequence/image budget cannot fit without destructive truncation. Do not begin RLVR until SFT produces a reproducible held-out gain.

### R14. Does the design generalize?

**Risk:** one extra domain is only a smoke test, and question-level splits can conceal memorization.

**Experiment:** select at least two materially different held-out sources, such as research papers and business reports, or use a separate page-retrieval corpus such as MMDocIR for the retrieval stages. Add only domain configuration and prompts.

**Decision:** zero pipeline-code changes. Report domain-specific results; do not average away a collapse on one domain.

---

## Component shipping rules

Before the first development run, record these provisional rules and do not change them after seeing held-out results:

- **Accuracy feature:** ship when paired end-task improvement is at least 1 absolute point and the bootstrap interval does not show a material regression.
- **Efficiency feature:** ship when latency or VRAM improves by at least 25% while end-task score remains within 0.5 point.
- **Subset guardrail:** no more than 1 absolute point loss on either answerable or unanswerable performance unless overall F1 improves by at least 2 points and the trade-off is explicitly accepted.
- **Stage metric rule:** a stage metric can diagnose a component, but end-task performance decides whether it remains.
- **Failure rule:** schema, OOM, timeout, and parse failures count as failed questions; they are never dropped from the denominator.

These thresholds are engineering defaults, not scientific facts. They may be revised once, using smoke/development variance, before the held-out run.

---

## Build order

### Gate 0: protocol and model feasibility

- Clone and pin the V2 repository and evaluator.
- Download the 134 upstream PDFs and verify the expected filenames.
- Create document-level smoke, development, and held-out manifests.
- Run all four model smoke tests on the 4090.
- Write `environment.lock`, `models.lock`, and a one-page feasibility record.

**Exit:** one command can reproduce each smoke output after a fresh process start.

### Milestone 1: direct baseline

- Implement PDF rendering, Qwen input construction, final response schema, run records, and V2 prediction export.
- Run the smoke set, then the development baseline.
- Freeze prompt, decoding, pixel budget, and evaluator settings.

**Exit:** a scored, reproducible direct baseline with no dropped rows.

### Milestone 2: retrieval

- Add image embedding cache and broad screening.
- Add reranking as a separable option.
- Resolve R3, R4, and R5 before selecting `candidate_k` and `evidence_k`.

**Exit:** evidence selection improves end-task score or clearly reduces compute without violating guardrails.

### Milestone 3: parsing

- Add Qianfan-OCR with page-level cache.
- Run images-only, Markdown-only, and combined ablations.

**Exit:** Qianfan earns its place; otherwise remove it.

### Milestone 4: verification

- Add constrained verifier output and quote checking.
- Tune the abstention policy only on development documents.

**Exit:** verification improves F1 and unsupported-answer behavior; otherwise collapse to one call.

### Milestone 5: held-out result

- Freeze code, prompts, model revisions, and configuration.
- Run the full held-out evaluation once.
- Publish overall, subset, retrieval, latency, and VRAM results with the exact protocol.

**Exit:** the harness beats the frozen direct baseline credibly.

### Milestone 6: training

- Build uncontaminated SFT examples from non-test documents.
- Run the 4090 memory probe and a tiny LoRA/QLoRA experiment.
- Expand only if the small run improves development results.
- Consider RLVR only after SFT and reward reliability are established.

**Exit:** a trained adapter beats the untrained full harness on held-out documents without degrading abstention or external-domain results.

---

## First coding session on the 4090

Do only this:

1. Create the minimal repository and the Pydantic contracts.
2. Pin the V2 evaluator and two source PDFs.
3. Lock the Qwen3.5-4B revision and its working CUDA environment.
4. Render one page and run one official-checkpoint multimodal inference.
5. Validate the result into `DraftAnswer` or a simpler baseline response model.
6. Write one prediction in the V2-required format and score it.
7. Record peak VRAM and commit the exact command.

Do not install the other three models until this path works end to end. Do not build training code. Do not add Pi-specific project abstractions; Pi is the coding agent, not a runtime dependency of the document harness.

## Second and third sessions

**Session 2:** run the stratified smoke set, make failures explicit, and establish the direct baseline.

**Session 3:** install only the 2B embedder, cache page vectors, and measure recall before touching reranking.

That order keeps every new component accountable to a measured problem.

---

## Prompt ceiling

V1 contains three generative prompts only:

1. **Parse:** faithful image-to-Markdown conversion; no question access and no summarization.
2. **Answer:** answer only from supplied pages; return answer, exact evidence quotes, and insufficiency flag.
3. **Verify:** independently extract supporting evidence, then accept, correct, or abstain.

Every prompt change must name the observed failure category, the development result before and after, and cross-domain regressions. No prompt may mention MMLongBench, question IDs, filenames, answer keys, or evidence labels.

---

## Training boundaries

Training is an optimization phase, not the starting point.

- Start with LoRA/QLoRA on the reasoner's language layers; freeze the vision encoder.
- Train answer and evidence behavior before verifier behavior.
- Use short, clean, evidence-linked examples before scaling sequence length.
- Never create positive targets from an unverified model answer.
- Keep source provenance and document hashes for every example.
- Use NeMo AutoModel's Qwen3.5 recipe as a starting implementation, but prove single-4090 memory behavior rather than assuming the published multi-GPU recipe maps directly.
- Do not fine-tune the retriever, reranker, or OCR model until its corresponding ablation identifies it as the bottleneck.
- Treat RLVR as optional. On one 4090, rollout generation and training may be too slow for useful iteration even if they technically fit.

---

## Decisions frozen now

- Primary evaluation: MMLongBench-Doc V2.
- Secondary comparison: original benchmark only under a separate protocol.
- Reasoner and initial verifier: `Qwen/Qwen3.5-4B`.
- Retriever: `Qwen/Qwen3-VL-Embedding-2B`.
- Optional reranker under test: `Qwen/Qwen3-VL-Reranker-2B`.
- Optional parser under test: `baidu/Qianfan-OCR`.
- Images and Markdown both reach the reasoner during the combined condition.
- Pydantic at all stage boundaries; Instructor only where it proves compatible.
- Three generative prompts maximum.
- No benchmark-specific routing.
- No tools in V1; one bounded page-request action is the only planned escape hatch.
- No training until the untrained harness and error taxonomy exist.
- 4090 execution is sequential by stage unless measurements prove safe co-residency useful.

## Decisions deliberately left open

- Exact direct-input representation and baseline score.
- Final screening and evidence page counts.
- Whether reranking ships.
- Whether Qianfan-OCR ships.
- Whether verification is a second call.
- Thinking mode per stage.
- Image pixel budgets and maximum context.
- Abstention calibration policy.
- LoRA/QLoRA configuration and which modules train.
- Whether one 4090 is adequate for useful RLVR.
- Which two external domains form the generalization test.

These are not omissions. They are decisions that require measurements.

## Stop conditions

Pause and redesign rather than adding complexity when any of these occurs:

- complete-evidence recall stays low even at a larger candidate set;
- the answerer fails despite receiving all annotated evidence pages;
- verification improves precision only by collapsing recall;
- OCR adds context and latency without end-task gain;
- the benchmark judge changes conclusions too often for small deltas to be meaningful;
- training gains disappear on document-level holdout;
- the 4090 requires truncation that removes essential visual evidence.

The likely fix depends on the failure: better retrieval for missing evidence, better reasoning or training when evidence is present, stronger abstention when evidence is absent, and a larger GPU only when memory—not architecture—is the blocker.

---

## Sources

- [Qwen3.5-4B official model card](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qianfan-OCR official model card](https://huggingface.co/baidu/Qianfan-OCR)
- [Qwen3-VL Embedding and Reranker repository](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [MMLongBench-Doc V2 repository](https://github.com/vectifyai/mmlongbench-doc-v2/blob/HEAD/README.md)
- [Original MMLongBench-Doc paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ae0e43289bffea0c1fa34633fc608e92-Abstract-Datasets_and_Benchmarks_Track.html)
- [DocAtlas paper](https://arxiv.org/abs/2608.07527v1)
- [DocAtlas public harness](https://github.com/microsoft/docatlas-harness/blob/HEAD/README.md)
- [NeMo AutoModel repository](https://github.com/nvidia-nemo/automodel/blob/HEAD/README.md)
- [NeMo vision-frame sharding guide](https://github.com/nvidia-nemo/automodel/blob/HEAD/docs/guides/cp-vision-frame-sharding.mdx)
- [MMDocIR paper](https://arxiv.org/abs/2501.08828v1)

## Final position

Start coding now, but begin with the feasibility and evaluation gates—not the complete pipeline. The 4090 removes the Mac bottleneck and is enough to establish a serious inference baseline. It does not remove the scientific risks: protocol mismatch, evidence recall, correlated verification errors, abstention calibration, data leakage, and uncertain training economics.

The shortest path to a real result is:

**one scored baseline -> one measured retrieval gain -> one measured parsing gain -> one measured verification gain -> then training.**

Anything that does not move the held-out end-task result, grounding quality, or cost is deleted.
