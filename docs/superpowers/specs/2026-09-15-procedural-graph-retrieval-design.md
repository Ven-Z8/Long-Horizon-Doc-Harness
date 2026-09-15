# Graph-Aware Evidence Retrieval Design

**Status:** Design only. This document authorizes no model download, GPU run, benchmark evaluation, or production-code change until the implementation plan is reviewed.

**Goal:** Add a provenance-preserving document graph and bounded graph-aware retrieval layer that improves cross-page evidence coverage for long-document QA without replacing the current staged baseline or introducing an open-ended agent loop.

**Predecessor:** The four-model staged harness on `feature/mmlongbench-v2-baseline`, currently at commit `014125c`.

**Related research:** The Google-affiliated preprint *Procedural Graphs: Self-Evolving Execution Structures for LLM Agents* motivates localized procedural guidance and validation-gated graph edits. This phase adopts the localized, explicit-state idea but does not claim to reproduce the paper or implement self-evolution.

## 1. Observed starting point

The repository already has a working CPU-testable staged harness for rendering, page retrieval, reranking, OCR, evidence packing, answering, verification, durable records, and RunPod setup. The current CPU suite passes 144 tests. The current retrieval and planning layers already have bounded candidate queues, expansion rounds, and page-level provenance, but they do not represent document structure or explicit cross-page relationships.

The operational recommendation is to run the existing one-page RunPod feasibility probe before any graph experiment. The probe establishes whether the pinned checkpoint, CUDA stack, image processor, and persistent-volume layout work. A full benchmark run remains gated on the 20-question smoke pair and resource inspection.

Current untracked documentation under `docs/pi/` and `docs/research/` is user-owned work and must remain untouched by this phase unless explicitly requested.

## 2. Problem and non-goals

### Problem

Embedding similarity can retrieve pages that are individually relevant while omitting a definition, qualification, exception, reference, or comparison page required to answer correctly. On a 200-page document, the system needs a compact way to follow document relationships without placing the entire document in every prompt.

### Non-goals

- Do not replace the direct-input, retrieval, or phase-two control baselines.
- Do not implement the Google paper's LLM-driven graph mutation, rejection memory, or multi-task self-evolution loop in this phase.
- Do not build a general-purpose knowledge graph, ontology editor, graph database, or web UI.
- Do not add LangChain, LangGraph, a vector database, or a new serving framework.
- Do not use benchmark evidence-page labels, question-type labels, reference answers, or evaluator metadata during inference.
- Do not parse every page with a question-conditioned prompt. Document indexing must be question-independent.
- Do not let document text alter tool policy, file access, prompts, or execution behavior.

## 3. Design principles

1. **Separate map from route.** The document graph records what the document says and where it says it. The procedural state records how the harness is currently investigating a question.
2. **Evidence before edges.** Every semantic edge must point to an exact source page and quote or deterministic structural origin. An ungrounded model guess is not a graph fact.
3. **Local context.** Retrieval expands only a bounded one- or two-hop neighborhood around question-relevant seed pages. The full graph is never serialized into the answer prompt.
4. **Baseline first.** Graph retrieval is an opt-in experiment. Existing configurations and historical artifacts remain byte-for-byte stable.
5. **CPU-testable mechanics.** Graph construction, serialization, validation, expansion, and selection use fake pages and embeddings in unit tests. GPU models are needed only for real indexing and retrieval measurements.
6. **Deterministic caps.** Every expansion has explicit limits for hops, candidates, selected pages, rounds, and wall time. Exhaustion is recorded as unresolved search, not as proof of unanswerability.
7. **Reproducible artifacts.** Graph identity includes source-document hashes, render/text fingerprints, extraction version, relation schema, and configuration hash. A stale graph cannot silently serve a new document.

## 4. Architecture

The current pipeline remains the default:

```text
PDF → render → embed/screen → rerank → OCR selected pages → answer → verify
```

The graph experiment adds an offline document-map artifact and inserts bounded expansion before reranking:

```text
PDF
  ↓
render and question-independent page/text index
  ↓
document graph (cached once per document)
  ↓
embedding seeds
  ↓
graph expansion (bounded neighbors and explicit references)
  ↓
rerank expanded candidates
  ↓
OCR and evidence bundle
  ↓
answer + verification
```

The graph is not required to contain a perfect entity inventory. The first version prioritizes high-value, auditable relationships:

- hierarchy: `contains`, `child_of`;
- order: `follows`, `adjacent_to`;
- explicit document references: `refers_to`;
- semantic support: `defines`, `supports`, `qualifies`, `contradicts`, `depends_on`.

Structural edges come from PDF outlines, headings, page order, table/figure labels, and explicit references. Semantic edges are created only from bounded structured extraction over question-independent page or section text and must retain provenance. If the source has no text layer, a separate one-time OCR/indexing job may supply text; the high-resolution answer OCR remains question-independent and page-bounded.

## 5. Data contracts

Add strict, versioned contracts in `src/doc_harness/documents/graph.py` and keep retrieval algorithms in `src/doc_harness/documents/graph_retrieval.py`. Existing contracts remain readable and public aliases remain compatible.

```python
from typing import Literal, Sequence
from pydantic import Field
from doc_harness.core.contracts import StrictModel

class GraphNode(StrictModel):
    node_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    kind: Literal["document", "section", "page", "block", "claim", "entity"]
    label: str = Field(min_length=1)
    page_ids: list[int] = Field(default_factory=list)
    source_quote: str | None = None
    source_locator: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

class GraphEdge(StrictModel):
    source_id: str = Field(min_length=1)
    relation: Literal[
        "contains", "child_of", "follows", "adjacent_to", "refers_to",
        "defines", "supports", "qualifies", "contradicts", "depends_on"
    ]
    target_id: str = Field(min_length=1)
    page_ids: list[int] = Field(default_factory=list)
    quote: str | None = None
    source_locator: str | None = None
    condition: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

class DocumentGraph(StrictModel):
    schema_version: Literal[1] = 1
    document_id: str = Field(min_length=1)
    source_sha256: str = Field(min_length=1)
    extraction_version: str = Field(min_length=1)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)

class GraphSearchState(StrictModel):
    seed_page_ids: list[int] = Field(default_factory=list)
    candidate_page_ids: list[int] = Field(default_factory=list)
    visited_node_ids: list[str] = Field(default_factory=list)
    active_node_id: str | None = None
    missing_evidence: list[str] = Field(default_factory=list)
    expansion_round: int = Field(default=0, ge=0)
    terminal_reason: str | None = None
```

The implementation must validate that node and edge endpoints exist, all page IDs belong to the graph's document, semantic edges have provenance, and serialized IDs are unique. Structural adjacency may use a deterministic `source_locator` such as `page-order` instead of a quote.

## 6. Indexing and caching

Graph construction is document-scoped and question-independent. A cached artifact directory contains:

```text
cache/graphs/<document_hash>/
  graph.json
  graph-manifest.json
  source-text.jsonl
```

The manifest records document hash, page/render fingerprints, parser or text-extraction identity, graph schema version, relation schema, extraction prompt/version if used, and configuration hash. A graph is reusable only when all identity fields match.

The initial graph builder should be deterministic wherever possible. It creates document/section/page hierarchy, page adjacency, and explicit references without an LLM. Semantic extraction is a separate optional step whose output is rejected if it lacks valid endpoints or provenance. The builder must not require a GPU for unit tests or for documents that already have usable text.

## 7. Retrieval and evidence selection

The graph-aware selector consumes the existing page embedding/reranking interfaces and remains model-independent:

```python
def expand_graph_candidates(
    seed_page_ids: Sequence[int],
    graph: DocumentGraph,
    *,
    max_hops: int,
    max_candidates: int,
    allowed_relations: Sequence[str],
) -> list[int]: ...

def select_connected_pages(
    ranked_pages: Sequence[RankedPage],
    graph: DocumentGraph,
    *,
    selected_k: int,
    require_connection: bool,
) -> list[RankedPage]: ...
```

The default experiment uses embedding seeds, expands at most two hops, caps the candidate union before reranking, and selects no more than the existing evidence-page budget. Deterministic ordering breaks score ties by page ID. The selector may prefer a connected evidence bundle, but it must not discard a higher-scoring isolated page unless the run record explains the connection rule.

The run record adds graph metadata without changing the prediction schema:

- graph manifest hash and schema version;
- seed page IDs;
- expanded candidate IDs and relation types;
- selected page IDs;
- graph search state and terminal reason;
- graph cache hit/miss and extraction status.

The answer and verifier receive the selected pages, OCR, images, and a compact serialized evidence path containing only selected nodes/edges with provenance. They do not receive hidden labels or the full graph.

## 8. Procedural investigation state

Reuse the existing bounded planning and expansion machinery rather than adding an autonomous loop. Extend the state with `active_node_id`, `visited_node_ids`, and `missing_evidence`. A question plan may say “locate definition,” “follow reference,” “check qualification,” or “compare parallel sections,” but these operations are hints for graph retrieval, not executable arbitrary tools.

The planner must fall back safely to the current heuristic plan if a model planning call fails. Graph expansion terminates on evidence resolution, hop/candidate/page/round limits, timeout, or no new nodes. A graph failure falls back to ordinary page retrieval and is recorded as a stage failure or cache miss; it must not become a semantic abstention.

Self-evolution is explicitly deferred. After development traces identify repeated missing-edge or wrong-route failures, a later phase may propose graph or route-policy edits. Any such phase must use a separate held-out validation set, structural checks, acceptance gating, and rejection memory; it must not mutate the factual graph from an unverified answer.

## 9. Configuration

Add a disabled-by-default nested configuration section, for example:

```toml
[graph]
enabled = false
max_hops = 2
max_candidates = 40
require_connection = false
allowed_relations = ["contains", "follows", "refers_to", "defines", "supports", "qualifies", "contradicts"]
extraction_version = "graph-v1"
```

The graph experiment configuration enables this section and uses a new run directory. Existing baseline and phase-two TOML files are not edited in place.

## 10. Failure handling and safety

- Missing or stale graph artifact: record a cache failure and use ordinary retrieval if configured; never silently use a graph from another document revision.
- Invalid edge endpoint, page ID, relation, or provenance: reject the edge and retain a structured extraction error.
- Graph expansion produces no new pages: terminate with `no_new_graph_candidates` and preserve the original seeds.
- Candidate cap reached: record omitted nodes/edges and continue with the deterministic cap.
- Model extraction timeout or malformed JSON: keep structural graph, mark semantic extraction incomplete, and continue without invented edges.
- Evidence bundle cannot fit the configured page/text/image budget: apply the existing omission ledger and let verification see the omission state.
- Document instructions that request tool use, credential access, or policy changes are treated as quoted content only.

## 11. Testing and measurement

### CPU tests

Add tests for:

- strict node, edge, graph, and search-state validation;
- duplicate IDs, unknown endpoints, cross-document pages, missing semantic provenance, and stale graph identity;
- deterministic hierarchy/reference construction;
- one- and two-hop expansion, relation filtering, candidate caps, tie breaking, and no-new-candidate termination;
- connected-page selection and preservation of isolated high-score pages under each configured policy;
- graph cache manifest mismatch and safe ordinary-retrieval fallback;
- graph metadata persistence in run records;
- no benchmark-only fields crossing into graph construction or inference requests.

The complete existing CPU suite must remain green.

### GPU/development experiment

After the one-page feasibility probe succeeds, run a matched development ablation with identical sample keys and judge settings:

1. current retrieval/reranking baseline;
2. graph-enabled candidate expansion;
3. graph-enabled connected evidence selection;
4. both graph changes together.

Report evidence-page recall, any-evidence recall, selected-page precision, cross-page answer accuracy, unsupported-answer rate, latency, peak VRAM, graph cache cost, and failures. Promote the graph only if the paired end-task result meets the repository's existing shipping rule and does not materially regress answerable/unanswerable behavior.

## 12. Delivery sequence

1. Run the existing RunPod verification and one-page GPU probe; record the environment without changing the graph design.
2. Review and approve the implementation plan derived from this spec.
3. Implement contracts, deterministic graph/index artifacts, and CPU tests.
4. Implement graph candidate expansion and connected selection behind the disabled-by-default flag.
5. Add run-record metadata and graph experiment configuration.
6. Run the full CPU suite and inspect the diff.
7. On RunPod, build one document graph, run the 20-question smoke pair, inspect graph artifacts, then run the matched development ablation.
8. Decide whether to keep graph retrieval. Do not implement self-evolution unless the ablation exposes a measured route-learning problem.

## 13. Acceptance criteria

The phase is complete when:

1. The graph schema is strict, versioned, provenance-preserving, and independently testable.
2. A document graph can be built, saved, loaded, and rejected when its identity is stale.
3. Graph expansion is bounded, deterministic, and safely falls back to ordinary retrieval.
4. The default baseline behavior is unchanged when `[graph].enabled = false`.
5. Selected graph evidence and omitted candidates are visible in run records.
6. No benchmark-only metadata enters graph construction, retrieval, prompts, or model requests.
7. The CPU suite passes with the new tests.
8. The development ablation reports whether graph expansion earns its compute and complexity; no score improvement is promised in advance.
