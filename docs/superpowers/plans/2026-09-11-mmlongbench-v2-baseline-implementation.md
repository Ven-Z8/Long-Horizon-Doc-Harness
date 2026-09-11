# MMLongBench-Doc V2 Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Build a runnable, reproducible direct-input Qwen3.5-4B baseline for MMLongBench-Doc V2 with typed contracts, PDF rendering, strict prediction validation, and model-independent tests.

**Architecture:** A small Python package separates contracts, configuration, rendering, protocol validation, model execution, and run records. The baseline runner receives a question plus selected page images and returns a validated `DraftAnswer`; the protocol layer exports full responses in V2’s required shape. Optional CUDA/model imports remain lazy so unit tests run offline.

**Tech Stack:** Python 3.12+, Pydantic 2, PyMuPDF, Pillow, pytest, TOML configuration, Hugging Face Transformers as an optional GPU extra.

**Spec:** `docs/superpowers/specs/2026-09-11-mmlongbench-v2-baseline-design.md`

## Global Constraints

- MMLongBench-Doc V2 is the primary evaluation; V1 and V2 scores are never mixed.
- Model requests must not contain reference answers, evidence labels, question types, or benchmark-only metadata.
- Internal page IDs are zero-based and all prediction keys are `(doc_id, normalized_question)`.
- Operational failures are explicit rows and are never silently dropped.
- Unit tests require no CUDA, model weights, PDFs, network, or evaluator credentials.
- Qwen3.5-4B is loaded lazily and only in the optional GPU path.
- Do not implement retrieval, OCR, reranking, verification, training, or agent loops in this plan.

---

### Task 1: Scaffold package and typed contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/doc_harness/__init__.py`
- Create: `src/doc_harness/contracts.py`
- Create: `tests/test_contracts.py`

**Interfaces:**
- Produces `Page`, `RankedPage`, `ParsedPage`, `EvidenceSpan`, `DraftAnswer`, `Verification`, `StageFailure`, `BenchmarkSample`, `ModelRequest`, `Prediction`, and `RunRecord` Pydantic models for all later tasks.

- [ ] **Step 1: Write the failing tests**

```python
from pydantic import ValidationError
import pytest

from doc_harness.contracts import DraftAnswer, EvidenceSpan, Page, StageFailure


def test_draft_answer_requires_insufficiency_for_missing_answer():
    with pytest.raises(ValidationError):
        DraftAnswer(answer=None, evidence=[], insufficient_evidence=False)


def test_page_rejects_negative_page_ids_and_zero_dimensions():
    with pytest.raises(ValidationError):
        Page(document_id="doc.pdf", page_id=-1, image_path="p.png", width=0,
             height=100, render_sha256="abc")


def test_evidence_span_and_failure_are_typed():
    span = EvidenceSpan(page_id=0, quote="Total debt")
    failure = StageFailure(stage="render", error_type="IOError",
                           message="missing PDF", retryable=False)
    assert span.page_id == 0
    assert failure.retryable is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_contracts.py`

Expected: FAIL because `doc_harness.contracts` does not exist.

- [ ] **Step 3: Implement the minimal contracts**

Define the enums and models named in the Interfaces block. Use Pydantic field constraints (`ge=0`, `gt=0`, `ge=1`), a model validator enforcing `answer is None` iff `insufficient_evidence` is true, and `extra="forbid"` on request models so benchmark-only fields cannot be smuggled in.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_contracts.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/doc_harness/__init__.py src/doc_harness/contracts.py tests/test_contracts.py
git commit -m "feat: add typed document harness contracts"
```

### Task 2: Add configuration and reproducibility hashing

**Files:**
- Create: `src/doc_harness/config.py`
- Create: `configs/baseline.toml`
- Create: `models.lock`
- Create: `environment.lock`
- Create: `tests/test_config.py`

**Interfaces:**
- `load_config(path: Path) -> HarnessConfig`
- `lock_hash(paths: Iterable[Path]) -> str`
- `HarnessConfig.effective_hash() -> str`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from doc_harness.config import HarnessConfig, load_config, lock_hash


def test_load_config_preserves_baseline_protocol(tmp_path: Path):
    config_path = tmp_path / "baseline.toml"
    config_path.write_text(
        '[model]\nmodel_id = "Qwen/Qwen3.5-4B"\nrevision = "rev"\n'
        '[render]\ndpi = 144\npage_policy = "explicit"\n'
        '[generation]\nmax_new_tokens = 128\ndo_sample = false\n'
    )
    config = load_config(config_path)
    assert config.model.model_id == "Qwen/Qwen3.5-4B"
    assert config.render.dpi == 144
    assert config.generation.do_sample is False
    assert len(config.effective_hash()) == 64


def test_lock_hash_changes_when_a_lock_changes(tmp_path: Path):
    lock = tmp_path / "lock"
    lock.write_text("one")
    first = lock_hash([lock])
    lock.write_text("two")
    assert lock_hash([lock]) != first
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_config.py`

Expected: FAIL because configuration code does not exist.

- [ ] **Step 3: Implement configuration**

Define nested Pydantic settings for model, render, generation, and paths. Parse TOML with the standard-library `tomllib`; resolve relative paths against the config file directory. Hash canonical JSON of effective settings and each lock file’s bytes with SHA-256.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_config.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_harness/config.py configs/baseline.toml models.lock environment.lock tests/test_config.py
git commit -m "feat: add reproducible baseline configuration"
```

### Task 3: Implement deterministic PDF rendering

**Files:**
- Create: `src/doc_harness/rendering.py`
- Create: `tests/test_rendering.py`

**Interfaces:**
- `render_pdf(pdf_path: Path, output_dir: Path, dpi: int = 144, page_ids: Sequence[int] | None = None) -> list[Page]`
- `validate_pages(pages: Sequence[Page]) -> None`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path
import pytest

from doc_harness.rendering import render_pdf, validate_pages


def test_render_pdf_rejects_an_unavailable_explicit_page(tmp_path: Path):
    pdf = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError):
        render_pdf(pdf, tmp_path / "pages", page_ids=[0])


def test_validate_pages_rejects_duplicate_page_ids():
    from doc_harness.contracts import Page
    pages = [
        Page(document_id="doc.pdf", page_id=0, image_path="a.png", width=10,
             height=10, render_sha256="a"),
        Page(document_id="doc.pdf", page_id=0, image_path="b.png", width=10,
             height=10, render_sha256="b"),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        validate_pages(pages)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_rendering.py`

Expected: FAIL because rendering code does not exist.

- [ ] **Step 3: Implement rendering**

Use PyMuPDF (`fitz`) imported inside `render_pdf`. Render pages at the requested DPI, save deterministic PNG names like `page-0000.png`, compute SHA-256 from the bytes, and return `Page` records with actual dimensions. Reject missing files, negative/out-of-range page IDs, nonpositive DPI, and duplicate output IDs.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_rendering.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_harness/rendering.py tests/test_rendering.py
git commit -m "feat: render PDFs into hashed page records"
```

### Task 4: Build benchmark protocol and redacted model requests

**Files:**
- Create: `src/doc_harness/protocol.py`
- Create: `tests/test_protocol.py`

**Interfaces:**
- `normalize_question(question: str) -> str`
- `build_request(sample: BenchmarkSample, pages: Sequence[Page], prompt: str, config_hash: str) -> ModelRequest`
- `validate_prediction_coverage(samples: Sequence[BenchmarkSample], predictions: Sequence[Prediction]) -> None`
- `export_v2_predictions(predictions: Sequence[Prediction], path: Path) -> None`
- `validate_evidence_quotes(draft: DraftAnswer, page_text: Mapping[int, str]) -> list[EvidenceSpan]`

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path
import pytest

from doc_harness.contracts import BenchmarkSample, DraftAnswer, EvidenceSpan, Page, Prediction
from doc_harness.protocol import (build_request, export_v2_predictions,
                                  normalize_question, validate_evidence_quotes,
                                  validate_prediction_coverage)


def sample(question=" What is total debt? "):
    return BenchmarkSample(doc_id="doc.pdf", question=question,
                           answer="100", evidence_pages=[0])


def test_normalize_question_collapses_whitespace():
    assert normalize_question("  a\\n  b ") == "a b"


def test_build_request_excludes_reference_metadata():
    request = build_request(sample(), [Page(document_id="doc.pdf", page_id=0,
        image_path="page.png", width=10, height=10, render_sha256="x")],
        prompt="Answer from the pages.", config_hash="hash")
    payload = request.model_dump()
    assert "answer" not in payload
    assert "evidence_pages" not in payload
    assert payload["question"] == "What is total debt?"


def test_prediction_coverage_rejects_duplicate_and_missing_rows():
    prediction = Prediction(doc_id="doc.pdf", question="What is total debt?",
                            response="100")
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_coverage([sample()], [prediction, prediction])
    with pytest.raises(ValueError, match="missing"):
        validate_prediction_coverage([sample(), sample("Other?")], [prediction])


def test_quote_validation_reports_only_present_quotes():
    draft = DraftAnswer(answer="100", insufficient_evidence=False,
                        evidence=[EvidenceSpan(page_id=0, quote="Total debt: 100"),
                                  EvidenceSpan(page_id=0, quote="absent")])
    valid = validate_evidence_quotes(draft, {0: "Total debt: 100"})
    assert [span.quote for span in valid] == ["Total debt: 100"]


def test_export_writes_v2_full_response(tmp_path: Path):
    out = tmp_path / "predictions.json"
    export_v2_predictions([Prediction(doc_id="doc.pdf", question="Q", response="A")], out)
    assert json.loads(out.read_text()) == [{"doc_id": "doc.pdf", "question": "Q", "response": "A"}]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_protocol.py`

Expected: FAIL because protocol code does not exist.

- [ ] **Step 3: Implement protocol functions**

Normalize questions by collapsing Unicode whitespace and trimming. Build `ModelRequest` only from caller-safe fields. Compare coverage using normalized keys and reject duplicate, missing, and unknown keys. Export only the V2 three-field objects. Validate quotes after normalizing line whitespace, returning valid spans and raising no exception for an invalid span so the caller can record a grounding failure.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_protocol.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_harness/protocol.py tests/test_protocol.py
git commit -m "feat: add benchmark protocol and request redaction"
```

### Task 5: Add lazy model runner and run records

**Files:**
- Create: `src/doc_harness/runner.py`
- Create: `src/doc_harness/records.py`
- Create: `tests/test_runner_and_records.py`

**Interfaces:**
- `class ModelRunner(Protocol): run(request: ModelRequest) -> DraftAnswer`
- `class FakeRunner: run(request: ModelRequest) -> DraftAnswer`
- `class QwenTransformersRunner: from_pretrained(...); run(request: ModelRequest) -> DraftAnswer`
- `write_run_record(record: RunRecord, path: Path) -> None`
- `read_run_records(path: Path) -> list[RunRecord]`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from doc_harness.contracts import DraftAnswer, ModelRequest, RunRecord
from doc_harness.records import read_run_records, write_run_record
from doc_harness.runner import FakeRunner


def request():
    return ModelRequest(document_id="doc.pdf", question="Q", page_ids=[0],
                        image_paths=["page.png"], prompt="P", config_hash="h")


def test_fake_runner_returns_valid_draft():
    answer = FakeRunner(answer="A").run(request())
    assert isinstance(answer, DraftAnswer)
    assert answer.answer == "A"


def test_run_record_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    record = RunRecord(run_id="r1", document_id="doc.pdf", question="Q",
                       status="ok", response="A", config_hash="h")
    write_run_record(record, path)
    assert read_run_records(path) == [record]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_runner_and_records.py`

Expected: FAIL because runner and record modules do not exist.

- [ ] **Step 3: Implement runners and JSONL records**

Implement `FakeRunner` without model dependencies. Implement `QwenTransformersRunner` with imports inside `from_pretrained`, use `AutoProcessor` and `AutoModelForImageTextToText`/the compatible multimodal class, construct messages from local images and the prompt, use deterministic generation, and parse strict JSON with one repair attempt into `DraftAnswer`. Raise a typed `StageFailure` at the CLI boundary for import, load, generation, or parse failures. Write one compact JSON object per line and reject malformed records on read.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_runner_and_records.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doc_harness/runner.py src/doc_harness/records.py tests/test_runner_and_records.py
git commit -m "feat: add lazy runners and JSONL run records"
```

### Task 6: Add CLI smoke path, docs, and complete verification

**Files:**
- Create: `src/doc_harness/cli.py`
- Create: `tests/test_cli.py`
- Create: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- `python -m doc_harness.cli smoke --question ... --image ... --output ...`
- `python -m doc_harness.cli export-v2 --samples ... --predictions ... --output ...`

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

from doc_harness.cli import run_smoke


def test_smoke_writes_record_and_prediction(tmp_path: Path):
    output = tmp_path / "run.jsonl"
    prediction = tmp_path / "prediction.json"
    run_smoke(question="What is shown?", document_id="doc.pdf",
              image_paths=[str(tmp_path / "page.png")],
              output_path=output, prediction_path=prediction)
    assert output.exists()
    assert json.loads(prediction.read_text())[0]["response"] == "stub answer"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --with pydantic --with pytest pytest -q tests/test_cli.py`

Expected: FAIL because the CLI does not exist.

- [ ] **Step 3: Implement the CLI and README**

Wire `run_smoke` to construct a safe request, use `FakeRunner` by default, validate the draft, write a `RunRecord`, and export a V2 prediction. Add argparse subcommands for the fake smoke path and V2 export. Document environment setup, fake smoke, optional GPU integration, artifact locations, and the exact external evaluator command requiring user-provided credentials.

- [ ] **Step 4: Run the complete offline test suite**

Run: `uv run --with pydantic --with pytest pytest -q`

Expected: PASS with no network or CUDA requirement.

- [ ] **Step 5: Run the CLI smoke command**

Run: `uv run --with pydantic python -m doc_harness.cli smoke --question 'What is shown?' --image /tmp/page.png --output /tmp/doc-harness-run.jsonl --prediction /tmp/doc-harness-predictions.json`

Expected: the command exits 0 and writes one JSONL run record plus one V2-shaped prediction. If `/tmp/page.png` is absent, the CLI must report a clear input error; the test uses a temporary fixture image.

- [ ] **Step 6: Commit**

```bash
git add src/doc_harness/cli.py tests/test_cli.py README.md pyproject.toml
git commit -m "feat: add reproducible baseline smoke CLI"
```

