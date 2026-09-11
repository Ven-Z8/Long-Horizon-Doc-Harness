from pathlib import Path

from doc_harness.config import load_config
from doc_harness.contracts import (
    DraftAnswer,
    EvidenceSpan,
    Page,
    ParsedPage,
    SafeQuestion,
    Verification,
    VerifyDecision,
)
from doc_harness.pipeline import PipelineServices, run_pipeline


def config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text('[model]\nmodel_id = "test"\nrevision = "rev"\n')
    return load_config(path)


def test_pipeline_runs_stages_and_preserves_page_provenance(tmp_path: Path):
    page = Page(
        document_id="doc.pdf",
        page_id=2,
        image_path=str(tmp_path / "page.png"),
        width=10,
        height=10,
        render_sha256="hash",
    )
    question = SafeQuestion(document_id="doc.pdf", question="What is total?")
    calls = []

    class Bundle:
        content_hash = "bundle-hash"

    services = PipelineServices(
        pages_for=lambda document_id: [page],
        select_pages=lambda q, pages: (calls.append("retrieve") or list(pages)),
        parse_page=lambda item: (
            calls.append("ocr")
            or ParsedPage(
                page_id=item.page_id,
                markdown="Total: 10",
                parser_model="ocr",
                parser_revision="rev",
                prompt_version="v1",
            )
        ),
        build_bundle=lambda q, pages, parsed, cfg: (calls.append("bundle") or Bundle()),
        answer=lambda q, bundle: (
            calls.append("answer")
            or DraftAnswer(
                answer="10",
                evidence=[EvidenceSpan(page_id=2, quote="Total: 10")],
                insufficient_evidence=False,
            )
        ),
        verify=lambda q, draft, bundle: (
            calls.append("verify")
            or Verification(
                decision=VerifyDecision.accept,
                final_answer="10",
                evidence=draft.evidence,
                reason="supported",
            )
        ),
    )
    result = run_pipeline(question, services, config(tmp_path))
    assert result.final_response == "10"
    assert result.selected_page_ids == [2]
    assert calls == ["retrieve", "ocr", "bundle", "answer", "verify"]


def test_pipeline_returns_explicit_failure_without_fabricating_answer(tmp_path: Path):
    question = SafeQuestion(document_id="missing.pdf", question="Q")
    services = PipelineServices(
        pages_for=lambda document_id: [],
        select_pages=lambda q, pages: [],
        parse_page=lambda page: None,
        build_bundle=lambda q, pages, parsed, cfg: None,
        answer=lambda q, bundle: (_ for _ in ()).throw(RuntimeError("model stopped")),
        verify=lambda q, draft, bundle: None,
    )
    result = run_pipeline(question, services, config(tmp_path))
    assert result.final_response == ""
    assert result.failures[0].stage == "retrieve"
