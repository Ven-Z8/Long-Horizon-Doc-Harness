import json
from pathlib import Path

import pytest

from doc_harness.contracts import (
    BenchmarkSample,
    DraftAnswer,
    EvidenceSpan,
    Page,
    Prediction,
)
from doc_harness.protocol import (
    build_request,
    export_v2_predictions,
    normalize_question,
    validate_evidence_quotes,
    validate_prediction_coverage,
)


def sample(question=" What is total debt? "):
    return BenchmarkSample(
        doc_id="doc.pdf", question=question, answer="100", evidence_pages=[0]
    )


def test_normalize_question_collapses_whitespace():
    assert normalize_question("  a\n  b ") == "a b"


def test_build_request_excludes_reference_metadata():
    request = build_request(
        sample(),
        [
            Page(
                document_id="doc.pdf",
                page_id=0,
                image_path="page.png",
                width=10,
                height=10,
                render_sha256="x",
            )
        ],
        prompt="Answer from the pages.",
        config_hash="hash",
    )
    payload = request.model_dump()
    assert "answer" not in payload
    assert "evidence_pages" not in payload
    assert payload["question"] == "What is total debt?"


def test_prediction_coverage_rejects_duplicate_and_missing_rows():
    prediction = Prediction(
        doc_id="doc.pdf", question="What is total debt?", response="100"
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_prediction_coverage([sample()], [prediction, prediction])
    with pytest.raises(ValueError, match="missing"):
        validate_prediction_coverage([sample(), sample("Other?")], [prediction])


def test_quote_validation_reports_only_present_quotes():
    draft = DraftAnswer(
        answer="100",
        insufficient_evidence=False,
        evidence=[
            EvidenceSpan(page_id=0, quote="Total debt: 100"),
            EvidenceSpan(page_id=0, quote="absent"),
        ],
    )
    valid = validate_evidence_quotes(draft, {0: "Total debt: 100"})
    assert [span.quote for span in valid] == ["Total debt: 100"]


def test_export_writes_v2_full_response(tmp_path: Path):
    out = tmp_path / "predictions.json"
    export_v2_predictions(
        [Prediction(doc_id="doc.pdf", question="Q", response="A")], out
    )
    assert json.loads(out.read_text()) == [
        {"doc_id": "doc.pdf", "question": "Q", "response": "A"}
    ]
