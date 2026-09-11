import json
from pathlib import Path

import pytest

from doc_harness.batch import load_v2_samples, run_v2_batch
from doc_harness.config import load_config
from doc_harness.runner import FakeRunner


def write_pdf(path: Path) -> None:
    fitz = pytest.importorskip("pymupdf")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Total debt: 100 million.")
    document.save(path)
    document.close()


def test_load_v2_samples_parses_string_list_fields(tmp_path: Path):
    path = tmp_path / "samples.json"
    path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "doc.pdf",
                    "doc_type": "report",
                    "question": "  What is total debt? ",
                    "answer": "100 million",
                    "evidence_pages": "[0]",
                    "evidence_sources": "['Table']",
                    "answer_format": "Str",
                }
            ]
        )
    )
    samples = load_v2_samples(path)
    assert samples[0].evidence_pages == [0]
    assert samples[0].evidence_sources == ["Table"]
    assert samples[0].answer_format == "Str"


def test_run_v2_batch_writes_predictions_and_resumes(tmp_path: Path):
    documents = tmp_path / "documents"
    documents.mkdir()
    write_pdf(documents / "doc.pdf")
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "doc.pdf",
                    "question": "Q1",
                    "answer": "A1",
                    "evidence_pages": "[0]",
                    "evidence_sources": "[]",
                },
                {
                    "doc_id": "doc.pdf",
                    "question": "Q2",
                    "answer": "A2",
                    "evidence_pages": "[0]",
                    "evidence_sources": "[]",
                },
            ]
        )
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[model]\nmodel_id = "test"\nrevision = "rev"\n'
        '[render]\ndpi = 72\npage_policy = "all"\n'
        '[generation]\nmax_new_tokens = 8\ndo_sample = false\n'
    )
    config = load_config(config_path)
    predictions_path = tmp_path / "predictions.json"
    records_path = tmp_path / "runs.jsonl"
    first = run_v2_batch(
        samples_path=samples_path,
        documents_dir=documents,
        output_path=predictions_path,
        records_path=records_path,
        render_dir=tmp_path / "rendered",
        config=config,
        runner=FakeRunner(answer="stub"),
    )
    assert [item.response for item in first] == ["stub", "stub"]
    assert len(records_path.read_text().splitlines()) == 2

    second = run_v2_batch(
        samples_path=samples_path,
        documents_dir=documents,
        output_path=predictions_path,
        records_path=records_path,
        render_dir=tmp_path / "rendered",
        config=config,
        runner=FakeRunner(answer="different"),
    )
    assert [item.response for item in second] == ["stub", "stub"]
    assert len(records_path.read_text().splitlines()) == 2


def test_run_v2_batch_retries_a_failed_row(tmp_path: Path):
    documents = tmp_path / "documents"
    documents.mkdir()
    write_pdf(documents / "doc.pdf")
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(
        json.dumps(
            [
                {
                    "doc_id": "doc.pdf",
                    "question": "Q1",
                    "answer": "A1",
                    "evidence_pages": "[0]",
                    "evidence_sources": "[]",
                }
            ]
        )
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text('[model]\nmodel_id = "test"\nrevision = "rev"\n')
    config = load_config(config_path)
    predictions_path = tmp_path / "predictions.json"
    records_path = tmp_path / "runs.jsonl"

    class FailingRunner:
        def run(self, request):
            raise RuntimeError("temporary")

    run_v2_batch(
        samples_path,
        documents,
        predictions_path,
        records_path,
        tmp_path / "rendered",
        config,
        FailingRunner(),
    )
    run_v2_batch(
        samples_path,
        documents,
        predictions_path,
        records_path,
        tmp_path / "rendered",
        config,
        FakeRunner(answer="recovered"),
    )
    assert json.loads(predictions_path.read_text())[0]["response"] == "recovered"
    assert len(records_path.read_text().splitlines()) == 2
