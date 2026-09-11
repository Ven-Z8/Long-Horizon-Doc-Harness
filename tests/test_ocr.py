from __future__ import annotations

from pathlib import Path

import pytest

from doc_harness.contracts import Page
from doc_harness.ocr import (
    ExtractionStatus,
    OCRParsedPage,
    parse_cached,
)


def _page(tmp_path: Path, *, document_id: str = "report.pdf", render_sha256: str = "render-a") -> Page:
    image = tmp_path / f"{document_id.replace('.', '-')}-page.png"
    image.write_bytes(b"synthetic image")
    return Page(
        document_id=document_id,
        page_id=0,
        image_path=str(image),
        width=100,
        height=100,
        render_sha256=render_sha256,
    )


class CountingParser:
    parser_model = "fake-ocr"
    parser_revision = "rev-1"
    prompt_version = "prompt-1"
    ocr_config_hash = "config-1"

    def __init__(self, result: object = "Total: 10") -> None:
        self.result = result
        self.calls = 0

    def parse(self, page: Page) -> object:
        self.calls += 1
        if isinstance(self.result, BaseException):
            raise self.result
        if callable(self.result):
            return self.result(page)
        return self.result


def test_parse_cached_hits_identical_identity_and_rejects_stale_identity(tmp_path: Path):
    parser = CountingParser()
    page = _page(tmp_path)

    first = parse_cached(page, parser, tmp_path / "cache")
    second = parse_cached(page, parser, tmp_path / "cache")
    assert first.extraction_status is ExtractionStatus.SUCCESS
    assert second.cache_key == first.cache_key
    assert parser.calls == 1
    assert first.document_id == page.document_id
    assert first.render_sha256 == page.render_sha256
    assert first.raw_output_path is not None

    changed_render = _page(tmp_path, render_sha256="render-b")
    parse_cached(changed_render, parser, tmp_path / "cache")
    assert parser.calls == 2

    parser.parser_revision = "rev-2"
    parse_cached(page, parser, tmp_path / "cache")
    assert parser.calls == 3

    parser.prompt_version = "prompt-2"
    parse_cached(page, parser, tmp_path / "cache")
    assert parser.calls == 4

    parser.ocr_config_hash = "config-2"
    parse_cached(page, parser, tmp_path / "cache")
    assert parser.calls == 5


def test_cache_identity_separates_documents_and_pages(tmp_path: Path):
    parser = CountingParser()
    cache = tmp_path / "cache"
    page_a = _page(tmp_path, document_id="a.pdf")
    page_b = _page(tmp_path, document_id="b.pdf")

    parsed_a = parse_cached(page_a, parser, cache)
    parsed_b = parse_cached(page_b, parser, cache)

    assert parsed_a.cache_key != parsed_b.cache_key
    assert len(list(cache.glob("*.json"))) == 2
    assert parser.calls == 2


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (None, ExtractionStatus.INVALID),
        ({"markdown": 12}, ExtractionStatus.INVALID),
        ("", ExtractionStatus.EMPTY),
    ],
)
def test_malformed_or_empty_extraction_is_never_cached_as_success(
    tmp_path: Path, result: object, status: ExtractionStatus
):
    parser = CountingParser(result)
    parsed = parse_cached(_page(tmp_path), parser, tmp_path / "cache")

    assert parsed.extraction_status is status
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_parser_exception_returns_deterministic_failed_status_without_cache(tmp_path: Path):
    parser = CountingParser(RuntimeError("worker stopped"))
    parsed = parse_cached(_page(tmp_path), parser, tmp_path / "cache")

    assert parsed.extraction_status is ExtractionStatus.FAILED
    assert parsed.error == "RuntimeError: worker stopped"
    assert list((tmp_path / "cache").glob("*.json")) == []


def test_incomplete_extraction_is_provenance_preserving(tmp_path: Path):
    parser = CountingParser(
        OCRParsedPage(
            document_id="report.pdf",
            page_id=0,
            markdown="first row\nsecond row",
            parser_model="fake-ocr",
            parser_revision="rev-1",
            prompt_version="prompt-1",
            render_sha256="render-a",
            extraction_status=ExtractionStatus.INCOMPLETE,
            ocr_config_hash="config-1",
        )
    )
    parsed = parse_cached(_page(tmp_path), parser, tmp_path / "cache")

    assert parsed.extraction_status is ExtractionStatus.INCOMPLETE
    assert parsed.page_id == 0
    assert parsed.document_id == "report.pdf"
    assert parsed.render_sha256 == "render-a"
    assert parsed.markdown == "first row\nsecond row"
    assert list((tmp_path / "cache").glob("*.json")) == []
