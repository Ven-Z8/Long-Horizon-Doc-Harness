from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from doc_harness.contracts import Page, SafeQuestion
from doc_harness.evidence import (
    EvidenceBudget,
    EvidenceRegion,
    build_bundle,
)
from doc_harness.ocr import ExtractionStatus, OCRParsedPage


def _page(tmp_path: Path, page_id: int, *, document_id: str = "report.pdf") -> Page:
    image = tmp_path / f"page-{page_id}.png"
    image.write_bytes(f"page {page_id}".encode())
    return Page(
        document_id=document_id,
        page_id=page_id,
        image_path=str(image),
        width=100,
        height=100,
        render_sha256=f"render-{page_id}",
    )


def _parsed(page: Page, text: str) -> OCRParsedPage:
    return OCRParsedPage(
        document_id=page.document_id,
        page_id=page.page_id,
        markdown=text,
        parser_model="fake-ocr",
        parser_revision="rev-1",
        prompt_version="prompt-1",
        render_sha256=page.render_sha256,
        extraction_status=ExtractionStatus.SUCCESS,
        ocr_config_hash="ocr-config",
    )


def _question() -> SafeQuestion:
    return SafeQuestion(document_id="report.pdf", question="What is the total?")


def test_bundle_preserves_page_ids_when_input_pages_are_reordered(tmp_path: Path):
    pages = [_page(tmp_path, 2), _page(tmp_path, 0)]
    parsed = [_parsed(pages[1], "page zero"), _parsed(pages[0], "page two")]
    budget = EvidenceBudget(
        max_pages=2,
        max_pixels_per_image=10_000,
        max_total_image_pixels=20_000,
        max_text_tokens=100,
        reserved_output_tokens=10,
    )

    bundle = build_bundle(_question(), pages, parsed, budget)

    assert bundle.included_page_ids == [2, 0]
    assert [image.page_id for image in bundle.images] == [2, 0]
    assert bundle.ocr_content == {0: "page zero", 2: "page two"}
    assert "page_id=2" in bundle.images[0].label
    assert bundle.content_hash


def test_evidence_region_requires_strict_normalized_bounds():
    region = EvidenceRegion(page_id=1, x0=0.1, y0=0.2, x1=0.8, y1=0.9)
    assert region.coordinates == (0.1, 0.2, 0.8, 0.9)

    with pytest.raises(ValidationError):
        EvidenceRegion(page_id=1, x0=0.0, y0=0.2, x1=0.0, y1=0.9)
    with pytest.raises(ValidationError):
        EvidenceRegion(page_id=1, coordinates=(0.1, 0.2, 1.1, 0.9))


def test_bundle_rejects_ocr_from_another_document(tmp_path: Path):
    page = _page(tmp_path, 0)
    foreign = _parsed(_page(tmp_path, 0, document_id="other.pdf"), "foreign")
    budget = EvidenceBudget(
        max_pages=1,
        max_pixels_per_image=10_000,
        max_total_image_pixels=10_000,
        max_text_tokens=100,
        reserved_output_tokens=10,
    )

    with pytest.raises(ValueError, match="different document"):
        build_bundle(_question(), [page], [foreign], budget)


def test_bundle_reports_image_and_text_budget_omissions_without_overflow(tmp_path: Path):
    pages = [_page(tmp_path, 0), _page(tmp_path, 1), _page(tmp_path, 2)]
    parsed = [_parsed(pages[0], "one two"), _parsed(pages[1], "three four")]
    budget = EvidenceBudget(
        max_pages=2,
        max_pixels_per_image=10_000,
        max_total_image_pixels=20_000,
        max_text_tokens=3,
        reserved_output_tokens=10,
    )

    bundle = build_bundle(_question(), pages, parsed, budget)

    assert len(bundle.images) <= budget.max_pages
    assert bundle.image_pixels <= budget.max_total_image_pixels
    assert bundle.text_tokens <= budget.max_text_tokens
    assert {item.page_id for item in bundle.omitted_items} >= {1, 2}
    assert any(item.reason == "text_token_budget" for item in bundle.omitted_items)


def test_bundle_omits_page_that_exceeds_per_image_budget(tmp_path: Path):
    page = _page(tmp_path, 0)
    parsed = [_parsed(page, "visible text")]
    budget = EvidenceBudget(
        max_pages=1,
        max_pixels_per_image=99,
        max_total_image_pixels=10_000,
        max_text_tokens=100,
        reserved_output_tokens=10,
    )

    bundle = build_bundle(_question(), [page], parsed, budget)

    assert bundle.images == []
    assert bundle.included_page_ids == []
    assert bundle.omitted_items[0].reason == "per_image_pixel_budget"


def test_bundle_accepts_focused_render_with_original_ocr_provenance(tmp_path: Path):
    source = _page(tmp_path, 0)
    focused = source.model_copy(
        update={
            "image_path": source.image_path,
            "width": 50,
            "height": 50,
            "render_sha256": "focused-hash",
            "source_render_sha256": source.render_sha256,
            "source_image_path": source.image_path,
        }
    )
    budget = EvidenceBudget(
        max_pages=1,
        max_pixels_per_image=2_500,
        max_total_image_pixels=2_500,
        max_text_tokens=100,
        reserved_output_tokens=10,
    )
    bundle = build_bundle(_question(), [focused], [_parsed(source, "visible")], budget)
    assert bundle.provenance[0].render_sha256 == source.render_sha256
    assert bundle.provenance[0].focused_render_sha256 == "focused-hash"
    assert bundle.provenance[0].source_image_path == source.image_path
    assert bundle.images[0].source_render_sha256 == source.render_sha256
