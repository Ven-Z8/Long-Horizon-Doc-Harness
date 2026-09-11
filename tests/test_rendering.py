from pathlib import Path

import pytest

from doc_harness.rendering import render_pdf, resize_page_for_budget, validate_pages


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


def test_resize_page_for_budget_preserves_source_render_identity(tmp_path: Path):
    image_module = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "page.png"
    image_module.new("RGB", (200, 100), color="white").save(image_path)
    from doc_harness.contracts import Page

    page = Page(
        document_id="doc.pdf",
        page_id=0,
        image_path=str(image_path),
        width=200,
        height=100,
        render_sha256="source-hash",
    )
    focused = resize_page_for_budget(page, 5_000, tmp_path / "focused")
    assert focused.width * focused.height <= 5_000
    assert focused.source_render_sha256 == "source-hash"
    assert focused.source_image_path == str(image_path)
    assert focused.render_sha256 != "source-hash"
