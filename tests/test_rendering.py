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
