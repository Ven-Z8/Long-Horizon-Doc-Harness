"""Deterministic PDF page rendering with content-addressed metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from .contracts import Page


def validate_pages(pages: Sequence[Page]) -> None:
    """Validate page identity and ordering invariants."""

    seen: set[tuple[str, int]] = set()
    for page in pages:
        key = (page.document_id, page.page_id)
        if key in seen:
            raise ValueError(f"duplicate page identity: {page.document_id}:{page.page_id}")
        seen.add(key)


def render_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 144,
    page_ids: Sequence[int] | None = None,
) -> list[Page]:
    """Render selected zero-based PDF pages to PNG files and return page records."""

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    try:
        import pymupdf as fitz
    except ImportError as exc:  # pragma: no cover - exercised in integration setup
        raise RuntimeError(
            "PDF rendering requires PyMuPDF; install the project's pdf extra"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    document_id = pdf_path.name
    scale = dpi / 72.0

    document = fitz.open(pdf_path)
    try:
        selected = list(range(len(document))) if page_ids is None else list(page_ids)
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate page IDs requested")
        if any(page_id < 0 or page_id >= len(document) for page_id in selected):
            raise IndexError("requested page ID is outside the PDF")

        pages: list[Page] = []
        for page_id in selected:
            pixmap = document[page_id].get_pixmap(
                matrix=fitz.Matrix(scale, scale), alpha=False
            )
            image_path = output_dir / f"page-{page_id:04d}.png"
            pixmap.save(str(image_path))
            image_bytes = image_path.read_bytes()
            pages.append(
                Page(
                    document_id=document_id,
                    page_id=page_id,
                    image_path=str(image_path),
                    width=pixmap.width,
                    height=pixmap.height,
                    render_sha256=hashlib.sha256(image_bytes).hexdigest(),
                )
            )
        validate_pages(pages)
        return pages
    finally:
        document.close()
