from pathlib import Path

from PIL import Image

from doc_harness.core.contracts import Page, SafeQuestion
from doc_harness.documents.evidence import EvidenceBudget, EvidenceRegion, build_bundle
from doc_harness.documents.rendering import materialize_region
from doc_harness.workflow.stages import required_ocr_pages


def test_new_window_page_is_ocr_work_even_when_not_initially_selected():
    page = Page(
        document_id="d.pdf", page_id=9, image_path="p9.png", width=10, height=10, render_sha256="hash9"
    )
    selected = required_ocr_pages({"question-a": [page], "question-b": [page]})
    assert [(item.document_id, item.page_id) for item in selected] == [("d.pdf", 9)]


def test_materialize_region_returns_real_pixel_crop(tmp_path):
    source = tmp_path / "page.png"
    image = Image.new("RGB", (100, 80), "black")
    for x in range(50, 100):
        for y in range(80):
            image.putpixel((x, y), (255, 0, 0))
    image.save(source)
    page = Page(
        document_id="d.pdf", page_id=0, image_path=str(source), width=100, height=80, render_sha256="source"
    )
    region = EvidenceRegion(page_id=0, x0=0.5, y0=0, x1=1, y1=1)
    cropped = materialize_region(page, region, tmp_path / "regions")
    with Image.open(cropped.image_path) as result:
        assert result.size == (50, 80)
        assert result.getpixel((0, 0)) == (255, 0, 0)


def test_bundle_uses_cropped_image_for_region(tmp_path):
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 80), "white").save(source)
    page = Page(
        document_id="d.pdf", page_id=0, image_path=str(source), width=100, height=80, render_sha256="source"
    )
    bundle = build_bundle(
        SafeQuestion(document_id="d.pdf", question="Q"),
        [page],
        [],
        EvidenceBudget(max_pages=1, max_pixels_per_image=10000, max_total_image_pixels=10000, max_text_tokens=10),
        regions={0: EvidenceRegion(page_id=0, x0=0, y0=0, x1=0.5, y1=1)},
    )
    assert bundle.images[0].image_path != str(source)
    with Image.open(bundle.images[0].image_path) as result:
        assert result.size == (50, 80)
