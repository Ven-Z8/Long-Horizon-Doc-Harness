"""Bounded, provenance-preserving evidence bundles for focused answering."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

from pydantic import AliasChoices, Field, model_validator

from ..core.contracts import Page, StrictModel
from .ocr import ExtractionStatus


class EvidenceBudget(StrictModel):
    """Hard limits used while assembling one question's evidence bundle."""

    max_pages: int = Field(default=6, gt=0)
    max_pixels_per_image: int = Field(
        default=1_048_576,
        gt=0,
        validation_alias=AliasChoices(
            "max_pixels_per_image", "per_image_pixels", "max_image_pixels"
        ),
    )
    max_total_image_pixels: int = Field(
        default=6_291_456,
        gt=0,
        validation_alias=AliasChoices(
            "max_total_image_pixels", "total_image_pixels"
        ),
    )
    max_text_tokens: int = Field(
        default=12_000,
        gt=0,
        validation_alias=AliasChoices("max_text_tokens", "text_tokens")
    )
    reserved_output_tokens: int = Field(
        default=1_024,
        ge=0,
        validation_alias=AliasChoices(
            "reserved_output_tokens", "output_tokens", "max_output_tokens"
        ),
    )

    @property
    def per_image_pixels(self) -> int:
        return self.max_pixels_per_image

    @property
    def total_image_pixels(self) -> int:
        return self.max_total_image_pixels

    @property
    def text_tokens(self) -> int:
        return self.max_text_tokens

    @property
    def output_tokens(self) -> int:
        return self.reserved_output_tokens


class EvidenceRegion(StrictModel):
    """A verified normalized crop rectangle on a source page."""

    page_id: int = Field(ge=0)
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def accept_coordinate_tuple(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "coordinates" in value:
            value = dict(value)
            coordinates = value.pop("coordinates")
            if not isinstance(coordinates, (tuple, list)) or len(coordinates) != 4:
                raise ValueError("coordinates must contain x0, y0, x1, and y1")
            value.update(dict(zip(("x0", "y0", "x1", "y1"), coordinates)))
        return value

    @model_validator(mode="after")
    def has_positive_area(self) -> "EvidenceRegion":
        if not self.x0 < self.x1:
            raise ValueError("x0 must be less than x1")
        if not self.y0 < self.y1:
            raise ValueError("y0 must be less than y1")
        return self

    @property
    def coordinates(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def area(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0)


class EvidenceImage(StrictModel):
    """An image or explicitly located crop with its source fingerprint."""

    document_id: str
    page_id: int = Field(ge=0)
    image_path: str
    render_sha256: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    pixel_count: int = Field(gt=0)
    region: EvidenceRegion | None = None
    label: str
    source_render_sha256: str | None = None
    source_image_path: str | None = None

    @model_validator(mode="after")
    def image_source_is_consistent(self) -> "EvidenceImage":
        if self.region is not None and self.region.page_id != self.page_id:
            raise ValueError("crop region page_id does not match image page_id")
        if self.region is None and self.pixel_count != self.width * self.height:
            raise ValueError("full-page pixel count is inconsistent with dimensions")
        if self.region is not None:
            expected = max(1, math.ceil(self.width * self.height * self.region.area))
            if self.pixel_count != expected:
                raise ValueError("crop pixel count is inconsistent with region bounds")
        return self


class EvidenceOmission(StrictModel):
    """An explicit record of content excluded by a hard bundle limit."""

    page_id: int | None = Field(default=None, ge=0)
    item: str
    reason: str


class EvidenceProvenance(StrictModel):
    """Source identity retained for each page admitted to a bundle."""

    document_id: str
    page_id: int = Field(ge=0)
    render_sha256: str
    image_path: str
    parser_model: str | None = None
    parser_revision: str | None = None
    prompt_version: str | None = None
    ocr_config_hash: str | None = None
    extraction_status: str | None = None
    focused_render_sha256: str | None = None
    source_image_path: str | None = None


def _estimate_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


class EvidenceBundle(StrictModel):
    """The bounded evidence passed to an answer or verification stage."""

    document_id: str
    question: str
    included_page_ids: list[int] = Field(
        default_factory=list,
        validation_alias=AliasChoices("included_page_ids", "page_ids"),
    )
    images: list[EvidenceImage] = Field(default_factory=list)
    regions: list[EvidenceRegion] = Field(default_factory=list)
    ocr_content: dict[int, str] = Field(default_factory=dict)
    omitted_items: list[EvidenceOmission] = Field(default_factory=list)
    provenance: dict[int, EvidenceProvenance] = Field(default_factory=dict)
    budget: EvidenceBudget
    image_pixels: int = Field(default=0, ge=0)
    text_tokens: int = Field(default=0, ge=0)
    content_hash: str

    @model_validator(mode="after")
    def obey_budget_and_provenance(self) -> "EvidenceBundle":
        ids = self.included_page_ids
        if len(set(ids)) != len(ids):
            raise ValueError("included page IDs must be unique")
        if len(ids) > self.budget.max_pages:
            raise ValueError("bundle exceeds maximum page budget")
        if self.image_pixels > self.budget.max_total_image_pixels:
            raise ValueError("bundle exceeds total image pixel budget")
        measured_pixels = sum(image.pixel_count for image in self.images)
        if measured_pixels != self.image_pixels:
            raise ValueError("bundle image pixel count is inconsistent")
        if any(image.pixel_count > self.budget.max_pixels_per_image for image in self.images):
            raise ValueError("bundle exceeds per-image pixel budget")
        measured_text = sum(_estimate_tokens(text) for text in self.ocr_content.values())
        if measured_text != self.text_tokens:
            raise ValueError("bundle text token count is inconsistent")
        if self.text_tokens > self.budget.max_text_tokens:
            raise ValueError("bundle exceeds text token budget")
        included = set(ids)
        image_ids = {image.page_id for image in self.images}
        if not image_ids <= included or not set(self.ocr_content) <= included:
            raise ValueError("bundle content references an excluded page")
        if not set(self.provenance) >= included:
            raise ValueError("bundle is missing page provenance")
        for page_id, provenance in self.provenance.items():
            if page_id != provenance.page_id:
                raise ValueError("provenance key does not match page ID")
            if provenance.document_id != self.document_id:
                raise ValueError("provenance belongs to a different document")
        return self

    @property
    def page_ids(self) -> list[int]:
        """Compatibility alias for callers using the shorter name."""

        return self.included_page_ids


def _parsed_value(parsed: object, name: str, default: Any = None) -> Any:
    if isinstance(parsed, Mapping):
        return parsed.get(name, default)
    return getattr(parsed, name, default)


def _status(parsed: object) -> ExtractionStatus:
    value = _parsed_value(parsed, "extraction_status", ExtractionStatus.SUCCESS)
    if isinstance(value, ExtractionStatus):
        return value
    try:
        return ExtractionStatus(str(value))
    except ValueError:
        return ExtractionStatus.INVALID


def _hash_bundle_payload(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_bundle(
    question: Any,
    pages: Sequence[Page],
    parsed: Sequence[object],
    budget: EvidenceBudget,
    *,
    regions: Mapping[int, EvidenceRegion | Sequence[EvidenceRegion]] | None = None,
) -> EvidenceBundle:
    """Build a deterministic bundle while recording every budget exclusion.

    Pages are processed in the caller's order and retain their internal
    zero-based IDs.  A region is used only when supplied explicitly; absent
    coordinates always mean a full source page.
    """

    if not isinstance(budget, EvidenceBudget):
        budget = EvidenceBudget.model_validate(budget)
    document_id = getattr(question, "document_id", None) or getattr(question, "doc_id", None)
    question_text = getattr(question, "question", None)
    if not isinstance(document_id, str) or not document_id:
        raise ValueError("question must include a document_id")
    if not isinstance(question_text, str):
        raise ValueError("question must include question text")

    page_by_id: dict[int, Page] = {}
    for page in pages:
        if page.document_id != document_id:
            raise ValueError("pages belong to a different document")
        if page.page_id in page_by_id:
            raise ValueError(f"duplicate page ID: {page.page_id}")
        page_by_id[page.page_id] = page

    parsed_by_id: dict[int, object] = {}
    for item in parsed:
        page_id = _parsed_value(item, "page_id")
        if not isinstance(page_id, int):
            raise ValueError("parsed page has no integer page ID")
        if page_id in parsed_by_id:
            raise ValueError(f"duplicate parsed page ID: {page_id}")
        parsed_document = _parsed_value(item, "document_id", "")
        if parsed_document not in (None, "", document_id):
            raise ValueError("parsed OCR belongs to a different document")
        if page_id not in page_by_id:
            raise ValueError(f"parsed page {page_id} is not in the selected pages")
        render_hash = _parsed_value(item, "render_sha256", "")
        valid_render_hashes = {
            page_by_id[page_id].render_sha256,
            page_by_id[page_id].source_render_sha256,
        }
        if render_hash not in (None, "") and render_hash not in valid_render_hashes:
            raise ValueError(f"parsed OCR for page {page_id} has a stale render")
        parsed_by_id[page_id] = item

    included: list[int] = []
    images: list[EvidenceImage] = []
    included_regions: list[EvidenceRegion] = []
    ocr_content: dict[int, str] = {}
    omissions: list[EvidenceOmission] = []
    provenance: dict[int, EvidenceProvenance] = {}
    image_pixels = 0
    text_tokens = 0

    for page in pages:
        if len(included) >= budget.max_pages:
            omissions.append(EvidenceOmission(page_id=page.page_id, item="page", reason="max_pages"))
            continue

        page_regions: list[EvidenceRegion | None]
        supplied_regions = regions.get(page.page_id) if regions else None
        if supplied_regions is None:
            page_regions = [None]
        elif isinstance(supplied_regions, EvidenceRegion):
            page_regions = [supplied_regions]
        else:
            page_regions = list(supplied_regions)
            if not page_regions:
                page_regions = [None]
        for region in page_regions:
            if region is not None and region.page_id != page.page_id:
                raise ValueError("crop region page_id does not match page")
            area = region.area if region is not None else 1.0
            pixels = max(1, math.ceil(page.width * page.height * area))
            if pixels > budget.max_pixels_per_image:
                omissions.append(
                    EvidenceOmission(page_id=page.page_id, item="image", reason="per_image_pixel_budget")
                )
                continue
            if image_pixels + pixels > budget.max_total_image_pixels:
                omissions.append(
                    EvidenceOmission(page_id=page.page_id, item="image", reason="total_image_pixel_budget")
                )
                continue
            images.append(
                EvidenceImage(
                    document_id=document_id,
                    page_id=page.page_id,
                    image_path=page.image_path,
                    render_sha256=page.render_sha256,
                    width=page.width,
                    height=page.height,
                    pixel_count=pixels,
                    region=region,
                    label=f"page_id={page.page_id}",
                    source_render_sha256=page.source_render_sha256,
                    source_image_path=page.source_image_path,
                )
            )
            if region is not None:
                included_regions.append(region)
            image_pixels += pixels

        # A page without an admitted image cannot be represented as focused
        # visual evidence.  Keep its exclusion explicit and do not smuggle OCR
        # text in under a different page identity.
        page_images = [image for image in images if image.page_id == page.page_id]
        if not page_images:
            continue
        included.append(page.page_id)
        parser_item = parsed_by_id.get(page.page_id)
        provenance[page.page_id] = EvidenceProvenance(
            document_id=document_id,
            page_id=page.page_id,
            render_sha256=page.source_render_sha256 or page.render_sha256,
            image_path=page.image_path,
            source_image_path=page.source_image_path or page.image_path,
            parser_model=_parsed_value(parser_item, "parser_model") if parser_item is not None else None,
            parser_revision=_parsed_value(parser_item, "parser_revision") if parser_item is not None else None,
            prompt_version=_parsed_value(parser_item, "prompt_version") if parser_item is not None else None,
            ocr_config_hash=_parsed_value(parser_item, "ocr_config_hash") if parser_item is not None else None,
            extraction_status=(
                _status(parser_item).value if parser_item is not None else None
            ),
            focused_render_sha256=(
                page.render_sha256 if page.source_render_sha256 is not None else None
            ),
        )

        item = parsed_by_id.get(page.page_id)
        if item is None:
            omissions.append(EvidenceOmission(page_id=page.page_id, item="ocr", reason="missing_extraction"))
            continue
        status = _status(item)
        markdown = _parsed_value(item, "markdown", "")
        if status not in (ExtractionStatus.SUCCESS, ExtractionStatus.INCOMPLETE) or not isinstance(markdown, str) or not markdown.strip():
            omissions.append(
                EvidenceOmission(page_id=page.page_id, item="ocr", reason=f"extraction_{status.value}")
            )
            continue
        tokens = _estimate_tokens(markdown)
        if text_tokens + tokens > budget.max_text_tokens:
            omissions.append(EvidenceOmission(page_id=page.page_id, item="ocr", reason="text_token_budget"))
            continue
        ocr_content[page.page_id] = markdown
        text_tokens += tokens

    payload = {
        "document_id": document_id,
        "question": question_text,
        "included_page_ids": included,
        "images": [image.model_dump(mode="json") for image in images],
        "regions": [region.model_dump(mode="json") for region in included_regions],
        "ocr_content": ocr_content,
        "omitted_items": [item.model_dump(mode="json") for item in omissions],
        "provenance": {str(key): value.model_dump(mode="json") for key, value in provenance.items()},
        "budget": budget.model_dump(mode="json"),
        "image_pixels": image_pixels,
        "text_tokens": text_tokens,
    }
    return EvidenceBundle(content_hash=_hash_bundle_payload(payload), **payload)
