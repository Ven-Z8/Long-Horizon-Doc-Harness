"""OCR adapters and provenance-preserving page extraction caches.

The module deliberately has no import-time dependency on a model runtime.  A
small ``PageParser`` protocol makes the cache usable with a fake parser in
unit tests and with the Qianfan adapter in a GPU worker.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from pydantic import Field, model_validator

from .contracts import Page, ParsedPage as ContractParsedPage, StrictModel


class ExtractionStatus(str, Enum):
    """The outcome of parsing one page.

    ``incomplete`` is intentionally distinct from ``success``: a truncated
    transcription must remain visible in traces and must not be mistaken for
    a complete OCR result.
    """

    SUCCESS = "success"
    EMPTY = "empty"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


class _CallableIdentity(str):
    """A string identity that also supports the older callable style."""

    def __call__(self) -> str:
        return str(self)


class OCRParsedPage(ContractParsedPage):
    """A parsed page with source and cache provenance.

    The original baseline ``contracts.ParsedPage`` remains unchanged for
    compatibility.  This extended type is what Stage 3 parsers and caches
    return; old contract instances are accepted and normalized by
    :func:`parse_cached`.
    """

    document_id: str = ""
    render_sha256: str = ""
    extraction_status: ExtractionStatus = Field(
        default=ExtractionStatus.SUCCESS,
        validation_alias="extraction_status",
    )
    raw_output_path: str | None = None
    ocr_config_hash: str = ""
    cache_key: str = ""
    error: str | None = None
    token_count: int = Field(default=0, ge=0)
    truncated: bool = False

    @model_validator(mode="after")
    def token_count_is_consistent(self) -> "OCRParsedPage":
        measured = _estimate_tokens(self.markdown)
        if self.token_count == 0 and measured:
            self.token_count = measured
        return self

    @property
    def status(self) -> ExtractionStatus:
        """Short alias used by callers that call the field simply ``status``."""

        return self.extraction_status

    @property
    def cache_identity(self) -> _CallableIdentity:
        """Return a stable identity for the source and parser configuration."""

        payload = {
            "document_id": self.document_id,
            "page_id": self.page_id,
            "render_sha256": self.render_sha256,
            "parser_model": self.parser_model,
            "parser_revision": self.parser_revision,
            "prompt_version": self.prompt_version,
            "ocr_config_hash": self.ocr_config_hash,
        }
        return _CallableIdentity(_sha256_json(payload))


# The name in the Stage 3 interface is ParsedPage.  Keep OCRParsedPage as a
# descriptive alias as well, which makes it clear that it extends the baseline
# contract when reading type annotations.
ParsedPage = OCRParsedPage


@runtime_checkable
class PageParser(Protocol):
    """Injectable parser interface used by cache and batch code."""

    def parse(self, page: Page) -> OCRParsedPage | ContractParsedPage | Mapping[str, Any] | str:
        ...


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    """Use a deterministic whitespace token estimate for cache metadata.

    Actual model tokenizers are optional and belong in the model worker.  The
    estimate is intentionally conservative enough for stable offline packing.
    """

    return len(re.findall(r"\S+", text))


def _parser_identity(parser: PageParser) -> dict[str, str]:
    """Read parser identity without requiring a concrete model implementation."""

    def attr(*names: str, default: str = "") -> str:
        for name in names:
            value = getattr(parser, name, None)
            if value is not None:
                return str(value)
        return default

    model = attr("parser_model", "model_id", "model", default=parser.__class__.__qualname__)
    revision = attr("parser_revision", "revision", default="unknown")
    prompt = attr("prompt_version", "prompt_hash", default="unknown")
    config = attr("ocr_config_hash", "config_hash", default="")
    if not config:
        config = _sha256_json(
            {
                "parser_model": model,
                "parser_revision": revision,
                "prompt_version": prompt,
            }
        )
    custom_identity = getattr(parser, "cache_identity", None)
    if callable(custom_identity):
        try:
            custom_value = custom_identity()
        except TypeError:
            custom_value = None
        if custom_value is not None:
            config = _sha256_json({"config": config, "adapter_identity": str(custom_value)})
    elif custom_identity is not None:
        config = _sha256_json({"config": config, "adapter_identity": str(custom_identity)})
    return {
        "parser_model": model,
        "parser_revision": revision,
        "prompt_version": prompt,
        "ocr_config_hash": config,
    }


def _cache_key(page: Page, identity: Mapping[str, str]) -> str:
    return _sha256_json(
        {
            "document_id": page.document_id,
            "page_id": page.page_id,
            "render_sha256": page.render_sha256,
            **dict(identity),
        }
    )


def _invalid_page(
    page: Page,
    identity: Mapping[str, str],
    status: ExtractionStatus,
    *,
    error: str | None = None,
    markdown: str = "",
    truncated: bool = False,
) -> OCRParsedPage:
    return OCRParsedPage(
        page_id=page.page_id,
        markdown=markdown,
        parser_model=identity["parser_model"],
        parser_revision=identity["parser_revision"],
        prompt_version=identity["prompt_version"],
        document_id=page.document_id,
        render_sha256=page.render_sha256,
        extraction_status=status,
        ocr_config_hash=identity["ocr_config_hash"],
        cache_key=_cache_key(page, identity),
        error=error,
        token_count=_estimate_tokens(markdown),
        truncated=truncated,
    )


def _normalize_result(
    page: Page,
    result: object,
    identity: Mapping[str, str],
    *,
    max_text_tokens: int | None = None,
) -> OCRParsedPage:
    """Validate an injected parser result and attach the source identity."""

    if isinstance(result, str):
        values: dict[str, Any] = {"markdown": result}
    elif isinstance(result, OCRParsedPage):
        values = result.model_dump()
    elif isinstance(result, ContractParsedPage):
        values = result.model_dump()
    elif isinstance(result, Mapping):
        values = dict(result)
    else:
        return _invalid_page(page, identity, ExtractionStatus.INVALID, error="parser returned a non-page result")

    markdown = values.get("markdown")
    if not isinstance(markdown, str):
        return _invalid_page(page, identity, ExtractionStatus.INVALID, error="parser markdown must be a string")

    supplied_document = values.get("document_id")
    if supplied_document not in (None, "", page.document_id):
        return _invalid_page(page, identity, ExtractionStatus.INVALID, error="parser returned a different document")
    supplied_page = values.get("page_id")
    if supplied_page not in (None, page.page_id):
        return _invalid_page(page, identity, ExtractionStatus.INVALID, error="parser returned a different page")
    supplied_render = values.get("render_sha256")
    if supplied_render not in (None, "", page.render_sha256):
        return _invalid_page(page, identity, ExtractionStatus.INVALID, error="parser returned a stale render")

    raw_status = values.get("extraction_status", values.get("status", ExtractionStatus.SUCCESS))
    try:
        status = raw_status if isinstance(raw_status, ExtractionStatus) else ExtractionStatus(str(raw_status))
    except ValueError:
        return _invalid_page(page, identity, ExtractionStatus.INVALID, error="parser returned an unknown extraction status")

    truncated = bool(values.get("truncated", False))
    if max_text_tokens is not None:
        if max_text_tokens <= 0:
            raise ValueError("max_text_tokens must be positive")
        token_words = markdown.split()
        if len(token_words) > max_text_tokens:
            markdown = " ".join(token_words[:max_text_tokens])
            truncated = True
            status = ExtractionStatus.INCOMPLETE

    if not markdown.strip() and status is ExtractionStatus.SUCCESS:
        status = ExtractionStatus.EMPTY
    if truncated and status is ExtractionStatus.SUCCESS:
        status = ExtractionStatus.INCOMPLETE

    return OCRParsedPage(
        page_id=page.page_id,
        markdown=markdown,
        parser_model=identity["parser_model"],
        parser_revision=identity["parser_revision"],
        prompt_version=identity["prompt_version"],
        document_id=page.document_id,
        render_sha256=page.render_sha256,
        extraction_status=status,
        raw_output_path=None,
        ocr_config_hash=identity["ocr_config_hash"],
        cache_key=_cache_key(page, identity),
        error=values.get("error"),
        token_count=_estimate_tokens(markdown),
        truncated=truncated,
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)


def parse_cached(
    page: Page,
    parser: PageParser,
    cache_dir: Path,
    *,
    max_text_tokens: int | None = None,
) -> OCRParsedPage:
    """Parse one page, reusing only a matching successful cache entry.

    Parser exceptions become a typed ``failed`` result.  Failed, empty,
    malformed, and incomplete results can still be inspected by the caller,
    but are never written as successful cache entries.
    """

    if not isinstance(page, Page):
        raise TypeError("page must be a Page")
    cache_dir = Path(cache_dir)
    identity = _parser_identity(parser)
    key = _cache_key(page, identity)
    cache_path = cache_dir / f"{key}.json"

    if cache_path.is_file():
        try:
            cached = OCRParsedPage.model_validate_json(cache_path.read_text())
            if (
                cached.cache_key == key
                and cached.cache_identity == key
                and cached.extraction_status is ExtractionStatus.SUCCESS
                and cached.raw_output_path is not None
                and Path(cached.raw_output_path).is_file()
            ):
                return cached
        except Exception:
            # A partial or old cache entry is a miss; do not expose it as OCR.
            pass

    try:
        result = parser.parse(page)
    except Exception as exc:  # parser workers should not take down a batch
        return _invalid_page(
            page,
            identity,
            ExtractionStatus.FAILED,
            error=f"{type(exc).__name__}: {exc}",
        )

    parsed = _normalize_result(page, result, identity, max_text_tokens=max_text_tokens)
    if parsed.extraction_status is not ExtractionStatus.SUCCESS:
        return parsed

    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / f"{key}.txt"
    raw_path.write_text(parsed.markdown, encoding="utf-8")
    parsed = parsed.model_copy(update={"raw_output_path": str(raw_path), "cache_key": key})
    _write_json_atomic(cache_path, parsed.model_dump(mode="json"))
    return parsed


def parse_many_cached(
    pages: list[Page], parser: PageParser, cache_dir: Path, *, max_text_tokens: int | None = None
) -> list[OCRParsedPage]:
    """Parse a stable, deduplicated page union while one parser is resident."""

    seen: set[tuple[str, int]] = set()
    unique: list[Page] = []
    for page in pages:
        key = (page.document_id, page.page_id)
        if key not in seen:
            seen.add(key)
            unique.append(page)
    return [parse_cached(page, parser, cache_dir, max_text_tokens=max_text_tokens) for page in unique]


# An intuitive alias for callers that describe the operation as batch OCR.
parse_batch_cached = parse_many_cached


class QianfanOCRParser:
    """Lazy Transformers adapter for the pinned Qianfan OCR checkpoint.

    The exact processor/model objects can be injected in tests or in a worker.
    ``from_pretrained`` is the only method that imports optional model
    dependencies, so importing this module remains CPU-only and offline-safe.
    """

    def __init__(
        self,
        processor: Any | None = None,
        model: Any | None = None,
        *,
        model_id: str = "baidu/Qianfan-OCR",
        revision: str = "623bf5d20d446abdb36606aa4547cd0c18886fe5",
        prompt_version: str = "qianfan-ocr-v1",
        ocr_config_hash: str = "",
        max_new_tokens: int = 4096,
        do_sample: bool = False,
    ) -> None:
        self.processor = processor
        self.model = model
        self.parser_model = model_id
        self.parser_revision = revision
        self.prompt_version = prompt_version
        self.ocr_config_hash = ocr_config_hash or _sha256_json(
            {
                "model_id": model_id,
                "revision": revision,
                "prompt_version": prompt_version,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
        )
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "baidu/Qianfan-OCR",
        revision: str = "623bf5d20d446abdb36606aa4547cd0c18886fe5",
        **kwargs: Any,
    ) -> "QianfanOCRParser":
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:  # pragma: no cover - older Transformers fallback
                from transformers import AutoModelForCausalLM as ModelClass
        except ImportError as exc:  # pragma: no cover - optional GPU integration
            raise RuntimeError("Qianfan OCR requires torch and transformers") from exc

        processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        model = ModelClass.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        return cls(processor, model, model_id=model_id, revision=revision, **kwargs)

    def parse(self, page: Page) -> OCRParsedPage:
        if self.processor is None or self.model is None:
            raise RuntimeError("QianfanOCRParser requires injected processor and model")
        try:
            from PIL import Image
            import torch
        except ImportError as exc:  # pragma: no cover - optional GPU integration
            raise RuntimeError("Qianfan OCR requires Pillow and torch") from exc

        prompt = "Transcribe this document page as faithful markdown. Preserve tables and layout."
        image = Image.open(Path(page.image_path)).convert("RGB")
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return OCRParsedPage(
                document_id=page.document_id,
                page_id=page.page_id,
                markdown=text,
                parser_model=self.parser_model,
                parser_revision=self.parser_revision,
                prompt_version=self.prompt_version,
                render_sha256=page.render_sha256,
                extraction_status=ExtractionStatus.SUCCESS if text.strip() else ExtractionStatus.EMPTY,
                ocr_config_hash=self.ocr_config_hash,
            )
        finally:
            image.close()
