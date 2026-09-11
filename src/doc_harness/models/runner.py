"""Model runner interfaces, including a dependency-light fake and lazy Qwen runner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol, Sequence

from ..core.contracts import (
    DraftAnswer,
    EvidenceSpan,
    GenerationResult,
    ModelRequest,
    StageFailure,
)


class ModelRunner(Protocol):
    def run(self, request: ModelRequest) -> DraftAnswer:
        """Generate a structured answer for a safe model request."""


class FakeRunner:
    """Deterministic runner used by offline smoke tests."""

    def __init__(self, answer: str | None = "stub answer") -> None:
        self.answer = answer

    def run(self, request: ModelRequest) -> DraftAnswer:
        del request
        insufficient = self.answer is None
        return DraftAnswer(
            answer=self.answer,
            evidence=[] if insufficient else [EvidenceSpan(page_id=0, quote=self.answer)],
            insufficient_evidence=insufficient,
        )


class QwenTransformersRunner:
    """Qwen3.5 runner with all heavyweight imports deferred until construction."""

    def __init__(
        self,
        processor,
        model,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        max_pixels: int | None = None,
    ):
        self.processor = processor
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.max_pixels = max_pixels

    def _finish_reason(self, generated: object) -> str:
        """Infer termination only from generated token IDs and known EOS IDs."""

        length = int(getattr(generated, "shape", (0, 0))[1])
        eos_ids: set[int] = set()
        generation_config = getattr(self.model, "generation_config", None)
        configured = getattr(generation_config, "eos_token_id", None)
        if configured is None:
            tokenizer = getattr(self.processor, "tokenizer", None)
            configured = getattr(tokenizer, "eos_token_id", None)
        if isinstance(configured, (list, tuple, set)):
            eos_ids.update(int(item) for item in configured)
        elif configured is not None:
            eos_ids.add(int(configured))
        if length:
            last = generated[0, length - 1]
            try:
                if int(last.item() if hasattr(last, "item") else last) in eos_ids:
                    return "eos"
            except (TypeError, ValueError):
                pass
        if length >= self.max_new_tokens:
            return "length"
        return "unknown"

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        max_pixels: int | None = None,
    ) -> "QwenTransformersRunner":
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:
                from transformers import AutoModelForMultimodalLM as ModelClass
        except ImportError as exc:  # pragma: no cover - requires optional GPU extra
            raise RuntimeError(
                "Qwen execution requires torch and transformers; install the gpu extra"
            ) from exc

        processor_kwargs = {}
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        processor = AutoProcessor.from_pretrained(
            model_id, revision=revision, **processor_kwargs
        )
        model = ModelClass.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        return cls(processor, model, max_new_tokens, do_sample, max_pixels)

    def run(self, request: ModelRequest) -> DraftAnswer:
        result = self.generate(request)
        if result.draft is None:
            if result.failure is not None:
                raise ValueError(result.failure.message)
            raise ValueError("model returned no parsed answer")
        return result.draft

    def generate(self, request: ModelRequest) -> GenerationResult:
        """Generate raw text and preserve parsing/termination state separately."""

        if len(request.page_ids) != len(request.image_paths):
            raise ValueError("page_ids and image_paths must have the same length")
        try:
            from PIL import Image
            import torch
        except ImportError as exc:  # pragma: no cover - requires optional GPU extra
            raise RuntimeError("Qwen execution requires Pillow and torch") from exc

        images = [Image.open(Path(path)).convert("RGB") for path in request.image_paths]
        try:
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": f"{request.prompt}\n\nQuestion: {request.question}"})
            messages = [{"role": "user", "content": content}]
            template_kwargs = {
                "add_generation_prompt": True,
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            }
            try:
                inputs = self.processor.apply_chat_template(
                    messages, enable_thinking=False, **template_kwargs
                )
            except TypeError:
                # Older Qwen processors do not expose the thinking switch.
                inputs = self.processor.apply_chat_template(messages, **template_kwargs)
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                )
            generated = output_ids[:, inputs["input_ids"].shape[1] :]
            text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
            if not text.strip():
                return GenerationResult(
                    raw_response="",
                    draft=None,
                    parse_status="empty",
                    finish_reason=self._finish_reason(generated),
                    input_tokens=int(inputs["input_ids"].numel()),
                    output_tokens=int(generated.numel()),
                )
            try:
                draft = parse_draft_answer(text)
            except ValueError as exc:
                return GenerationResult(
                    raw_response=text,
                    draft=None,
                    parse_status="invalid",
                    finish_reason=self._finish_reason(generated),
                    input_tokens=int(inputs["input_ids"].numel()),
                    output_tokens=int(generated.numel()),
                    failure=StageFailure(
                        stage="parse",
                        error_type=type(exc).__name__,
                        message=str(exc),
                        retryable=True,
                    ),
                )
            return GenerationResult(
                raw_response=text,
                draft=draft,
                parse_status="valid",
                finish_reason=self._finish_reason(generated),
                input_tokens=int(inputs["input_ids"].numel()),
                output_tokens=int(generated.numel()),
            )
        finally:
            for image in images:
                image.close()


def parse_draft_answer(text: str) -> DraftAnswer:
    """Parse strict JSON, accepting one common fenced-JSON repair form."""

    candidate = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    objects: list[object] = []
    try:
        objects.append(json.loads(candidate))
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", candidate):
            try:
                payload, _ = decoder.raw_decode(candidate[match.start() :])
            except json.JSONDecodeError:
                continue
            objects.append(payload)

    required = {"answer", "evidence", "insufficient_evidence"}
    for payload in objects:
        if isinstance(payload, dict) and required.issubset(payload):
            try:
                return DraftAnswer.model_validate(payload)
            except Exception:
                continue
    raise ValueError(
        "model response does not contain a valid DraftAnswer JSON object"
    )
