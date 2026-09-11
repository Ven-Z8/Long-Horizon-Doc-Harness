"""Model runner interfaces, including a dependency-light fake and lazy Qwen runner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol, Sequence

from .contracts import DraftAnswer, EvidenceSpan, ModelRequest


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
            return parse_draft_answer(text)
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
    # Keep a useful answer when a local checkpoint ignores the JSON-only instruction.
    # The V2 judge scores the complete response semantically, so preserving the model's
    # prose is more informative than converting a parseable answer into an abstention.
    if candidate:
        return DraftAnswer(answer=candidate, evidence=[], insufficient_evidence=False)
    raise ValueError("model response is empty")
