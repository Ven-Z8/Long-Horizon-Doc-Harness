"""Model runner interfaces, including a dependency-light fake and lazy Qwen runner."""

from __future__ import annotations

import json
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

    def __init__(self, processor, model, max_new_tokens: int = 256, do_sample: bool = False):
        self.processor = processor
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        revision: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
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

        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = ModelClass.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        return cls(processor, model, max_new_tokens, do_sample)

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
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
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

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("model response is not valid JSON") from exc
    return DraftAnswer.model_validate(payload)
