"""Load, render, and fingerprint packaged prompt resources."""

from __future__ import annotations

import hashlib
from importlib import resources
from string import Template
from typing import Mapping


class PromptResourceError(ValueError):
    """Raised when a prompt resource or its render fields is invalid."""


_PROMPT_NAMES = (
    "system",
    "planning",
    "retrieval",
    "reranking",
    "ocr",
    "answer",
    "verification",
    "repair",
    "synthesis",
)
_SETS = {"control", "v2"}


def _resource(name: str, prompt_set: str):
    if prompt_set not in _SETS:
        raise PromptResourceError(f"unknown prompt set: {prompt_set}")
    if name not in _PROMPT_NAMES:
        raise PromptResourceError(f"unknown prompt name: {name}")
    return resources.files(__package__).joinpath(prompt_set, f"{name}.txt")


def load_prompt(name: str, prompt_set: str) -> str:
    resource = _resource(name, prompt_set)
    try:
        text = resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        raise PromptResourceError(f"prompt resource is unavailable: {prompt_set}/{name}.txt") from exc
    if not text.strip():
        raise PromptResourceError(f"prompt resource is empty: {prompt_set}/{name}.txt")
    return text


def prompt_inventory(prompt_set: str) -> dict[str, str]:
    """Return resource filenames and SHA-256 content hashes."""

    return {
        f"{name}.txt": hashlib.sha256(load_prompt(name, prompt_set).encode("utf-8")).hexdigest()
        for name in _PROMPT_NAMES
    }


def render_prompt(name: str, prompt_set: str, fields: Mapping[str, str]) -> str:
    template = Template(load_prompt(name, prompt_set))
    identifiers = set(template.get_identifiers())
    supplied = set(fields)
    missing = sorted(identifiers - supplied)
    extra = sorted(supplied - identifiers)
    if missing:
        raise PromptResourceError(f"missing fields for {name}: {', '.join(missing)}")
    if extra:
        raise PromptResourceError(f"unexpected fields for {name}: {', '.join(extra)}")
    try:
        rendered = template.substitute({key: str(value) for key, value in fields.items()})
    except (KeyError, ValueError) as exc:
        raise PromptResourceError(f"could not render prompt {prompt_set}/{name}.txt") from exc
    if not rendered.strip():
        raise PromptResourceError(f"rendered prompt is empty: {prompt_set}/{name}.txt")
    return rendered
