"""Versioned trusted instructions used by model-facing stages."""

from .registry import PromptResourceError, load_prompt, prompt_inventory, render_prompt

__all__ = ["PromptResourceError", "load_prompt", "prompt_inventory", "render_prompt"]
