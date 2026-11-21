"""EvoSynth generation prompt loader.

Prompt text for the EvoSynth generation pipeline lives under
``prompts/task_generation/`` (project-relative, override with
``EVOSYNTH_PROMPTS_DIR``). Code composes those files with runtime values at
call time; no prompt wording is defined inline in Python.

Placeholders use the ``{{name}}`` form and are applied with ``str.replace`` so
that literal JSON braces and other punctuation inside the prompt text are never
interpreted. ``render`` leaves unknown placeholders untouched when the caller
passes ``strict=False``.
"""
from __future__ import annotations

from functools import lru_cache

from evosynth.utils import prompts_dir

__all__ = ["PROMPT_SUBDIR", "load_prompt", "render"]

PROMPT_SUBDIR = "task_generation"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the raw text of one prompt file (cached; files are read-only).

    The file content is the prompt byte for byte, including any trailing
    newline the prompt string is meant to end with. Call sites that need an
    exact match to a previously inline string must therefore keep each file
    byte-exact.
    """
    path = prompts_dir() / PROMPT_SUBDIR / name
    if not path.is_file():
        raise FileNotFoundError(f"evosynth generation prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def render(name: str, /, *, strict: bool = True, **values: object) -> str:
    """Return one prompt file with ``{{key}}`` placeholders substituted.

    ``strict`` raises when a placeholder survives substitution, which catches
    typos between the prompt file and its call site.
    """
    text = load_prompt(name)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    if strict and "{{" in text:
        raise ValueError(
            f"unsubstituted placeholder in prompt {name!r}: "
            f"{text[text.index('{{'):text.index('{{') + 40]!r}")
    return text
