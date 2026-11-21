"""Shared model-specific sampling parameter normalization."""

from __future__ import annotations

from typing import Any, Mapping


FIXED_TEMPERATURE_MODELS = frozenset({"kimi-k3", "kimi-k2.7-code"})


def normalize_sampling_kwargs(
    model: str,
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(kwargs)
    if str(model) in FIXED_TEMPERATURE_MODELS and "temperature" in normalized:
        normalized["temperature"] = 1
    return normalized


__all__ = ["FIXED_TEMPERATURE_MODELS", "normalize_sampling_kwargs"]
