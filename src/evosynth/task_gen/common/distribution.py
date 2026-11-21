"""Curriculum-aware sampling helpers for task generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


UNIFORM_FLOOR = 0.3


def load_curriculum_biases(
    curriculum_path: str | Path | None,
    axes: Iterable[str],
) -> dict[str, dict[str, float]]:
    """Load supported curriculum axes, keeping the strongest duplicate bias."""
    biases: dict[str, dict[str, float]] = {str(axis): {} for axis in axes}
    if not curriculum_path:
        return biases
    path = Path(curriculum_path)
    if not path.exists():
        return biases
    try:
        with open(path) as fh:
            curriculum = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return biases
    for entry in curriculum.get("sampling_biases", []):
        axis = entry.get("axis")
        value = entry.get("value")
        if axis in biases and value:
            weight = float(entry.get("bias_weight", 1.0))
            biases[axis][value] = max(biases[axis].get(value, 1.0), weight)
    return biases


def weighted_pick(
    rng,
    options: Mapping[str, float],
    *,
    used: Mapping[str, int],
    biases: Mapping[str, float] | None = None,
    alpha: float = 1.0,
    uniform_floor: float = UNIFORM_FLOOR,
) -> str:
    """Pick by base weight, curriculum bias, and inverse coverage frequency."""
    axis_biases = biases or {}
    apply_bias = bool(axis_biases) and rng.random() > uniform_floor
    weights = {
        key: (
            weight * (axis_biases.get(key, 1.0) if apply_bias else 1.0)
        ) / (1 + used.get(key, 0)) ** alpha
        for key, weight in options.items()
    }
    total = sum(weights.values())
    target = rng.random() * total
    accumulated = 0.0
    for key, weight in weights.items():
        accumulated += weight
        if target <= accumulated:
            return key
    return list(options)[-1]


__all__ = ["UNIFORM_FLOOR", "load_curriculum_biases", "weighted_pick"]
