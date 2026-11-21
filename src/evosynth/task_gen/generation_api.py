"""Compatibility facade for the EvoSynth task-generation API.

实现按职责位于 :mod:`evosynth.task_gen.generation`；公开入口继续
从本模块获取 ``generate_domain``。
"""
from __future__ import annotations

from .generation.pipeline import (
    EVOSYNTH_CONTRACT_FILES,
    EVOSYNTH_PIPELINE_VERSION,
    generate_domain,
)

__all__ = [
    "EVOSYNTH_CONTRACT_FILES",
    "EVOSYNTH_PIPELINE_VERSION",
    "generate_domain",
]
