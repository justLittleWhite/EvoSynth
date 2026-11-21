"""Canonical JSON serialization shared by task-generation modules."""

from __future__ import annotations

import json
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = ["canonical_json"]
