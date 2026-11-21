"""Fail-closed loader for declarative semantic tool specifications.

Registry snapshots in the legacy tree use several equivalent shapes (a keyed
``tools`` object, a list, or a single tool object).  Normalising those shapes
at one boundary prevents each caller from inventing a slightly different
meaning for a missing tool id or version.  The loader only constructs
``ToolSemanticSpec`` objects; executable lenses remain outside the snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .runtime import (
    SemanticRuntimeError,
    ToolSemanticRegistry,
    ToolSemanticSpec,
)


_CONTAINER_KEYS = ("tools", "specs", "semantic_tools", "contracts")


def _qualified_parts(key: str, default_version: str) -> tuple[str, str]:
    text = str(key).strip()
    if not text:
        raise SemanticRuntimeError("semantic tool key cannot be empty")
    if "@" in text:
        tool_id, version = text.rsplit("@", 1)
        if not tool_id or not version:
            raise SemanticRuntimeError(f"invalid qualified semantic tool key: {text!r}")
        return tool_id, version
    return text, default_version


def iter_spec_dicts(raw: Any) -> Iterable[dict[str, Any]]:
    """Yield explicit spec mappings from all supported snapshot shapes."""

    if isinstance(raw, ToolSemanticRegistry):
        for item in raw.to_dict().values():
            yield dict(item)
        return
    if isinstance(raw, ToolSemanticSpec):
        yield raw.to_dict()
        return
    if isinstance(raw, (str, bytes, bytearray, Path)):
        raise SemanticRuntimeError("semantic specs must be decoded before loading")
    if isinstance(raw, Mapping):
        if raw.get("tool_id") or raw.get("id"):
            item = dict(raw)
            if not item.get("tool_id"):
                item["tool_id"] = item.pop("id")
            yield item
            return
        container = None
        for key in _CONTAINER_KEYS:
            if key in raw:
                container = raw[key]
                break
        if container is None:
            # A keyed mapping is accepted, but arbitrary metadata is not
            # silently ignored: every value must be a tool object.
            container = raw
        default_version = str(raw.get("version", raw.get("spec_version", "1")))
        if isinstance(container, Mapping):
            for key, value in container.items():
                if not isinstance(value, Mapping):
                    raise SemanticRuntimeError(f"semantic spec {key!r} must be an object")
                item = dict(value)
                if not item.get("tool_id"):
                    tool_id, version = _qualified_parts(str(key), str(item.get("version", default_version)))
                    item["tool_id"] = tool_id
                    item.setdefault("version", version)
                elif "@" in str(key):
                    key_tool, key_version = _qualified_parts(str(key), default_version)
                    if str(item["tool_id"]) != key_tool:
                        raise SemanticRuntimeError(
                            f"semantic spec key/id mismatch: {key!r} != {item['tool_id']!r}"
                        )
                    if item.get("version") is not None and str(item["version"]) != key_version:
                        raise SemanticRuntimeError(
                            f"semantic spec key/version mismatch: {key!r} != {item['version']!r}"
                        )
                    item["version"] = key_version
                yield item
            return
        if isinstance(container, (list, tuple)):
            for item in container:
                if not isinstance(item, Mapping):
                    raise SemanticRuntimeError("each semantic spec in a list must be an object")
                yield dict(item)
            return
        raise SemanticRuntimeError("semantic spec container must be a mapping or list")
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if not isinstance(item, Mapping):
                raise SemanticRuntimeError("each semantic spec in a list must be an object")
            yield dict(item)
        return
    raise SemanticRuntimeError(f"unsupported semantic spec container: {type(raw)!r}")


def load_semantic_registry(raw: Any) -> ToolSemanticRegistry:
    specs = [ToolSemanticSpec.from_dict(item) for item in iter_spec_dicts(raw)]
    if not specs:
        raise SemanticRuntimeError("semantic spec snapshot contains no tools")
    return ToolSemanticRegistry(specs)


def load_semantic_registry_file(path: str | Path) -> ToolSemanticRegistry:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SemanticRuntimeError(f"cannot read semantic spec snapshot {path}: {exc}") from exc
    return load_semantic_registry(raw)


__all__ = ["iter_spec_dicts", "load_semantic_registry", "load_semantic_registry_file"]
