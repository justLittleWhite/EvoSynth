"""Small data-driven lens adapter used by semantic contract packages.

The evosynth core only knows how to execute a ``ToolLens``.  This adapter
lets a contract package describe common JSON record layouts without adding a
domain/tool branch to that core.  It intentionally supports a narrow set of
operations and raises on missing paths instead of guessing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

from .environment import ToolLens
from .runtime import QueryIR, SemanticRuntimeError


class DeclarativeLensError(SemanticRuntimeError):
    """Raised when a declarative lens rule cannot be applied."""


def _format(value: Any, *, index: int) -> Any:
    if isinstance(value, str):
        return value.replace("{index}", str(index))
    if isinstance(value, Mapping):
        return {str(k): _format(v, index=index) for k, v in value.items()}
    if isinstance(value, list):
        return [_format(v, index=index) for v in value]
    return value


def pointer_get(value: Any, pointer: str) -> Any:
    """Read an RFC6901 pointer; relative pointers are allowed for records."""

    if pointer in (None, "", "/"):
        return value
    if not isinstance(pointer, str):
        raise DeclarativeLensError("JSON pointer must be a string")
    current = value
    for raw in pointer.lstrip("/").split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise DeclarativeLensError(f"pointer not found: {pointer}")
    return current


def _descriptor(descriptor: Any, *, query: QueryIR, item: Any, index: int) -> Any:
    if not isinstance(descriptor, Mapping):
        return _format(copy.deepcopy(descriptor), index=index)
    if "literal" in descriptor:
        return _format(copy.deepcopy(descriptor["literal"]), index=index)
    origin = str(descriptor.get("from", descriptor.get("source", "item"))).lower()
    if origin in {"item", "record", "observation"}:
        source = item
    elif origin in {"query", "arguments", "args"}:
        source = {"arguments": query.arguments, "bindings": query.entity_bindings,
                  "qualifiers": query.qualifiers}
    else:
        raise DeclarativeLensError(f"unsupported descriptor source: {origin}")
    path = _format(descriptor.get("path", "/"), index=index)
    return copy.deepcopy(pointer_get(source, path))


def _claim_field(rule: Mapping[str, Any], name: str, *, query: QueryIR,
                 item: Any, index: int, default: Any = None) -> Any:
    if name not in rule:
        return default
    return _descriptor(rule[name], query=query, item=item, index=index)


@dataclass(frozen=True)
class DeclarativeLens:
    """Compile an extraction/render description into a generic ``ToolLens``."""

    tool_id: str
    contract: Mapping[str, Any]

    def _records(self, observation: Any) -> list[Any]:
        pointer = self.contract.get("collection_pointer", self.contract.get("records_pointer"))
        if pointer is None:
            return [observation]
        records = pointer_get(observation, pointer)
        if not isinstance(records, list):
            raise DeclarativeLensError("collection pointer must resolve to a list")
        return records

    def extract(self, query: QueryIR, observation: Any) -> list[Mapping[str, Any]]:
        rule = self.contract.get("claim", self.contract.get("claim_rule", {}))
        if not isinstance(rule, Mapping):
            raise DeclarativeLensError("claim rule must be an object")
        claims: list[Mapping[str, Any]] = []
        collection_pointer = self.contract.get("collection_pointer", self.contract.get("records_pointer"))
        value_pointer = rule.get("value_pointer", rule.get("value_path"))
        if value_pointer is not None and not isinstance(value_pointer, str):
            raise DeclarativeLensError("value_pointer must be a string")
        for index, item in enumerate(self._records(observation)):
            value = _claim_field(rule, "value", query=query, item=item, index=index)
            if value is None and value_pointer is not None:
                value = pointer_get(item, _format(value_pointer, index=index))
            if value is None and "value" not in rule and value_pointer is None:
                raise DeclarativeLensError("claim rule needs value or value_pointer")
            claim: dict[str, Any] = {
                "subject": _claim_field(rule, "subject", query=query, item=item, index=index),
                "predicate": _claim_field(rule, "predicate", query=query, item=item, index=index),
                "value": value,
            }
            # Claim identity belongs to ClaimLedger.  A lens may provide an
            # explicit external id, but positional defaults such as claim-0
            # collide across calls and make split/merged views inequivalent.
            if "claim_id" in rule:
                claim_id = _claim_field(
                    rule, "claim_id", query=query, item=item, index=index,
                )
                if claim_id is None:
                    raise DeclarativeLensError("claim_id descriptor resolved to null")
                claim["claim_id"] = str(claim_id)
            for name in ("unit", "value_type", "precision", "uncertainty"):
                field = _claim_field(rule, name, query=query, item=item, index=index)
                if field is not None:
                    claim[name] = field
            qualifiers = rule.get("qualifiers")
            if isinstance(qualifiers, Mapping):
                claim["qualifiers"] = {
                    str(key): _descriptor(value, query=query, item=item, index=index)
                    for key, value in qualifiers.items()
                }
            elif qualifiers is not None:
                claim["qualifiers"] = _descriptor(qualifiers, query=query, item=item, index=index)
            pointer = rule.get("output_pointer", value_pointer)
            if pointer is not None:
                prefix = str(collection_pointer or "").rstrip("/")
                suffix = _format(str(pointer), index=index).lstrip("/")
                claim["provenance"] = [{"output_pointer": f"{prefix}/{index}/{suffix}" if collection_pointer else f"/{suffix}"}]
            claims.append(claim)
        return claims

    def render(self, query: QueryIR, claims: list[Any]) -> Any:
        render = self.contract.get("render", {"type": "claims"})
        if not isinstance(render, Mapping):
            raise DeclarativeLensError("render rule must be an object")
        mode = str(render.get("type", "records")).lower()
        if mode in {"claims", "identity"}:
            return [copy.deepcopy(item.to_dict() if hasattr(item, "to_dict") else item) for item in claims]
        if mode in {"value", "scalar"}:
            if len(claims) != 1:
                raise DeclarativeLensError("scalar render requires one claim")
            return copy.deepcopy(getattr(claims[0], "value", claims[0]))
        if mode not in {"records", "record", "object", "singleton"}:
            raise DeclarativeLensError(f"unsupported render type: {mode}")
        key = str(render.get("records_key", "results"))
        fields = render.get("fields", {"value": {"claim": "value"}})
        if not isinstance(fields, Mapping):
            raise DeclarativeLensError("render fields must be an object")
        records = []
        for claim in claims:
            record: dict[str, Any] = {}
            for name, descriptor in fields.items():
                if isinstance(descriptor, Mapping) and "claim" in descriptor:
                    source = str(descriptor["claim"])
                    value = getattr(claim, source, None)
                    if isinstance(claim, Mapping):
                        value = claim.get(source)
                    elif value is None and hasattr(claim, "to_dict"):
                        value = claim.to_dict().get(source)
                elif isinstance(descriptor, Mapping) and "qualifier" in descriptor:
                    qualifiers = (claim.get("qualifiers", {}) if isinstance(claim, Mapping)
                                  else getattr(claim, "qualifiers", {})) or {}
                    value = qualifiers.get(descriptor["qualifier"])
                else:
                    value = descriptor
                record[str(name)] = copy.deepcopy(value)
            records.append(record)
        if mode in {"record", "object", "singleton"}:
            if len(records) != 1:
                raise DeclarativeLensError("record render requires one claim")
            return records[0]
        return {key: records}

    def as_tool_lens(self) -> ToolLens:
        return ToolLens(self.tool_id, self.extract, self.render)


__all__ = ["DeclarativeLens", "DeclarativeLensError", "pointer_get"]
