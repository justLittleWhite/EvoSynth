"""Generic semantic query/runtime layer for the EvoSynth pipeline.

The legacy runtime keys observations by raw tool arguments.  This module makes
the semantic boundary explicit: a tool contract normalizes a call into a
``QueryIR`` and a world ledger stores entities/claims separately from the
tool's rendered observation.  Contracts are declarative and can be proposed
from schemas, but callers must validate them before using them for synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
import base64
import copy
import hashlib
import json
import math
from typing import Any, Mapping

from .store import ClaimLedger, EntityLedger


class SemanticRuntimeError(ValueError):
    """Base error for invalid semantic contracts or queries."""


class AmbiguousEntityError(SemanticRuntimeError):
    """Raised when a selector maps to more than one entity."""


def _jsonable(value: Any) -> Any:
    """Convert arguments to JSON without changing their semantic value.

    The legacy implementation formatted coordinates to 15 significant digits
    and converted them to strings.  That made distinct geometry collide and
    handed lenses the wrong type.  Canonicalisation is deliberately lossless;
    contracts that want a measurement tolerance must state it explicitly.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    # Semantic records and enum-like values can expose a stable structural
    # representation.  Prefer it over ``repr`` so observations remain
    # serialisable and hashable across processes.
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if converted is not value:
            return _jsonable(converted)
    if hasattr(value, "value") and not isinstance(value, (str, bytes, bytearray)):
        try:
            converted = value.value
            if converted is not value:
                return _jsonable(converted)
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        values = [_jsonable(v) for v in value]
        return sorted(values, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SemanticRuntimeError("non-finite numeric argument")
        return value
    if isinstance(value, Decimal):
        return {"__decimal__": format(value, "f")}
    if isinstance(value, (datetime, date)):
        return {"__datetime__" if isinstance(value, datetime) else "__date__": value.isoformat()}
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    raise SemanticRuntimeError(f"unsupported argument value type: {type(value)!r}")


def _digest(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryIR:
    """Canonical semantic representation of one tool query."""

    tool_id: str
    operator: str
    arguments: Mapping[str, Any]
    entity_bindings: Mapping[str, str] = field(default_factory=dict)
    predicates: tuple[str, ...] = ()
    predicate_arguments: Mapping[str, Any] = field(default_factory=dict)
    qualifiers: Mapping[str, Any] = field(default_factory=dict)
    projection: tuple[str, ...] = ()
    ordering: tuple[str, ...] = ()
    cardinality: str = "unknown"
    completeness: str = "unknown"
    unresolved_selectors: tuple[str, ...] = ()
    # ``unresolved_selectors`` is retained for audit/debugging even when a
    # contract explicitly permits literal selectors.  Only selectors listed
    # here are hard failures for an environment using ``reject_unresolved``.
    blocking_unresolved_selectors: tuple[str, ...] = ()
    # Schema-shaped arguments before EntityID substitution. Generators use
    # these to call a real tool/world model; semantic identity and caching use
    # ``arguments`` and ``entity_bindings``.
    public_arguments: Mapping[str, Any] = field(default_factory=dict)

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "operator": self.operator,
            "arguments": _jsonable(self.arguments),
            "entity_bindings": _jsonable(self.entity_bindings),
            "predicates": list(self.predicates),
            "predicate_arguments": _jsonable(self.predicate_arguments),
            "qualifiers": _jsonable(self.qualifiers),
            "projection": list(self.projection),
            "ordering": list(self.ordering),
            "cardinality": self.cardinality,
            "completeness": self.completeness,
            "unresolved_selectors": list(self.unresolved_selectors),
            "blocking_unresolved_selectors": list(self.blocking_unresolved_selectors),
        }

    @property
    def digest(self) -> str:
        return _digest(self.canonical_dict())

    def equivalent_to(self, other: "QueryIR") -> bool:
        """Semantic equality, deliberately stricter than text similarity."""
        return self.canonical_dict() == other.canonical_dict()


@dataclass(frozen=True)
class ToolSemanticSpec:
    """Declarative semantic contract for a tool.

    A spec does not contain Python callbacks.  ``argument_aliases`` and
    ``defaults`` handle syntax-level normalization; the remaining fields
    describe the claims and scope that a renderer/extractor must implement.
    """

    tool_id: str
    version: str = "1"
    operator: str = "lookup"
    entity_args: tuple[str, ...] = ()
    predicate: str | None = None
    predicate_args: Mapping[str, str] = field(default_factory=dict)
    qualifier_args: tuple[str, ...] = ()
    projection: tuple[str, ...] = ()
    ordering: tuple[str, ...] = ()
    cardinality: str = "unknown"
    completeness: str = "unknown"
    defaults: Mapping[str, Any] = field(default_factory=dict)
    argument_aliases: Mapping[str, str] = field(default_factory=dict)
    unordered_args: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    # Per-argument selector semantics.  ``strict`` (the default) requires a
    # registered entity; ``literal`` preserves structured values verbatim;
    # ``resolve_or_literal`` resolves known selectors and retains unknown ones.
    # Kept at the end of the dataclass to preserve legacy positional callers.
    selector_policies: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Freeze nested contract containers at construction time.  ``to_dict``
        # can then safely return detached data without allowing registry drift.
        # Accept a mapping in ``entity_args`` as a small ergonomic shorthand,
        # e.g. ``{"locations": "literal"}``, while keeping the serialized
        # contract explicit and backwards compatible with the tuple form.
        raw_entity_args = self.entity_args
        inferred_policies: dict[str, str] = {}
        if isinstance(raw_entity_args, Mapping):
            inferred_policies = {str(name): str(mode) for name, mode in raw_entity_args.items()}
            normalized_entity_args = tuple(inferred_policies)
        else:
            normalized_entity_args = tuple(raw_entity_args or ())
        policies = {str(name): str(mode) for name, mode in (self.selector_policies or {}).items()}
        for name, mode in inferred_policies.items():
            policies.setdefault(name, mode)
        object.__setattr__(self, "predicate_args", dict(self.predicate_args or {}))
        object.__setattr__(self, "defaults", dict(self.defaults or {}))
        object.__setattr__(self, "argument_aliases", dict(self.argument_aliases or {}))
        object.__setattr__(self, "entity_args", normalized_entity_args)
        object.__setattr__(self, "selector_policies", policies)
        object.__setattr__(self, "qualifier_args", tuple(self.qualifier_args or ()))
        object.__setattr__(self, "projection", tuple(self.projection or ()))
        object.__setattr__(self, "ordering", tuple(self.ordering or ()))
        object.__setattr__(self, "unordered_args", tuple(self.unordered_args or ()))
        object.__setattr__(self, "capabilities", tuple(self.capabilities or ()))

    @property
    def qualified_tool_id(self) -> str:
        return f"{self.tool_id}@{self.version}"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ToolSemanticSpec":
        if not isinstance(raw, Mapping):
            raise SemanticRuntimeError("semantic spec must be an object")
        if not raw.get("tool_id"):
            raise SemanticRuntimeError("semantic spec requires tool_id")
        raw_entity_args = raw.get("entity_args", ())
        inferred_policies: dict[str, str] = {}
        if isinstance(raw_entity_args, Mapping):
            inferred_policies = {str(name): str(mode) for name, mode in raw_entity_args.items()}
            entity_args = tuple(inferred_policies)
        else:
            entity_args = tuple(str(x) for x in raw_entity_args)
        raw_policies = raw.get(
            "selector_policies",
            raw.get("selector_policy", raw.get("entity_selector_policies", {})),
        )
        if isinstance(raw_policies, str):
            raw_policies = {name: raw_policies for name in entity_args}
        policies = dict(inferred_policies)
        policies.update({str(name): str(mode) for name, mode in (raw_policies or {}).items()})
        # These aliases make schema-generated contracts readable without
        # changing the core semantics; they are converted to the same policy
        # map and validated below.
        for name in raw.get("structured_entity_args", ()) or ():
            policies[str(name)] = "literal"
        for name in raw.get("allow_unresolved_entity_args", ()) or ():
            policies[str(name)] = "resolve_or_literal"
        return cls(
            tool_id=str(raw["tool_id"]),
            version=str(raw.get("version", "1")),
            operator=str(raw.get("operator", "lookup")),
            entity_args=entity_args,
            selector_policies=policies,
            predicate=raw.get("predicate"),
            predicate_args=dict(raw.get("predicate_args", {})),
            qualifier_args=tuple(str(x) for x in raw.get("qualifier_args", raw.get("qualifiers", ()))),
            projection=tuple(str(x) for x in raw.get("projection", ())),
            ordering=tuple(str(x) for x in raw.get("ordering", ())),
            cardinality=str(raw.get("cardinality", "unknown")),
            completeness=str(raw.get("completeness", "unknown")),
            defaults=dict(raw.get("defaults", {})),
            argument_aliases=dict(raw.get("argument_aliases", {})),
            unordered_args=tuple(str(x) for x in raw.get("unordered_args", ())),
            capabilities=tuple(str(x) for x in raw.get("capabilities", ())),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.tool_id or "/" not in self.tool_id:
            errors.append("tool_id must be a fully-qualified server/tool identifier")
        if self.cardinality not in {"unknown", "one", "many", "set"}:
            errors.append(f"unsupported cardinality: {self.cardinality}")
        if self.completeness not in {"unknown", "local", "closed", "unbounded"}:
            errors.append(f"unsupported completeness: {self.completeness}")
        allowed_selector_policies = {
            "strict", "resolve", "literal", "structured", "resolve_or_literal", "optional",
        }
        for name, mode in self.selector_policies.items():
            if name not in self.entity_args:
                errors.append(f"selector policy references non-entity argument: {name}")
            if mode not in allowed_selector_policies:
                errors.append(f"unsupported selector policy for {name}: {mode}")
        if self.operator in {"threshold", "top_k", "nearest", "identify"} and self.completeness == "unknown":
            errors.append("selection operators require an explicit completeness contract")
        for alias, target in self.argument_aliases.items():
            if not alias or not target:
                errors.append("argument aliases cannot contain empty names")
        aliases = {str(k): str(v) for k, v in self.argument_aliases.items()}
        for start in aliases:
            seen: set[str] = set()
            cursor = start
            while cursor in aliases:
                if cursor in seen:
                    errors.append(f"argument alias cycle:{start}")
                    break
                seen.add(cursor)
                cursor = aliases[cursor]
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "version": self.version,
            "operator": self.operator,
            "entity_args": list(self.entity_args),
            "selector_policies": dict(self.selector_policies),
            "predicate": self.predicate,
            "predicate_args": dict(self.predicate_args),
            "qualifier_args": list(self.qualifier_args),
            "projection": list(self.projection),
            "ordering": list(self.ordering),
            "cardinality": self.cardinality,
            "completeness": self.completeness,
            "defaults": json.loads(json.dumps(_jsonable(self.defaults), ensure_ascii=False)),
            "argument_aliases": dict(self.argument_aliases),
            "unordered_args": list(self.unordered_args),
            "capabilities": list(self.capabilities),
            "qualified_tool_id": self.qualified_tool_id,
        }


class ToolSemanticRegistry:
    """Registry for validated, fully-qualified semantic tool contracts."""

    def __init__(self, specs: list[ToolSemanticSpec] | Mapping[str, Any] | None = None):
        self._specs: dict[str, ToolSemanticSpec] = {}
        if isinstance(specs, Mapping):
            # Keep the public constructor ergonomic while routing all shape
            # variants through the single fail-closed loader.
            from .spec import iter_spec_dicts
            values = [ToolSemanticSpec.from_dict(item) for item in iter_spec_dicts(specs)]
        else:
            values = list(specs or [])
        for spec in values:
            self.register(spec)

    def register(self, spec: ToolSemanticSpec) -> None:
        errors = spec.validate()
        if errors:
            raise SemanticRuntimeError("; ".join(errors))
        key = spec.qualified_tool_id
        if key in self._specs:
            raise SemanticRuntimeError(f"duplicate semantic tool: {key}")
        self._specs[key] = spec

    def get(self, tool_id: str, version: str | None = None) -> ToolSemanticSpec:
        if version is not None:
            key = f"{tool_id}@{version}"
        elif "@" in tool_id:
            key = tool_id
        else:
            matches = [s for s in self._specs.values() if s.tool_id == tool_id]
            if len(matches) != 1:
                raise SemanticRuntimeError(f"tool version is ambiguous: {tool_id}")
            return matches[0]
        try:
            return self._specs[key]
        except KeyError as exc:
            raise SemanticRuntimeError(f"unknown semantic tool: {key}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {key: spec.to_dict() for key, spec in sorted(self._specs.items())}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | list[Any]) -> "ToolSemanticRegistry":
        from .spec import iter_spec_dicts
        return cls([ToolSemanticSpec.from_dict(item) for item in iter_spec_dicts(raw)])

    @classmethod
    def from_json(cls, value: str | bytes) -> "ToolSemanticRegistry":
        try:
            raw = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SemanticRuntimeError(f"invalid semantic registry JSON: {exc}") from exc
        return cls.from_dict(raw)


class SemanticRuntime:
    """Normalize calls and record typed claims without changing API clients."""

    def __init__(self, registry: ToolSemanticRegistry | None = None,
                 entities: EntityLedger | None = None,
                 claims: ClaimLedger | None = None):
        self.registry = registry or ToolSemanticRegistry()
        self.entities = entities or EntityLedger()
        self.claims = claims or ClaimLedger()
        self.observations: dict[str, dict[str, Any]] = {}

    def normalize_args(self, spec: ToolSemanticSpec, raw_args: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(raw_args, Mapping):
            raise SemanticRuntimeError("tool arguments must be an object")
        raw = dict(raw_args)
        aliases = {str(old): str(new) for old, new in spec.argument_aliases.items()}
        args: dict[str, Any] = {}
        for name, value in raw.items():
            target = name
            seen: set[str] = set()
            while target in aliases:
                if target in seen:
                    raise SemanticRuntimeError(f"argument alias cycle at {target}")
                seen.add(target)
                target = aliases[target]
            if target in args and _jsonable(args[target]) != _jsonable(value):
                raise SemanticRuntimeError(f"conflicting alias arguments for canonical field: {target}")
            args[target] = value
        for name, value in spec.defaults.items():
            args.setdefault(name, value)
        for name in spec.unordered_args:
            if name in args and isinstance(args[name], list):
                args[name] = sorted(args[name], key=lambda x: json.dumps(_jsonable(x), ensure_ascii=False, sort_keys=True))
        return _jsonable(args)

    def _resolve_entity(self, selector: Any) -> tuple[str | None, str | None]:
        """Resolve every selector form supported by :class:`EntityLedger`.

        The old runtime special-cased only ``{"entity_id": ...}``, so an
        equivalent ``id``, ``name`` or ``stable_id`` selector silently became
        a different query.  Delegating to ``resolve_detailed`` keeps all
        selector forms on one semantic path while still rejecting an explicit
        unknown entity id rather than treating it as a free-form name.
        """

        # Normalized query records and evaluator evidence commonly carry the
        # canonical ID as a plain string.  Treat a registered ID as explicit
        # identity before attempting alias resolution; otherwise replaying a
        # canonical source_call would incorrectly become unresolved.
        if isinstance(selector, str) and self.entities.get(selector) is not None:
            return str(selector), None

        explicit_entity_id = (
            isinstance(selector, Mapping) and selector.get("entity_id") is not None
        )
        if explicit_entity_id:
            entity_id = str(selector["entity_id"])
            if self.entities.get(entity_id) is None:
                raise SemanticRuntimeError(f"unknown entity_id: {entity_id}")

        resolution = self.entities.resolve_detailed(selector)
        if resolution.status == "ambiguous":
            raise AmbiguousEntityError(
                f"selector maps to {len(resolution.entity_ids)} entities: {selector!r}"
            )
        if resolution.status == "resolved":
            return str(resolution.entity_ids[0]), None

        # Preserve the original selector in the unresolved marker for audit
        # output.  ``_jsonable`` is already guaranteed by ``normalize_args``.
        marker = selector if isinstance(selector, str) else json.dumps(
            _jsonable(selector), ensure_ascii=False, sort_keys=True,
        )
        return None, str(marker)

    def normalize_call(self, tool_id: str, raw_args: Mapping[str, Any], version: str | None = None) -> QueryIR:
        spec = self.registry.get(tool_id, version)
        args = self.normalize_args(spec, raw_args)
        public_arguments = copy.deepcopy(args)
        bindings: dict[str, str] = {}
        unresolved: list[str] = []
        blocking_unresolved: list[str] = []
        for name in spec.entity_args:
            if name not in args:
                continue
            value = args[name]
            values = value if isinstance(value, list) else [value]
            resolved_values: list[Any] = []
            policy = spec.selector_policies.get(name, "strict")
            # ``structured`` and ``literal`` are aliases for a value that is
            # intentionally not an EntityLedger selector.  This is useful for
            # coordinates and other tool-level structured inputs.
            literal_policy = policy in {"literal", "structured"}
            for item in values:
                if literal_policy:
                    eid, unresolved_selector = None, None
                else:
                    eid, unresolved_selector = self._resolve_entity(item)
                if eid:
                    resolved_values.append(eid)
                else:
                    resolved_values.append(item)
                    if unresolved_selector is not None:
                        marker = f"{name}={unresolved_selector}"
                        unresolved.append(marker)
                        if policy in {"strict", "resolve"}:
                            blocking_unresolved.append(marker)
            args[name] = resolved_values if isinstance(value, list) else resolved_values[0]
            bindings[name] = json.dumps(_jsonable(args[name]), ensure_ascii=False, sort_keys=True)
        qualifiers = {name: args[name] for name in spec.qualifier_args if name in args}
        predicate_arguments = {
            str(name): args[arg_name]
            for name, arg_name in spec.predicate_args.items()
            if arg_name in args
        }
        return QueryIR(
            tool_id=spec.qualified_tool_id,
            operator=spec.operator,
            arguments=args,
            entity_bindings=bindings,
            predicates=((spec.predicate,) if spec.predicate else ()),
            predicate_arguments=predicate_arguments,
            qualifiers=qualifiers,
            projection=spec.projection,
            ordering=spec.ordering,
            cardinality=spec.cardinality,
            completeness=spec.completeness,
            unresolved_selectors=tuple(unresolved),
            blocking_unresolved_selectors=tuple(blocking_unresolved),
            public_arguments=public_arguments,
        )

    def record_observation(self, call_id: str, query: QueryIR, observation: Any,
                           claim_ids: list[str] | tuple[str, ...] = ()) -> None:
        if not call_id:
            raise SemanticRuntimeError("call_id is required")
        if call_id in self.observations:
            prior = self.observations[call_id]
            if prior["query_digest"] != query.digest:
                raise SemanticRuntimeError(f"call_id reused for a different query: {call_id}")
            incoming = {
                "query_digest": query.digest,
                "tool_id": query.tool_id,
                "observation": _jsonable(observation),
                "claim_ids": list(claim_ids),
            }
            prior_cmp = {key: prior[key] for key in incoming}
            if _jsonable(prior_cmp) != incoming:
                raise SemanticRuntimeError(f"call_id reused with different observation: {call_id}")
            return
        self.observations[call_id] = {
            "call_id": call_id,
            "query_digest": query.digest,
            "tool_id": query.tool_id,
            "observation": _jsonable(observation),
            "claim_ids": list(claim_ids),
        }

    def export(self) -> dict[str, Any]:
        return {
            "tool_specs": self.registry.to_dict(),
            "observations": copy.deepcopy(self.observations),
            "entities": self.entities.to_dict(),
            "claims": self.claims.to_dict(),
        }
