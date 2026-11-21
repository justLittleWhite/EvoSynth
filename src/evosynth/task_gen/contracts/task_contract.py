"""Task contracts for the semantic evosynth synthesis runtime.

The old open-domain pipeline treated a reference trace as the task's answer
key.  This module deliberately keeps the task specification independent from
that trace.  A :class:`TaskContract` describes bindings, the finite universe
over which a collection operation is meaningful, closed hidden choices, and a
small declarative answer predicate.  A verifier can therefore accept a
different (but semantically valid) witness without seeing the reference
answer.

The classes are intentionally dependency free.  They accept dictionaries
exported by a ClaimLedger, which lets the semantic store evolve without making
the task format depend on a particular implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
import math
import re
from typing import Any, Mapping


_MISSING = object()


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(value, key=lambda item: repr(item)))
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _as_bool(value: Any, *, field_name: str) -> bool:
    """Parse booleans without Python's surprising ``bool('false')`` rule."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    raise TypeError(f"{field_name} must be a boolean")


def _jsonable(value: Any) -> Any:
    """Return a JSON-safe copy while preserving useful scalar types."""

    if value is _MISSING:
        return None
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict())
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    # Enum-like values are common in ledgers.
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.value)
        except Exception:  # pragma: no cover - defensive for foreign enums
            pass
    return value


@dataclass(frozen=True)
class EntityBinding:
    """A named entity binding used by a task.

    ``entity_ids`` is intentionally separate from a human-readable selector.
    A selector can be retained for audit purposes, but it is never treated as
    proof of identity.  By default a binding used in an exact-answer task must
    be singleton; witness-style tasks may explicitly set ``singleton=False``.
    """

    variable: str
    entity_ids: tuple[str, ...] = ()
    selector: Mapping[str, Any] | None = None
    singleton: bool = True
    exposed: bool = True
    scope_id: str | None = None

    @classmethod
    def from_dict(cls, value: Any, variable: str | None = None) -> "EntityBinding":
        if isinstance(value, cls):
            return value
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return cls.from_dict(value.to_dict())
        if isinstance(value, str):
            return cls(variable=variable or "entity", entity_ids=(value,))
        if value is None:
            return cls(variable=variable or "entity")
        if not isinstance(value, Mapping):
            raise TypeError(f"entity binding must be a mapping, got {type(value)!r}")
        var = str(value.get("variable", value.get("name", variable or "entity")))
        raw_ids = value.get("entity_ids", value.get("ids", _MISSING))
        if raw_ids is _MISSING:
            raw_ids = value.get("entity_id", _MISSING)
        if raw_ids is _MISSING:
            # ``candidates`` is accepted as an identity list when its values
            # are strings; structured candidates remain selectors/audit data.
            candidates = value.get("candidates", ())
            if isinstance(candidates, (list, tuple)) and all(
                isinstance(x, str) for x in candidates
            ):
                raw_ids = candidates
            else:
                raw_ids = ()
        ids: list[str] = []
        for item in _as_tuple(raw_ids):
            if isinstance(item, Mapping):
                ident = item.get("entity_id", item.get("id"))
                if ident is not None:
                    ids.append(str(ident))
            elif item is not None:
                ids.append(str(item))
        singleton = value.get("singleton", value.get("require_singleton", True))
        return cls(
            variable=var,
            entity_ids=tuple(dict.fromkeys(ids)),
            selector=copy.deepcopy(value.get("selector")),
            singleton=_as_bool(singleton, field_name="binding.singleton"),
            exposed=_as_bool(value.get("exposed", True), field_name="binding.exposed"),
            scope_id=value.get("scope_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "variable": self.variable,
            "entity_ids": list(self.entity_ids),
            "singleton": self.singleton,
            "exposed": self.exposed,
        }
        if self.selector is not None:
            result["selector"] = _jsonable(self.selector)
        if self.scope_id is not None:
            result["scope_id"] = self.scope_id
        return result

    @property
    def cardinality(self) -> int:
        return len(self.entity_ids)


@dataclass(frozen=True)
class CandidateScope:
    """The universe available to a finite/exhaustive answer predicate."""

    entity_ids: tuple[str, ...] = ()
    complete: bool = False
    finite: bool | None = None
    scope_id: str | None = None
    description: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "CandidateScope | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, (list, tuple, set, frozenset)):
            value = {"entity_ids": list(value), "complete": False}
        if not isinstance(value, Mapping):
            raise TypeError(f"candidate scope must be a mapping, got {type(value)!r}")
        raw_ids = value.get(
            "entity_ids",
            value.get("ids", value.get("candidate_ids", value.get("entities", value.get("candidates", ())))),
        )
        ids: list[str] = []
        for item in _as_tuple(raw_ids):
            if isinstance(item, Mapping):
                ident = item.get("entity_id", item.get("id"))
                if ident is not None:
                    ids.append(str(ident))
            elif item is not None:
                ids.append(str(item))
        finite_raw = value.get("finite", value.get("is_finite"))
        finite = (_as_bool(finite_raw, field_name="candidate_scope.finite")
                  if finite_raw is not None else bool(ids))
        return cls(
            entity_ids=tuple(dict.fromkeys(ids)),
            complete=_as_bool(
                value.get("complete", value.get("exhaustive", value.get("is_complete", False))),
                field_name="candidate_scope.complete",
            ),
            finite=finite,
            scope_id=value.get("scope_id", value.get("id")),
            description=value.get("description"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "entity_ids": list(self.entity_ids),
            "complete": self.complete,
            "finite": self.finite,
        }
        if self.scope_id is not None:
            result["scope_id"] = self.scope_id
        if self.description is not None:
            result["description"] = self.description
        return result

    @property
    def is_finite_complete(self) -> bool:
        return bool(self.finite) and self.complete


@dataclass(frozen=True)
class HiddenChoice:
    """A latent generation choice that may affect claims or the answer.

    A hidden choice is safe only when its domain and selected value are closed
    in the task contract.  ``closed`` is explicit rather than inferred from a
    reference trajectory: inference must fail closed if a generator forgot to
    expose or constrain a choice.
    """

    choice_id: str
    options: tuple[Any, ...] = ()
    selected: Any = _MISSING
    closed: bool = False
    exposed: bool = False
    affects: tuple[str, ...] = ()
    resolution: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "HiddenChoice":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"hidden choice must be a mapping, got {type(value)!r}")
        selected = value["selected"] if "selected" in value else _MISSING
        options = value.get("options", value.get("domain", value.get("candidates", ())))
        affects = value.get("affects", value.get("dependencies", ()))
        return cls(
            choice_id=str(value.get("choice_id", value.get("id", value.get("name", "")))),
            options=_as_tuple(copy.deepcopy(options)),
            selected=(copy.deepcopy(selected) if selected is not _MISSING else _MISSING),
            closed=_as_bool(value.get("closed", value.get("resolved", False)), field_name="hidden_choice.closed"),
            exposed=_as_bool(value.get("exposed", False), field_name="hidden_choice.exposed"),
            affects=tuple(str(x) for x in _as_tuple(affects)),
            resolution=value.get("resolution"),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "choice_id": self.choice_id,
            "options": _jsonable(self.options),
            "closed": self.closed,
            "exposed": self.exposed,
            "affects": list(self.affects),
        }
        if self.selected is not _MISSING:
            result["selected"] = _jsonable(self.selected)
        if self.resolution is not None:
            result["resolution"] = self.resolution
        return result


@dataclass(frozen=True)
class ClaimProvenance:
    """A pointer from a typed claim to one concrete observation."""

    call_id: str | None = None
    source_call: Any = None
    output_pointer: str | None = None
    observation_hash: str | None = None
    world_id: str | None = None

    @classmethod
    def from_dict(cls, value: Any) -> "ClaimProvenance | None":
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return cls.from_dict(value.to_dict())
        # ClaimLedger exports may retain more than one observation.  A
        # CertificateClaim has one primary provenance slot; callers that need
        # alternatives can still pass one claim per provenance.  Selecting the
        # first entry here keeps compact certificates backwards compatible.
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            return cls.from_dict(value[0])
        if not isinstance(value, Mapping):
            # A bare call identifier is useful in compact ledgers.
            return cls(call_id=str(value))
        return cls(
            call_id=(str(value["call_id"]) if value.get("call_id") is not None else None),
            source_call=copy.deepcopy(value.get("source_call", value.get("call"))),
            output_pointer=value.get("output_pointer", value.get("pointer", value.get("json_pointer"))),
            observation_hash=value.get("observation_hash", value.get("evidence_hash")),
            world_id=(str(value["world_id"]) if value.get("world_id") is not None else None),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.call_id is not None:
            result["call_id"] = self.call_id
        if self.source_call is not None:
            result["source_call"] = _jsonable(self.source_call)
        if self.output_pointer is not None:
            result["output_pointer"] = self.output_pointer
        if self.observation_hash is not None:
            result["observation_hash"] = self.observation_hash
        if self.world_id is not None:
            result["world_id"] = self.world_id
        return result

    @property
    def is_bound(self) -> bool:
        return bool(self.call_id or self.source_call) and self.output_pointer is not None


@dataclass(frozen=True)
class CertificateClaim:
    """A typed claim required by a task certificate.

    ``value`` is a crystallized world value, not an answer key.  The verifier
    uses it only to check that the candidate's claim and observation agree with
    the contract's provenance.  The answer predicate is evaluated against the
    candidate claims, allowing alternative valid answers.
    """

    claim_id: str
    value: Any = _MISSING
    provenance: ClaimProvenance | None = None
    subject: Any = None
    predicate: str | None = None
    qualifiers: Mapping[str, Any] | None = None
    unit: str | None = None
    tolerance: float | None = None
    value_type: str | None = None
    precision: Any = None
    uncertainty: Any = None
    world_id: str | None = None
    required: bool = True
    # ``exact`` pins one concrete reference observation. ``semantic`` accepts
    # any candidate observation that proves the same typed claim in the same
    # world through the declared tool, allowing equivalent call shapes.
    provenance_policy: str = "exact"

    @classmethod
    def from_dict(cls, value: Any, claim_id: str | None = None) -> "CertificateClaim":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return cls(claim_id=str(claim_id or ""), value=copy.deepcopy(value))
        ident = str(value.get("claim_id", value.get("id", claim_id or "")))
        raw_prov = value.get("provenance")
        if raw_prov is None and any(k in value for k in ("call_id", "source_call", "output_pointer", "pointer")):
            raw_prov = {
                k: value[k]
                for k in ("call_id", "source_call", "output_pointer", "pointer", "observation_hash", "world_id")
                if k in value
            }
        return cls(
            claim_id=ident,
            value=copy.deepcopy(value["value"]) if "value" in value else _MISSING,
            provenance=ClaimProvenance.from_dict(raw_prov),
            subject=copy.deepcopy(value.get("subject")) if value.get("subject") is not None else None,
            predicate=(str(value["predicate"]) if value.get("predicate") is not None else None),
            qualifiers=copy.deepcopy(value.get("qualifiers", value.get("context"))),
            unit=value.get("unit"),
            tolerance=(float(value["tolerance"]) if value.get("tolerance") is not None else None),
            value_type=(str(value["value_type"]) if value.get("value_type") is not None
                        else (str(value["type"]) if value.get("type") is not None else None)),
            precision=copy.deepcopy(value.get("precision")),
            uncertainty=copy.deepcopy(value.get("uncertainty")),
            world_id=(str(value["world_id"]) if value.get("world_id") is not None else None),
            required=_as_bool(value.get("required", True), field_name="certificate_claim.required"),
            provenance_policy=str(
                value.get("provenance_policy", value.get("provenance_mode", "exact"))
            ).lower(),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"claim_id": self.claim_id, "required": self.required}
        if self.value is not _MISSING:
            result["value"] = _jsonable(self.value)
        if self.provenance is not None:
            result["provenance"] = self.provenance.to_dict()
        if self.subject is not None:
            result["subject"] = _jsonable(self.subject)
        if self.predicate is not None:
            result["predicate"] = self.predicate
        if self.qualifiers is not None:
            result["qualifiers"] = _jsonable(self.qualifiers)
        if self.unit is not None:
            result["unit"] = self.unit
        if self.tolerance is not None:
            result["tolerance"] = self.tolerance
        if self.value_type is not None:
            result["value_type"] = self.value_type
        if self.precision is not None:
            result["precision"] = _jsonable(self.precision)
        if self.uncertainty is not None:
            result["uncertainty"] = _jsonable(self.uncertainty)
        if self.world_id is not None:
            result["world_id"] = self.world_id
        if self.provenance_policy != "exact":
            result["provenance_policy"] = self.provenance_policy
        return result


@dataclass(frozen=True)
class Certificate:
    claims: tuple[CertificateClaim, ...] = ()
    version: str = "1"
    # Additional machine-checkable metadata is retained, but never interpreted
    # as a hidden expected answer by the verifier.
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Any) -> "Certificate":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, (list, tuple)):
            raw_claims = value
            metadata: Mapping[str, Any] = {}
            version = "1"
        elif isinstance(value, Mapping):
            raw_claims = value.get("claims", value.get("facts", ()))
            version = str(value.get("version", "1"))
            metadata = {
                str(k): copy.deepcopy(v)
                for k, v in value.items()
                if k not in {"claims", "facts", "version"}
            }
            # Compact form: {claim_id: {value, provenance}}.
            if isinstance(raw_claims, Mapping):
                raw_claims = [dict(v, claim_id=k) if isinstance(v, Mapping) else {"claim_id": k, "value": v}
                              for k, v in raw_claims.items()]
        else:
            raise TypeError(f"certificate must be a mapping/list, got {type(value)!r}")
        claims = tuple(CertificateClaim.from_dict(item) for item in raw_claims or ())
        return cls(claims=claims, version=version, metadata=metadata)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "version": self.version,
            "claims": [claim.to_dict() for claim in self.claims],
        }
        metadata = _jsonable(self.metadata)
        if isinstance(metadata, Mapping):
            collisions = set(result).intersection(metadata)
            if collisions:
                raise ValueError(f"certificate metadata collides with structural fields: {sorted(collisions)}")
            result.update(metadata)
        return result

    def by_id(self) -> dict[str, CertificateClaim]:
        return {claim.claim_id: claim for claim in self.claims}


@dataclass(frozen=True, init=False)
class AnswerPredicate:
    """Small declarative DSL used to evaluate candidate answers.

    The custom initializer accepts either ``AnswerPredicate("equals", {...})``
    or convenient keyword arguments such as
    ``AnswerPredicate("equals", left=..., right=...)``.  Nested mappings are
    kept opaque until evaluation so unknown operations can safely produce
    ``INDETERMINATE`` rather than being guessed.
    """

    op: str
    args: Mapping[str, Any]

    def __init__(self, op: str, args: Mapping[str, Any] | None = None, **kwargs: Any) -> None:
        merged: dict[str, Any] = {}
        if args is not None:
            if not isinstance(args, Mapping):
                raise TypeError("AnswerPredicate args must be a mapping")
            merged.update(copy.deepcopy(dict(args)))
        merged.update(copy.deepcopy(kwargs))
        object.__setattr__(self, "op", str(op).strip().lower())
        object.__setattr__(self, "args", merged)

    @classmethod
    def from_dict(cls, value: Any) -> "AnswerPredicate":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            op = value.get("op", value.get("kind", value.get("type")))
            if op is None:
                # A mapping without an operator is an explicit literal object.
                return cls("literal", {"value": copy.deepcopy(dict(value))})
            args = {str(k): copy.deepcopy(v) for k, v in value.items() if k not in {"op", "kind", "type"}}
            return cls(str(op), args)
        return cls("literal", {"value": copy.deepcopy(value)})

    def to_dict(self) -> dict[str, Any]:
        result = dict(_jsonable(self.args))
        # Nested caller data must not be able to replace the actual operator.
        result["op"] = self.op
        return result

    def referenced_claim_ids(self) -> set[str]:
        found: set[str] = set()

        def walk(value: Any) -> None:
            if isinstance(value, AnswerPredicate):
                walk(value.to_dict())
            elif isinstance(value, Mapping):
                op = str(value.get("op", value.get("kind", ""))).lower()
                if op in {"claim", "fact", "claim_value"}:
                    ident = value.get("claim_id", value.get("id", value.get("claim")))
                    if isinstance(ident, str) and not ident.startswith("$"):
                        found.add(ident)
                for item in value.values():
                    walk(item)
            elif isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    walk(item)

        # Include this predicate's own operator. Walking only ``args`` misses
        # a top-level claim or set predicate.
        walk(self.to_dict())
        return found

    def requires_complete_scope(self) -> bool:
        """Whether this predicate claims to enumerate a complete collection."""

        exhaustive_ops = {"all_members", "all_candidates", "exhaustive", "exact_set"}

        def walk(value: Any) -> bool:
            if isinstance(value, AnswerPredicate):
                return walk(value.to_dict())
            if not isinstance(value, Mapping):
                if isinstance(value, (list, tuple, set, frozenset)):
                    return any(walk(x) for x in value)
                return False
            op = str(value.get("op", value.get("kind", ""))).lower()
            if op in exhaustive_ops:
                return True
            if op == "set_predicate":
                # A plain set_predicate is a witness/subset predicate.  An
                # exhaustive task must opt in with mode=all/exact or
                # require_complete=true; otherwise an open universe would be
                # silently treated as complete.
                mode = str(value.get("mode", value.get("quantifier", "subset"))).lower()
                # ``exact`` and ``all`` require a bounded complete universe;
                # subset/any are witness predicates and do not.
                if bool(value.get("require_complete", False)) or mode in {"all", "exact", "equals", "exhaustive"}:
                    return True
            return any(walk(x) for x in value.values())

        return walk(self.to_dict())


@dataclass(frozen=True)
class TaskContract:
    """Complete, serializable contract for one evosynth task."""

    task_id: str
    answer_predicate: AnswerPredicate
    bindings: Mapping[str, EntityBinding] = field(default_factory=dict)
    candidate_scope: CandidateScope | None = None
    hidden_choices: tuple[HiddenChoice, ...] = ()
    certificate: Certificate = field(default_factory=Certificate)
    require_singleton_bindings: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = "2"

    @classmethod
    def from_dict(cls, value: Any) -> "TaskContract":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"task contract must be a mapping, got {type(value)!r}")
        raw_bindings = value.get("bindings", value.get("entity_bindings", {})) or {}
        bindings: dict[str, EntityBinding] = {}
        if isinstance(raw_bindings, Mapping):
            for name, binding in raw_bindings.items():
                item = EntityBinding.from_dict(binding, variable=str(name))
                bindings[item.variable] = item
        else:
            for binding in raw_bindings:
                item = EntityBinding.from_dict(binding)
                bindings[item.variable] = item
        raw_hidden = value.get("hidden_choices", value.get("hidden_choice_closure", ())) or ()
        if isinstance(raw_hidden, Mapping):
            raw_hidden = [dict(item, choice_id=name) if isinstance(item, Mapping) else {"choice_id": name, "selected": item}
                          for name, item in raw_hidden.items()]
        hidden = tuple(HiddenChoice.from_dict(item) for item in raw_hidden)
        metadata = {
            str(k): copy.deepcopy(v)
            for k, v in value.items()
            if k not in {
                "task_id", "id", "answer_predicate", "predicate", "bindings", "entity_bindings",
                "candidate_scope", "candidate_universe", "universe", "scope", "hidden_choices", "hidden_choice_closure",
                "certificate", "require_singleton_bindings", "singleton_bindings", "version",
            }
        }
        predicate_raw = value.get("answer_predicate", value.get("predicate", _MISSING))
        return cls(
            task_id=str(value.get("task_id", value.get("id", ""))),
            answer_predicate=(
                AnswerPredicate("missing", {})
                if predicate_raw is _MISSING
                else AnswerPredicate.from_dict(predicate_raw)
            ),
            bindings=bindings,
            candidate_scope=CandidateScope.from_dict(
                value.get("candidate_scope", value.get("candidate_universe", value.get("scope", value.get("universe"))))
            ),
            hidden_choices=hidden,
            certificate=Certificate.from_dict(value.get("certificate", value.get("claims"))),
            require_singleton_bindings=_as_bool(
                value.get("require_singleton_bindings", value.get("singleton_bindings", True)),
                field_name="task.require_singleton_bindings",
            ),
            metadata=metadata,
            version=str(value.get("version", "2")),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": self.task_id,
            "answer_predicate": self.answer_predicate.to_dict(),
        }
        if self.version != "2":
            result["version"] = self.version
        if self.bindings:
            result["bindings"] = {
                name: binding.to_dict() for name, binding in self.bindings.items()
            }
        if self.hidden_choices:
            result["hidden_choices"] = [choice.to_dict() for choice in self.hidden_choices]
        if self.certificate.claims or self.certificate.metadata:
            result["certificate"] = self.certificate.to_dict()
        if not self.require_singleton_bindings:
            result["require_singleton_bindings"] = self.require_singleton_bindings
        if self.candidate_scope is not None:
            result["candidate_scope"] = self.candidate_scope.to_dict()
        metadata = _jsonable(self.metadata)
        if isinstance(metadata, Mapping):
            collisions = set(result).intersection(metadata)
            if collisions:
                raise ValueError(f"task metadata collides with structural fields: {sorted(collisions)}")
            result.update(metadata)
        return result

    def to_json(self, **kwargs: Any) -> str:
        kwargs.setdefault("sort_keys", True)
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_json(cls, value: str | bytes) -> "TaskContract":
        return cls.from_dict(json.loads(value))

    def hidden_choice_closure(self) -> list[str]:
        """Return machine-readable closure violations (empty means closed)."""

        errors: list[str] = []
        seen: set[str] = set()
        for choice in self.hidden_choices:
            ident = choice.choice_id
            if not ident:
                errors.append("hidden_choice_missing_id")
                continue
            if ident in seen:
                errors.append(f"hidden_choice_duplicate:{ident}")
            seen.add(ident)
            if not choice.closed:
                errors.append(f"hidden_choice_unclosed:{ident}")
            if choice.selected is _MISSING:
                errors.append(f"hidden_choice_unbounded:{ident}")
            if choice.selected is not _MISSING and choice.options and not any(
                _safe_equal(choice.selected, option) for option in choice.options
            ):
                errors.append(f"hidden_choice_selection_outside_domain:{ident}")
        return errors

    def requires_complete_scope(self) -> bool:
        return self.answer_predicate.requires_complete_scope()

    def validate(self) -> list[str]:
        """Validate proof obligations without consulting a reference answer.

        The return value is a list to keep the API convenient for generators;
        callers that need a boolean can use ``not contract.validate()``.
        """

        errors: list[str] = []
        if not self.task_id:
            errors.append("task_id_missing")
        if not self.answer_predicate.op:
            errors.append("answer_predicate_missing")
        elif self.answer_predicate.op == "missing":
            errors.append("answer_predicate_missing")
        known_ops = {
            "answer", "candidate", "candidate_answer", "literal", "const", "constant", "claim", "fact",
            "claim_value", "var", "variable", "ref", "item", "candidate_item", "element",
            "equals", "eq", "not_equals", "ne", "within", "between", "contains", "member_of", "in",
            "set_predicate", "all_members", "all_candidates", "exact_set", "exhaustive", "and", "all_of",
            "or", "any_of", "not", "derived", "arithmetic", "field", "get", "length", "count", "sum",
            "min", "max", "abs", "round", "add", "sub", "mul", "div", "gt", "gte", "lt", "lte",
            "ge", "le",
        }

        def check_ops(value: Any) -> None:
            if isinstance(value, AnswerPredicate):
                op = value.op
                if op not in known_ops:
                    errors.append(f"answer_predicate_unknown_op:{op}")
                check_ops(value.args)
            elif isinstance(value, Mapping):
                if "op" in value or "kind" in value:
                    op = str(value.get("op", value.get("kind", ""))).lower()
                    if op not in known_ops:
                        errors.append(f"answer_predicate_unknown_op:{op}")
                    if op == "set_predicate":
                        if value.get("allowed", value.get("allowed_values", value.get("members"))) is None \
                                and value.get("element_predicate", value.get("predicate", value.get("condition"))) is None:
                            errors.append("set_predicate_condition_missing")
                for item in value.values():
                    check_ops(item)
            elif isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    check_ops(item)

        check_ops(self.answer_predicate)

        for name, binding in self.bindings.items():
            if not name or not binding.variable:
                errors.append("binding_missing_variable")
            if self.require_singleton_bindings and binding.singleton and binding.cardinality != 1:
                kind = "missing" if binding.cardinality == 0 else "ambiguous"
                errors.append(f"binding_{kind}:{binding.variable}")
            if self.candidate_scope and binding.entity_ids and self.candidate_scope.entity_ids:
                outside = set(binding.entity_ids) - set(self.candidate_scope.entity_ids)
                if outside:
                    errors.append(f"binding_outside_candidate_scope:{binding.variable}")

        if self.requires_complete_scope():
            scope = self.candidate_scope
            if scope is None:
                errors.append("candidate_scope_missing_for_exhaustive_predicate")
            elif not scope.is_finite_complete:
                errors.append("candidate_scope_not_finite_and_complete")
            else:
                scope_ids = set(scope.entity_ids)

                def check_universes(value: Any) -> None:
                    if isinstance(value, Mapping):
                        op = str(value.get("op", value.get("kind", ""))).lower()
                        universe = value.get("universe")
                        exhaustive = {"set_predicate", "all_members", "all_candidates", "exact_set", "exhaustive"}
                        if op in exhaustive and universe is not None and isinstance(
                            universe, (list, tuple, set, frozenset)
                        ) and {str(item) for item in universe} != scope_ids:
                            errors.append("predicate_universe_differs_from_candidate_scope")
                        for item in value.values():
                            check_universes(item)
                    elif isinstance(value, (list, tuple, set, frozenset)):
                        for item in value:
                            check_universes(item)

                check_universes(self.answer_predicate.to_dict())

        errors.extend(self.hidden_choice_closure())

        cert_ids: set[str] = set()
        for claim in self.certificate.claims:
            if not claim.claim_id:
                errors.append("certificate_claim_missing_id")
            elif claim.claim_id in cert_ids:
                errors.append(f"certificate_claim_duplicate:{claim.claim_id}")
            cert_ids.add(claim.claim_id)
            if claim.provenance_policy not in {"exact", "semantic"}:
                errors.append(
                    f"certificate_provenance_policy_invalid:{claim.claim_id}:"
                    f"{claim.provenance_policy}"
                )
            elif claim.required and claim.provenance is None:
                errors.append(f"certificate_provenance_missing:{claim.claim_id}")
            elif claim.required and claim.provenance is not None:
                if claim.provenance_policy == "exact" and not claim.provenance.is_bound:
                    errors.append(f"certificate_provenance_unbound:{claim.claim_id}")
                elif claim.provenance_policy == "semantic":
                    source = claim.provenance.source_call
                    tool = source.get("tool", source.get("name")) if isinstance(source, Mapping) else None
                    if not isinstance(tool, str) or not tool.strip():
                        errors.append(
                            f"certificate_semantic_provenance_tool_missing:{claim.claim_id}"
                        )
            if claim.required and claim.value is _MISSING:
                errors.append(f"certificate_value_missing:{claim.claim_id}")
            if claim.tolerance is not None and (
                not math.isfinite(claim.tolerance) or claim.tolerance < 0
            ):
                errors.append(f"certificate_tolerance_invalid:{claim.claim_id}")
            if claim.world_id is not None and (
                not isinstance(claim.world_id, str) or not claim.world_id.strip()
            ):
                errors.append(f"certificate_world_id_invalid:{claim.claim_id}")
            if claim.provenance is not None:
                digest = claim.provenance.observation_hash
                if digest is not None and not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                    errors.append(f"certificate_observation_hash_invalid:{claim.claim_id}")
                if claim.world_id is not None and claim.provenance.world_id is not None \
                        and claim.world_id != claim.provenance.world_id:
                    errors.append(f"certificate_world_id_conflict:{claim.claim_id}")

        referenced = self.answer_predicate.referenced_claim_ids()
        missing_cert = referenced - cert_ids
        for claim_id in sorted(missing_cert):
            # A claim can be supplied by a live ledger, but a publishable task
            # still needs a provenance-bearing certificate entry.
            errors.append(f"predicate_claim_not_certified:{claim_id}")
        return list(dict.fromkeys(errors))

    def validate_or_raise(self) -> None:
        errors = self.validate()
        if errors:
            raise ValueError("invalid task contract: " + "; ".join(errors))

    def referenced_claim_ids(self) -> set[str]:
        return self.answer_predicate.referenced_claim_ids()


def _safe_equal(left: Any, right: Any) -> bool:
    try:
        # JSON round-tripping makes mappings with non-string keys comparable
        # while avoiding surprising object identity checks.
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return _jsonable(left) == _jsonable(right)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return list(left) == list(right)
        return bool(left == right)
    except Exception:
        return repr(left) == repr(right)


__all__ = [
    "AnswerPredicate",
    "CandidateScope",
    "Certificate",
    "CertificateClaim",
    "ClaimProvenance",
    "EntityBinding",
    "HiddenChoice",
    "TaskContract",
]
