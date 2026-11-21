"""Entity and typed-claim ledgers for the semantic evosynth runtime.

The legacy evosynth implementation used a digest of raw tool arguments as a
fact identity.  This module deliberately puts the semantic identity one layer
below a tool call:

* :class:`EntityLedger` stores canonical entity identifiers and aliases.  An
  alias may point to several entities; it is never used to silently merge
  homonyms.
* :class:`ClaimLedger` stores typed, qualified claims.  Provenance points to a
  concrete call and output pointer, while the claim key is independent of the
  presentation chosen by a tool.

Both ledgers are deterministic and JSON serialisable.  Writes are optimistic
transactions: a transaction started at revision ``r`` fails if another writer
has committed revision ``r + 1`` before it commits.  A conflicting value for
the same semantic claim key is rejected by default rather than allowing a
second parallel world to be created.

No domain or tool-specific normalisation is performed here.  In particular,
coordinates and geometry are retained exactly as supplied; callers that need
a measurement tolerance must encode it as claim metadata instead of relying
on implicit rounding in an identity function.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import unicodedata
from typing import Any, Iterator, Literal

from evosynth.task_gen.common.json_utils import canonical_json as _encode_canonical_json


SCHEMA_VERSION = "semantic_store"


class SemanticStoreError(ValueError):
    """Base class for invalid semantic-store operations."""


class EntityConflictError(SemanticStoreError):
    """Raised when an entity id/key is reused with incompatible identity."""

    def __init__(self, message: str, *, entity_id: str | None = None,
                 existing: Any = None, incoming: Any = None):
        super().__init__(message)
        self.entity_id = entity_id
        self.existing = existing
        self.incoming = incoming


class EntityNotFoundError(SemanticStoreError):
    """Raised by strict entity resolution when no candidate exists."""


class AmbiguousEntityError(SemanticStoreError):
    """Raised by strict entity resolution when multiple candidates exist."""

    def __init__(self, message: str, *, query: Any = None,
                 candidates: Iterable[str] = ()):
        super().__init__(message)
        self.query = query
        self.candidates = tuple(candidates)


class ClaimConflictError(SemanticStoreError):
    """Raised when one semantic claim key receives incompatible values."""

    def __init__(self, message: str, *, key: str | None = None,
                 existing: Any = None, incoming: Any = None):
        super().__init__(message)
        self.key = key
        self.existing = existing
        self.incoming = incoming


class TransactionConflictError(SemanticStoreError):
    """Raised when an optimistic transaction observes a changed revision."""


class SerializationError(SemanticStoreError):
    """Raised for malformed or unsupported ledger serialisation."""


_MISSING = object()
_WS_RE = re.compile(r"\s+")


def _normalise_text(value: Any) -> str:
    """Conservatively normalise a selector/token without fuzzy matching."""

    return _WS_RE.sub(" ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def normalize_alias(value: Any) -> str:
    """Return the index form used for aliases.

    Only Unicode compatibility, case, and whitespace are normalised.  Punctuation,
    numbers, and coordinates are intentionally preserved to avoid near-miss
    entity merges.
    """

    if value is None:
        return ""
    return _normalise_text(value)


def _json_ready(value: Any) -> Any:
    """Convert supported values to a stable, JSON-compatible representation."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SerializationError("NaN and infinity are not valid claim values")
        # Do not round.  Python's JSON encoder preserves the supplied float's
        # shortest representation, which is enough to keep exact coordinates
        # distinct while remaining deterministic across ledger instances.
        return value
    if isinstance(value, Decimal):
        return {"__decimal__": format(value, "f")}
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, date):
        return {"__date__": value.isoformat()}
    if isinstance(value, bytes):
        # Bytes are not expected in tool claims, but a tagged form keeps the
        # serializer explicit instead of falling back to repr(address).
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_ready(item) for item in value]
        return sorted(items, key=lambda item: _canonical_json(item))
    # Dataclasses and foreign value objects may explicitly expose a JSON form;
    # do not use repr because it commonly contains process-specific addresses.
    method = getattr(value, "to_dict", None)
    if callable(method):
        converted = method()
        if converted is not value:
            return _json_ready(converted)
    raise SerializationError(f"unsupported value type: {type(value)!r}")


def _restore_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_value(item) for item in value]
    if isinstance(value, Mapping):
        if set(value) == {"__decimal__"}:
            return Decimal(str(value["__decimal__"]))
        if set(value) == {"__datetime__"}:
            return datetime.fromisoformat(str(value["__datetime__"]))
        if set(value) == {"__date__"}:
            return date.fromisoformat(str(value["__date__"]))
        if set(value) == {"__bytes_hex__"}:
            return bytes.fromhex(str(value["__bytes_hex__"]))
        return {str(key): _restore_value(item) for key, item in value.items()}
    return value


def _canonical_json(value: Any) -> str:
    return _encode_canonical_json(_json_ready(value))


def canonical_json(value: Any) -> str:
    """Public canonical JSON helper used by manifests and tests."""

    return _canonical_json(value)


def deterministic_id(prefix: str, value: Any, *, length: int = 32) -> str:
    """Create an order-independent identifier from semantic material."""

    if not prefix or not re.fullmatch(r"[a-z][a-z0-9_]*", prefix):
        raise SemanticStoreError(f"invalid id prefix: {prefix!r}")
    digest = hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:length]}"


def _copy(value: Any) -> Any:
    """Copy values without ever handing callers an internal mutable object."""

    return copy.deepcopy(value)


def _normalise_predicate(value: Any) -> str:
    result = _normalise_text(value)
    if not result:
        raise SemanticStoreError("claim predicate cannot be empty")
    return result


def _coerce_provenance(value: Any) -> "Provenance":
    if isinstance(value, Provenance):
        return value
    if isinstance(value, Mapping):
        return Provenance.from_dict(value)
    if value is None:
        return Provenance()
    return Provenance(call_id=str(value))


def _coerce_provenance_list(value: Any) -> tuple["Provenance", ...]:
    if value is None:
        return ()
    if isinstance(value, (Provenance, Mapping, str)):
        return (_coerce_provenance(value),)
    return tuple(_coerce_provenance(item) for item in value)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Evidence binding for a claim.

    ``call_id`` and ``output_pointer`` are kept separately so a verifier can
    mechanically locate the exact field that supplied a claim.  ``source_call``
    may contain a structured tool call; it is audit metadata, not an identity
    key.
    """

    call_id: str | None = None
    output_pointer: str | None = None
    source_call: Any = None
    observation_hash: str | None = None
    world_id: str | None = None
    tool_id: str | None = None
    observed_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Provenance is part of the evidence boundary.  Copy nested payloads
        # here so mutating a caller-owned source-call dictionary cannot rewrite
        # an already materialized claim.
        if self.source_call is not None:
            object.__setattr__(self, "source_call", _copy(self.source_call))
        object.__setattr__(self, "metadata", _copy(dict(self.metadata or {})))
        if self.call_id is not None:
            object.__setattr__(self, "call_id", str(self.call_id))
        if self.output_pointer is not None:
            object.__setattr__(self, "output_pointer", str(self.output_pointer))

    @classmethod
    def from_dict(cls, value: Any) -> "Provenance":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            return cls(call_id=str(value))
        source_call = value.get("source_call", value.get("call"))
        return cls(
            call_id=(str(value["call_id"]) if value.get("call_id") is not None else None),
            output_pointer=value.get("output_pointer", value.get("pointer", value.get("json_pointer"))),
            source_call=_copy(source_call),
            observation_hash=value.get("observation_hash", value.get("evidence_hash")),
            world_id=value.get("world_id"),
            tool_id=value.get("tool_id"),
            observed_at=value.get("observed_at", value.get("timestamp")),
            metadata=_copy(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in ("call_id", "output_pointer", "source_call", "observation_hash",
                     "world_id", "tool_id", "observed_at"):
            value = getattr(self, name)
            if value is not None:
                result[name] = _json_ready(value)
        if self.metadata:
            result["metadata"] = _json_ready(self.metadata)
        return result

    @property
    def is_bound(self) -> bool:
        return bool(self.call_id or self.source_call) and bool(self.output_pointer)

    @property
    def canonical_key(self) -> str:
        return _canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class EntityRecord:
    entity_id: str
    entity_type: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    stable_ids: Mapping[str, str] = field(default_factory=dict)
    identity: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    provenance: tuple[Provenance, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.entity_id).strip():
            raise SemanticStoreError("entity_id cannot be empty")
        if not str(self.entity_type).strip():
            raise SemanticStoreError("entity_type cannot be empty")
        if not str(self.canonical_name).strip():
            raise SemanticStoreError("canonical_name cannot be empty")
        object.__setattr__(self, "entity_id", str(self.entity_id))
        object.__setattr__(self, "entity_type", str(self.entity_type))
        object.__setattr__(self, "canonical_name", str(self.canonical_name))
        aliases = tuple(dict.fromkeys(str(item) for item in self.aliases if str(item).strip()))
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "stable_ids", _copy(dict(self.stable_ids or {})))
        object.__setattr__(self, "identity", _copy(dict(self.identity or {})))
        object.__setattr__(self, "attributes", _copy(dict(self.attributes or {})))
        object.__setattr__(self, "provenance", _coerce_provenance_list(self.provenance))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EntityRecord":
        return cls(
            entity_id=str(value["entity_id"]),
            entity_type=str(value.get("entity_type", value.get("type", "entity"))),
            canonical_name=str(value.get("canonical_name", value.get("canonical_label", value.get("name", "")))),
            aliases=tuple(value.get("aliases", ())),
            stable_ids=dict(value.get("stable_ids", value.get("external_ids", {})) or {}),
            identity=_restore_value(value.get("identity", {})),
            attributes=_restore_value(value.get("attributes", {})),
            provenance=_coerce_provenance_list(value.get("provenance")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name,
            "aliases": list(self.aliases),
            "stable_ids": _json_ready(self.stable_ids),
            "identity": _json_ready(self.identity),
            "attributes": _json_ready(self.attributes),
            "provenance": [item.to_dict() for item in self.provenance],
        }


def _clone_entity_record(record: EntityRecord) -> EntityRecord:
    """Return a detached record for public API boundaries."""

    return EntityRecord.from_dict(record.to_dict())


@dataclass(frozen=True, slots=True)
class EntityResolution:
    query: Any
    normalized_query: str
    status: Literal["resolved", "ambiguous", "not_found"]
    entity_ids: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def entity_id(self) -> str | None:
        return self.entity_ids[0] if self.status == "resolved" and self.entity_ids else None

    @property
    def is_ambiguous(self) -> bool:
        return self.status == "ambiguous"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": _json_ready(self.query),
            "normalized_query": self.normalized_query,
            "status": self.status,
            "entity_ids": list(self.entity_ids),
            "reason": self.reason,
        }


class EntityLedger:
    """Canonical entity catalog with conservative alias resolution."""

    def __init__(self, records: Iterable[EntityRecord | Mapping[str, Any]] | None = None,
                 *, revision: int = 0):
        self._entities: dict[str, EntityRecord] = {}
        self._aliases: dict[str, set[str]] = {}
        self._stable_index: dict[tuple[str, str], set[str]] = {}
        self._identity_index: dict[str, set[str]] = {}
        self._alias_provenance: dict[str, set[str]] = {}
        self._revision = int(revision)
        self._lock = threading.RLock()
        for raw in records or ():
            record = raw if isinstance(raw, EntityRecord) else EntityRecord.from_dict(raw)
            self._insert_record(record, aliases=record.aliases, bump=False)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def _identity_material(self, entity_type: str, canonical_name: str,
                           stable_ids: Mapping[str, Any], identity: Mapping[str, Any] | None) -> dict[str, Any]:
        # A caller-provided identity/stable id is the only basis for merging
        # records whose display names differ.  Without one, the canonical name
        # itself remains the deterministic declaration key.
        material: dict[str, Any] = {"entity_type": _normalise_text(entity_type)}
        if stable_ids:
            material["stable_ids"] = {str(k): str(v) for k, v in sorted(stable_ids.items(), key=lambda item: str(item[0]))}
        if identity:
            material["identity"] = _json_ready(identity)
        elif not stable_ids:
            material["canonical_name"] = _normalise_text(canonical_name)
        return material

    def _insert_record(self, record: EntityRecord, *, aliases: Iterable[str] = (), bump: bool = True) -> EntityRecord:
        # Never retain a caller-owned dataclass instance.  The frozen marker on
        # EntityRecord only protects attribute assignment, not nested mappings.
        record = _clone_entity_record(record)
        existing = self._entities.get(record.entity_id)
        before = existing.to_dict() if existing is not None else None
        incoming_name = record.canonical_name
        if existing is not None:
            if _normalise_text(existing.entity_type) != _normalise_text(record.entity_type):
                raise EntityConflictError("entity id reused with a different type", entity_id=record.entity_id,
                                          existing=existing, incoming=record)
            # Explicit IDs may be used to add aliases/provenance, but changing
            # identity data is a conflict rather than an implicit merge.
            if existing.identity and record.identity and _canonical_json(existing.identity) != _canonical_json(record.identity):
                raise EntityConflictError("entity identity changed", entity_id=record.entity_id,
                                          existing=existing, incoming=record)
            merged_aliases = tuple(dict.fromkeys(
                existing.aliases + record.aliases + tuple(aliases)
                + ((incoming_name,) if normalize_alias(incoming_name) != normalize_alias(existing.canonical_name) else ())
            ))
            merged_stable = dict(existing.stable_ids)
            for key, value in record.stable_ids.items():
                if key in merged_stable and str(merged_stable[key]) != str(value):
                    raise EntityConflictError("stable identifier changed", entity_id=record.entity_id,
                                              existing=existing, incoming=record)
                merged_stable[str(key)] = str(value)
            merged_attrs = dict(existing.attributes)
            for key, value in record.attributes.items():
                if key in merged_attrs and _canonical_json(merged_attrs[key]) != _canonical_json(value):
                    raise EntityConflictError(f"attribute conflict: {key}", entity_id=record.entity_id,
                                              existing=existing, incoming=record)
                merged_attrs[str(key)] = _copy(value)
            merged_prov = list(existing.provenance)
            seen_prov = {item.canonical_key for item in merged_prov}
            for item in record.provenance:
                if item.canonical_key not in seen_prov:
                    merged_prov.append(item)
                    seen_prov.add(item.canonical_key)
            record = EntityRecord(
                entity_id=existing.entity_id,
                entity_type=existing.entity_type,
                canonical_name=existing.canonical_name,
                aliases=merged_aliases,
                stable_ids=merged_stable,
                identity=existing.identity or record.identity,
                attributes=merged_attrs,
                provenance=tuple(merged_prov),
            )
        # Stable identifiers are explicit identity assertions.  Reusing one
        # for another entity is a conflict, not an ambiguous display alias.
        for namespace, value in record.stable_ids.items():
            stable_key = (_normalise_text(namespace), _normalise_text(value))
            owners = self._stable_index.get(stable_key, set()) - {record.entity_id}
            if owners:
                prior = self._entities[sorted(owners)[0]]
                raise EntityConflictError(
                    f"stable identifier {namespace}:{value} already belongs to another entity",
                    entity_id=record.entity_id, existing=prior, incoming=record,
                )
        self._entities[record.entity_id] = record
        all_aliases = tuple(record.aliases) + tuple(aliases) + (record.canonical_name,)
        alias_changed = False
        for alias in all_aliases:
            alias_changed = self._add_alias_unlocked(alias, record.entity_id) or alias_changed
        for namespace, value in record.stable_ids.items():
            key = (_normalise_text(namespace), _normalise_text(value))
            self._stable_index.setdefault(key, set()).add(record.entity_id)
        if record.identity:
            self._identity_index.setdefault(_canonical_json(record.identity), set()).add(
                record.entity_id
            )
        changed = before is None or before != record.to_dict() or alias_changed
        if bump and changed:
            self._revision += 1
        return record

    def _add_alias_unlocked(self, alias: Any, entity_id: str,
                            provenance: Iterable[Provenance] = ()) -> bool:
        norm = normalize_alias(alias)
        if not norm:
            raise SemanticStoreError("alias cannot be empty")
        ids = self._aliases.setdefault(norm, set())
        before_ids = len(ids)
        ids.add(entity_id)
        keys = self._alias_provenance.setdefault(norm, set())
        before_keys = len(keys)
        keys.update(item.canonical_key for item in provenance)
        return len(ids) != before_ids or len(keys) != before_keys

    def register(self, entity_type: str | EntityRecord | Mapping[str, Any] | None = None,
                 canonical_name: str | None = None, *, canonical_label: str | None = None,
                 aliases: Iterable[str] = (),
                 stable_ids: Mapping[str, Any] | None = None,
                 external_ids: Mapping[str, Any] | None = None,
                 identity: Mapping[str, Any] | None = None,
                 attributes: Mapping[str, Any] | None = None,
                 provenance: Iterable[Provenance | Mapping[str, Any] | str] = (),
                 entity_id: str | None = None) -> EntityRecord:
        """Register an entity, merging only an identical deterministic key.

        Pass ``identity`` or ``stable_ids`` when two same-named entities must
        remain distinct.  A shared alias is allowed and resolves as ambiguous.
        """

        if canonical_name is None:
            canonical_name = canonical_label
        elif canonical_label is not None and normalize_alias(canonical_name) != normalize_alias(canonical_label):
            raise SemanticStoreError("canonical_name and canonical_label disagree")
        if stable_ids is None:
            stable_ids = external_ids
        elif external_ids is not None and _canonical_json(stable_ids) != _canonical_json(external_ids):
            raise SemanticStoreError("stable_ids and external_ids disagree")
        if isinstance(entity_type, EntityRecord):
            record = entity_type
            with self._lock:
                return _clone_entity_record(self._insert_record(record, aliases=aliases))
        if isinstance(entity_type, Mapping):
            raw = dict(entity_type)
            if canonical_name is not None:
                raw.setdefault("canonical_name", canonical_name)
            if entity_id is not None:
                raw["entity_id"] = entity_id
            raw_type = raw.get("entity_type", raw.get("type", "entity"))
            raw_name = raw.get("canonical_name", raw.get("canonical_label", raw.get("name", "")))
            raw_stable = raw.get("stable_ids", raw.get("external_ids", stable_ids or {}))
            raw_identity = raw.get("identity", identity or {})
            raw_id = raw.get("entity_id") or deterministic_id(
                "ent", self._identity_material(str(raw_type), str(raw_name), raw_stable, raw_identity)
            )
            record = EntityRecord.from_dict({
                "entity_id": raw_id,
                "entity_type": raw_type,
                "canonical_name": raw_name,
                "aliases": tuple(raw.get("aliases", aliases or ())),
                "stable_ids": raw_stable,
                "identity": raw_identity,
                "attributes": raw.get("attributes", attributes or {}),
                "provenance": raw.get("provenance", provenance or ()),
            })
            with self._lock:
                return _clone_entity_record(self._insert_record(record, aliases=aliases))
        if entity_type is None:
            raise SemanticStoreError("entity_type is required")
        if canonical_name is None:
            raise SemanticStoreError("canonical_name is required")
        stable = dict(stable_ids or {})
        ident = dict(identity or {})
        material = self._identity_material(str(entity_type), str(canonical_name), stable, ident)
        eid = str(entity_id or deterministic_id("ent", material))
        record = EntityRecord(
            entity_id=eid,
            entity_type=str(entity_type),
            canonical_name=str(canonical_name),
            aliases=tuple(aliases or ()), stable_ids=stable, identity=ident,
            attributes=dict(attributes or {}),
            provenance=tuple(_coerce_provenance(item) for item in (provenance or ())),
        )
        with self._lock:
            return _clone_entity_record(self._insert_record(record))

    def register_alias(self, alias: str, entity_id: str,
                       *, provenance: Iterable[Provenance | Mapping[str, Any] | str] = ()) -> None:
        with self._lock:
            if entity_id not in self._entities:
                raise EntityNotFoundError(f"unknown entity: {entity_id}")
            prov = _coerce_provenance_list(provenance)
            changed = self._add_alias_unlocked(alias, entity_id, prov)
            record = self._entities[entity_id]
            if normalize_alias(alias) not in {normalize_alias(item) for item in record.aliases}:
                self._entities[entity_id] = EntityRecord(
                    entity_id=record.entity_id, entity_type=record.entity_type,
                    canonical_name=record.canonical_name,
                    aliases=record.aliases + (str(alias),), stable_ids=record.stable_ids,
                    identity=record.identity, attributes=record.attributes,
                    provenance=record.provenance,
                )
                changed = True
            if changed:
                self._revision += 1

    def get(self, entity_id: str) -> EntityRecord | None:
        with self._lock:
            record = self._entities.get(str(entity_id))
            return _clone_entity_record(record) if record is not None else None

    def require(self, entity_id: str) -> EntityRecord:
        record = self.get(entity_id)
        if record is None:
            raise EntityNotFoundError(f"unknown entity: {entity_id}")
        return record

    def all(self) -> tuple[EntityRecord, ...]:
        with self._lock:
            return tuple(_clone_entity_record(self._entities[key]) for key in sorted(self._entities))

    def resolve_detailed(self, selector: Any, *, entity_type: str | None = None,
                         constraints: Mapping[str, Any] | None = None) -> EntityResolution:
        constraints = dict(constraints or {})
        if isinstance(selector, EntityRecord):
            selector = selector.entity_id
        raw_query = selector
        explicit_id: str | None = None
        stable_query: tuple[str, str] | None = None
        identity_query: str | None = None
        if isinstance(selector, Mapping):
            if selector.get("entity_id") is not None:
                explicit_id = str(selector["entity_id"])
            elif selector.get("id") is not None:
                # ``id`` is an explicit identity selector just like
                # ``entity_id``; an unknown id must not silently degrade into
                # a free-form alias lookup.
                explicit_id = str(selector["id"])
            elif selector.get("stable_id") is not None:
                namespace = str(selector.get("namespace", selector.get("stable_namespace", "")))
                stable_query = (_normalise_text(namespace), _normalise_text(selector["stable_id"]))
            elif selector.get("alias") is not None or selector.get("name") is not None:
                selector = selector.get("alias", selector.get("name", ""))
            else:
                # Structured selectors (for example exact coordinates) are
                # identity values, not fuzzy aliases. No rounding is applied.
                identity_query = _canonical_json(selector)
        with self._lock:
            if explicit_id is not None:
                ids = [explicit_id] if explicit_id in self._entities else []
            elif stable_query is not None:
                ids = sorted(self._stable_index.get(stable_query, ()))
            elif identity_query is not None:
                ids = sorted(self._identity_index.get(identity_query, ()))
            else:
                norm = normalize_alias(selector)
                ids = sorted(self._aliases.get(norm, ()))
            if entity_type is not None:
                typ = _normalise_text(entity_type)
                ids = [eid for eid in ids if _normalise_text(self._entities[eid].entity_type) == typ]
            if constraints:
                filtered: list[str] = []
                for eid in ids:
                    record = self._entities[eid]
                    ok = True
                    for key, expected in constraints.items():
                        if key in record.attributes:
                            actual = record.attributes[key]
                        elif key in record.identity:
                            actual = record.identity[key]
                        elif key in record.stable_ids:
                            actual = record.stable_ids[key]
                        else:
                            ok = False
                            break
                        if _canonical_json(actual) != _canonical_json(expected):
                            ok = False
                            break
                    if ok:
                        filtered.append(eid)
                ids = filtered
        norm_query = normalize_alias(selector)
        if len(ids) == 1:
            return EntityResolution(raw_query, norm_query, "resolved", tuple(ids))
        if len(ids) > 1:
            return EntityResolution(raw_query, norm_query, "ambiguous", tuple(ids), "alias has multiple entities")
        return EntityResolution(raw_query, norm_query, "not_found", (), "no matching entity")

    def resolve(self, selector: Any, *, entity_type: str | None = None,
                constraints: Mapping[str, Any] | None = None, strict: bool = False) -> str | list[str] | None:
        """Resolve a selector for runtime compatibility.

        A singleton returns its ID, an ambiguous alias returns sorted IDs, and
        an unknown selector returns ``None``.  ``strict=True`` raises instead.
        """

        result = self.resolve_detailed(selector, entity_type=entity_type, constraints=constraints)
        if result.status == "resolved":
            return result.entity_id
        if result.status == "ambiguous":
            if strict:
                raise AmbiguousEntityError(f"selector maps to {len(result.entity_ids)} entities: {selector!r}",
                                           query=selector, candidates=result.entity_ids)
            return list(result.entity_ids)
        if strict:
            raise EntityNotFoundError(f"unknown entity selector: {selector!r}")
        return None

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "revision": self._revision,
                "entities": [self._entities[key].to_dict() for key in sorted(self._entities)],
                "aliases": {key: sorted(value) for key, value in sorted(self._aliases.items())},
                "alias_provenance": {
                    key: sorted(value) for key, value in sorted(self._alias_provenance.items()) if value
                },
            }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EntityLedger":
        if not isinstance(value, Mapping):
            raise SerializationError("entity ledger must be an object")
        records = [EntityRecord.from_dict(item) for item in value.get("entities", ())]
        ledger = cls(records, revision=int(value.get("revision", 0)))
        # Preserve explicitly serialised aliases, including aliases that were
        # registered after the record snapshot was produced.
        with ledger._lock:
            for alias, ids in (value.get("aliases", {}) or {}).items():
                for eid in ids:
                    if eid in ledger._entities:
                        ledger._add_alias_unlocked(alias, str(eid))
            for alias, entries in (value.get("alias_provenance", {}) or {}).items():
                ledger._alias_provenance.setdefault(str(alias), set()).update(str(item) for item in entries)
        return ledger

    def dumps(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False)

    def save(self, path: str | Path) -> None:
        _atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "EntityLedger":
        return cls.from_dict(_load_json(path))

    def transaction(self) -> "EntityTransaction":
        return EntityTransaction(self)


@dataclass(frozen=True, slots=True, init=False)
class QualifiedClaim:
    """A typed claim with subject IDs, qualifiers, value and provenance."""

    subjects: tuple[str, ...]
    predicate: str
    value: Any
    qualifiers: Mapping[str, Any]
    unit: str | None
    value_type: str
    precision: Any
    uncertainty: Any
    provenance: tuple[Provenance, ...]
    claim_id: str
    world_id: str | None

    def __init__(self, subjects: str | Iterable[str] | None = None,
                 predicate: str = "", value: Any = None,
                 qualifiers: Mapping[str, Any] | None = None,
                 unit: str | None = None, value_type: str | None = None,
                 precision: Any = None, uncertainty: Any = None,
                 provenance: Iterable[Provenance | Mapping[str, Any] | str] = (),
                 claim_id: str | None = None, world_id: str | None = None,
                 *, subject: str | Iterable[str] | None = None,
                 scope: Mapping[str, Any] | None = None) -> None:
        if subjects is None:
            subjects = subject
        elif subject is not None and tuple(_as_subjects(subjects)) != tuple(_as_subjects(subject)):
            raise SemanticStoreError("subjects and subject disagree")
        subject_tuple = tuple(_as_subjects(subjects))
        if not subject_tuple:
            raise SemanticStoreError("claim requires at least one subject")
        pred = _normalise_predicate(predicate)
        q = dict(qualifiers if qualifiers is not None else (scope or {}))
        prov = _coerce_provenance_list(provenance)
        inferred = str(value_type or _infer_value_type(value))
        semantic = {
            "world_id": world_id,
            "subjects": list(subject_tuple), "predicate": pred,
            "value": _json_ready(value), "qualifiers": _json_ready(q),
            "unit": unit, "value_type": inferred,
            "precision": _json_ready(precision), "uncertainty": _json_ready(uncertainty),
        }
        ident = str(claim_id or deterministic_id("clm", semantic))
        object.__setattr__(self, "subjects", subject_tuple)
        object.__setattr__(self, "predicate", pred)
        object.__setattr__(self, "value", _copy(value))
        object.__setattr__(self, "qualifiers", _copy(q))
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "value_type", inferred)
        object.__setattr__(self, "precision", _copy(precision))
        object.__setattr__(self, "uncertainty", _copy(uncertainty))
        object.__setattr__(self, "provenance", prov)
        object.__setattr__(self, "claim_id", ident)
        object.__setattr__(self, "world_id", world_id)

    @property
    def subject(self) -> str:
        return self.subjects[0]

    @property
    def key_material(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "subjects": list(self.subjects), "predicate": self.predicate,
            "qualifiers": _json_ready(self.qualifiers), "unit": self.unit,
        }

    @property
    def claim_key(self) -> str:
        return deterministic_id("key", self.key_material)

    @property
    def semantic_material(self) -> dict[str, Any]:
        return {
            **self.key_material, "value": _json_ready(self.value),
            "value_type": self.value_type, "precision": _json_ready(self.precision),
            "uncertainty": _json_ready(self.uncertainty),
        }

    @property
    def fingerprint(self) -> str:
        return deterministic_id("fp", self.semantic_material)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, world_id: str | None = None) -> "QualifiedClaim":
        if not isinstance(value, Mapping):
            raise SerializationError("claim must be an object")
        raw_world = value.get("world_id")
        if world_id is not None and raw_world is not None and str(raw_world) != str(world_id):
            raise SemanticStoreError(
                f"claim world_id {raw_world!r} conflicts with ledger world_id {world_id!r}"
            )
        raw_subjects = value.get("subjects", value.get("subject", value.get("entity_id")))
        return cls(
            subjects=raw_subjects,
            predicate=str(value.get("predicate", "")),
            value=_restore_value(_copy(value.get("value"))),
            qualifiers=_restore_value(_copy(value.get("qualifiers", value.get("context", {}))) or {}),
            unit=value.get("unit"), value_type=value.get("value_type", value.get("type")),
            precision=_restore_value(_copy(value.get("precision"))),
            uncertainty=_restore_value(_copy(value.get("uncertainty"))),
            provenance=_coerce_provenance_list(value.get("provenance")),
            claim_id=value.get("claim_id"), world_id=(world_id if world_id is not None else raw_world),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "claim_id": self.claim_id,
            "subjects": list(self.subjects),
            "predicate": self.predicate,
            "value": _json_ready(self.value),
            "value_type": self.value_type,
            "qualifiers": _json_ready(self.qualifiers),
            "provenance": [item.to_dict() for item in self.provenance],
        }
        # A singleton ``subject`` is convenient for the task contract while
        # ``subjects`` remains the lossless representation for relations.
        if len(self.subjects) == 1:
            result["subject"] = self.subjects[0]
        if self.unit is not None:
            result["unit"] = self.unit
        if self.precision is not None:
            result["precision"] = _json_ready(self.precision)
        if self.uncertainty is not None:
            result["uncertainty"] = _json_ready(self.uncertainty)
        if self.world_id is not None:
            result["world_id"] = self.world_id
        return result

    # Lightweight Mapping-like access helps generic renderers consume claims
    # without importing this implementation's concrete class.
    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self) -> Iterator[str]:
        return iter(self.to_dict().keys())


def _as_subjects(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (set, frozenset)):
        # Set-valued relations must have a process-independent identity.
        values = tuple(sorted(value, key=lambda item: _canonical_json(item)))
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        values = tuple(value)
    else:
        values = (value,)
    result = tuple(str(item) for item in values if str(item).strip())
    return result


def _infer_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, (float, Decimal)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple, set, frozenset)):
        return "array"
    return type(value).__name__


class ClaimLedger:
    """Typed claim store with deterministic IDs and conflict detection."""

    def __init__(self, claims: Iterable[QualifiedClaim | Mapping[str, Any]] | None = None,
                 *, revision: int = 0):
        self._claims: dict[str, QualifiedClaim] = {}
        self._by_key: dict[str, set[str]] = {}
        self._conflicts: dict[str, set[str]] = {}
        self._revision = int(revision)
        self._lock = threading.RLock()
        for raw in claims or ():
            self._put_unlocked(self._coerce(raw), allow_conflict=True, bump=False)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def _coerce(self, raw: QualifiedClaim | Mapping[str, Any], *, world_id: str | None = None) -> QualifiedClaim:
        if isinstance(raw, QualifiedClaim):
            if world_id is not None and raw.world_id is not None and raw.world_id != world_id:
                raise SemanticStoreError(
                    f"claim world_id {raw.world_id!r} conflicts with ledger world_id {world_id!r}"
                )
            effective_world = world_id if world_id is not None else raw.world_id
            return QualifiedClaim(
                subjects=raw.subjects, predicate=raw.predicate, value=raw.value,
                qualifiers=raw.qualifiers, unit=raw.unit, value_type=raw.value_type,
                precision=raw.precision, uncertainty=raw.uncertainty,
                provenance=raw.provenance, claim_id=raw.claim_id, world_id=effective_world,
            )
        if not isinstance(raw, Mapping):
            raise SemanticStoreError("claim must be QualifiedClaim or mapping")
        return QualifiedClaim.from_dict(raw, world_id=world_id)

    @staticmethod
    def _detached(claim: QualifiedClaim) -> QualifiedClaim:
        """Return a deep copy so reads cannot mutate the authoritative ledger."""

        return QualifiedClaim(
            subjects=claim.subjects, predicate=claim.predicate, value=claim.value,
            qualifiers=claim.qualifiers, unit=claim.unit, value_type=claim.value_type,
            precision=claim.precision, uncertainty=claim.uncertainty,
            provenance=claim.provenance, claim_id=claim.claim_id, world_id=claim.world_id,
        )

    def _merge_same(self, prior: QualifiedClaim, incoming: QualifiedClaim) -> QualifiedClaim:
        seen = {item.canonical_key for item in prior.provenance}
        merged = list(prior.provenance)
        for item in incoming.provenance:
            if item.canonical_key not in seen:
                merged.append(item)
                seen.add(item.canonical_key)
        # Keep the first claim's ID and semantic value; provenance is the only
        # mutable evidence field merged on an idempotent write.
        return QualifiedClaim(
            subjects=prior.subjects, predicate=prior.predicate, value=prior.value,
            qualifiers=prior.qualifiers, unit=prior.unit, value_type=prior.value_type,
            precision=prior.precision, uncertainty=prior.uncertainty,
            provenance=tuple(merged), claim_id=prior.claim_id,
            world_id=prior.world_id or incoming.world_id,
        )

    def _put_unlocked(self, claim: QualifiedClaim, *, allow_conflict: bool = False,
                      bump: bool = True) -> QualifiedClaim:
        key = claim.claim_key
        same_id = self._claims.get(claim.claim_id)
        if same_id is not None and same_id.fingerprint != claim.fingerprint:
            raise ClaimConflictError(
                f"claim id {claim.claim_id} reused for different semantic content",
                key=key, existing=same_id, incoming=claim,
            )
        ids = self._by_key.setdefault(key, set())
        for existing_id in tuple(ids):
            prior = self._claims[existing_id]
            if prior.fingerprint == claim.fingerprint:
                merged = self._merge_same(prior, claim)
                self._claims[existing_id] = merged
                if bump and merged != prior:
                    self._revision += 1
                return merged
        if ids and not allow_conflict:
            prior = self._claims[sorted(ids)[0]]
            raise ClaimConflictError(
                f"conflicting values for semantic claim key {key}",
                key=key, existing=prior, incoming=claim,
            )
        self._claims[claim.claim_id] = claim
        ids.add(claim.claim_id)
        if len(ids) > 1:
            self._conflicts.setdefault(key, set()).update(ids)
        if bump:
            self._revision += 1
        return claim

    def put(self, claim: QualifiedClaim | Mapping[str, Any], *, world_id: str | None = None,
            allow_conflict: bool = False) -> QualifiedClaim:
        coerced = self._coerce(claim, world_id=world_id)
        with self._lock:
            return self._detached(self._put_unlocked(coerced, allow_conflict=allow_conflict))

    def put_many(self, claims: Iterable[QualifiedClaim | Mapping[str, Any]], *,
                 world_id: str | None = None, allow_conflict: bool = False) -> tuple[QualifiedClaim, ...]:
        tx = self.transaction()
        for claim in claims:
            tx.put(claim, world_id=world_id, allow_conflict=allow_conflict)
        return tx.commit()

    def get(self, claim_id_or_key: str | None = None, *, subjects: Any = None,
            subject: Any = None, predicate: str | None = None,
            qualifiers: Mapping[str, Any] | None = None, unit: str | None = None,
            world_id: str | None = None) -> QualifiedClaim | None:
        with self._lock:
            if subjects is not None or subject is not None or predicate is not None:
                raw_subjects = subjects if subjects is not None else subject
                if predicate is None:
                    raise SemanticStoreError("predicate is required for semantic lookup")
                wanted_subjects = tuple(_as_subjects(raw_subjects))
                wanted_predicate = _normalise_predicate(predicate)
                wanted_qualifiers = qualifiers or {}
                matches = [
                    item for item in self._claims.values()
                    if item.subjects == wanted_subjects
                    and item.predicate == wanted_predicate
                    and item.unit == unit
                    and _canonical_json(item.qualifiers) == _canonical_json(wanted_qualifiers)
                    and (world_id is None or item.world_id == world_id)
                ]
                if not matches:
                    return None
                if len(matches) > 1:
                    raise ClaimConflictError(
                        "semantic lookup matched multiple world/value claims",
                        existing=tuple(matches),
                    )
                return self._detached(matches[0])
            else:
                if claim_id_or_key is None:
                    return None
                key_or_id = str(claim_id_or_key)
                if key_or_id in self._claims:
                    item = self._claims[key_or_id]
                    if world_id is not None and item.world_id != world_id:
                        return None
                    return self._detached(item)
                key = key_or_id
            ids = self._by_key.get(key, set())
            if not ids:
                return None
            if len(ids) > 1:
                raise ClaimConflictError(f"semantic claim key has {len(ids)} conflicting values", key=key,
                                          existing=tuple(self._claims[item] for item in sorted(ids)))
            item = self._claims[next(iter(ids))]
            if world_id is not None and item.world_id != world_id:
                return None
            return self._detached(item)

    def require(self, claim_id: str) -> QualifiedClaim:
        claim = self.get(claim_id)
        if claim is None:
            raise SemanticStoreError(f"unknown claim: {claim_id}")
        return claim

    def query(self, *, subjects: Any = None, subject: Any = None,
              predicate: str | None = None, qualifiers: Mapping[str, Any] | None = None,
              unit: str | None = None, world_id: str | None = None,
              include_conflicts: bool = True) -> tuple[QualifiedClaim, ...]:
        wanted_subjects = tuple(_as_subjects(subjects if subjects is not None else subject)) if (subjects is not None or subject is not None) else None
        wanted_pred = _normalise_predicate(predicate) if predicate is not None else None
        with self._lock:
            values = list(self._claims.values())
        result: list[QualifiedClaim] = []
        for claim in values:
            if wanted_subjects is not None and claim.subjects != wanted_subjects:
                continue
            if wanted_pred is not None and claim.predicate != wanted_pred:
                continue
            if qualifiers is not None and _canonical_json(claim.qualifiers) != _canonical_json(qualifiers):
                continue
            if unit is not None and claim.unit != unit:
                continue
            if world_id is not None and claim.world_id != world_id:
                continue
            if not include_conflicts and claim.claim_key in self._conflicts:
                continue
            result.append(self._detached(claim))
        return tuple(sorted(result, key=lambda item: item.claim_id))

    def conflicts(self) -> dict[str, tuple[QualifiedClaim, ...]]:
        with self._lock:
            return {
                key: tuple(self._detached(self._claims[item]) for item in sorted(ids))
                for key, ids in sorted(self._conflicts.items())
            }

    def all(self) -> tuple[QualifiedClaim, ...]:
        with self._lock:
            return tuple(self._detached(self._claims[key]) for key in sorted(self._claims))

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema_version": SCHEMA_VERSION,
                "revision": self._revision,
                "claims": [self._claims[key].to_dict() for key in sorted(self._claims)],
                "conflicts": {key: sorted(ids) for key, ids in sorted(self._conflicts.items())},
            }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClaimLedger":
        if not isinstance(value, Mapping):
            raise SerializationError("claim ledger must be an object")
        ledger = cls(revision=int(value.get("revision", 0)))
        with ledger._lock:
            for item in value.get("claims", ()):
                ledger._put_unlocked(QualifiedClaim.from_dict(item), allow_conflict=True, bump=False)
            # Retain conflict metadata even if an older export omitted one of
            # the claims (the claims themselves remain the source of truth).
            for key, ids in (value.get("conflicts", {}) or {}).items():
                known = {str(item) for item in ids if str(item) in ledger._claims}
                if len(known) > 1:
                    ledger._conflicts[str(key)] = known
        return ledger

    def dumps(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent, allow_nan=False)

    def save(self, path: str | Path) -> None:
        _atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "ClaimLedger":
        return cls.from_dict(_load_json(path))

    def transaction(self) -> "ClaimTransaction":
        return ClaimTransaction(self)


class EntityTransaction:
    """Optimistic transaction wrapper for entity registrations/aliases."""

    def __init__(self, ledger: EntityLedger):
        self.ledger = ledger
        self.base_revision = ledger.revision
        self._operations: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise TransactionConflictError("transaction is already closed")

    def register(self, *args: Any, **kwargs: Any) -> EntityRecord:
        self._check_open()
        # Compute an ID immediately for useful callers, but defer mutation.
        if args and isinstance(args[0], EntityRecord):
            record = args[0]
        elif args and isinstance(args[0], Mapping):
            record = EntityRecord.from_dict(dict(args[0]))
        else:
            if len(args) < 2 and kwargs.get("canonical_name") is None:
                raise SemanticStoreError("canonical_name is required")
            entity_type = args[0] if args else kwargs.pop("entity_type")
            canonical_name = args[1] if len(args) > 1 else kwargs.pop("canonical_name")
            stable = dict(kwargs.get("stable_ids", {}) or {})
            ident = dict(kwargs.get("identity", {}) or {})
            material = self.ledger._identity_material(str(entity_type), str(canonical_name), stable, ident)
            record = EntityRecord(
                entity_id=str(kwargs.get("entity_id") or deterministic_id("ent", material)),
                entity_type=str(entity_type), canonical_name=str(canonical_name),
                aliases=tuple(kwargs.get("aliases", ()) or ()), stable_ids=stable,
                identity=ident, attributes=dict(kwargs.get("attributes", {}) or {}),
                provenance=tuple(_coerce_provenance(item) for item in (kwargs.get("provenance", ()) or ())),
            )
        self._operations.append(("register", (record,), {"aliases": kwargs.get("aliases", ())}))
        return record

    def register_alias(self, alias: str, entity_id: str, **kwargs: Any) -> None:
        self._check_open()
        self._operations.append(("alias", (alias, entity_id), kwargs))

    def commit(self) -> tuple[EntityRecord, ...]:
        self._check_open()
        with self.ledger._lock:
            if self.ledger._revision != self.base_revision:
                self._closed = True
                raise TransactionConflictError(
                    f"entity ledger changed from revision {self.base_revision} to {self.ledger._revision}"
                )
            snapshot = self.ledger.to_dict()
            result: list[EntityRecord] = []
            try:
                for op, args, kwargs in self._operations:
                    if op == "register":
                        result.append(self.ledger._insert_record(args[0], aliases=kwargs.get("aliases", ())))
                    else:
                        self.ledger.register_alias(*args, **kwargs)
            except Exception:
                restored = EntityLedger.from_dict(snapshot)
                self.ledger._entities = restored._entities
                self.ledger._aliases = restored._aliases
                self.ledger._stable_index = restored._stable_index
                self.ledger._alias_provenance = restored._alias_provenance
                self.ledger._revision = restored._revision
                self._closed = True
                raise
            self._closed = True
            return tuple(_clone_entity_record(item) for item in result)

    def rollback(self) -> None:
        self._closed = True
        self._operations.clear()

    def __enter__(self) -> "EntityTransaction":
        self._check_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


class ClaimTransaction:
    """Optimistic, atomic transaction for claim writes."""

    def __init__(self, ledger: ClaimLedger):
        self.ledger = ledger
        self.base_revision = ledger.revision
        self._operations: list[tuple[QualifiedClaim, bool]] = []
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise TransactionConflictError("transaction is already closed")

    def put(self, claim: QualifiedClaim | Mapping[str, Any], *, world_id: str | None = None,
            allow_conflict: bool = False) -> QualifiedClaim:
        self._check_open()
        coerced = self.ledger._coerce(claim, world_id=world_id)
        self._operations.append((coerced, allow_conflict))
        return coerced

    def commit(self) -> tuple[QualifiedClaim, ...]:
        self._check_open()
        with self.ledger._lock:
            if self.ledger._revision != self.base_revision:
                self._closed = True
                raise TransactionConflictError(
                    f"claim ledger changed from revision {self.base_revision} to {self.ledger._revision}"
                )
            snapshot = self.ledger.to_dict()
            result: list[QualifiedClaim] = []
            try:
                for claim, allow_conflict in self._operations:
                    result.append(self.ledger._put_unlocked(claim, allow_conflict=allow_conflict))
            except Exception:
                restored = ClaimLedger.from_dict(snapshot)
                self.ledger._claims = restored._claims
                self.ledger._by_key = restored._by_key
                self.ledger._conflicts = restored._conflicts
                self.ledger._revision = restored._revision
                self._closed = True
                raise
            self._closed = True
            return tuple(self.ledger._detached(item) for item in result)

    def rollback(self) -> None:
        self._closed = True
        self._operations.clear()

    def __enter__(self) -> "ClaimTransaction":
        self._check_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()


class SemanticStore:
    """Convenience container for an entity and claim ledger snapshot."""

    def __init__(self, entities: EntityLedger | None = None, claims: ClaimLedger | None = None,
                 *, world_id: str = "world-local"):
        if not world_id:
            raise SemanticStoreError("world_id is required")
        self.entities = entities or EntityLedger()
        self.claims = claims or ClaimLedger()
        self.world_id = world_id

    @property
    def revision(self) -> tuple[int, int]:
        return self.entities.revision, self.claims.revision

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "world_id": self.world_id,
                "entities": self.entities.to_dict(), "claims": self.claims.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticStore":
        return cls(EntityLedger.from_dict(value.get("entities", {})),
                   ClaimLedger.from_dict(value.get("claims", {})),
                   world_id=str(value.get("world_id", "world-local")))

    def save(self, path: str | Path) -> None:
        _atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "SemanticStore":
        return cls.from_dict(_load_json(path))


def _atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise SerializationError(f"cannot save ledger: {target}: {exc}") from exc


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            value = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SerializationError(f"cannot load ledger: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SerializationError("ledger JSON root must be an object")
    return dict(value)


__all__ = [
    "SCHEMA_VERSION", "SemanticStoreError", "EntityConflictError",
    "EntityNotFoundError", "AmbiguousEntityError", "ClaimConflictError",
    "TransactionConflictError", "SerializationError", "Provenance",
    "EntityRecord", "EntityResolution", "EntityLedger", "QualifiedClaim",
    "ClaimLedger", "EntityTransaction", "ClaimTransaction", "SemanticStore",
    "normalize_alias", "canonical_json", "deterministic_id",
]
