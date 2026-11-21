"""Contract-driven semantic environment.

Unlike the legacy ``OpenEnvironment``, this runtime does not let a mutable
conversation history define facts.  A lens receives a normalized ``QueryIR``
and must return typed claims; the ledger is the source of truth and the lens
only renders a tool-specific observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import threading
from typing import Any, Callable, Mapping

from .runtime import QueryIR, SemanticRuntime, SemanticRuntimeError, _jsonable
from .store import ClaimLedger, Provenance, QualifiedClaim


class LensError(SemanticRuntimeError):
    """Raised when a tool lens cannot produce a valid semantic view."""


@dataclass(frozen=True)
class LensResult:
    observation: Any
    # ``None`` distinguishes an omitted convenience field from an explicit
    # empty claim list.  When supplied, the list must exactly match
    # ``ToolLens.extract``; omission simply delegates to the extractor.
    claims: tuple[Mapping[str, Any], ...] | None = None


@dataclass(frozen=True)
class ToolLens:
    """Declarative contract plus executable extract/render adapters.

    The adapters are intentionally injected by a tool contract package.  The
    core environment never branches on a domain or tool name.
    """

    tool_id: str
    extract: Callable[[QueryIR, Any], list[Mapping[str, Any]]]
    render: Callable[[QueryIR, list[Any]], Any]
    # Complex relation tools may provide their own selector.  Most entity-read
    # tools use the generic subject/predicate resolver in SemanticEnvironment.
    select: Callable[[QueryIR, ClaimLedger, str], list[QualifiedClaim]] | None = None


def _claim_id(claim: Any) -> str:
    if isinstance(claim, Mapping):
        value = claim.get("claim_id")
        if value:
            return str(value)
    value = getattr(claim, "claim_id", None)
    if value:
        return str(value)
    raise LensError("lens returned a claim without claim_id")


def _pointer_get(value: Any, pointer: str | None) -> tuple[bool, Any]:
    """Resolve an RFC6901 pointer against a rendered observation."""

    if pointer in (None, "", "/"):
        return True, value
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return False, None
    current = value
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, (list, tuple)) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def _same_value(left: Any, right: Any) -> bool:
    return json.dumps(
        _jsonable(left), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ) == json.dumps(
        _jsonable(right), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def _canonical_json(value: Any) -> str:
    """Canonical representation for comparing source-call records."""

    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


class SemanticEnvironment:
    """Materialize a frozen semantic world through registered tool lenses."""

    def __init__(self, runtime: SemanticRuntime, world_id: str = "world-local",
                 *, reject_unresolved: bool = True):
        if not world_id:
            raise LensError("world_id is required")
        self.runtime = runtime
        self.world_id = world_id
        self.lenses: dict[str, ToolLens] = {}
        self.calls: dict[str, dict[str, Any]] = {}
        self.evidence: list[dict[str, Any]] = []
        self.reject_unresolved = reject_unresolved
        # A state snapshot must never race another call and roll back its
        # successful audit records.  The claim ledger has its own lock, but
        # calls/evidence/observations form a larger transaction boundary.
        self._lock = threading.RLock()

    def _snapshot_state(self) -> tuple[dict[str, Any], list[dict[str, Any]],
                                      dict[str, dict[str, Any]], dict[str, Any]]:
        """Capture every mutable surface touched by one materialisation.

        Claim writes, the semantic query cache, evidence, and runtime
        observations form one audit record.  Saving only the claim ledger can
        leave a cache entry or observation that points at claims rolled back
        after a renderer/serializer failure.
        """

        return (
            copy.deepcopy(self.calls),
            copy.deepcopy(self.evidence),
            copy.deepcopy(self.runtime.observations),
            self.runtime.claims.to_dict(),
        )

    def _restore_state(self, snapshot: tuple[dict[str, Any], list[dict[str, Any]],
                                             dict[str, dict[str, Any]], dict[str, Any]]) -> None:
        calls, evidence, observations, claim_snapshot = snapshot
        self.calls.clear()
        self.calls.update(calls)
        self.evidence[:] = evidence
        self.runtime.observations.clear()
        self.runtime.observations.update(observations)

        # Preserve the caller's ClaimLedger object identity while restoring
        # its complete internal indexes and revision.
        restored = type(self.runtime.claims).from_dict(claim_snapshot)
        with self.runtime.claims._lock:
            self.runtime.claims._claims = restored._claims
            self.runtime.claims._by_key = restored._by_key
            self.runtime.claims._conflicts = restored._conflicts
            self.runtime.claims._revision = restored._revision

    @staticmethod
    def _observation_hash(observation: Any) -> str:
        """Return the cross-process hash used by evidence/provenance checks."""

        payload = json.dumps(
            _jsonable(observation), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _claim_with_provenance(claim: QualifiedClaim,
                               *, call_id: str, source_call: Mapping[str, Any],
                               world_id: str, tool_id: str,
                               observation_hash: str,
                               allow_unbound: bool) -> QualifiedClaim:
        """Bind a staged claim to this call without changing its value.

        A claim may already carry provenance supplied by an extractor.  Missing
        audit fields are filled, while an additional current-call entry is
        retained when the extractor points at an older call.  This preserves
        history and guarantees that a freshly materialized claim has a path to
        the evidence emitted by this environment call.
        """

        provenance = tuple(claim.provenance)
        if not provenance:
            if allow_unbound:
                return claim
            raise LensError(
                f"claim {claim.claim_id} requires provenance with output_pointer"
            )
        bound: list[Provenance] = []
        for item in provenance:
            if item.output_pointer is None:
                if not allow_unbound:
                    raise LensError(
                        f"claim {claim.claim_id} requires provenance output_pointer"
                    )
                bound.append(item)
                continue
            if item.world_id is not None and str(item.world_id) != str(world_id):
                raise LensError(
                    f"claim {claim.claim_id} provenance world_id conflicts with environment"
                )
            if item.observation_hash is not None and str(item.observation_hash) != observation_hash:
                raise LensError(
                    f"claim {claim.claim_id} provenance observation_hash mismatch"
                )
            bound_item = Provenance(
                call_id=item.call_id or call_id,
                source_call=(item.source_call if item.source_call is not None else source_call),
                output_pointer=item.output_pointer,
                observation_hash=(item.observation_hash or observation_hash),
                world_id=(item.world_id or world_id),
                tool_id=(item.tool_id or tool_id),
                observed_at=item.observed_at,
                metadata=item.metadata,
            )
            bound.append(bound_item)
            # If an extractor referred to a prior call, add a current-call
            # witness rather than silently rewriting its historical record.  A
            # mismatched source/tool is treated the same way so every fresh
            # materialization has a provenance path to its own evidence.
            source_matches = (
                item.source_call is None
                or json.dumps(_jsonable(item.source_call), ensure_ascii=False,
                              sort_keys=True, separators=(",", ":"))
                == json.dumps(_jsonable(source_call), ensure_ascii=False,
                              sort_keys=True, separators=(",", ":"))
            )
            if (
                item.call_id not in (None, call_id)
                or not source_matches
                or (item.tool_id is not None and item.tool_id != tool_id)
            ):
                bound.append(Provenance(
                    call_id=call_id,
                    source_call=source_call,
                    output_pointer=item.output_pointer,
                    observation_hash=observation_hash,
                    world_id=world_id,
                    tool_id=tool_id,
                    observed_at=item.observed_at,
                    metadata=item.metadata,
                ))
        return QualifiedClaim(
            subjects=claim.subjects,
            predicate=claim.predicate,
            value=claim.value,
            qualifiers=claim.qualifiers,
            unit=claim.unit,
            value_type=claim.value_type,
            precision=claim.precision,
            uncertainty=claim.uncertainty,
            provenance=tuple(bound),
            claim_id=claim.claim_id,
            world_id=claim.world_id or world_id,
        )

    def _stage_claims(self, raw_claims: tuple[Any, ...], *, world_id: str) -> tuple[Any, ...]:
        """Coerce claims without mutating the authoritative ledger."""

        tx = self.runtime.claims.transaction()
        try:
            staged = tuple(tx.put(raw_claim, world_id=world_id) for raw_claim in raw_claims)
        finally:
            # ``put`` only queues operations; explicitly close the transaction
            # so a failed materialization cannot accidentally commit it later.
            tx.rollback()
        return staged

    def register_lens(self, lens: ToolLens) -> None:
        with self._lock:
            spec = self.runtime.registry.get(lens.tool_id)
            if spec.qualified_tool_id != lens.tool_id and lens.tool_id != spec.tool_id:
                raise LensError(f"lens/spec mismatch: {lens.tool_id}")
            self.lenses[spec.qualified_tool_id] = lens

    def generate_from_claims(
        self,
        query: QueryIR,
        source_claims: ClaimLedger,
        *,
        world_id: str | None = None,
    ) -> LensResult:
        """Render a query from an immutable semantic world claim store.

        This is the claim-centric counterpart to an exact observation cache.
        The query may split, merge, reorder, or repeat entity selectors; the
        selected claims are rendered in the normalized query order.  The
        source ledger is only read.  ``call`` subsequently extracts the view
        into the session ledger, so the solver receives observations but never
        direct access to frozen claims.
        """

        if not isinstance(source_claims, ClaimLedger):
            raise LensError("source_claims must be a ClaimLedger")
        spec = self.runtime.registry.get(query.tool_id)
        lens = self.lenses.get(query.tool_id)
        if lens is None:
            raise LensError(f"no semantic lens registered for {query.tool_id}")
        source_world = str(world_id or self.world_id)
        # A custom selector is executable contract code, but it must not
        # receive the authoritative mutable ledger.  Give it a detached
        # snapshot and later attest every returned claim against the original
        # semantic fingerprints.
        source_digest = _canonical_json(source_claims.to_dict())
        readonly_source = ClaimLedger.from_dict(source_claims.to_dict())

        if lens.select is not None:
            selected = lens.select(query, readonly_source, source_world)
            if not isinstance(selected, (list, tuple)):
                raise LensError("lens.select must return a list of claims")
            claims = list(selected)
        else:
            claims = self._select_claims(query, spec, readonly_source, source_world)

        if _canonical_json(source_claims.to_dict()) != source_digest:
            raise LensError("custom claim selector mutated the authoritative world ledger")

        if not claims:
            raise LensError("semantic query matched no frozen world claims")
        for claim in claims:
            if not isinstance(claim, QualifiedClaim):
                raise LensError("claim selector returned a non-QualifiedClaim value")
            if claim.world_id != source_world:
                raise LensError(
                    f"claim {claim.claim_id} belongs to world {claim.world_id!r}, "
                    f"expected {source_world!r}"
                )
            trusted_matches = source_claims.query(
                subjects=claim.subjects,
                predicate=claim.predicate,
                qualifiers=claim.qualifiers,
                unit=claim.unit,
                world_id=source_world,
                include_conflicts=False,
            )
            if len(trusted_matches) != 1 or trusted_matches[0].fingerprint != claim.fingerprint:
                raise LensError(
                    f"claim selector returned a value not present in frozen world: {claim.claim_id}"
                )
        try:
            observation = lens.render(query, claims)
        except Exception as exc:
            raise LensError(f"claim rendering failed: {exc}") from exc
        return LensResult(copy.deepcopy(observation))

    def extract_evidence_claims(
        self,
        source_call: Mapping[str, Any],
        observation: Any,
        *,
        world_id: str | None = None,
    ) -> tuple[QualifiedClaim, ...]:
        """Re-extract claims from an observation using this pinned lens.

        The verifier can install this method as its semantic evidence
        attestor.  It deliberately performs no generator call and writes to a
        throw-away ledger, so checking untrusted solver evidence cannot mutate
        the session/world store.  A caller must keep the environment's lens
        package immutable for the duration of verification.
        """

        if not isinstance(source_call, Mapping):
            raise LensError("semantic evidence source_call must be an object")
        tool_id = source_call.get("tool", source_call.get("name"))
        if tool_id is None:
            raise LensError("semantic evidence source_call lacks tool")
        raw_args = source_call.get("args", source_call.get("arguments", {}))
        query = self.runtime.normalize_call(str(tool_id), raw_args)
        if self.reject_unresolved and query.blocking_unresolved_selectors:
            raise LensError(
                "semantic evidence contains unresolved selectors: "
                + ", ".join(query.blocking_unresolved_selectors)
            )
        lens = self.lenses.get(query.tool_id)
        if lens is None:
            raise LensError(f"no semantic lens registered for {query.tool_id}")
        spec = self.runtime.registry.get(query.tool_id)
        try:
            extracted = lens.extract(query, observation)
        except Exception as exc:
            raise LensError(f"semantic evidence extraction failed: {exc}") from exc
        if not isinstance(extracted, (list, tuple)):
            raise LensError("lens.extract must return a list of claims")
        raw_claims = tuple(extracted)
        if spec.cardinality == "one" and len(raw_claims) != 1:
            raise LensError(
                f"semantic evidence claim cardinality mismatch: expected one, got {len(raw_claims)}"
            )
        scratch = ClaimLedger()
        tx = scratch.transaction()
        try:
            staged = [tx.put(item, world_id=str(world_id or self.world_id)) for item in raw_claims]
            # Commit to the scratch ledger to canonicalize IDs and merge an
            # intentional repeated selector, then return the authoritative
            # unique records.  A conflicting duplicate remains an error.
            self._validate_render_claims(staged)
            canonical = tx.commit()
        except Exception:
            tx.rollback()
            raise
        self._validate_claim_subjects(spec, list(canonical))
        for claim in staged:
            for provenance in claim.provenance:
                ok, observed = _pointer_get(observation, provenance.output_pointer)
                if not ok or not _same_value(observed, claim.value):
                    raise LensError(
                        f"semantic evidence pointer does not contain claim {claim.claim_id}"
                    )
        return tuple(canonical)

    def attest_evidence(
        self,
        expected_claim: Mapping[str, Any] | None = None,
        candidate_claim: Mapping[str, Any] | None = None,
        evidence: Mapping[str, Any] | None = None,
        *,
        source_claims: ClaimLedger,
        expected_tool: str | None = None,
        expected_world_id: str | None = None,
        require_session_record: bool = True,
        expected: Mapping[str, Any] | None = None,
        candidate: Mapping[str, Any] | None = None,
    ) -> bool:
        """Attest a candidate observation against an evaluator-owned ledger.

        Re-extraction alone only proves that an untrusted payload has the
        right shape; a solver could still guess the value.  This method adds
        the missing trust anchor: every re-extracted claim must have exactly
        one non-conflicting fingerprint in ``source_claims``.  The source
        ledger is read-only and should be loaded from the pinned world bundle.
        """

        expected_claim = expected_claim if expected_claim is not None else expected
        candidate_claim = candidate_claim if candidate_claim is not None else candidate
        if not isinstance(source_claims, ClaimLedger):
            raise LensError("source_claims must be a ClaimLedger")
        if not require_session_record:
            raise LensError(
                "semantic attestation requires an evaluator-owned sealed session record"
            )
        if not isinstance(expected_claim, Mapping) or not isinstance(candidate_claim, Mapping):
            return False
        if not isinstance(evidence, Mapping):
            return False
        raw_provenance = candidate_claim.get("provenance", candidate_claim.get("provenances", ()))
        if isinstance(raw_provenance, Mapping):
            raw_provenance = [raw_provenance]
        if not isinstance(raw_provenance, (list, tuple)):
            return False
        output = next(
            (evidence[key] for key in ("output", "observation", "result", "response", "value") if key in evidence),
            evidence,
        )
        # The contract/evaluator chooses the world and tool.  Candidate
        # fields are only checked against those values, never used to select
        # an alternate source snapshot.
        expected_provenance = expected_claim.get("provenance")
        if isinstance(expected_provenance, Mapping):
            expected_source = expected_provenance.get("source_call", expected_provenance.get("call"))
            if expected_tool is None and isinstance(expected_source, Mapping):
                expected_tool = expected_source.get("tool", expected_source.get("name"))
            if expected_world_id is None and expected_provenance.get("world_id") is not None:
                expected_world_id = str(expected_provenance["world_id"])
        if expected_tool is not None:
            expected_tool = str(expected_tool)
        world_id = str(expected_world_id or self.world_id)
        if candidate_claim.get("world_id") is not None and str(candidate_claim["world_id"]) != world_id:
            return False
        if evidence.get("world_id") is not None and str(evidence["world_id"]) != world_id:
            return False
        expected_id = str(expected_claim.get("claim_id", ""))
        candidate_id = str(candidate_claim.get("claim_id", ""))
        if candidate_id and expected_id and candidate_id != expected_id:
            return False
        authoritative_target = (
            source_claims.get(expected_id, world_id=world_id) if expected_id else None
        )
        if authoritative_target is None:
            expected_subjects = expected_claim.get(
                "subjects", expected_claim.get("subject")
            )
            expected_predicate = expected_claim.get("predicate")
            if expected_subjects is None or expected_predicate is None:
                return False
            target_matches = source_claims.query(
                subjects=expected_subjects,
                predicate=str(expected_predicate),
                qualifiers=expected_claim.get("qualifiers"),
                unit=expected_claim.get("unit"),
                world_id=world_id,
                include_conflicts=False,
            )
            if len(target_matches) != 1:
                return False
            authoritative_target = target_matches[0]
        # A conflicted source ledger is not an authoritative world.  Looking
        # up a claim by ID alone would otherwise let one side of a same-key
        # value conflict pass merely because its ID happened to be selected.
        conflicting_keys = source_claims.conflicts()
        if authoritative_target.claim_key in conflicting_keys:
            return False
        for raw_item in raw_provenance:
            if not isinstance(raw_item, Mapping):
                continue
            source_call = raw_item.get("source_call", raw_item.get("call"))
            if not isinstance(source_call, Mapping):
                continue
            source_tool = source_call.get("tool", source_call.get("name"))
            if expected_tool is not None and str(source_tool) != expected_tool:
                continue
            evidence_call_id = evidence.get("call_id", evidence.get("id"))
            if raw_item.get("call_id") is not None and str(raw_item["call_id"]) != str(evidence_call_id):
                continue
            evidence_source = evidence.get("source_call", evidence.get("call"))
            if evidence_source is None and evidence.get("tool") is not None:
                evidence_source = {
                    "tool": evidence.get("tool"),
                    "args": evidence.get("args", evidence.get("arguments", {})),
                }
            if evidence_source is None or _canonical_json(source_call) != _canonical_json(evidence_source):
                continue
            evidence_hash = self._observation_hash(output)
            if raw_item.get("observation_hash") is not None \
                    and str(raw_item["observation_hash"]) != evidence_hash:
                continue
            if evidence.get("observation_hash") is not None \
                    and str(evidence["observation_hash"]) != evidence_hash:
                continue
            pointer = raw_item.get("output_pointer", raw_item.get("pointer", raw_item.get("json_pointer")))
            ok, pointed_value = _pointer_get(output, pointer)
            if not ok:
                continue
            if "value" in candidate_claim and not _same_value(pointed_value, candidate_claim["value"]):
                continue
            # In-process generalized evaluation must consume an evaluator-owned
            # session record, not a solver-supplied copy of an observation.
            if require_session_record:
                trusted = False
                for item in self.evidence:
                    if str(item.get("call_id")) != str(evidence_call_id):
                        continue
                    trusted_output = item.get("output", item.get("observation"))
                    trusted_hash = self._observation_hash(trusted_output)
                    if trusted_hash != evidence_hash:
                        continue
                    if item.get("observation_hash") is not None \
                            and str(item["observation_hash"]) != trusted_hash:
                        continue
                    trusted_source = item.get("source_call", item.get("call"))
                    # A session record without its canonical source call is
                    # not an evaluator-owned witness.  Treat it as forged
                    # rather than allowing a query-digest coincidence.
                    if not isinstance(trusted_source, Mapping):
                        continue
                    if _canonical_json(trusted_source) != _canonical_json(source_call):
                        continue
                    if item.get("world_id") is not None and str(item["world_id"]) != world_id:
                        continue
                    try:
                        trusted_query = self.runtime.normalize_call(
                            str(source_tool),
                            trusted_source.get("args", trusted_source.get("arguments", {})),
                        )
                    except Exception:
                        continue
                    observation_record = self.runtime.observations.get(str(evidence_call_id))
                    if not isinstance(observation_record, Mapping) \
                            or observation_record.get("query_digest") != trusted_query.digest:
                        continue
                    if observation_record.get("tool_id") != trusted_query.tool_id:
                        continue
                    trusted = True
                    break
                if not trusted:
                    continue
            try:
                extracted = self.extract_evidence_claims(
                    source_call, output, world_id=str(world_id)
                )
            except Exception:
                continue
            for observed in extracted:
                # A candidate cannot redirect a claim to an arbitrary
                # same-valued field or another row; the pointer and semantic
                # identity must belong to this exact extracted claim.
                observed_pointers = {
                    provenance.output_pointer
                    for provenance in observed.provenance
                    if provenance.output_pointer is not None
                }
                if pointer not in observed_pointers:
                    continue
                candidate_subject = candidate_claim.get("subject")
                if candidate_subject is not None:
                    observed_subject = (
                        observed.subject if len(observed.subjects) == 1
                        else list(observed.subjects)
                    )
                    if not _same_value(candidate_subject, observed_subject):
                        continue
                if candidate_claim.get("subjects") is not None \
                        and not _same_value(candidate_claim["subjects"], list(observed.subjects)):
                    continue
                if candidate_claim.get("predicate") is not None \
                        and str(candidate_claim["predicate"]) != observed.predicate:
                    continue
                if candidate_claim.get("qualifiers") is not None \
                        and not _same_value(candidate_claim["qualifiers"], observed.qualifiers):
                    continue
                if candidate_claim.get("unit") is not None and candidate_claim["unit"] != observed.unit:
                    continue
                if candidate_claim.get("value_type") is not None \
                        and str(candidate_claim["value_type"]) != observed.value_type:
                    continue
                if candidate_claim.get("precision") is not None \
                        and not _same_value(candidate_claim["precision"], observed.precision):
                    continue
                if candidate_claim.get("uncertainty") is not None \
                        and not _same_value(candidate_claim["uncertainty"], observed.uncertainty):
                    continue
                if expected_claim.get("subject") is not None:
                    observed_subject = (
                        observed.subject if len(observed.subjects) == 1
                        else list(observed.subjects)
                    )
                    if not _same_value(expected_claim["subject"], observed_subject):
                        continue
                if expected_claim.get("predicate") is not None \
                        and str(expected_claim["predicate"]) != observed.predicate:
                    continue
                if expected_claim.get("qualifiers") is not None \
                        and not _same_value(expected_claim["qualifiers"], observed.qualifiers):
                    continue
                if expected_claim.get("unit") is not None and expected_claim["unit"] != observed.unit:
                    continue
                if expected_claim.get("value_type") is not None \
                        and str(expected_claim["value_type"]) != observed.value_type:
                    continue
                if expected_claim.get("precision") is not None \
                        and not _same_value(expected_claim["precision"], observed.precision):
                    continue
                if expected_claim.get("uncertainty") is not None \
                        and not _same_value(expected_claim["uncertainty"], observed.uncertainty):
                    continue
                if "value" in candidate_claim and not _same_value(
                    observed.value, candidate_claim.get("value")
                ):
                    continue
                if "value" in expected_claim and not _same_value(
                    observed.value, expected_claim.get("value")
                ):
                    continue
                pinned = source_claims.get(observed.claim_id, world_id=str(world_id))
                if pinned is None:
                    # Explicit lens IDs are presentation metadata; when they
                    # differ from a deterministic store ID, resolve the
                    # trusted fact by its complete semantic key instead.
                    matches = source_claims.query(
                        subjects=observed.subjects,
                        predicate=observed.predicate,
                        qualifiers=observed.qualifiers,
                        unit=observed.unit,
                        world_id=str(world_id),
                        include_conflicts=False,
                    )
                    if len(matches) == 1:
                        pinned = matches[0]
                if pinned is None \
                        or pinned.claim_key in conflicting_keys \
                        or pinned.fingerprint != observed.fingerprint:
                    continue
                if authoritative_target.fingerprint != observed.fingerprint:
                    continue
                return True
        return False

    def make_semantic_attestor(self, source_claims: ClaimLedger) -> Callable[..., bool]:
        """Seal a detached evaluator-owned world snapshot for verification.

        The returned callback never consults a caller-owned mutable ledger.
        This prevents accidentally passing the solver/session ledger as the
        semantic trust anchor and detects unexpected mutation of the sealed
        snapshot before each attestation.
        """

        if not isinstance(source_claims, ClaimLedger):
            raise LensError("source_claims must be a ClaimLedger")
        snapshot = ClaimLedger.from_dict(source_claims.to_dict())
        digest = _canonical_json(snapshot.to_dict())

        def attestor(*, expected: Mapping[str, Any], candidate: Mapping[str, Any],
                     evidence: Mapping[str, Any]) -> bool:
            if _canonical_json(snapshot.to_dict()) != digest:
                raise LensError("sealed semantic source snapshot was mutated")
            return self.attest_evidence(
                expected=expected,
                candidate=candidate,
                evidence=evidence,
                source_claims=snapshot,
            )

        return attestor

    def _select_claims(
        self,
        query: QueryIR,
        spec: Any,
        source_claims: ClaimLedger,
        world_id: str,
    ) -> list[QualifiedClaim]:
        """Fail-closed default selector for entity/predicate claim tools."""

        predicates = tuple(str(item) for item in query.predicates if str(item))
        if not predicates:
            raise LensError("claim-backed rendering requires a declared predicate")

        values: list[Any] = []
        for name in spec.entity_args:
            if name not in query.arguments:
                raise LensError(f"semantic query is missing entity argument {name!r}")
            values.append(query.arguments[name])

        def resolve_subject(value: Any, *, argument: str) -> str:
            """Resolve both normalized IDs and structured/public selectors.

            ``normalize_call`` normally replaces strict selectors with an
            ``EntityID``.  A contract may intentionally keep a selector
            literal (for example a coordinate object), however; claim-backed
            rendering must still address the same canonical entity without a
            domain-specific branch.  Resolution is conservative: ambiguous
            and unknown selectors fail closed.
            """

            text_value = str(value)
            if self.runtime.entities.get(text_value) is not None:
                return text_value
            resolution = self.runtime.entities.resolve_detailed(value)
            if resolution.status == "resolved" and resolution.entity_id is not None:
                return str(resolution.entity_id)
            if resolution.status == "ambiguous":
                raise LensError(
                    f"query selector for {argument!r} is ambiguous: {value!r}"
                )
            raise LensError(
                f"query selector for {argument!r} is not a registered entity: {value!r}"
            )

        subject_groups: list[tuple[str, ...]] = []
        if not values:
            # Predicate-only tools are supported when the result is
            # unambiguous. ClaimLedger's stable ordering is preserved below.
            subject_groups = []
        elif len(values) == 1:
            raw = values[0]
            items = raw if isinstance(raw, list) else [raw]
            subject_groups = [
                (resolve_subject(item, argument=str(spec.entity_args[0])),)
                for item in items
            ]
        else:
            if any(isinstance(item, list) for item in values):
                raise LensError(
                    "multi-argument list relations require a ToolLens.select callback"
                )
            subject_groups = [
                tuple(
                    resolve_subject(item, argument=str(name))
                    for name, item in zip(spec.entity_args, values)
                )
            ]

        selected: list[QualifiedClaim] = []
        if subject_groups:
            for subjects in subject_groups:
                for predicate in predicates:
                    matches = source_claims.query(
                        subjects=subjects,
                        predicate=predicate,
                        world_id=world_id,
                        include_conflicts=False,
                    )
                    if len(matches) != 1:
                        raise LensError(
                            "frozen claim lookup is missing or ambiguous for "
                            f"subjects={subjects!r}, predicate={predicate!r}: {len(matches)} matches"
                        )
                    selected.append(matches[0])
        else:
            for predicate in predicates:
                matches = source_claims.query(
                    predicate=predicate,
                    world_id=world_id,
                    include_conflicts=False,
                )
                if not matches:
                    raise LensError(f"frozen claim lookup found no predicate {predicate!r}")
                selected.extend(matches)

        if spec.cardinality == "one" and len(selected) != 1:
            raise LensError(
                f"claim-backed cardinality mismatch: expected one, got {len(selected)}"
            )
        return selected

    def _validate_claim_subjects(self, spec: Any, claims: list[QualifiedClaim]) -> None:
        """Reject ghost subjects for contracts that declare entity semantics."""

        capabilities = {str(item) for item in spec.capabilities}
        requires_registered = bool(spec.entity_args) or "entity-read" in capabilities
        if not requires_registered or "allow-unregistered-subjects" in capabilities:
            return
        for claim in claims:
            for subject in claim.subjects:
                if self.runtime.entities.get(subject) is None:
                    raise LensError(
                        f"claim {claim.claim_id} references unregistered entity subject {subject}"
                    )

    def call(self, tool_id: str, raw_args: Mapping[str, Any], *, call_id: str,
             generate: Callable[[QueryIR], LensResult] | None = None) -> Any:
        with self._lock:
            return self._call_locked(tool_id, raw_args, call_id=call_id, generate=generate)

    def _call_locked(self, tool_id: str, raw_args: Mapping[str, Any], *, call_id: str,
                     generate: Callable[[QueryIR], LensResult] | None = None) -> Any:
        if not call_id:
            raise LensError("call_id is required")
        query = self.runtime.normalize_call(tool_id, raw_args)
        if self.reject_unresolved and query.blocking_unresolved_selectors:
            raise LensError(
                "semantic query contains unresolved entity selectors: "
                + ", ".join(query.blocking_unresolved_selectors)
            )
        if call_id in self.runtime.observations:
            prior = self.runtime.observations[call_id]
            if prior.get("query_digest") != query.digest:
                raise LensError(f"call_id reused for a different query: {call_id}")
            # Even an identical replay must receive a fresh call id.  Reusing
            # an id would make evidence multiplicity and provenance ambiguous.
            raise LensError(f"call_id already materialized: {call_id}")
        snapshot = self._snapshot_state()
        cached = self.calls.get(query.digest)
        try:
            spec = self.runtime.registry.get(query.tool_id)
            allow_unbound = bool(
                {str(item) for item in spec.capabilities}
                & {"opaque_exact", "allow_unbound_claims"}
            )
            if cached is not None:
                # Equivalent query spellings still create a distinct auditable
                # call and provenance entry; only semantic materialisation is
                # reused.  The whole branch is transactional as well.
                claim_ids = list(cached["claim_ids"])
                cached_observation = copy.deepcopy(cached["observation"])
                computed_cached_hash = self._observation_hash(cached_observation)
                stored_cached_hash = cached.get("observation_hash")
                if stored_cached_hash is not None and str(stored_cached_hash) != computed_cached_hash:
                    raise LensError("cached observation_hash mismatch: does not match observation")
                cached_hash = computed_cached_hash
                source_call = {"tool": query.tool_id, "args": query.arguments}
                provenance_templates = cached.get("claim_provenance")
                if not isinstance(provenance_templates, Mapping):
                    raise LensError("cached semantic claims lack provenance templates")
                for claim_id in claim_ids:
                    prior = self.runtime.claims.require(claim_id)
                    if prior.world_id != self.world_id:
                        raise LensError(f"cached claim {claim_id} belongs to another world")
                    raw_templates = provenance_templates.get(claim_id)
                    if not isinstance(raw_templates, list):
                        raise LensError(f"cached claim {claim_id} lacks provenance templates")
                    template_claim = QualifiedClaim(
                        subjects=prior.subjects,
                        predicate=prior.predicate,
                        value=prior.value,
                        qualifiers=prior.qualifiers,
                        unit=prior.unit,
                        value_type=prior.value_type,
                        precision=prior.precision,
                        uncertainty=prior.uncertainty,
                        provenance=tuple(Provenance.from_dict(item) for item in raw_templates),
                        claim_id=prior.claim_id,
                        world_id=prior.world_id,
                    )
                    bound = self._claim_with_provenance(
                        template_claim,
                        call_id=call_id,
                        source_call=source_call,
                        world_id=self.world_id,
                        tool_id=query.tool_id,
                        observation_hash=cached_hash,
                        allow_unbound=allow_unbound,
                    )
                    self._validate_current_witnesses(
                        bound, observation=cached_observation,
                        call_id=call_id, observation_hash=cached_hash,
                        allow_unbound=allow_unbound,
                    )
                    self.runtime.claims.put(bound, world_id=self.world_id)
                observation = cached_observation
                self.runtime.record_observation(call_id, query, observation, claim_ids)
                self.evidence.append({"call_id": call_id, "source_call": source_call,
                                      "output": copy.deepcopy(observation),
                                      "observation_hash": cached_hash,
                                      "world_id": self.world_id})
                return observation

            lens = self.lenses.get(query.tool_id)
            if lens is None:
                raise LensError(f"no semantic lens registered for {query.tool_id}")
            if generate is None:
                raise LensError("missing generator for an unmaterialized semantic query")
            result = generate(query)
            if not isinstance(result, LensResult):
                raise LensError("generator must return LensResult")
            try:
                extracted = lens.extract(query, result.observation)
            except Exception as exc:
                raise LensError(f"claim extraction failed: {exc}") from exc
            if not isinstance(extracted, (list, tuple)):
                raise LensError("lens.extract must return a list of claims")
            extracted_tuple = tuple(extracted)
            if spec.cardinality == "one" and len(extracted_tuple) != 1:
                raise LensError(
                    f"lens claim cardinality mismatch: expected one, got {len(extracted_tuple)}"
                )

            # The extractor is the executable semantic authority.  A
            # generator may include an explicit claim list for convenience,
            # but it must match the extractor byte-for-byte (including the
            # empty-list case); otherwise fabricated claims could bypass the
            # lens.  ``None`` means the generator omitted the optional list.
            explicit = result.claims
            if explicit is not None:
                if not isinstance(explicit, (list, tuple)):
                    raise LensError("generator claims must be a list or tuple")
                explicit_tuple = tuple(explicit)
                if _stable_claims(explicit_tuple) != _stable_claims(extracted_tuple):
                    raise LensError("generator claims disagree with lens.extract")
            raw_claims = extracted_tuple

            # Render against canonical, uncommitted claims first.  The
            # resulting observation supplies the hash used to bind provenance;
            # only then do we commit the claims atomically.
            canonical_claims = list(self._stage_claims(raw_claims, world_id=self.world_id))
            self._validate_claim_subjects(spec, canonical_claims)
            # ClaimLedger assigns deterministic IDs when an extractor omits
            # them.  Repeated selectors are valid (``[A, A, B]``): retain the
            # repeated rows for rendering, but reject a lens that maps one
            # claim ID to different semantic values.  Distinct IDs for one
            # semantic claim are also rejected because they would make the
            # authoritative identity depend on presentation order.
            self._validate_render_claims(canonical_claims)
            observation = lens.render(query, canonical_claims)
            observation = copy.deepcopy(observation)
            observation_hash = self._observation_hash(observation)
            source_call = {"tool": query.tool_id, "args": query.arguments}
            bound_claims = [
                self._claim_with_provenance(
                    claim,
                    call_id=call_id,
                    source_call=source_call,
                    world_id=self.world_id,
                    tool_id=query.tool_id,
                    observation_hash=observation_hash,
                    allow_unbound=allow_unbound,
                )
                for claim in canonical_claims
            ]
            for claim in bound_claims:
                self._validate_current_witnesses(
                    claim, observation=observation,
                    call_id=call_id, observation_hash=observation_hash,
                    allow_unbound=allow_unbound,
                )
            # Collapse repeated semantic claims only at the ledger boundary.
            # Their provenance pointers are merged, so every displayed row
            # remains independently auditable while the session store keeps a
            # single authoritative fingerprint.
            unique_bound_claims = self._aggregate_bound_claims(bound_claims)
            tx = self.runtime.claims.transaction()
            for claim in unique_bound_claims:
                tx.put(claim, world_id=self.world_id)
            canonical = tx.commit()
            claim_ids = [_claim_id(claim) for claim in canonical]
            cache_provenance: dict[str, list[dict[str, Any]]] = {}
            for committed in canonical:
                current = [
                    item for item in committed.provenance
                    if item.call_id == call_id and item.observation_hash == observation_hash
                ]
                cache_provenance[committed.claim_id] = [
                    {
                        "output_pointer": item.output_pointer,
                        **({"observed_at": item.observed_at} if item.observed_at is not None else {}),
                        **({"metadata": copy.deepcopy(item.metadata)} if item.metadata else {}),
                    }
                    for item in current
                ]
            # Store by semantic query, not by the caller's raw argument spelling.
            self.calls[query.digest] = {
                "query": query.canonical_dict(),
                "observation": copy.deepcopy(observation),
                "observation_hash": observation_hash,
                "claim_ids": list(claim_ids),
                # Cache only pointer templates for this exact rendered view.
                # The authoritative claim may later gain provenance from a
                # different query shape with a different observation hash.
                "claim_provenance": cache_provenance,
            }
            self.runtime.record_observation(call_id, query, observation, claim_ids)
            self.evidence.append({
                "call_id": call_id,
                "source_call": {"tool": query.tool_id, "args": query.arguments},
                "output": copy.deepcopy(observation),
                "observation_hash": observation_hash,
                "world_id": self.world_id,
            })
            return observation
        except Exception:
            # A failed lens/render/observation must not leave a half-materialized
            # semantic world or dangling audit references.
            self._restore_state(snapshot)
            raise

    @staticmethod
    def _validate_current_witnesses(
        claim: QualifiedClaim,
        *,
        observation: Any,
        call_id: str,
        observation_hash: str,
        allow_unbound: bool,
    ) -> None:
        """Ensure current-call provenance points to the extracted claim value."""

        current = [
            item for item in claim.provenance
            if item.call_id == call_id and item.observation_hash == observation_hash
        ]
        if not current:
            if allow_unbound:
                return
            raise LensError(f"claim {claim.claim_id} lacks current-call provenance")
        for item in current:
            ok, observed = _pointer_get(observation, item.output_pointer)
            if not ok:
                raise LensError(
                    f"claim {claim.claim_id} provenance pointer is absent from observation"
                )
            if not _same_value(observed, claim.value):
                raise LensError(
                    f"claim {claim.claim_id} provenance value disagrees with observation"
                )

    @staticmethod
    def _validate_render_claims(claims: list[QualifiedClaim]) -> None:
        """Validate duplicate rows before rendering a materialized view."""

        by_id: dict[str, QualifiedClaim] = {}
        by_key: dict[str, QualifiedClaim] = {}
        for claim in claims:
            prior = by_id.get(claim.claim_id)
            if prior is not None and prior.fingerprint != claim.fingerprint:
                raise LensError(
                    f"claim_id {claim.claim_id} is reused for conflicting semantic values"
                )
            by_id.setdefault(claim.claim_id, claim)
            prior_key = by_key.get(claim.claim_key)
            if prior_key is not None and prior_key.fingerprint != claim.fingerprint:
                raise LensError(
                    "lens returned conflicting values for one semantic claim key"
                )
            if prior_key is not None and prior_key.claim_id != claim.claim_id:
                raise LensError("lens returned duplicate semantic claims")
            by_key.setdefault(claim.claim_key, claim)

    @staticmethod
    def _aggregate_bound_claims(claims: list[QualifiedClaim]) -> list[QualifiedClaim]:
        """Merge repeated rows while preserving every distinct provenance path."""

        grouped: dict[str, QualifiedClaim] = {}
        order: list[str] = []
        for claim in claims:
            prior = grouped.get(claim.claim_id)
            if prior is None:
                grouped[claim.claim_id] = claim
                order.append(claim.claim_id)
                continue
            if prior.fingerprint != claim.fingerprint:
                # Normally caught by ``_validate_render_claims``; retain the
                # guard for callers that reuse this helper independently.
                raise LensError(
                    f"claim_id {claim.claim_id} is reused for conflicting semantic values"
                )
            seen = {item.canonical_key for item in prior.provenance}
            merged = list(prior.provenance)
            for item in claim.provenance:
                if item.canonical_key not in seen:
                    merged.append(item)
                    seen.add(item.canonical_key)
            grouped[claim.claim_id] = QualifiedClaim(
                subjects=prior.subjects,
                predicate=prior.predicate,
                value=prior.value,
                qualifiers=prior.qualifiers,
                unit=prior.unit,
                value_type=prior.value_type,
                precision=prior.precision,
                uncertainty=prior.uncertainty,
                provenance=tuple(merged),
                claim_id=prior.claim_id,
                world_id=prior.world_id,
            )
        return [grouped[item] for item in order]

    def export(self) -> dict[str, Any]:
        with self._lock:
            return {
                "world_id": self.world_id,
                "calls": copy.deepcopy(self.calls),
                "evidence": copy.deepcopy(self.evidence),
                "runtime": self.runtime.export(),
            }


def _stable_claims(claims: tuple[Any, ...]) -> str:
    """Compare extracted and explicitly supplied claims without object identity."""

    values = []
    for claim in claims:
        if hasattr(claim, "to_dict"):
            claim = claim.to_dict()
        values.append(claim)
    return json.dumps(values, ensure_ascii=False, sort_keys=True, default=str,
                      separators=(",", ":"))
