"""Immutable runtime/world contracts for the current EvoSynth pipeline.

The original evosynth pipeline read a mutable global registry at inference
time.  A task could consequently be generated under one set of tool semantics
and evaluated under another.  This module provides a small, dependency-free
manifest which pins every input that can change the meaning of a world:
registry, semantic contracts, code used to normalise and render claims,
prompts/models, runtime endpoints, and the world revision.

The module deliberately does not import or modify the old pipeline at import
time.  ``RuntimeConfig.capture`` reads its constants lazily, so importing this
module never starts a client or makes a network request.  Verification is
fail-closed: malformed manifests, missing files, unsupported source objects,
and digest mismatches raise ``ManifestMismatchError`` (or return ``False``
when ``raise_on_error=False`` is explicitly requested).
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import tempfile
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, ClassVar

from ..version import EVOSYNTH_VERSION


SCHEMA_VERSION = "evosynth_manifest"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()
_SECRET_FIELD_NAMES = frozenset(
    {"api_key", "agent_key", "world_model_key", "token", "password", "secret"}
)


class ManifestError(RuntimeError):
    """Base class for all manifest errors."""


class ManifestValidationError(ManifestError, ValueError):
    """Raised when a manifest or a source cannot be represented safely."""


class ManifestMismatchError(ManifestError):
    """Raised when a frozen manifest does not match the runtime/world."""


class ManifestIOError(ManifestError):
    """Raised for malformed or non-atomically readable manifest files."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _check_digest(value: str, field: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value.lower()):
        raise ManifestValidationError(
            f"{field} must be a lowercase SHA-256 hex digest"
        )
    return value.lower()


def _freeze_value(value: Any) -> Any:
    """Recursively freeze mappings/sequences used by immutable dataclasses."""

    if isinstance(value, Mapping):
        frozen = {str(k): _freeze_value(v) for k, v in value.items()}
        return types.MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(v) for v in value)
    if isinstance(value, set):
        # Sorting by canonical JSON gives deterministic ordering for mixed
        # scalar types without relying on Python's incomparable ordering.
        vals = [_freeze_value(v) for v in value]
        return tuple(sorted(vals, key=lambda item: _canonical_json(item)))
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _thaw_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(v) for v in value]
    return value


def _reject_secret_fields(value: Any, *, path: str = "extras") -> None:
    """Prevent credentials from entering a persisted runtime manifest."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).strip().lower()
            if key_text in _SECRET_FIELD_NAMES:
                raise ManifestValidationError(
                    f"runtime {path}.{key_text} looks like a secret; store only a digest"
                )
            _reject_secret_fields(item, path=f"{path}.{key_text}")
    elif isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            _reject_secret_fields(item, path=f"{path}[{index}]")


def _canonical_value(value: Any) -> Any:
    """Convert a JSON-like value into a deterministic JSON representation."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestValidationError("NaN and infinity are not valid manifest data")
        # JSON distinguishes -0.0 in some implementations; semantic specs do
        # not, so normalise it before hashing.
        return 0.0 if value == 0 else value
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise ManifestValidationError(
                    f"manifest mapping key {key!r} is not JSON-compatible"
                )
            canonical_key = str(key)
            if canonical_key in result:
                raise ManifestValidationError(
                    f"manifest mapping has colliding keys after JSON canonicalisation: {canonical_key!r}"
                )
            result[canonical_key] = _canonical_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        vals = [_canonical_value(item) for item in value]
        return sorted(vals, key=lambda item: _canonical_json(item))
    # Explicit conversion hooks are preferred over repr(), whose address and
    # process state make a digest non-reproducible.
    for method_name in ("to_manifest", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            converted = method()
            if converted is value:
                break
            return _canonical_value(converted)
    raise ManifestValidationError(
        f"unsupported manifest value type {type(value).__module__}.{type(value).__qualname__}; "
        "provide a mapping, bytes, path, or explicit digest"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _looks_like_path(value: str) -> bool:
    suffixes = (
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".txt",
        ".md",
        ".prompt",
        ".py",
    )
    return (
        os.sep in value
        or (os.altsep is not None and os.altsep in value)
        or value.startswith(".")
        or value.endswith(suffixes)
    )


def _hash_file(path: Path) -> str:
    if not path.exists():
        raise ManifestMismatchError(f"manifest source file is missing: {path}")
    if not path.is_file():
        raise ManifestMismatchError(f"manifest source is not a regular file: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ManifestMismatchError(f"cannot read manifest source {path}: {exc}") from exc

    # JSON formatting is not part of a registry's semantics.  Canonicalising
    # valid JSON lets a harmless reformat retain the same world contract while
    # still detecting every semantic edit.  Other files are byte-addressed.
    if path.suffix.lower() == ".json":
        try:
            def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, value in pairs:
                    if key in result:
                        raise ManifestValidationError(
                            f"JSON source {path} contains duplicate key {key!r}"
                        )
                    result[key] = value
                return result

            parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
            return _sha256(_canonical_json(parsed).encode("utf-8"))
        except UnicodeDecodeError:
            pass
        except json.JSONDecodeError:
            pass
    return _sha256(raw)


def _callable_payload(value: Callable[..., Any]) -> Mapping[str, Any]:
    """Return stable callable material, refusing opaque callables."""

    code = getattr(value, "__code__", None)
    try:
        source = inspect.getsource(value)
    except (OSError, IOError, TypeError):
        source = None
    payload: dict[str, Any] = {
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
    }
    if source is not None:
        payload["source"] = inspect.cleandoc(source)
    if code is not None:
        payload.update(
            {
                "bytecode_b64": base64.b64encode(code.co_code).decode("ascii"),
                "consts": _canonical_value(code.co_consts),
                "names": list(code.co_names),
                "defaults": _canonical_value(getattr(value, "__defaults__", None)),
                "kwdefaults": _canonical_value(getattr(value, "__kwdefaults__", None)),
            }
        )
        closure = getattr(value, "__closure__", None)
        if closure:
            cells = []
            for cell in closure:
                try:
                    cells.append(_canonical_value(cell.cell_contents))
                except ValueError as exc:
                    raise ManifestValidationError("callable has an empty closure cell") from exc
            payload["closure"] = cells
    if source is None and code is None:
        raise ManifestValidationError(
            f"opaque callable {value!r} cannot be hashed safely; pass an explicit digest"
        )
    return payload


def hash_source(source: Any) -> str:
    """Return a deterministic SHA-256 digest for a manifest source.

    Paths and path-like strings are content-addressed.  JSON files are
    canonicalised; all other files are hashed byte-for-byte.  Unsupported
    objects raise instead of falling back to ``repr``.
    """

    if source is _MISSING or source is None:
        raise ManifestValidationError("manifest source is missing")
    if isinstance(source, Path):
        return _hash_file(source)
    if isinstance(source, os.PathLike):
        return _hash_file(Path(source))
    if isinstance(source, str):
        candidate = Path(source)
        if candidate.exists():
            return _hash_file(candidate)
        if _looks_like_path(source):
            raise ManifestMismatchError(f"manifest source file is missing: {source}")
        return _sha256(source.encode("utf-8"))
    if isinstance(source, bytes):
        return _sha256(source)
    if callable(source):
        return _sha256(_canonical_json(_callable_payload(source)).encode("utf-8"))
    return _sha256(_canonical_json(source).encode("utf-8"))


def _source_ref(source: Any) -> str | None:
    if isinstance(source, Path):
        return str(source.resolve())
    if isinstance(source, os.PathLike):
        return str(Path(source).resolve())
    if isinstance(source, str) and Path(source).exists():
        return str(Path(source).resolve())
    return None


def _read_runtime_defaults() -> dict[str, Any]:
    """读取当前任务生成运行时配置。"""

    try:
        generation_runtime = importlib.import_module(
            "evosynth.task_gen.generation.runtime")
        configured = generation_runtime.runtime_config()
        if configured.agent_api_key and configured.world_api_key:
            return {
                "agent_url": configured.agent_base_url,
                "agent_model": configured.agent_model,
                "world_model_url": configured.world_base_url,
            }
    except (ImportError, AttributeError) as exc:
        raise ManifestValidationError(
            "task generation runtime is not configured"
        ) from exc
    raise ManifestValidationError("task generation runtime is not configured")


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """The network/model settings under which a world is interpreted."""

    agent_url: str
    agent_model: str
    world_model_url: str
    world_model: str | None = None
    extras: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("agent_url", "agent_model", "world_model_url"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ManifestValidationError(f"runtime.{field_name} must be a non-empty string")
        if self.world_model is not None and not isinstance(self.world_model, str):
            raise ManifestValidationError("runtime.world_model must be a string or null")
        if not isinstance(self.extras, Mapping):
            raise ManifestValidationError("runtime.extras must be a mapping")
        _reject_secret_fields(self.extras)
        object.__setattr__(self, "extras", _freeze_value(dict(self.extras)))

    @classmethod
    def capture(
        cls,
        *,
        agent_url: str | None = None,
        agent_model: str | None = None,
        world_model_url: str | None = None,
        world_model: str | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> "RuntimeConfig":
        defaults = _read_runtime_defaults()
        return cls(
            agent_url=defaults["agent_url"] if agent_url is None else agent_url,
            agent_model=defaults["agent_model"] if agent_model is None else agent_model,
            world_model_url=(
                defaults["world_model_url"] if world_model_url is None else world_model_url
            ),
            world_model=world_model,
            extras={} if extras is None else extras,
        )

    from_current = capture

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RuntimeConfig":
        if not isinstance(data, Mapping):
            raise ManifestValidationError("runtime must be an object")
        # Accept the explicit ``agent_api_url`` spelling when reading a
        # hand-authored manifest, while serialisation always uses agent_url.
        agent_url = data.get("agent_url", data.get("agent_api_url"))
        extras = data.get("extras", {})
        if extras is None:
            extras = {}
        return cls(
            agent_url=agent_url,
            agent_model=data.get("agent_model"),
            world_model_url=data.get("world_model_url"),
            world_model=data.get("world_model"),
            extras=extras,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "agent_url": self.agent_url,
            "agent_model": self.agent_model,
            "world_model_url": self.world_model_url,
            "world_model": self.world_model,
        }
        if self.extras:
            result["extras"] = _thaw_value(self.extras)
        else:
            result["extras"] = {}
        return result

    def digest(self) -> str:
        return _sha256(_canonical_json(self.to_dict()).encode("utf-8"))

    @property
    def hash(self) -> str:
        return self.digest()

    @property
    def api_url(self) -> str:
        return self.agent_url

    def verify_current(self, *, raise_on_error: bool = True) -> bool:
        try:
            current = RuntimeConfig.capture()
            if self.to_dict() != current.to_dict():
                raise ManifestMismatchError(
                    "runtime configuration differs from the pinned pipeline constants"
                )
            return True
        except (ManifestError, ImportError, OSError) as exc:
            if raise_on_error:
                if isinstance(exc, ManifestError):
                    raise
                raise ManifestMismatchError(str(exc)) from exc
            return False


@dataclasses.dataclass(frozen=True, slots=True)
class WorldManifest:
    """An immutable, content-addressed world contract."""

    schema_version: str
    world_id: str
    world_revision: str
    world_revision_hash: str
    registry_hash: str
    semantic_spec_hash: str
    normalizer_hash: str
    renderer_hash: str
    prompt_hash: str
    model_hash: str
    runtime: RuntimeConfig
    runtime_hash: str
    source_refs: Mapping[str, str] = dataclasses.field(default_factory=dict)
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    integrity_hash: str = ""

    SCHEMA_VERSION: ClassVar[str] = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ManifestValidationError(
                f"unsupported manifest schema {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        for field_name in ("world_id", "world_revision"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ManifestValidationError(f"{field_name} must be a non-empty string")
        for field_name in (
            "world_revision_hash",
            "registry_hash",
            "semantic_spec_hash",
            "normalizer_hash",
            "renderer_hash",
            "prompt_hash",
            "model_hash",
            "runtime_hash",
        ):
            object.__setattr__(self, field_name, _check_digest(getattr(self, field_name), field_name))
        if not isinstance(self.runtime, RuntimeConfig):
            raise ManifestValidationError("runtime must be a RuntimeConfig")
        if self.runtime_hash != self.runtime.digest():
            raise ManifestValidationError("runtime_hash does not match runtime")
        if not isinstance(self.source_refs, Mapping):
            raise ManifestValidationError("source_refs must be a mapping")
        if not isinstance(self.metadata, Mapping):
            raise ManifestValidationError("metadata must be a mapping")
        object.__setattr__(self, "source_refs", _freeze_value(dict(self.source_refs)))
        object.__setattr__(self, "metadata", _freeze_value(dict(self.metadata)))
        calculated = _sha256(_canonical_json(self._payload_dict()).encode("utf-8"))
        if self.integrity_hash:
            supplied = _check_digest(self.integrity_hash, "integrity_hash")
            if supplied != calculated:
                raise ManifestValidationError("manifest integrity hash is invalid")
            object.__setattr__(self, "integrity_hash", supplied)
        else:
            object.__setattr__(self, "integrity_hash", calculated)

    def _payload_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "world_id": self.world_id,
            "world_revision": self.world_revision,
            "world_revision_hash": self.world_revision_hash,
            "registry_hash": self.registry_hash,
            "semantic_spec_hash": self.semantic_spec_hash,
            "normalizer_hash": self.normalizer_hash,
            "renderer_hash": self.renderer_hash,
            "prompt_hash": self.prompt_hash,
            "model_hash": self.model_hash,
            "runtime": self.runtime.to_dict(),
            "runtime_hash": self.runtime_hash,
            "source_refs": _thaw_value(self.source_refs),
            "metadata": _thaw_value(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._payload_dict()
        result["integrity_hash"] = self.integrity_hash
        return result

    as_dict = to_dict

    @classmethod
    def capture(
        cls,
        *,
        registry: Any = _MISSING,
        semantic_spec: Any = _MISSING,
        normalizer: Any = _MISSING,
        renderer: Any = _MISSING,
        prompt: Any = _MISSING,
        model: Any = _MISSING,
        world_revision: Any = "genesis",
        world_id: str = EVOSYNTH_VERSION,
        runtime: RuntimeConfig | Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        allow_unset: bool = False,
        # Explicit paths are useful to callers that want the source kind to be
        # obvious.  They are aliases, not map-specific behaviour.
        registry_path: os.PathLike[str] | str | None = None,
        semantic_spec_path: os.PathLike[str] | str | None = None,
        normalizer_path: os.PathLike[str] | str | None = None,
        renderer_path: os.PathLike[str] | str | None = None,
        prompt_path: os.PathLike[str] | str | None = None,
        model_path: os.PathLike[str] | str | None = None,
        registry_hash: str | None = None,
        semantic_spec_hash: str | None = None,
        normalizer_hash: str | None = None,
        renderer_hash: str | None = None,
        prompt_hash: str | None = None,
        model_hash: str | None = None,
        world_revision_hash: str | None = None,
    ) -> "WorldManifest":
        runtime_obj = runtime
        if runtime_obj is None:
            runtime_obj = RuntimeConfig.capture()
        elif isinstance(runtime_obj, Mapping):
            runtime_obj = RuntimeConfig.from_dict(runtime_obj)
        elif not isinstance(runtime_obj, RuntimeConfig):
            raise ManifestValidationError("runtime must be RuntimeConfig or a mapping")

        artifact_values: dict[str, Any] = {
            "registry": registry,
            "semantic_spec": semantic_spec,
            "normalizer": normalizer,
            "renderer": renderer,
            "prompt": prompt,
            "model": model,
        }
        artifact_paths = {
            "registry": registry_path,
            "semantic_spec": semantic_spec_path,
            "normalizer": normalizer_path,
            "renderer": renderer_path,
            "prompt": prompt_path,
            "model": model_path,
        }
        explicit_hashes: dict[str, str | None] = {
            "registry": registry_hash,
            "semantic_spec": semantic_spec_hash,
            "normalizer": normalizer_hash,
            "renderer": renderer_hash,
            "prompt": prompt_hash,
            "model": model_hash,
        }

        # The current prompt is a stable default because it is part of the
        # existing pipeline contract.  Other semantic inputs must be supplied
        # explicitly; silently hashing an empty placeholder would make a
        # manifest appear valid while omitting the actual implementation.
        if artifact_values["prompt"] is _MISSING and artifact_paths["prompt"] is not None:
            artifact_values["prompt"] = artifact_paths["prompt"]
        if artifact_values["prompt"] is _MISSING:
            try:
                old_pipeline = importlib.import_module("evosynth.task_gen.world.constants")
                discovered = getattr(old_pipeline, "WM_PROMPT_PATH", None)
                if discovered is not None and Path(discovered).exists():
                    artifact_values["prompt"] = discovered
            except (ImportError, AttributeError):
                pass

        # A model source defaults to the pinned runtime descriptor.  This is a
        # deterministic local value and does not query either model endpoint.
        if artifact_values["model"] is _MISSING and artifact_paths["model"] is not None:
            artifact_values["model"] = artifact_paths["model"]
        if artifact_values["model"] is _MISSING:
            artifact_values["model"] = {
                "agent_model": runtime_obj.agent_model,
                "world_model": runtime_obj.world_model,
                "agent_url": runtime_obj.agent_url,
                "world_model_url": runtime_obj.world_model_url,
            }

        source_refs: dict[str, str] = {}
        digests: dict[str, str] = {}
        for name, value in artifact_values.items():
            path_alias = artifact_paths[name]
            if value is _MISSING and path_alias is not None:
                value = path_alias
            override = explicit_hashes[name]
            if override is not None:
                digests[name] = _check_digest(override, f"{name}_hash")
            elif value is _MISSING or value is None:
                if not allow_unset:
                    raise ManifestValidationError(
                        f"{name} source is required; pass allow_unset=True only for a deliberate opaque contract"
                    )
                digests[name] = _sha256(f"__unset__:{name}".encode("utf-8"))
            else:
                digests[name] = hash_source(value)
                ref = _source_ref(value)
                if ref is not None:
                    source_refs[name] = ref

        if world_revision_hash is not None:
            revision_digest = _check_digest(world_revision_hash, "world_revision_hash")
        else:
            revision_digest = hash_source(world_revision)
            revision_ref = _source_ref(world_revision)
            if revision_ref is not None:
                source_refs["world_revision"] = revision_ref

        return cls(
            schema_version=SCHEMA_VERSION,
            world_id=world_id,
            world_revision=(
                str(world_revision)
                if not isinstance(world_revision, (Path, os.PathLike))
                else str(Path(world_revision))
            ),
            world_revision_hash=revision_digest,
            registry_hash=digests["registry"],
            semantic_spec_hash=digests["semantic_spec"],
            normalizer_hash=digests["normalizer"],
            renderer_hash=digests["renderer"],
            prompt_hash=digests["prompt"],
            model_hash=digests["model"],
            runtime=runtime_obj,
            runtime_hash=runtime_obj.digest(),
            source_refs=source_refs,
            metadata={} if metadata is None else metadata,
        )

    from_sources = capture

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorldManifest":
        if not isinstance(data, Mapping):
            raise ManifestIOError("manifest JSON root must be an object")
        schema = data.get("schema_version")
        if schema != SCHEMA_VERSION:
            raise ManifestIOError(
                f"unsupported or missing manifest schema {schema!r}; refusing to load"
            )
        runtime_data = data.get("runtime")
        if runtime_data is None:
            runtime_data = {
                key: data.get(key)
                for key in ("agent_url", "agent_model", "world_model_url", "world_model", "extras")
                if key in data
            }
        # Require all fields rather than filling defaults from the live
        # process.  This is the key fail-closed property for old/corrupt files.
        required = (
            "world_id",
            "world_revision",
            "world_revision_hash",
            "registry_hash",
            "semantic_spec_hash",
            "normalizer_hash",
            "renderer_hash",
            "prompt_hash",
            "model_hash",
            "runtime_hash",
            "integrity_hash",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise ManifestIOError(f"manifest is missing required fields: {', '.join(missing)}")
        return cls(
            schema_version=schema,
            world_id=data["world_id"],
            world_revision=data["world_revision"],
            world_revision_hash=data["world_revision_hash"],
            registry_hash=data["registry_hash"],
            semantic_spec_hash=data["semantic_spec_hash"],
            normalizer_hash=data["normalizer_hash"],
            renderer_hash=data["renderer_hash"],
            prompt_hash=data["prompt_hash"],
            model_hash=data["model_hash"],
            runtime=RuntimeConfig.from_dict(runtime_data),
            runtime_hash=data["runtime_hash"],
            source_refs=data.get("source_refs", {}),
            metadata=data.get("metadata", {}),
            integrity_hash=data["integrity_hash"],
        )

    def freeze(self) -> "WorldManifest":
        """Return this already-frozen manifest (convenient builder boundary)."""

        return self

    @property
    def hash(self) -> str:
        return self.integrity_hash

    @property
    def manifest_hash(self) -> str:
        return self.integrity_hash

    # Digest aliases keep integration code readable and make the serialized
    # names stable even if a caller prefers ``*_digest`` terminology.
    registry_digest = property(lambda self: self.registry_hash)
    semantic_digest = property(lambda self: self.semantic_spec_hash)
    semantic_spec_digest = property(lambda self: self.semantic_spec_hash)
    normalizer_digest = property(lambda self: self.normalizer_hash)
    renderer_digest = property(lambda self: self.renderer_hash)
    prompt_digest = property(lambda self: self.prompt_hash)
    model_digest = property(lambda self: self.model_hash)
    world_revision_digest = property(lambda self: self.world_revision_hash)

    def save(self, path: os.PathLike[str] | str) -> Path:
        """Atomically persist the manifest, preventing zero-byte snapshots."""

        target = Path(path)
        if target.exists() and target.is_dir():
            raise ManifestIOError(f"manifest target is a directory: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    self.to_dict(),
                    stream,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except (OSError, TypeError, ValueError) as exc:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise ManifestIOError(f"cannot atomically save manifest {target}: {exc}") from exc
        return target

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "WorldManifest":
        target = Path(path)
        try:
            raw = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ManifestIOError(f"cannot read manifest {target}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManifestIOError(f"manifest is not valid JSON: {target}") from exc
        return cls.from_dict(data)

    def verify(
        self,
        *,
        registry: Any = _MISSING,
        semantic_spec: Any = _MISSING,
        normalizer: Any = _MISSING,
        renderer: Any = _MISSING,
        prompt: Any = _MISSING,
        model: Any = _MISSING,
        world_revision: Any = _MISSING,
        runtime: RuntimeConfig | Mapping[str, Any] | None = None,
        check_runtime: bool = True,
        require_sources: bool = False,
        raise_on_error: bool = True,
    ) -> bool:
        """Verify integrity and any supplied/current sources.

        ``verify`` always checks the manifest's own integrity and runtime (by
        default).  Artifact sources are taken from explicit arguments or the
        path references captured at creation.  Set ``require_sources=True`` to
        reject an inline manifest when callers did not provide its sources.
        """

        issues: list[str] = []
        try:
            expected_integrity = _sha256(_canonical_json(self._payload_dict()).encode("utf-8"))
            if expected_integrity != self.integrity_hash:
                issues.append("manifest integrity hash mismatch")
        except ManifestError as exc:
            issues.append(str(exc))

        if check_runtime:
            try:
                runtime_obj: RuntimeConfig
                if runtime is None:
                    runtime_obj = RuntimeConfig.capture()
                elif isinstance(runtime, Mapping):
                    runtime_obj = RuntimeConfig.from_dict(runtime)
                elif isinstance(runtime, RuntimeConfig):
                    runtime_obj = runtime
                else:
                    raise ManifestValidationError("runtime must be RuntimeConfig or a mapping")
                if runtime_obj.digest() != self.runtime_hash:
                    issues.append("runtime configuration digest mismatch")
            except ManifestError as exc:
                issues.append(str(exc))

        supplied: dict[str, Any] = {
            "registry": registry,
            "semantic_spec": semantic_spec,
            "normalizer": normalizer,
            "renderer": renderer,
            "prompt": prompt,
            "model": model,
            "world_revision": world_revision,
        }
        expected: dict[str, str] = {
            "registry": self.registry_hash,
            "semantic_spec": self.semantic_spec_hash,
            "normalizer": self.normalizer_hash,
            "renderer": self.renderer_hash,
            "prompt": self.prompt_hash,
            "model": self.model_hash,
            "world_revision": self.world_revision_hash,
        }
        for name, value in supplied.items():
            if value is _MISSING:
                ref = self.source_refs.get(name)
                if ref:
                    value = Path(ref)
                elif require_sources:
                    issues.append(f"source for {name} was not supplied")
                    continue
                else:
                    continue
            try:
                actual = hash_source(value)
            except ManifestError as exc:
                issues.append(f"{name}: {exc}")
                continue
            if actual != expected[name]:
                issues.append(
                    f"{name} digest mismatch (expected {expected[name]}, got {actual})"
                )

        if issues:
            if raise_on_error:
                raise ManifestMismatchError("; ".join(issues))
            return False
        return True

    def verify_or_raise(self, **kwargs: Any) -> None:
        self.verify(**kwargs, raise_on_error=True)

    def is_valid(self, **kwargs: Any) -> bool:
        kwargs["raise_on_error"] = False
        try:
            return self.verify(**kwargs)
        except ManifestError:
            return False

    def assert_compatible(self, other: "WorldManifest") -> None:
        if not isinstance(other, WorldManifest):
            raise ManifestMismatchError("can only compare a WorldManifest to another WorldManifest")
        if self.integrity_hash != other.integrity_hash:
            raise ManifestMismatchError(
                f"world manifests differ: {self.integrity_hash} != {other.integrity_hash}"
            )


def capture(**kwargs: Any) -> WorldManifest:
    """Functional convenience wrapper around :meth:`WorldManifest.capture`."""

    return WorldManifest.capture(**kwargs)


def freeze(value: Any = _MISSING, **kwargs: Any) -> WorldManifest:
    """Freeze an existing manifest/mapping or capture one from sources."""

    if isinstance(value, WorldManifest):
        return value
    if isinstance(value, Mapping) and not kwargs:
        return WorldManifest.from_dict(value)
    if value is not _MISSING:
        kwargs.setdefault("registry", value)
    return WorldManifest.capture(**kwargs)


def load(path: os.PathLike[str] | str) -> WorldManifest:
    return WorldManifest.load(path)


def verify(manifest: WorldManifest, **kwargs: Any) -> bool:
    if not isinstance(manifest, WorldManifest):
        raise ManifestValidationError("verify expects a WorldManifest")
    return manifest.verify(**kwargs)


__all__ = [
    "SCHEMA_VERSION",
    "ManifestError",
    "ManifestValidationError",
    "ManifestMismatchError",
    "ManifestIOError",
    "RuntimeConfig",
    "WorldManifest",
    "hash_source",
    "capture",
    "freeze",
    "load",
    "verify",
]
