"""Create and verify immutable EvoSynth task/world bundles.

The bundle is the boundary between synthesis and inference.  Every source
that can change semantic identity is copied or content-addressed, and all
paths inside a revision are relative to the bundle root so a bundle can be
relocated safely.  The legacy pipeline, endpoint constants, and model ports
are deliberately left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from .fingerprint import canonical_fingerprint, fingerprint
from .manifest import RuntimeConfig, WorldManifest, hash_source
from ..version import EVOSYNTH_VERSION, WORLD_REVISION_SCHEMA
from evosynth.task_gen.generation.tool_info import dumps_compact


def file_hash(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BundleError(RuntimeError):
    """Raised when an immutable bundle is incomplete or has drifted."""


_REVISION_SCHEMA = WORLD_REVISION_SCHEMA


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _inside(root: Path, path: str | Path, *, field: str) -> Path:
    """Resolve a bundle-relative path and reject traversal/symlink escape."""

    root_real = _resolved(root)
    raw = Path(path)
    if raw.is_absolute():
        # Absolute paths in a revision are not relocatable and can point at a
        # mutable live journal.  Refuse them even when currently inside root.
        raise BundleError(f"{field} must be relative to bundle root: {path}")
    candidate = (root_real / raw).resolve()
    try:
        candidate.relative_to(root_real)
    except ValueError as exc:
        raise BundleError(f"{field} escapes bundle root: {path}") from exc
    return candidate


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n",
            encoding="utf-8",
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _copy_source(src: Path, dst: Path) -> Path:
    source = _resolved(src)
    if not source.exists() or not source.is_file():
        raise BundleError(f"source does not exist or is not a file: {src}")
    destination = Path(dst)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination.resolve():
        raise BundleError(f"bundle source and destination are identical: {src}")
    shutil.copy2(source, destination)
    return destination


def _model_descriptor(runtime: RuntimeConfig) -> dict[str, Any]:
    return {
        "agent_model": runtime.agent_model,
        "world_model": runtime.world_model,
        "agent_url": runtime.agent_url,
        "world_model_url": runtime.world_model_url,
    }


def _framework_payload() -> dict[str, Any]:
    # Keep both the per-file map and aggregate digest in the bundle.  The map
    # makes drift diagnostics actionable; the manifest hashes the full value.
    payload = fingerprint()
    payload["aggregate_sha256"] = canonical_fingerprint()
    return payload


@dataclass(frozen=True)
class PinnedBundle:
    root: Path
    manifest: WorldManifest
    registry_path: Path
    semantic_spec_path: Path
    world_revision_path: Path
    prompt_path: Path
    framework_sources_path: Path

    def _read_revision(self) -> Mapping[str, Any]:
        try:
            revision = json.loads(self.world_revision_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BundleError(f"invalid world revision file: {exc}") from exc
        if not isinstance(revision, Mapping) or revision.get("schema_version") != _REVISION_SCHEMA:
            raise BundleError("world revision has an unsupported or missing schema")
        if str(revision.get("world_id", "")) != self.manifest.world_id:
            raise BundleError("world revision world_id does not match manifest")
        records = revision.get("files")
        if not isinstance(records, list):
            raise BundleError("world revision files must be a list")
        return revision

    def _verify_revision_files(self, revision: Mapping[str, Any]) -> None:
        seen: set[str] = set()
        for item in revision["files"]:
            if not isinstance(item, Mapping):
                raise BundleError("world revision file record must be an object")
            relative = item.get("path")
            digest = item.get("sha256")
            if not isinstance(relative, str) or not relative or relative in seen:
                raise BundleError(f"invalid or duplicate world revision path: {relative!r}")
            seen.add(relative)
            if not isinstance(digest, str) or len(digest) != 64:
                raise BundleError(f"invalid world revision digest for {relative}")
            source = _inside(self.root, relative, field="world revision path")
            if not source.is_file():
                raise BundleError(f"world revision source is missing: {relative}")
            if file_hash(source) != digest.lower():
                raise BundleError(f"world revision source drift: {relative}")
            size = item.get("size")
            if size is not None and (not isinstance(size, int) or size != source.stat().st_size):
                raise BundleError(f"world revision size mismatch: {relative}")

        # Contract/world source directories are closed sets.  An undeclared
        # file could otherwise be imported by an adapter even though it is not
        # covered by the manifest hash.
        for directory in ("world_inputs", "contract_sources"):
            source_root = self.root / directory
            if not source_root.exists():
                continue
            if not source_root.is_dir():
                raise BundleError(f"bundle source path is not a directory: {directory}")
            for path in source_root.rglob("*"):
                if path.is_file() or path.is_symlink():
                    relative = path.relative_to(self.root).as_posix()
                    if relative not in seen:
                        raise BundleError(f"undeclared bundle source: {relative}")

    def verify(self, *, check_runtime: bool = True,
               check_framework: bool = True) -> None:
        required_files = (
            self.registry_path,
            self.semantic_spec_path,
            self.world_revision_path,
            self.prompt_path,
            self.framework_sources_path,
        )
        for path in required_files:
            if not path.is_file():
                raise BundleError(f"bundle file is missing: {path}")

        current_framework = _framework_payload()
        try:
            recorded_framework = json.loads(
                self.framework_sources_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BundleError(f"invalid framework source snapshot: {exc}") from exc
        if check_framework and recorded_framework != current_framework:
            raise BundleError("EvoSynth framework source drift")

        model = _model_descriptor(self.manifest.runtime)
        # Infer may run after later serialization-only edits.  World integrity
        # still checks against the snapshot pinned at generation time.
        framework = current_framework if check_framework else recorded_framework
        self.manifest.verify(
            registry=self.registry_path,
            semantic_spec=self.semantic_spec_path,
            normalizer=framework,
            renderer=framework,
            prompt=self.prompt_path,
            model=model,
            world_revision=self.world_revision_path,
            check_runtime=check_runtime,
            require_sources=True,
        )
        revision = self._read_revision()
        self._verify_revision_files(revision)


def pin_bundle(*, root: str | Path, registry_path: str | Path,
               semantic_spec: Mapping[str, Any] | list[Any],
               world_files: Iterable[str | Path] = (),
               contract_files: Iterable[str | Path] = (),
               prompt_path: str | Path | None = None,
               world_id: str = EVOSYNTH_VERSION,
               world_revision: str = "genesis",
               metadata: Mapping[str, Any] | None = None) -> PinnedBundle:
    """Snapshot all semantic inputs needed by a generation/inference run."""

    if metadata is not None and not isinstance(metadata, Mapping):
        raise BundleError("bundle metadata must be an object")
    manifest_metadata = dict(metadata or {})
    reserved_metadata = {"framework_source_schema", "framework_source_sha256"}
    collisions = reserved_metadata.intersection(manifest_metadata)
    if collisions:
        raise BundleError(
            "bundle metadata uses reserved framework fields: "
            + ", ".join(sorted(collisions))
        )

    root = _resolved(root)
    root.mkdir(parents=True, exist_ok=True)
    registry_dst = _copy_source(Path(registry_path), root / "registry.snapshot.json")
    spec_dst = root / "semantic_spec.json"
    spec_dst.parent.mkdir(parents=True, exist_ok=True)
    spec_dst.write_text(dumps_compact(semantic_spec) + "\n", encoding="utf-8")

    # Pin the existing prompt and capture its endpoint/model settings without
    # creating a client or changing any API/port constants.
    if prompt_path is not None:
        prompt_src = Path(prompt_path)
    else:
        try:
            from evosynth.task_gen.world.constants import WM_PROMPT_PATH
            prompt_src = Path(WM_PROMPT_PATH)
        except (ImportError, AttributeError) as exc:
            raise BundleError(
                f"cannot locate the existing world-model prompt: {exc}") from exc
    prompt_dst = _copy_source(prompt_src, root / "prompt.snapshot.txt")

    revision_records: list[dict[str, Any]] = []

    def snapshot_sources(sources: Iterable[str | Path], *, kind: str, directory: str) -> None:
        for index, source in enumerate(tuple(sources)):
            source_path = _resolved(source)
            if not source_path.exists() or not source_path.is_file():
                # A declared input that disappears is a broken world, not an
                # optional file.  The old implementation silently skipped it.
                raise BundleError(f"{kind} revision source is missing: {source}")
            relative = Path(directory) / f"{index:04d}_{source_path.name}"
            copied = _copy_source(source_path, root / relative)
            revision_records.append({
                "path": relative.as_posix(),
                "kind": kind,
                # Keep only a non-authoritative label; an absolute live path
                # would make the relocatable bundle depend on its creator's
                # filesystem layout.
                "source_name": source_path.name,
                "sha256": file_hash(copied),
                "size": copied.stat().st_size,
            })

    snapshot_sources(world_files, kind="world", directory="world_inputs")
    # Executable tool lenses are versioned contract-package inputs, not core
    # framework code.  Snapshot them separately so custom adapters cannot
    # drift while the generic runtime fingerprint remains unchanged.
    snapshot_sources(contract_files, kind="contract", directory="contract_sources")

    revision_dst = root / "world_revision.json"
    _write_json(revision_dst, {
        "schema_version": _REVISION_SCHEMA,
        "label": str(world_revision),
        "world_id": str(world_id),
        "files": revision_records,
    })
    framework = _framework_payload()
    framework_dst = root / "framework_sources.json"
    _write_json(framework_dst, framework)

    runtime = RuntimeConfig.capture()
    model = _model_descriptor(runtime)
    manifest_metadata.update({
        "framework_source_schema": framework.get("schema_version"),
        "framework_source_sha256": framework.get("aggregate_sha256"),
    })
    manifest = WorldManifest.capture(
        world_id=str(world_id),
        world_revision="world_revision.json",
        world_revision_hash=hash_source(revision_dst),
        registry_hash=hash_source(registry_dst),
        semantic_spec_hash=hash_source(spec_dst),
        normalizer_hash=hash_source(framework),
        renderer_hash=hash_source(framework),
        prompt_hash=hash_source(prompt_dst),
        model_hash=hash_source(model),
        runtime=runtime,
        metadata=manifest_metadata,
        allow_unset=False,
    )
    manifest.save(root / "world_manifest.json")
    bundle = PinnedBundle(
        root=root,
        manifest=manifest,
        registry_path=root / "registry.snapshot.json",
        semantic_spec_path=root / "semantic_spec.json",
        world_revision_path=revision_dst,
        prompt_path=prompt_dst,
        framework_sources_path=framework_dst,
    )
    bundle.verify()
    return bundle


def load_bundle(root: str | Path, *, check_runtime: bool = True,
                check_framework: bool = True) -> PinnedBundle:
    root = _resolved(root)
    manifest = WorldManifest.load(root / "world_manifest.json")
    bundle = PinnedBundle(
        root=root,
        manifest=manifest,
        registry_path=root / "registry.snapshot.json",
        semantic_spec_path=root / "semantic_spec.json",
        world_revision_path=root / "world_revision.json",
        prompt_path=root / "prompt.snapshot.txt",
        framework_sources_path=root / "framework_sources.json",
    )
    bundle.verify(check_runtime=check_runtime, check_framework=check_framework)
    return bundle


__all__ = ["BundleError", "PinnedBundle", "load_bundle", "pin_bundle"]
