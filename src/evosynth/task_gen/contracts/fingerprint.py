"""Content fingerprints for the executable EvoSynth semantic framework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..version import FRAMEWORK_SOURCE_SCHEMA


# Every module that can change query identity, materialisation, task meaning,
# or verification is covered.  Keeping this list data-only makes the bundle
# boundary auditable and avoids hashing just the caller that happened to pin a
# bundle.
FRAMEWORK_FILES: tuple[str, ...] = (
    "__init__.py",
    "contracts/compat.py",
    "contracts/fingerprint.py",
    "contracts/manifest.py",
    "contracts/task_bundle.py",
    "contracts/task_contract.py",
    "contracts/versioned_bundle.py",
    "validation/semantic/declarative_lens.py",
    "validation/semantic/environment.py",
    "validation/semantic/runtime.py",
    "validation/semantic/spec.py",
    "validation/semantic/store.py",
    "version.py",
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(*, root: Path | None = None) -> dict[str, Any]:
    base = (root or _root()).resolve()
    files: dict[str, str] = {}
    for name in FRAMEWORK_FILES:
        path = base / name
        if not path.is_file():
            raise FileNotFoundError(f"missing framework source: {path}")
        files[name] = _file_digest(path)
    return {"schema_version": FRAMEWORK_SOURCE_SCHEMA, "files": files}


def canonical_fingerprint(*, root: Path | None = None) -> str:
    payload = json.dumps(fingerprint(root=root), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["FRAMEWORK_FILES", "fingerprint", "canonical_fingerprint"]
