"""Persist a task together with its world manifest id."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .task_contract import TaskContract
from .manifest import ManifestMismatchError, WorldManifest
from ..version import EVOSYNTH_VERSION


@dataclass(frozen=True)
class TaskBundle:
    task_id: str
    task_natural: str
    world_manifest_hash: str
    world_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    task_contract: TaskContract | None = None
    reference_witness: tuple[Mapping[str, Any], ...] = ()
    pipeline_version: str = EVOSYNTH_VERSION

    @classmethod
    def create(cls, *, task_id: str, task_natural: str,
               manifest: WorldManifest, metadata: Mapping[str, Any] | None = None,
               reference_witness: tuple[Mapping[str, Any], ...] = ()) -> "TaskBundle":
        if not str(task_id).strip():
            raise ValueError("task_id is required")
        if not task_natural.strip():
            raise ValueError("task_natural is required")
        payload = dict(metadata or {})
        answer = payload.get("correct_answer")
        if not isinstance(answer, Mapping) or not str(answer.get("answer") or "").strip():
            raise ValueError("correct_answer is required")
        return cls(
            task_id=str(task_id),
            task_natural=task_natural,
            world_manifest_hash=manifest.manifest_hash,
            world_id=manifest.world_id,
            metadata=payload,
            reference_witness=tuple(reference_witness),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_version": self.pipeline_version,
            "task_id": self.task_id,
            "task_natural": self.task_natural,
            "world_manifest_hash": self.world_manifest_hash,
            "world_id": self.world_id,
            "metadata": dict(self.metadata),
        }

    def solver_payload(self, *, tool_schemas: Any = None,
                       answer_schema: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return the public task view safe to place in solver messages."""

        return {
            "task_id": self.task_id,
            "task_natural": self.task_natural,
            "tool_schemas": json.loads(json.dumps(tool_schemas or [], ensure_ascii=False)),
            "answer_schema": json.loads(json.dumps(answer_schema or {}, ensure_ascii=False)),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskBundle":
        pipeline_version = value.get("pipeline_version")
        if pipeline_version != EVOSYNTH_VERSION:
            raise ValueError(
                "unsupported EvoSynth task bundle version "
                f"{pipeline_version!r}; expected {EVOSYNTH_VERSION!r}"
            )
        raw_contract = value.get("task_contract")
        return cls(
            task_id=str(value["task_id"]),
            task_natural=str(value["task_natural"]),
            world_manifest_hash=str(value["world_manifest_hash"]),
            world_id=str(value["world_id"]),
            metadata=dict(value.get("metadata") or {}),
            task_contract=(
                TaskContract.from_dict(raw_contract) if raw_contract else None
            ),
            reference_witness=tuple(value.get("reference_witness") or ()),
        )

    def verify_manifest(self, manifest: WorldManifest, *, verify_sources: bool = False) -> None:
        if manifest.manifest_hash != self.world_manifest_hash or manifest.world_id != self.world_id:
            raise ManifestMismatchError("task bundle and world manifest do not match")
        if verify_sources:
            manifest.verify()

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self.to_dict(), stream, ensure_ascii=False, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return target

    @classmethod
    def load(cls, path: str | Path) -> "TaskBundle":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["TaskBundle"]
