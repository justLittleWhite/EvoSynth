"""EvoSynth adapter for the semantic validation runtime."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping, Sequence

from evosynth.task_gen.common.synthesis import CallRecord, ExecutionTrace

from .environment import LensResult


class SemanticWorldBackend:
    """为任务生成提供基于 ``SemanticEnvironment`` 的世界后端。

    它不依赖某个地图/金融工具：实体、声明和事实结晶继续由传入的
    ``SemanticEnvironment`` 与 ``generate`` 回调处理。该适配器仅补齐统一
    的 call-id、审计轨迹和冻结快照。
    """

    kind = "open"

    def __init__(self, environment: Any, *, generate: Any = None,
                 tool_schemas: Sequence[Mapping[str, Any]] = ()):
        self.environment = environment
        self.generate = generate
        self._tool_schemas = [copy.deepcopy(dict(item)) for item in tool_schemas]
        self._call_index = 0
        self.call_log: list[dict[str, Any]] = []

    def bootstrap(self, *, seed: int | None = None,
                  spec: Mapping[str, Any] | None = None) -> Any:
        del seed, spec
        return self.snapshot()

    def tool_schemas(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._tool_schemas)

    def call(self, tool: str, arguments: Mapping[str, Any], *,
             source: str = "execution") -> str:
        self._call_index += 1
        call_id = f"{source}-{self._call_index:04d}"
        try:
            semantic_tool = tool
            registry = getattr(
                getattr(self.environment, "runtime", None), "registry", None)
            if registry is not None:
                try:
                    registry.get(semantic_tool)
                except Exception:
                    # Public OpenAI schemas expose the unqualified function
                    # name, whereas semantic contracts use server/tool. Map
                    # only an unambiguous suffix; never guess between servers.
                    specs = list(getattr(registry, "_specs", {}).values())
                    matches = [
                        spec.tool_id for spec in specs
                        if str(spec.tool_id).rsplit("/", 1)[-1] == tool
                    ]
                    if len(matches) != 1:
                        raise
                    semantic_tool = matches[0]

            def generator(query: Any) -> Any:
                if self.generate is None:
                    raise RuntimeError("semantic backend has no materializer")
                # Prefer QueryIR so custom backends can inspect semantic
                # bindings, then fall back to public schema-shaped arguments.
                try:
                    generated = self.generate(query)
                except (TypeError, AttributeError):
                    generated = self.generate(
                        semantic_tool,
                        getattr(query, "public_arguments", query.arguments),
                    )
                return (
                    generated
                    if isinstance(generated, LensResult)
                    else LensResult(generated)
                )

            value = self.environment.call(
                semantic_tool,
                dict(arguments),
                call_id=call_id,
                generate=generator,
            )
            observation = (
                json.dumps(value, ensure_ascii=False)
                if not isinstance(value, str)
                else value
            )
        except Exception as exc:
            observation = f"Error: {type(exc).__name__}: {exc}"
        self.call_log.append({
            "tool": tool,
            "args": dict(arguments),
            "observation": observation,
            "source": source,
            "provenance": "semantic_ledger",
        })
        return observation

    def snapshot(self) -> Any:
        runtime = getattr(self.environment, "runtime", None)
        if runtime is not None and hasattr(runtime, "export"):
            return copy.deepcopy(runtime.export())
        return copy.deepcopy(self.call_log)

    def diff(self, before: Any, after: Any) -> Mapping[str, Any]:
        if before == after:
            return {"changed": False, "claims_added": 0, "entities_added": 0}

        def count(value: Any, key: str) -> int:
            return (
                len((value or {}).get(key, {}))
                if isinstance(value, Mapping)
                else 0
            )

        return {
            "changed": True,
            "claims_added": count(after, "claims") - count(before, "claims"),
            "entities_added": count(after, "entities") - count(before, "entities"),
        }

    def replay(self, reference: Sequence[Mapping[str, Any]], *,
               source: str = "replay") -> ExecutionTrace:
        trace = ExecutionTrace()
        for i, step in enumerate(reference, 1):
            before = self.snapshot()
            obs = self.call(
                str(step["tool"]),
                step.get("arguments") or {},
                source=source,
            )
            after = self.snapshot()
            trace.calls.append(CallRecord(
                i,
                str(step["tool"]),
                step.get("arguments") or {},
                obs,
                not obs.startswith("Error:"),
                source,
                "semantic_ledger",
                call_id=(
                    str(step["call_id"])
                    if step.get("call_id")
                    else None
                ),
                depends_on=tuple(
                    str(item) for item in step.get("depends_on", ())
                ),
                produces=tuple(str(item) for item in step.get("produces", ())),
                kind=str(step.get("kind", "tool")),
                state_changed=before != after,
            ))
        trace.final_snapshot = self.snapshot()
        return trace

    def freeze(self) -> Any:
        return self.snapshot()


__all__ = ["SemanticWorldBackend"]
