"""任务生成流程使用的共享数据结构、执行轨迹和质量门禁。

领域相关的世界后端通过 :class:`WorldBackend` 注入；本模块负责记录调用、
计算复杂度、构造任务计划并执行通用的结构化校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import copy
import json
import re
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class CallRecord:
    """一次后端调用的规范化记录。"""

    step: int
    tool: str
    arguments: Mapping[str, Any]
    observation: str = ""
    success: bool = True
    source: str = "execution"
    provenance: str | None = None
    # ``call_id`` and ``depends_on`` describe the real data-flow graph.  Call
    # order is deliberately not treated as dependency: two independent reads
    # remain depth one even if they happened one after another.
    call_id: str | None = None
    depends_on: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    relevant: bool = True
    kind: str = "tool"
    # ``None`` is reserved for legacy/imported traces that were recorded before
    # per-call snapshots existed.  New production paths must set a real boolean.
    # Backward compatibility deliberately treats ``None`` as the old
    # success-based semantics; an explicit ``False`` can never fill mutation
    # complexity or enter the executable reference witness.
    state_changed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "tool": self.tool,
            "arguments": copy.deepcopy(dict(self.arguments)),
            "observation": self.observation,
            "success": self.success,
            "source": self.source,
            "provenance": self.provenance,
            "call_id": self.call_id,
            "depends_on": list(self.depends_on),
            "produces": list(self.produces),
            "relevant": self.relevant,
            "kind": self.kind,
            "state_changed": self.state_changed,
        }

    @property
    def identity(self) -> str:
        return self.call_id or f"step-{self.step}"

    @property
    def is_derived(self) -> bool:
        return self.kind in {"compute", "derive", "decision"} \
            or self.provenance in {"local_compute", "derived"}


@dataclass(frozen=True)
class PersonaRef:
    """PersonHub 中一次可追溯的人设采样。"""

    text: str
    line: int | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DerivationNode:
    """任务语义图中的一个领域观测、计算或决策节点。"""

    node_id: str
    operation: str
    inputs: tuple[str, ...] = ()
    kind: str = "derive"
    output_field: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operation": self.operation,
            "inputs": list(self.inputs),
            "kind": self.kind,
            "output_field": self.output_field,
            "metadata": copy.deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True)
class DerivationGraph:
    """与领域无关的有向无环派生图。

    图描述“答案如何由世界观测得到”，不是调用顺序提示。领域适配器可以
    把节点编译为文件变更、语义声明表达式或本地计算，但不能伪造依赖边。
    """

    nodes: tuple[DerivationNode, ...] = ()

    def validate(self) -> tuple[str, ...]:
        seen: set[str] = set()
        reasons: list[str] = []
        for node in self.nodes:
            if not node.node_id or node.node_id in seen:
                reasons.append(f"duplicate_or_empty_node:{node.node_id}")
                continue
            missing = [item for item in node.inputs if item not in seen]
            if missing:
                reasons.append(f"forward_or_missing_dependency:{node.node_id}:{','.join(missing)}")
            if node.node_id in node.inputs:
                reasons.append(f"self_dependency:{node.node_id}")
            if node.kind != "observation" and not node.inputs:
                reasons.append(f"ungrounded_derived_node:{node.node_id}")
            seen.add(node.node_id)
        return tuple(dict.fromkeys(reasons))

    @property
    def dependency_depth(self) -> int:
        depths: dict[str, int] = {}
        for node in self.nodes:
            depths[node.node_id] = 1 + max(
                (depths.get(parent, 0) for parent in node.inputs), default=0)
        return max(depths.values(), default=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "dependency_depth": self.dependency_depth,
            "validation": list(self.validate()),
        }


@dataclass(frozen=True)
class TaskPlan:
    """任务计划；领域细节保留在 ``payload``。"""

    task_id: str
    seed: int
    domain: str
    persona: PersonaRef
    archetype: str
    archetype_desc: str
    tier: str
    constraints: tuple[str, ...]
    target_complexity: Mapping[str, Any]
    derivation: DerivationGraph
    payload: Mapping[str, Any] = field(default_factory=dict)
    raw_spec: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "domain": self.domain,
            "persona": self.persona.text,
            "persona_line": self.persona.line,
            "archetype": self.archetype,
            "archetype_desc": self.archetype_desc,
            "tier": self.tier,
            "constraints": list(self.constraints),
            "target_complexity": copy.deepcopy(dict(self.target_complexity)),
            "derivation": self.derivation.to_dict(),
            "payload": copy.deepcopy(dict(self.payload)),
            "raw_spec": copy.deepcopy(dict(self.raw_spec)),
        }


@dataclass(frozen=True)
class UserGoalContract:
    """真实用户目的合同，领域细节保留在 ``payload``。

    Archetype 只描述隐藏推理结构；公开条件必须来自真实用户目的。去重、
    证据覆盖和重放一致性等构造器义务保留在 ``hidden_invariants``，不得由
    题面自然化阶段读取。

    This contract remains available for deserializing older task artifacts.
    Current synthesis derives the public task from the sampled capability,
    persona, tool set, and materialized world.
    """

    scenario: str
    decision_or_need: str
    target_answer_semantics: str
    answer_mode: str = "observed_value"
    public_conditions: tuple[str, ...] = ()
    hidden_invariants: tuple[str, ...] = ()
    known_inputs: tuple[str, ...] = ()
    success_consequence: str = ""
    applicable_constraints: tuple[str, ...] = ()
    compatible_terminal_properties: tuple[Mapping[str, Any], ...] = ()

    def public_view(self) -> dict[str, Any]:
        """Return only information that may influence public task wording."""
        return {
            "scenario": self.scenario,
            "decision_or_need": self.decision_or_need,
            "target_answer_semantics": self.target_answer_semantics,
            "public_conditions": list(self.public_conditions),
            "known_inputs": list(self.known_inputs),
            "success_consequence": self.success_consequence,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "decision_or_need": self.decision_or_need,
            "target_answer_semantics": self.target_answer_semantics,
            "answer_mode": self.answer_mode,
            "public_conditions": list(self.public_conditions),
            "hidden_invariants": list(self.hidden_invariants),
            "known_inputs": list(self.known_inputs),
            "success_consequence": self.success_consequence,
            "applicable_constraints": list(self.applicable_constraints),
            "compatible_terminal_properties": [
                copy.deepcopy(dict(item))
                for item in self.compatible_terminal_properties
            ],
        }


@dataclass
class ExecutionTrace:
    """执行先行循环的可序列化轨迹。"""

    calls: list[CallRecord] = field(default_factory=list)
    conclusion: str = ""
    initial_snapshot: Any = None
    final_snapshot: Any = None

    @property
    def successful_calls(self) -> list[CallRecord]:
        return [c for c in self.calls if c.success]

    @property
    def relevant_calls(self) -> list[CallRecord]:
        return [c for c in self.calls if c.relevant]

    @property
    def relevant_successful_calls(self) -> list[CallRecord]:
        return [c for c in self.calls if c.success and c.relevant]

    @staticmethod
    def _is_mutation(call: CallRecord) -> bool:
        return call.kind == "mutation" or call.tool in {
            "write_file", "edit_file", "move_file", "delete_file",
        }

    @property
    def no_effect_mutation_calls(self) -> list[CallRecord]:
        """Successful mutation-intent calls proven not to change authority state."""
        return [c for c in self.relevant_successful_calls
                if self._is_mutation(c) and c.state_changed is False]

    @property
    def reference_calls(self) -> list[CallRecord]:
        """Executable witness calls, excluding audited no-effect mutations.

        Reads and setup calls remain because they can be necessary executable
        context.  Legacy mutation records with ``state_changed=None`` retain the
        old behavior and are included.
        """
        return [c for c in self.relevant_successful_calls
                if not (self._is_mutation(c) and c.state_changed is False)]

    @property
    def complexity_calls(self) -> list[CallRecord]:
        """Calls allowed to contribute to complexity metrics."""
        return self.reference_calls

    @property
    def mutation_calls(self) -> list[CallRecord]:
        return [c for c in self.reference_calls if self._is_mutation(c)]

    def reference_solution(self) -> list[dict[str, Any]]:
        """从真实成功调用导出参考解（保留原有 pipeline 字段名）。"""
        result: list[dict[str, Any]] = []
        for i, c in enumerate(self.reference_calls):
            item = {"step": i + 1, "tool": c.tool, "arguments": dict(c.arguments)}
            if c.call_id is not None:
                item["call_id"] = c.call_id
            if c.depends_on:
                item["depends_on"] = list(c.depends_on)
            if c.produces:
                item["produces"] = list(c.produces)
            if c.kind != "tool":
                item["kind"] = c.kind
            result.append(item)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": [c.to_dict() for c in self.calls],
            "conclusion": self.conclusion,
            "initial_snapshot": copy.deepcopy(self.initial_snapshot),
            "final_snapshot": copy.deepcopy(self.final_snapshot),
        }

    @classmethod
    def from_call_log(cls, call_log: Iterable[Mapping[str, Any]], **kwargs: Any) -> "ExecutionTrace":
        """兼容旧环境 ``call_log``（字段 tool/args/observation/source）。"""
        calls: list[CallRecord] = []
        for i, raw in enumerate(call_log, 1):
            obs = str(raw.get("observation", ""))
            state_changed = raw.get("state_changed")
            if not isinstance(state_changed, bool):
                state_changed = None
            calls.append(CallRecord(
                step=i, tool=str(raw.get("tool", "")),
                arguments=dict(raw.get("args") or raw.get("arguments") or {}),
                observation=obs,
                success=bool(raw.get("success", not obs.startswith("Error:"))),
                source=str(raw.get("source", "execution")),
                provenance=raw.get("provenance"),
                call_id=(str(raw["call_id"]) if raw.get("call_id") else None),
                depends_on=tuple(str(x) for x in raw.get("depends_on", ())),
                produces=tuple(str(x) for x in raw.get("produces", ())),
                relevant=bool(raw.get("relevant", True)),
                kind=str(raw.get("kind", "tool")),
                state_changed=state_changed))
        kwargs.pop("evidence_call_ids", None)
        return cls(calls=calls, **kwargs)


@dataclass(frozen=True)
class ComplexityMetrics:
    """可跨域比较的实际复杂度指标。"""

    calls: int
    successful_calls: int
    mutations: int
    distinct_tools: int
    entities: int
    dependency_depth: int
    unique_calls: int = 0
    derived_calls: int = 0
    grounded_derived_calls: int = 0
    dependent_calls: int = 0
    knowledge_tools: int = 0
    logic_steps: int = 0
    facet_count: int = 0
    branch_count: int = 0
    derived_step_count: int = 0
    join_count: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"calls": self.calls, "successful_calls": self.successful_calls,
                "mutations": self.mutations, "distinct_tools": self.distinct_tools,
                "entities": self.entities, "dependency_depth": self.dependency_depth,
                "unique_calls": self.unique_calls, "derived_calls": self.derived_calls,
                "grounded_derived_calls": self.grounded_derived_calls,
                "dependent_calls": self.dependent_calls,
                "knowledge_tools": self.knowledge_tools,
                "logic_steps": self.logic_steps, "facet_count": self.facet_count,
                "branch_count": self.branch_count,
                "derived_step_count": self.derived_step_count,
                "join_count": self.join_count}


def _entity_count(calls: Sequence[CallRecord]) -> int:
    """保守估计实体数，不依赖具体 domain 字段名。"""
    values: set[str] = set()
    ignored = {"", "unknown", "null", "none", "true", "false"}
    for call in calls:
        # setup/noop 可以是可执行见证中的必要前置，但不代表任务实际处理了
        # 一个领域实体（例如文件系统工具的 create_directory 辅助目录状态）。
        if call.kind in {"setup", "noop"}:
            continue
        def add_selector(value: Any, prefix: str = "") -> None:
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                text = str(value).strip().lower()
                if text not in ignored:
                    values.add(prefix + text)
            elif isinstance(value, (list, tuple, set, frozenset)):
                for item in value:
                    add_selector(item, prefix)

        def add_observed_ids(value: Any) -> None:
            if isinstance(value, Mapping):
                rid = value.get("rid") or value.get("resource_id")
                if isinstance(rid, (str, int)):
                    add_selector(rid, "resource:")
                ident = value.get("id")
                object_type = value.get("object")
                if isinstance(ident, (str, int)):
                    if isinstance(object_type, str) and object_type:
                        add_selector(f"{object_type}/{ident}", "resource:")
                    elif "/" in str(ident):
                        add_selector(ident, "resource:")
                for item in value.values():
                    add_observed_ids(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    add_observed_ids(item)

        # WorldBackend 已把真实资源选择器归一写入 produces。优先消费这一
        # 环境无关证据，避免实体计数依赖 path/symbol 等少数参数字段名。
        # produces 只来自选择器参数，不包含正文载荷，因此不会把内容词误算为实体。
        if call.produces:
            for selector in call.produces:
                add_selector(selector, "resource:")
            try:
                observed = json.loads(call.observation)
            except Exception:
                observed = None
            add_observed_ids(observed)
            continue

        def visit(v: Any) -> None:
            if isinstance(v, Mapping):
                for k, item in v.items():
                    key_lower = str(k).lower()
                    # Coordinates are structured entity selectors rather than
                    # ordinary scalar arguments.  Count each coordinate pair
                    # as one entity, while retaining the generic token rules
                    # for finance/search/directions domains.
                    if key_lower in {"locations", "coordinates", "points"} and isinstance(item, (list, tuple)):
                        for point in item:
                            if isinstance(point, Mapping) and {"latitude", "longitude"} <= set(point):
                                values.add("coord:" + str(point["latitude"]) + "," + str(point["longitude"]))
                            else:
                                visit(point)
                        continue
                    if {"latitude", "longitude"} <= set(v) and key_lower in {"latitude", "longitude"}:
                        # The enclosing mapping is handled by its locations
                        # parent; do not count each scalar separately.
                        continue
                    # Closed World 的“实体”是被选中的资源，例如
                    # path/rid；编辑内容、替换文本等载荷不是实体。
                    if key_lower in {
                        "path", "paths", "source", "destination", "rid",
                        "resource_id", "resource_ids",
                    }:
                        add_selector(item, "resource:")
                    elif any(x in key_lower for x in
                             ("symbol", "ticker", "place", "address", "entity",
                              "query", "origin", "destination")):
                        add_selector(item)
                    visit(item)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    visit(item)
        visit(call.arguments)
    return len(values)


def _stable_call_key(call: CallRecord) -> str:
    try:
        args = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except Exception:
        args = repr(call.arguments)
    return f"{call.tool}\0{args}"


def dependency_violations(trace: ExecutionTrace) -> tuple[str, ...]:
    """检查依赖只指向更早、成功且有信息含量的节点。"""
    seen: dict[str, CallRecord] = {}
    reasons: list[str] = []
    for call in trace.complexity_calls:
        ident = call.identity
        if ident in seen:
            reasons.append(f"duplicate_call_id:{ident}")
        for parent in call.depends_on:
            if parent == ident:
                reasons.append(f"self_dependency:{ident}")
            elif parent not in seen:
                reasons.append(f"forward_or_missing_dependency:{ident}:{parent}")
        seen[ident] = call
    return tuple(dict.fromkeys(reasons))


def _dependency_analysis(calls: Sequence[CallRecord]) -> tuple[int, int, int]:
    depths: dict[str, int] = {}
    grounded: dict[str, bool] = {}
    grounded_derived = 0
    dependent = 0
    for call in calls:
        parents = [item for item in call.depends_on if item in depths]
        if parents:
            dependent += 1
        depths[call.identity] = 1 + max((depths[item] for item in parents), default=0)
        if call.is_derived:
            grounded[call.identity] = bool(parents) and any(grounded[item] for item in parents)
            if grounded[call.identity]:
                grounded_derived += 1
        else:
            # A successful non-derived call is a world observation root.
            grounded[call.identity] = True
    return max(depths.values(), default=0), grounded_derived, dependent


def dependency_chain_depth(calls: Sequence[CallRecord]) -> int:
    """Return the longest dependency path, measured in call nodes."""
    depth, _, _ = _dependency_analysis(calls)
    return depth


def measure_complexity(trace: ExecutionTrace) -> ComplexityMetrics:
    # A successful mutation-intent call with an explicit unchanged snapshot is
    # still visible in the audit trace, but contributes to no complexity axis.
    # ``state_changed=None`` remains compatible with historical traces.
    calls = trace.complexity_calls
    successful = trace.complexity_calls
    depth, grounded_derived, dependent = _dependency_analysis(successful)
    derived = [call for call in successful if call.is_derived]
    knowledge = [call for call in successful if not call.is_derived]
    return ComplexityMetrics(
        calls=len(calls), successful_calls=len(successful),
        mutations=len(trace.mutation_calls),
        distinct_tools=len({c.tool for c in successful}),
        entities=_entity_count(knowledge), dependency_depth=depth,
        unique_calls=len({_stable_call_key(call) for call in successful}),
        derived_calls=len(derived), grounded_derived_calls=grounded_derived,
        dependent_calls=dependent,
        knowledge_tools=len({call.tool for call in knowledge}))


def _observed_information_facets(trace: ExecutionTrace) -> set[str]:
    """从真实观测的叶字段提取属性覆盖，不依赖领域或工具名。"""
    facets: set[str] = set()

    def visit(tool: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(item, (Mapping, list, tuple)):
                    visit(tool, item)
                else:
                    facets.add(f"field:{tool.casefold()}:{str(key).casefold()}")
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(tool, item)
        else:
            facets.add(f"field:{tool.casefold()}:value")

    for call in trace.relevant_successful_calls:
        try:
            value = json.loads(call.observation)
        except Exception:
            value = call.observation
        # 部分工具把对象包装为 JSON 字符串。
        if isinstance(value, str) and value.strip()[:1] in "[{":
            try:
                value = json.loads(value)
            except Exception:
                pass
        visit(call.tool, value)
    return facets


def _logic_operation_evidence(operation: str, trace: ExecutionTrace,
                              metrics: ComplexityMetrics,
                              certificate: Mapping[str, Any] | None = None) -> bool:
    """从真实执行轨迹判断逻辑步骤，不依赖额外的证书字段。"""
    op = operation.casefold()
    if op == "collect":
        return metrics.successful_calls >= 2
    if op == "resolve":
        return metrics.entities >= 1
    if op == "verify":
        return metrics.unique_calls >= 2
    successful = trace.relevant_successful_calls
    if op == "join":
        return any(call.depends_on for call in successful) or len({call.tool for call in successful}) >= 2
    if op in {"synthesize", "scan"}:
        return len({call.tool for call in successful}) >= 2 or len(successful) >= 2
    return len(successful) >= 2


@runtime_checkable
class WorldBackend(Protocol):
    """任务生成世界后端的最小接口。"""

    kind: str

    def bootstrap(self, *, seed: int | None = None, spec: Mapping[str, Any] | None = None) -> Any: ...
    def tool_schemas(self) -> list[dict[str, Any]]: ...
    def call(self, tool: str, arguments: Mapping[str, Any], *, source: str = "execution") -> str: ...
    def snapshot(self) -> Any: ...
    def diff(self, before: Any, after: Any) -> Mapping[str, Any]: ...
    def replay(self, reference: Sequence[Mapping[str, Any]], *, source: str = "replay") -> ExecutionTrace: ...
    def freeze(self) -> Any: ...


class _AdapterBase:
    kind = "unknown"

    def __init__(self, environment: Any):
        self.environment = environment
        self._initial: Any = None

    def bootstrap(self, *, seed: int | None = None, spec: Mapping[str, Any] | None = None) -> Any:
        del seed, spec
        self._initial = self.snapshot()
        return copy.deepcopy(self._initial)

    def tool_schemas(self) -> list[dict[str, Any]]:
        fn = getattr(self.environment, "agent_tools_schema", None)
        return list(fn() if callable(fn) else [])

    def call(self, tool: str, arguments: Mapping[str, Any], *, source: str = "execution") -> str:
        return str(self.environment.call(tool, dict(arguments), source=source))

    def snapshot(self) -> Any:
        raise NotImplementedError

    def diff(self, before: Any, after: Any) -> Mapping[str, Any]:
        if before == after:
            return {"changed": False}
        return {"changed": True}

    def replay(self, reference: Sequence[Mapping[str, Any]], *, source: str = "replay") -> ExecutionTrace:
        trace = ExecutionTrace()
        for i, step in enumerate(reference, 1):
            before = self.snapshot()
            try:
                obs = self.call(str(step["tool"]), step.get("arguments") or {}, source=source)
                after = self.snapshot()
                trace.calls.append(CallRecord(
                    i, str(step["tool"]), step.get("arguments") or {}, obs,
                    not obs.startswith("Error:"), source,
                    call_id=(str(step["call_id"]) if step.get("call_id") else None),
                    depends_on=tuple(str(x) for x in step.get("depends_on", ())),
                    produces=tuple(str(x) for x in step.get("produces", ())),
                    kind=str(step.get("kind", "tool")),
                    state_changed=before != after))
            except Exception as exc:  # replay 必须记录失败而非吞掉审计信息
                after = self.snapshot()
                trace.calls.append(CallRecord(
                    i, str(step.get("tool", "")), step.get("arguments") or {},
                    f"Error: {exc}", False, source,
                    call_id=(str(step["call_id"]) if step.get("call_id") else None),
                    depends_on=tuple(str(x) for x in step.get("depends_on", ())),
                    produces=tuple(str(x) for x in step.get("produces", ())),
                    kind=str(step.get("kind", "tool")),
                    state_changed=before != after))
        trace.final_snapshot = self.snapshot()
        return trace

    def freeze(self) -> Any:
        return copy.deepcopy(self.snapshot())


class ClosedWorldBackend(_AdapterBase):
    """包装旧 ``SynthesisEnvironment``，保持文件系统语义不变。"""
    kind = "closed"

    _MUTATION_TOOLS = {
        "write_file", "edit_file", "move_file", "delete_file",
    }

    def call(self, tool: str, arguments: Mapping[str, Any], *,
             source: str = "execution") -> str:
        """Execute once and attach authoritative per-call effect evidence."""
        before = self.snapshot()
        observation = super().call(tool, arguments, source=source)
        after = self.snapshot()
        call_log = getattr(self.environment, "call_log", None)
        if isinstance(call_log, list) and call_log:
            row = call_log[-1]
            if (row.get("tool") == tool
                    and row.get("source", source) == source):
                row["state_changed"] = before != after
        return observation

    def snapshot(self) -> Any:
        ws_id = getattr(self.environment, "ws_id", None)
        graph = getattr(self.environment, "graph", None)
        if graph is not None and ws_id is not None:
            return copy.deepcopy(graph.ws_files(ws_id))
        return copy.deepcopy(getattr(self.environment, "files", {}))

    def diff(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "created": sorted(set(after) - set(before)),
            "deleted": sorted(set(before) - set(after)),
            "modified": sorted(k for k in set(before) & set(after) if before[k] != after[k]),
            "unchanged": sorted(k for k in set(before) & set(after) if before[k] == after[k]),
        }

    def replay(self, reference: Sequence[Mapping[str, Any]], *,
               source: str = "replay") -> ExecutionTrace:
        """从 ``bootstrap`` 捕获的初始快照在影子 workspace 上重放。

        只读开放域可以在同一后端上再查一次，但闭合域的
        ``move/delete/edit`` 若在执行终态上重放会必然失败。参考执行应在同一初始快照的影子空间中进行。
        这里对 ``SynthesisEnvironment`` 所暴露的 ``graph/ws_id``
        做最小结构适配，不把 filesystem 语义复制到共享核心。

        对不支持影子空间的自定义后端，纯只读参考解仍保留
        旧兼容路径；含变更的参考解则显式失败，防止将终态原地
        重放误报为可验证。
        """
        graph = getattr(self.environment, "graph", None)
        ws_id = getattr(self.environment, "ws_id", None)
        servers = getattr(self.environment, "servers", None)
        initial = copy.deepcopy(self._initial)

        if graph is not None and ws_id is not None and isinstance(initial, Mapping):
            workspaces = getattr(graph, "data", {}).get("workspaces")
            if isinstance(workspaces, dict):
                prefix = f"{ws_id}/__shared_replay__"
                index = 1
                replay_id = f"{prefix}{index}"
                while replay_id in workspaces:
                    index += 1
                    replay_id = f"{prefix}{index}"
                workspaces[replay_id] = {
                    "calls": {}, "files": copy.deepcopy(dict(initial)), "mutations": [],
                }
                try:
                    env_type = type(self.environment)
                    kwargs: dict[str, Any] = {}
                    if hasattr(self.environment, "verbose"):
                        kwargs["verbose"] = False
                    shadow_env = env_type(graph, replay_id, list(servers or ()), **kwargs)
                    shadow_backend = ClosedWorldBackend(shadow_env)
                    trace = _AdapterBase.replay(
                        shadow_backend, reference, source=source)
                    # ``_AdapterBase.replay`` 已在删除影子空间前捕获终态。
                    return trace
                except Exception as exc:
                    calls = [CallRecord(
                        step=i, tool=str(step.get("tool", "")),
                        arguments=step.get("arguments") or {},
                        observation=f"Error: isolated replay failed: {exc}",
                        success=False, source=source,
                    ) for i, step in enumerate(reference, 1)]
                    return ExecutionTrace(calls=calls, final_snapshot=self.snapshot())
                finally:
                    workspaces.pop(replay_id, None)

        mutating = any(str(step.get("tool", "")) in self._MUTATION_TOOLS
                       for step in reference)
        if not mutating:
            return super().replay(reference, source=source)
        calls = [CallRecord(
            step=i, tool=str(step.get("tool", "")),
            arguments=step.get("arguments") or {},
            observation="Error: closed-world mutation replay requires an isolated workspace",
            success=False, source=source,
        ) for i, step in enumerate(reference, 1)]
        return ExecutionTrace(calls=calls, final_snapshot=self.snapshot())


class EvoSynthBackend(_AdapterBase):
    """包装旧 ``OpenEnvironment``；事实结晶仍由其 registry/lens 负责。"""
    kind = "open"

    def snapshot(self) -> Any:
        graph = getattr(self.environment, "graph", None)
        if graph is not None:
            return copy.deepcopy(getattr(graph, "data", {}))
        return copy.deepcopy(getattr(self.environment, "call_log", []))

    def diff(self, before: Any, after: Any) -> Mapping[str, Any]:
        return {"changed": before != after, "before": before, "after": after}


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: ComplexityMetrics

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reasons": list(self.reasons),
                "metrics": self.metrics.to_dict()}


class SharedQualityGate:
    """跨域统一门禁，拒绝单工具/单调用和题面泄漏。"""

    def __init__(self, *, enforce_complexity: bool = True):
        # 默认值保持 Closed World 的原有行为。EvoSynth 在构造阶段按
        # 采样规格引导复杂度，事后只需让门禁裁决硬正确性。
        self.enforce_complexity = enforce_complexity

    def evaluate(self, trace: ExecutionTrace, *, spec: Mapping[str, Any] | None = None,
                 task_text: str = "", tool_schemas: Sequence[Mapping[str, Any]] = (),
                 derivation: DerivationGraph | None = None,
                 private_values: Iterable[Any] = (),
                 public_values: Iterable[Any] = (),
                 semantic_certificate: Mapping[str, Any] | None = None) -> QualityReport:
        if isinstance(spec, SynthesisSpec):
            spec = dict(spec.raw) or {
                "read_calls_range": spec.target_calls,
                "entities_range": spec.target_entities,
            }
        spec = spec or {}
        semantic_certificate = semantic_certificate or {}
        m = measure_complexity(trace)
        reasons: list[str] = []
        if trace.no_effect_mutation_calls:
            reasons.append("no_effect_mutation")
        # 阈值制门禁(当前版本):失败调用占比≤15%且仅含 failed_tool_call/no_effect_mutation → 放行
        failed_calls = [c for c in trace.calls if not c.success]
        if spec.get("reject_failed_calls") and failed_calls:
            total_calls = len(trace.calls)
            failed_ratio = len(failed_calls) / total_calls if total_calls > 0 else 0
            # 超过15%阈值,或虽在阈值内但混有其它拒因类型 → 拒绝
            if failed_ratio > 0.15:
                reasons.append("failed_tool_call")
        # 成功调用数下限门禁已删除(当前实现)。原实现是
        # ``m.successful_calls < max(2, int(read_calls_range[0]))``,即拿档位
        # 观测下限(EvoSynth S=6)当发布门禁——正是"复杂度必须来自
        # target_calls 最低值"的旧口径,用户已明确否决:难度必须来自题面的
        # 跨类目条件组合,不是调用次数。这里只保留一条不数次数、不数工具的
        # 存在性下限:没有任何一次成功观测就不可能存在被观测支撑的答案,
        # 而下方的 require_nontrivial_logic / grounded derived /
        # duplicate_call_used_as_complexity / empty_task 全部以"至少有一次
        # 真实调用"为前提,没有这条下限时它们会对着空轨迹空转。
        if self.enforce_complexity and not m.successful_calls:
            reasons.append("no_successful_call")
        mutation_range = spec.get("mutation_chain_range")
        if self.enforce_complexity and isinstance(mutation_range, (list, tuple)) and mutation_range:
            # 保留文件系统任务的基础门槛：总调用数不能代替
            # 真正的状态变更链，否则多次 list/read 会伪造复杂度。
            required_mutations = max(1, int(mutation_range[0]))
            if m.mutations < required_mutations:
                reasons.append(f"mutations<{required_mutations}")
            if len(mutation_range) > 1:
                maximum_mutations = int(mutation_range[1])
                if m.mutations > maximum_mutations:
                    reasons.append(f"mutations>{maximum_mutations}")
        entities_range = spec.get("target_entities", spec.get("entities_range"))
        if (self.enforce_complexity
                and isinstance(entities_range, (list, tuple)) and entities_range):
            if m.entities < int(entities_range[0]):
                reasons.append(f"entities<{int(entities_range[0])}")
        # distinct_tools 下限门禁已删除(当前实现)。原实现从
        # target_tools 取一个下限,缺省时还会因 computation 是 join /
        # multi-step arithmetic 把下限抬到 2,再用
        # ``m.distinct_tools < required_tools`` 拒绝提交。诊断:事后数不同
        # 工具个数不是难度的代理,它只逼出冗余调用(同一信息换一个接口再取
        # 一次),并让"看起来多工具"的题从门禁溜过。多工具应是题面语义要求
        # (条件跨 ≥2 个类目)的结果;该要求已上移到 prompt 层,由采样/题面
        # 负责,这里不再用任何计数代替。m.distinct_tools 仍作为指标保留在
        # ComplexityMetrics 里供分析。
        if self.enforce_complexity and spec.get("require_nontrivial_logic"):
            certificate_rows = [item for item in
                                (semantic_certificate.get("claims") or ())
                                if isinstance(item, Mapping)]
            certificate_call_ids = {
                str((item.get("provenance") or {}).get("call_id"))
                for item in certificate_rows
                if (item.get("provenance") or {}).get("call_id")}
            # 当前实现:唯一诚实且可被证书证明的"非平凡"证据,是终局答
            # 案建立在 ≥2 条互相独立的事实之上(证书里两条不同 provenance 的
            # claim)。原来的第二条分支 ``successful_calls>=2 and
            # distinct_tools>=2`` 是伪装成逻辑条件的工具计数,已删除;原来的
            # ``logical_archetype``(archetype != lookup)合取项也是恒真的死
            # 代码——该 spec 键在整个代码库里无人写入,取值恒为空串。
            # 判据只看证书本身证明了什么,不数工具、不数调用次数。
            if len(certificate_call_ids) < 2:
                reasons.append("missing_nontrivial_logic")
        target_depth = int(spec.get("target_depth", 1) or 1)
        if self.enforce_complexity and m.dependency_depth < target_depth:
            reasons.append(f"dependency_depth<{target_depth}")
        reasons.extend(dependency_violations(trace))
        if m.derived_calls and m.grounded_derived_calls < m.derived_calls:
            reasons.append("ungrounded_derived_call")
        if spec.get("require_grounded_derivation") and not m.grounded_derived_calls:
            reasons.append("missing_grounded_derivation")
        if (self.enforce_complexity and spec.get("reject_duplicate_queries")
                and m.unique_calls < m.successful_calls):
            reasons.append("duplicate_call_used_as_complexity")
        if derivation is not None:
            reasons.extend(derivation.validate())
            if (self.enforce_complexity
                    and derivation.dependency_depth != m.dependency_depth):
                reasons.append(
                    f"trace_plan_depth_mismatch:{m.dependency_depth}!={derivation.dependency_depth}")
        names = set()
        for schema in tool_schemas:
            fn = schema.get("function", schema) if isinstance(schema, Mapping) else {}
            if isinstance(fn, Mapping) and fn.get("name"):
                names.add(str(fn["name"]).lower())
        # 工具公开名可能本身就是普通自然语言动词（例如 ``search``、
        # ``fetch``、``read``）。裸词出现在用户请求中不等于泄漏实现，
        # 否则“搜索某主题”会被误判。只有代码/接口上下文，或带命名空间的
        # 机器标识，才构成工具名泄漏；显式“调用 search 工具”仍会被拦截。
        leaked = []
        lowered_task = task_text.lower()
        for name in names:
            if not name:
                continue
            escaped = re.escape(name)
            explicit = (
                rf"`{escaped}`", rf"[\"']{escaped}[\"']",
                rf"\b{escaped}\s*\(",
                rf"\b(?:tool|function|api|interface)\s+{escaped}\b",
                rf"\b{escaped}\s+(?:tool|function|api|interface)\b",
                rf"(?:工具|函数|接口|调用)\s*{escaped}\b",
            )
            machine_name = ("__" in name or "/" in name or
                            ("-" in name and len(name) > 4))
            if any(re.search(pattern, lowered_task) for pattern in explicit):
                leaked.append(name)
            elif machine_name and re.search(
                    rf"(?<![\w.-]){escaped}(?![\w.-])", lowered_task):
                leaked.append(name)
        if leaked:
            reasons.append("tool_name_leak:" + ",".join(sorted(leaked)[:3]))
        # 题面只能表达目标和约束，不应泄漏工具参数或调用形状。
        parameters = set()
        for schema in tool_schemas:
            fn = schema.get("function", schema) if isinstance(schema, Mapping) else {}
            if isinstance(fn, Mapping):
                parameters.update(str(k).lower() for k in
                                  ((fn.get("parameters") or {}).get("properties") or {}))
        # Ordinary domain nouns ("locations", "ticker", "origin") may be
        # necessary to identify the user's entities.  Treat a parameter as a
        # leak only when it is exposed in code/assignment form or explicitly
        # called a parameter; plain prose is not an API hint.
        parameter_leaks = []
        # A field name is not a leak merely because it is an ordinary word in
        # the request (``content``, ``files``, ``draft`` and ``type`` are all
        # common domain language).  Require an unambiguous API/code context:
        # an assignment, a JSON object key, a function-call argument, or an
        # explicit statement that the word is a parameter/field.  In
        # particular, do not treat prose such as ``type: description`` or a
        # bolded noun as an interface specification.
        lowered_task = task_text.lower()
        for parameter in parameters:
            if len(parameter) <= 2:
                continue
            escaped = re.escape(parameter)
            patterns = (
                # Python/JS-style assignment, or a named argument in a call.
                rf"(?<![\w.-]){escaped}\s*=",
                rf"\([^)]*\b{escaped}\s*=",
                # JSON/mapping key: require an opening delimiter so ordinary
                # prose like ``type: description`` is not classified as code.
                rf"[{{[,]]\s*[\"']{escaped}[\"']\s*:",
                # Explicit API terminology (English and Chinese).
                rf"\b(?:parameter|argument|field|property|key)\s+[`\"']?{escaped}\b",
                rf"(?:参数|实参|字段|属性|键)\s*[`\"']?{escaped}\b",
            )
            if any(re.search(pattern, lowered_task) for pattern in patterns):
                parameter_leaks.append(parameter)
        if parameter_leaks:
            reasons.append("parameter_name_leak:" + ",".join(sorted(parameter_leaks)[:3]))
        lowered = task_text.lower()
        explicit_order = any(token in lowered for token in (
            "先调用", "再调用", "按顺序调用", "依次调用", "first call", "then call"))
        explicit_order = explicit_order or bool(re.search(
            r"\b(?:call|invoke|use)\b.{0,60}\bin order\b", lowered))
        if explicit_order:
            reasons.append("call_order_leak")
        if not task_text.strip():
            reasons.append("empty_task")

        def primitives(values: Iterable[Any]) -> list[Any]:
            output: list[Any] = []
            def visit(value: Any) -> None:
                if isinstance(value, Mapping):
                    for item in value.values():
                        visit(item)
                elif isinstance(value, (list, tuple, set, frozenset)):
                    for item in value:
                        visit(item)
                elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
                    output.append(value)
            for value in values:
                visit(value)
            return output

        public_tokens = {str(value).strip().lower() for value in primitives(public_values)}
        public_phrases = tuple(item for item in public_tokens if item)
        answer_leaks: list[str] = []
        for value in primitives(private_values):
            token = str(value).strip()
            if not token:
                continue
            lowered_token = token.lower()
            # A derived winner may legitimately be one member of a public
            # candidate string (for example ``Stone Harbor`` inside the public
            # input ``Stone Harbor, New Jersey``). Publishing the candidate is
            # not publishing the observed winning relation. Only values absent
            # from every public input remain private leak candidates.
            public_candidate = any(
                re.search(rf"(?<![\w]){re.escape(lowered_token)}(?![\w])", phrase)
                for phrase in public_phrases)
            if lowered_token in public_tokens or public_candidate:
                continue
            if isinstance(value, str):
                # Short/generic strings are too collision-prone to be a useful
                # mechanical leak signal. Long frozen names and identifiers are
                # checked as complete case-insensitive phrases.
                if len(token) >= 5 and token.lower() in lowered:
                    answer_leaks.append(token[:40])
            else:
                # Ignore small integers commonly used for counts/precision.
                try:
                    if float(value).is_integer() and abs(float(value)) < 20:
                        continue
                except Exception:
                    pass
                pattern = rf"(?<![\d.]){re.escape(token)}(?![\d.])"
                if re.search(pattern, task_text):
                    answer_leaks.append(token[:40])
        if answer_leaks:
            reasons.append("private_value_leak:" + ",".join(answer_leaks[:3]))
        return QualityReport(not reasons, tuple(dict.fromkeys(reasons)), m)


@dataclass(frozen=True)
class SynthesisSpec:
    """两类世界共用的任务规格最小投影。

    ``raw`` 保留未规范化的输入字段，调用方可以无损地传入原始 spec。
    """

    archetype: str
    tier: str
    constraints: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)
    # 以下字段保存生成器的审计元数据；缺失时使用默认值。
    persona: str = ""
    persona_line: int | None = None
    target_calls: tuple[int, int] = (2, 4)
    target_entities: tuple[int, int] = (1, 1)
    target_tools: tuple[int, int] = (1, 1)
    target_depth: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SynthesisSpec":
        return cls(
            archetype=str(value.get("archetype", "unknown")),
            tier=str(value.get("tier", "S")),
            constraints=tuple(str(x) for x in value.get("constraints", ())),
            raw=dict(value), persona=str(value.get("persona", "")),
            persona_line=value.get("persona_line"),
            target_calls=tuple(value.get("target_calls", value.get("read_calls_range", (2, 4)))),
            target_entities=tuple(value.get("target_entities", value.get("entities_range", (1, 1)))),
            target_tools=tuple(value.get("target_tools", (1, 1))),
            target_depth=int(value.get("target_depth", 1)))

    @property
    def target_complexity(self) -> dict[str, Any]:
        return {"calls": list(self.target_calls), "entities": list(self.target_entities),
                "tools": list(self.target_tools), "dependency_depth": self.target_depth,
                "constraints": len(self.constraints)}


class ExecutionFirstSynthesizer:
    """共享执行先行驱动器。

    它刻意不包含任何 LLM prompt：整理/research agent 由上层提供
    ``executor(backend, spec)``，驱动器负责冻结初始态、收集真实调用、计算
    复杂度并执行统一门禁。这使 Open 与 Closed 只在 bootstrap/call 上不同。
    """

    def __init__(self, backend: WorldBackend, quality_gate: SharedQualityGate | None = None):
        self.backend = backend
        self.quality_gate = quality_gate or SharedQualityGate()

    def synthesize(self, spec: SynthesisSpec,
                   executor: Any, *, task_builder: Any = None,
                   metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """任务生成的统一 ``synthesize`` 入口。

        ``executor`` 可以返回规范化 ``ExecutionTrace``，也可以返回旧式调用
        列表/记录；题面构造器只接收规格、轨迹和世界差分，便于统一执行后
        反向提案。
        """
        raw = spec.raw or {"archetype": spec.archetype, "tier": spec.tier,
                           "constraints": list(spec.constraints),
                           "read_calls_range": spec.target_calls}
        initial = self.backend.bootstrap(spec=raw)
        result = executor(self.backend, raw)
        if isinstance(result, ExecutionTrace):
            trace = result
        elif isinstance(result, Mapping) and "calls" in result:
            trace = ExecutionTrace.from_call_log(
                result.get("calls") or [],
                conclusion=str(result.get("conclusion", "")))
        elif isinstance(result, Iterable) and not isinstance(result, (str, bytes, Mapping)):
            trace = ExecutionTrace.from_call_log(result)
        else:
            trace = ExecutionTrace.from_call_log(
                getattr(self.backend, "call_log", []) or
                getattr(getattr(self.backend, "environment", None), "call_log", []),
                conclusion=str(result.get("conclusion", "") if isinstance(result, Mapping) else ""))
        trace.initial_snapshot = initial
        trace.final_snapshot = self.backend.snapshot()
        world_diff = self.backend.diff(initial, trace.final_snapshot)
        task_text = task_builder(spec, trace, world_diff) if callable(task_builder) else ""
        report = self.quality_gate.evaluate(trace, spec=spec, task_text=task_text,
                                            tool_schemas=self.backend.tool_schemas())
        return {
            "trace": trace,
            "task_natural": task_text,
            "reference_solution": trace.reference_solution(),
            "world_diff": copy.deepcopy(dict(world_diff or {})),
            "quality": report.to_dict(),
            "metadata": {"persona": spec.persona, "persona_line": spec.persona_line,
                         "archetype": spec.archetype, "tier": spec.tier,
                         "constraints": list(spec.constraints),
                         "target_complexity": spec.target_complexity,
                         "actual_complexity": report.metrics.to_dict(),
                         **dict(metadata or {})},
            "world": self.backend.freeze(),
        }

    def run(self, spec: Mapping[str, Any] | SynthesisSpec,
            executor: Any, *, task_text: str = "") -> dict[str, Any]:
        raw_spec = spec.raw if isinstance(spec, SynthesisSpec) else dict(spec)
        initial = self.backend.bootstrap(spec=raw_spec)
        result = executor(self.backend, raw_spec)
        if isinstance(result, ExecutionTrace):
            trace = result
        elif isinstance(result, Mapping) and "calls" in result:
            trace = ExecutionTrace.from_call_log(result["calls"],
                                                 conclusion=str(result.get("conclusion", "")))
        else:
            # 兼容旧 execute_organization/run_research 返回的结果：调用日志
            # 可由 environment 暴露，且绝不凭空制造参考解。
            trace = ExecutionTrace.from_call_log(
                getattr(getattr(self.backend, "environment", None), "call_log", []),
                conclusion=str((result or {}).get("conclusion", "") if isinstance(result, Mapping) else ""))
        trace.initial_snapshot = initial
        trace.final_snapshot = self.backend.snapshot()
        report = self.quality_gate.evaluate(
            trace, spec=raw_spec, task_text=task_text,
            tool_schemas=self.backend.tool_schemas())
        return {
            "trace": trace,
            "reference_solution": trace.reference_solution(),
            "initial_snapshot": initial,
            "final_snapshot": trace.final_snapshot,
            "diff": self.backend.diff(initial, trace.final_snapshot),
            "complexity": report.metrics.to_dict(),
            "quality": report.to_dict(),
        }


@dataclass(frozen=True)
class SynthesisHooks:
    """共享七阶段流程中的领域策略。

    领域实现可以提供不同的执行提示和证书编译器，但阶段顺序、轨迹来源、
    自然化后的门禁与重放规则由 :class:`ExecutionFirstPipeline` 固定。
    """

    execute: Callable[[TaskPlan, WorldBackend], ExecutionTrace]
    propose: Callable[[TaskPlan, ExecutionTrace, Mapping[str, Any]], Mapping[str, Any]]
    compile_certificate: Callable[[TaskPlan, ExecutionTrace, Mapping[str, Any], Mapping[str, Any]], Any]
    naturalize: Callable[[TaskPlan, Mapping[str, Any]], str]
    public_values: Callable[[TaskPlan], Iterable[Any]] = lambda _plan: ()
    private_values: Callable[[TaskPlan, ExecutionTrace, Any], Iterable[Any]] = lambda _plan, _trace, _cert: ()
    replay: Callable[[TaskPlan, WorldBackend, ExecutionTrace], Mapping[str, Any]] | None = None
    # 有些 Closed World 的真实依赖只能在执行后由实际资源参数确定。此 hook
    # 允许从完整 trace 编译派生图；未提供时继续使用执行前 plan.derivation，
    # 因而对现有 Open/Closed 调用方完全兼容。
    compile_derivation: Callable[[TaskPlan, ExecutionTrace], DerivationGraph] | None = None


class ExecutionFirstPipeline:
    """任务生成流程的完整执行顺序。

    唯一允许发生在题面产生前的事实来源是 ``execute`` 的成功调用。证书从
    同一轨迹和冻结 outcome 编译；自然化之后才做泄漏/复杂度门禁；最后对
    参考见证做无决策重放。调用方不能跳过或重新排序这些阶段。
    """

    def __init__(self, backend: WorldBackend,
                 quality_gate: SharedQualityGate | None = None):
        self.backend = backend
        self.quality_gate = quality_gate or SharedQualityGate()

    def construct(self, plan: TaskPlan, hooks: SynthesisHooks) -> dict[str, Any]:
        raw_spec = dict(plan.raw_spec)
        raw_spec.setdefault("archetype", plan.archetype)
        raw_spec.setdefault("tier", plan.tier)
        raw_spec.setdefault("constraints", list(plan.constraints))
        target = dict(plan.target_complexity)
        raw_spec.setdefault("target_calls", target.get("calls", (2, 99)))
        raw_spec.setdefault("read_calls_range", target.get("calls", (2, 99)))
        raw_spec.setdefault("target_entities", target.get("entities", (1, 99)))
        raw_spec.setdefault("entities_range", target.get("entities", (1, 99)))
        raw_spec.setdefault("target_tools", target.get("tools", (1, 99)))
        raw_spec.setdefault("target_depth", target.get("dependency_depth", 1))

        initial = self.backend.bootstrap(seed=plan.seed, spec=raw_spec)
        trace = hooks.execute(plan, self.backend)
        if not isinstance(trace, ExecutionTrace):
            raise TypeError("execute hook must return ExecutionTrace")
        trace.initial_snapshot = copy.deepcopy(initial)
        trace.final_snapshot = self.backend.snapshot()
        derivation = (
            hooks.compile_derivation(plan, trace)
            if hooks.compile_derivation is not None
            else plan.derivation
        )
        if not isinstance(derivation, DerivationGraph):
            raise TypeError("compile_derivation hook must return DerivationGraph")
        # execute 之前的 plan 仍只含目标；后续各阶段统一看到由真实执行编译出的
        # effective_plan，避免领域 hook 又回退到占位派生图。
        effective_plan = replace(plan, derivation=derivation)
        outcome = {
            "diff": copy.deepcopy(dict(self.backend.diff(initial, trace.final_snapshot) or {})),
            "frozen_world": self.backend.freeze(),
        }
        proposal = dict(hooks.propose(effective_plan, trace, outcome) or {})
        proposal_text = str(
            proposal.get("task_instruct")
            or proposal.get("task_natural")
            or proposal.get("task")
            or ""
        ).strip()
        proposal_reason = None
        if not proposal_text:
            status = str(proposal.get("_proposal_parse_status") or "missing_task_text")
            proposal_reason = f"proposal_invalid:{status}"
        # ``propose`` may normalize trace annotations. Recompile the derivation
        # graph afterward so later stages always consume the final trace view.
        if hooks.compile_derivation is not None:
            derivation = hooks.compile_derivation(effective_plan, trace)
            effective_plan = replace(effective_plan, derivation=derivation)
        if proposal_reason:
            # 解析失败或空提案必须立即停止题面阶段。严禁把空字符串送入
            # naturalize，让模型仅凭 persona 幻写一个与执行无关的新任务。
            certificate = {}
            task_natural = ""
        else:
            certificate = hooks.compile_certificate(effective_plan, trace, outcome, proposal)
            task_natural = str(hooks.naturalize(effective_plan, proposal) or "").strip()
        report = self.quality_gate.evaluate(
            trace, spec=raw_spec, task_text=task_natural,
            tool_schemas=self.backend.tool_schemas(),
            # 闭合状态型任务的证书由 state diff 承载，不必
            # 伪造“答案派生图”。有显式派生节点时仍严格校验。
            derivation=derivation if derivation.nodes else None,
            private_values=hooks.private_values(effective_plan, trace, certificate),
            public_values=hooks.public_values(effective_plan),
            semantic_certificate=(certificate if isinstance(certificate, Mapping) else None),
        )

        replay = (
            dict(hooks.replay(effective_plan, self.backend, trace) or {})
            if hooks.replay is not None
            else self._default_replay(trace)
        )
        reasons = list(report.reasons)
        if proposal_reason:
            reasons.append(proposal_reason)
        if not replay.get("passed", False):
            reasons.append("mechanical_replay_failed")
        quality = report.to_dict()
        quality["reasons"] = list(dict.fromkeys(reasons))
        quality["accepted"] = not quality["reasons"]
        return {
            "plan": effective_plan.to_dict(),
            "trace": trace,
            "proposal": proposal,
            "task_natural": task_natural,
            "certificate": certificate,
            "reference_solution": trace.reference_solution(),
            "outcome": outcome,
            "replay": replay,
            "quality": quality,
            "metadata": {
                "persona": plan.persona.text,
                "persona_line": plan.persona.line,
                "archetype": plan.archetype,
                "tier": plan.tier,
                "constraints": list(plan.constraints),
                "target_complexity": target,
                "actual_complexity": report.metrics.to_dict(),
                "derivation": derivation.to_dict(),
            },
        }

    def _default_replay(self, trace: ExecutionTrace) -> dict[str, Any]:
        replay_trace = self.backend.replay(trace.reference_solution(), source="replay")
        expected = trace.reference_calls
        actual = replay_trace.reference_calls
        same = len(expected) == len(actual) and all(
            left.tool == right.tool
            and dict(left.arguments) == dict(right.arguments)
            and left.observation == right.observation
            for left, right in zip(expected, actual)
        )
        return {
            "passed": same and len(actual) == len(replay_trace.calls),
            "expected_calls": len(expected),
            "successful_calls": len(actual),
            "observations_identical": same,
        }


__all__ = ["CallRecord", "ExecutionTrace", "ComplexityMetrics", "measure_complexity",
           "dependency_violations", "dependency_chain_depth", "PersonaRef",
           "DerivationNode", "DerivationGraph", "TaskPlan",
           "WorldBackend", "ClosedWorldBackend", "EvoSynthBackend", "QualityReport",
           "SharedQualityGate", "SynthesisSpec", "ExecutionFirstSynthesizer",
           "SynthesisHooks", "ExecutionFirstPipeline"]
