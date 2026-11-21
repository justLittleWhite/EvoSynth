"""把任意 ``tool_info`` 编译为开放域的声明式工具表。

本模块只解释工具声明，不包含 map、finance、search 或 medical 的工具名配方。
领域到 server 的分组由调用方给出；任务类型、复杂度和调用路径仍由共享
``SynthesisSpec``、分布控制器和研究 agent 决定。

输入同时兼容常见的 OpenAI 工具 schema 格式和项目内部 registry 格式：

``{server: [{type: function, function: {...}}]}``
``{servers: {server: {tools: [...]}}}``
``{domain: {server: [{type: function, function: {...}}]}}``
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from evosynth.task_gen.common.distribution import weighted_pick


_NON_KNOWLEDGE_NAMES = {"calculate", "calculator"}
_REASONING_NAMES = {"think", "reason", "reflect", "plan"}


def dumps_compact(value: Any) -> str:
    """编译产物用的单行 JSON：可被机器回读，且不浪费 prompt/落盘换行。"""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return text or "tool"


def _tool_kind(name: str, description: str, parameters: Mapping[str, Any]) -> str:
    name_lower = name.casefold()
    lowered = f"{name} {description}".casefold()
    if name_lower in _REASONING_NAMES:
        return "reasoning"
    # 函数名比长描述中的交叉引用更可靠。例如 fetch 的说明常出现
    # “search result”，不能因此把 fetch 误分类为 search。
    if any(token in name_lower for token in ("search", "finder", "lookup")):
        return "search"
    if any(token in name_lower for token in ("fetch", "getter", "detail", "read")):
        return "detail"
    if any(token in lowered for token in ("search", "find ", "lookup", "query")):
        return "search"
    if any(token in lowered for token in ("fetch", "getter", "get detail", "retrieve", "read ")):
        return "detail"
    if any(token in lowered for token in ("predict", "estimate", "score")):
        return "derive"
    required = set(parameters.get("required") or ())
    if required:
        return "detail"
    return "lookup"


@dataclass(frozen=True)
class CompiledTool:
    domain: str
    server: str
    name: str
    public_name: str
    description: str
    parameters: Mapping[str, Any]
    output_schema: Any
    kind: str

    @property
    def tool_id(self) -> str:
        return f"{self.server}/{self.name}"

    @property
    def knowledge(self) -> bool:
        return self.kind != "reasoning"

    def public_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.public_name,
            "description": self.description or self.name,
            "parameters": copy.deepcopy(dict(self.parameters)),
        }}

    def registry_record(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": copy.deepcopy(dict(self.parameters)),
                "output_schema": copy.deepcopy(self.output_schema)}


@dataclass(frozen=True)
class CompiledOpenDomain:
    name: str
    tools: tuple[CompiledTool, ...]
    registry: Mapping[str, Any]

    @property
    def servers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(tool.server for tool in self.tools))

    @property
    def knowledge_tools(self) -> tuple[CompiledTool, ...]:
        return tuple(tool for tool in self.tools if tool.knowledge)

    def public_schemas(self, selected: Iterable[str] | None = None) -> list[dict[str, Any]]:
        chosen = set(selected or ())
        tools = self.tools if not chosen else tuple(
            tool for tool in self.tools if tool.public_name in chosen or tool.tool_id in chosen)
        return [tool.public_schema() for tool in tools]

    def by_public_name(self, name: str) -> CompiledTool:
        matches = [tool for tool in self.tools if tool.public_name == name]
        if len(matches) != 1:
            raise KeyError(f"unknown or ambiguous public tool: {name}")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "servers": list(self.servers),
                "tools": [{"tool_id": tool.tool_id, "public_name": tool.public_name,
                           "kind": tool.kind}
                          for tool in self.tools]}


def _tool_entries(record: Any) -> list[Any] | None:
    """Return a tool list when ``record`` is one server, not a domain group."""
    if isinstance(record, list):
        return record
    if isinstance(record, Mapping):
        tools = record.get("tools")
        if isinstance(tools, list):
            return tools
    return None


def _iter_server_records(raw: Mapping[str, Any]) -> Iterable[tuple[str, list[Any]]]:
    """Yield ``(server, tool entries)`` for flat pools and domain-grouped pools.

    A domain group is ``{domain: {server: [tools...]}}``. Server names stay the
    inner keys, so a category label is never treated as an MCP server.
    """
    body = raw.get("servers") if isinstance(raw.get("servers"), Mapping) else raw
    if not isinstance(body, Mapping):
        return
    for key, record in body.items():
        if str(key) in {"metadata", "version"}:
            continue
        entries = _tool_entries(record)
        if entries is not None:
            if entries:
                yield str(key), entries
            continue
        if not isinstance(record, Mapping):
            continue
        for server, inner in record.items():
            inner_entries = _tool_entries(inner)
            if inner_entries:
                yield str(server), inner_entries


def _iter_source(raw: Mapping[str, Any], selected_servers: Sequence[str] | None
                 ) -> Iterable[tuple[str, Mapping[str, Any]]]:
    selected = set(selected_servers or ())
    for server, entries in _iter_server_records(raw):
        if selected and server not in selected:
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            function = entry.get("function") if isinstance(entry.get("function"), Mapping) else entry
            if function.get("name"):
                yield server, function


def _infer_domain(path_or_raw: str | Path | Mapping[str, Any]) -> str:
    """Infer a label when callers provide only ``tool_info``.

    The label is metadata used in task IDs and manifests; it never selects a
    tool recipe.  A file stem is preferred for reproducibility, while an
    in-memory registry falls back to its first server key.
    """
    if isinstance(path_or_raw, (str, Path)):
        stem = Path(path_or_raw).stem.strip()
        if stem:
            return stem
    if isinstance(path_or_raw, Mapping):
        servers = list_mcp_servers(path_or_raw)
        if servers:
            return servers[0]
    return "open_domain"


def list_mcp_servers(path_or_raw: str | Path | Mapping[str, Any]) -> list[str]:
    """按文件中的出现顺序列出带工具声明的 MCP server。"""
    if isinstance(path_or_raw, Mapping):
        raw = path_or_raw
    else:
        raw = json.loads(Path(path_or_raw).read_text(encoding="utf-8"))
    return [server for server, _entries in _iter_server_records(raw)]


def load_mcp_pool(path: str | Path) -> dict[str, dict[str, list[Any]]]:
    """读取 ``{domain: {server: [tool, ...]}}`` 的 MCP 池。server 名必须全池唯一。"""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read mcp pool {source}: {exc}") from exc
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"mcp pool must be a non-empty JSON object: {source}")
    pool: dict[str, dict[str, list[Any]]] = {}
    seen: dict[str, str] = {}
    for domain, group in raw.items():
        if not isinstance(group, Mapping):
            raise ValueError(
                f"mcp pool domain {domain!r} must map server names to tool lists: {source}")
        pool[str(domain)] = {}
        for server, record in group.items():
            entries = _tool_entries(record)
            if entries is None:
                raise ValueError(f"mcp pool server {domain}/{server} has no tool list: {source}")
            if str(server) in seen:
                raise ValueError(
                    f"mcp pool server {server!r} appears in both "
                    f"{seen[str(server)]!r} and {domain!r}: {source}")
            seen[str(server)] = str(domain)
            pool[str(domain)][str(server)] = entries
    return pool


def _function_of(entry: Any) -> Mapping[str, Any]:
    if isinstance(entry, Mapping):
        function = entry.get("function")
        return function if isinstance(function, Mapping) else entry
    return {}


def _knowledge_functions(entries: Iterable[Any]) -> list[Mapping[str, Any]]:
    out = []
    for entry in entries:
        function = _function_of(entry)
        name = str(function.get("name") or "")
        if not name or name.casefold() in _NON_KNOWLEDGE_NAMES \
                or name.casefold() in _REASONING_NAMES:
            continue
        out.append(function)
    return out


def sample_pool_servers(pool: Mapping[str, Mapping[str, Any]], seed: int,
                        per_domain: int = 3, *,
                        used: Mapping[str, int] | None = None,
                        alpha: float = 1.0) -> dict[str, list[str]]:
    """每个 domain 内抽 ``per_domain`` 个 server（不足则全取）。

    只在含知识工具的 server 里抽；同一 seed、同一 ``used`` 的结果可复现。

    ``used`` 为 ``None`` 时每个 domain 内均匀随机抽 ``per_domain`` 个（历史行为）。

    ``used`` 为 ``{server: 已被使用次数}`` 时按频率控制抽取，且**跨 domain 统一抽**：
    候选总数仍是 ``per_domain × 含知识 server 的 domain 数``，但在全部 server
    上不放回地逐个抽，每个 server 权重 ``1 / (1 + used) ** alpha``，用得越少越
    优先，与 archetype 的 coverage 逆频率是同一机制。这样 server 少的 domain
    不会因为"每个 domain 固定出 per_domain 个候选"而被反复选中。未出现在
    ``used`` 里的 server 次数按 0 计。返回值仍按 domain 分组，保持池内顺序。
    """
    if per_domain < 1:
        raise ValueError("per_domain must be positive")
    eligible_by_domain: dict[str, list[str]] = {}
    for domain, group in pool.items():
        eligible = [str(server) for server, entries in group.items()
                    if _knowledge_functions(entries)]
        if eligible:
            eligible_by_domain[str(domain)] = eligible
    if used is None:
        return {
            domain: random.Random(f"{seed}:{domain}").sample(
                eligible, min(per_domain, len(eligible)))
            for domain, eligible in eligible_by_domain.items()}

    remaining: dict[str, float] = {}
    domain_of: dict[str, str] = {}
    for domain, eligible in eligible_by_domain.items():
        for name in eligible:
            remaining.setdefault(name, 1.0)
            domain_of.setdefault(name, domain)
    take = min(per_domain * len(eligible_by_domain), len(remaining))
    rng = random.Random(f"{seed}:global")
    chosen: set[str] = set()
    for _ in range(take):
        pick = weighted_pick(rng, remaining, used=used, alpha=alpha)
        chosen.add(pick)
        del remaining[pick]
    sampled: dict[str, list[str]] = {}
    for domain, eligible in eligible_by_domain.items():
        group = [name for name in eligible if name in chosen and domain_of[name] == domain]
        if group:
            sampled[domain] = group
    return sampled


STALL_FACTOR = 0.1


@dataclass(frozen=True)
class AnchorPick:
    """一次采样给出的锚点 server 与可选同伴。

    ``anchor`` 是本组必须覆盖的 server（由覆盖缺口决定，不由 LLM 决定）；
    ``companions`` 只是可选搭档，LLM 可以取其中任意子集。
    """

    anchor: str
    domain: str
    companions: tuple[str, ...]

    def candidates(self) -> tuple[str, ...]:
        return (self.anchor, *self.companions)


def pool_server_index(pool: Mapping[str, Mapping[str, Any]]) -> tuple[
        dict[str, str], dict[str, list[str]]]:
    """返回 ``(domain_of, tools_of)``，只含带知识工具的 server。

    ``tools_of[server]`` 是该 server 知识工具的 public_name，与
    ``compile_tool_info`` 生成的名字一致，可直接和 coverage 里的 ``tool_fn`` 对齐。
    """
    domain_of: dict[str, str] = {}
    tools_of: dict[str, list[str]] = {}
    for domain, group in pool.items():
        for server, entries in group.items():
            functions = _knowledge_functions(entries)
            if not functions:
                continue
            domain_of[str(server)] = str(domain)
            tools_of[str(server)] = [
                f"{_slug(str(server))}__{_slug(str(fn['name']))}" for fn in functions]
    return domain_of, tools_of


def _draw(rng: random.Random, weights: Mapping[str, float]) -> str:
    total = sum(weights.values())
    target = rng.random() * total
    accumulated = 0.0
    for key, weight in weights.items():
        accumulated += weight
        if target <= accumulated:
            return key
    return list(weights)[-1]


def sample_anchor_candidates(
        pool: Mapping[str, Mapping[str, Any]], seed: int, *,
        tool_counts: Mapping[str, int],
        domain_counts: Mapping[str, int],
        server_pending: Mapping[str, int] | None = None,
        stalled: Iterable[str] = (),
        companions: int = 3,
        alpha: float = 1.0,
        server_bias: Mapping[str, float] | None = None) -> AnchorPick:
    """按覆盖缺口选锚点 server，再配若干可选同伴。

    分两层，每层都是「目标份额 × 已覆盖量的逆频率」：

    1. domain：目标份额 ∝ ``sqrt(该 domain 的知识工具数)``，权重
       ``sqrt(T_d) / (1 + domain_counts[d]) ** alpha``。开方让工具多的 domain
       份额更大，但不会按工具数线性放大而淹没小 domain。
    2. domain 内 server：权重 ``sqrt(n_tools) × 工具平均缺口 / (1 + pending)``，
       工具平均缺口是该 server 各工具 ``1 / (1 + tool_counts[t]) ** alpha`` 的均值。
       这样用工具覆盖量而不是 server 被选次数来驱动，工具多的 server 不会被稀释。

    ``tool_counts`` 与 ``domain_counts`` 应是已发布任务的计数（调用方可再加上在途
    预留量）。``server_pending`` 是在途预留量。``stalled`` 里的 server 权重乘
    ``STALL_FACTOR``：它们多次做锚点却一直没产出任务，不再占满采样预算。

    同伴在其余 server 里按同一权重不放回抽取，跨 domain。同一 ``seed`` 与同样的
    计数下结果可复现。
    """
    if companions < 0:
        raise ValueError("companions must be non-negative")
    domain_of, tools_of = pool_server_index(pool)
    if not domain_of:
        raise ValueError("mcp pool has no server with knowledge tools")
    pending = server_pending or {}
    stalled_set = set(stalled)
    rng = random.Random(f"{seed}:anchor")

    def server_weight(server: str) -> float:
        tools = tools_of[server]
        gap = sum(1.0 / (1 + tool_counts.get(tool, 0)) ** alpha
                  for tool in tools) / len(tools)
        weight = (len(tools) ** 0.5) * gap / (1 + pending.get(server, 0)) ** alpha
        weight *= (server_bias or {}).get(server, 1.0)  # feedback 的锚点兼容率偏好
        return weight * (STALL_FACTOR if server in stalled_set else 1.0)

    servers_by_domain: dict[str, list[str]] = {}
    for server, domain in domain_of.items():
        servers_by_domain.setdefault(domain, []).append(server)
    domain_weight = {
        domain: (sum(len(tools_of[s]) for s in servers) ** 0.5)
        / (1 + domain_counts.get(domain, 0)) ** alpha
        for domain, servers in servers_by_domain.items()}
    domain = _draw(rng, domain_weight)
    anchor = _draw(rng, {s: server_weight(s) for s in servers_by_domain[domain]})

    rest = {s: server_weight(s) for s in domain_of if s != anchor}
    chosen: list[str] = []
    for _ in range(min(companions, len(rest))):
        pick = _draw(rng, rest)
        chosen.append(pick)
        del rest[pick]
    return AnchorPick(anchor=anchor, domain=domain, companions=tuple(chosen))


def _brief_description(text: Any) -> str:
    """只留功能描述：去掉 docstring 里的参数段，折叠空白。"""
    body = re.split(r"\n\s*(?:Args|Arguments|Parameters|Params)\s*:", str(text or ""),
                    maxsplit=1)[0]
    return " ".join(body.split())


def _server_lookup(pool: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {str(server): entries for group in pool.values()
            for server, entries in group.items()}


def render_server_catalog(pool: Mapping[str, Mapping[str, Any]],
                          servers: Iterable[str]) -> str:
    """把候选 server 渲染成 prompt 用文本：只含函数名与 description。

    ::

        Servers:
        [server]
        - function1: <desc>
        - function2: <desc>
    """
    lookup = _server_lookup(pool)
    lines = ["Servers:"]
    for server in servers:
        lines.append(f"[{server}]")
        for function in _knowledge_functions(lookup[str(server)]):
            name = str(function["name"])
            lines.append(f"- {name}: {_brief_description(function.get('description')) or name}")
    return "\n".join(lines)


def count_knowledge_tools(pool: Mapping[str, Mapping[str, Any]],
                          servers: Iterable[str]) -> int:
    lookup = _server_lookup(pool)
    return sum(len(_knowledge_functions(lookup[str(server)])) for server in servers)


def _coerce_parameter_schema(server: str, name: str, parameters: Any) -> dict[str, Any]:
    """把工具参数统一成 JSON Schema object。

    池里一部分 playground 导出把 parameters 写成
    ``[{name, required, type}, ...]``，而不是 ``{type: object, properties}``。
    只转写声明，不补缺失字段的语义。
    """
    if not parameters:
        return {"type": "object", "properties": {}}
    if isinstance(parameters, Mapping):
        return copy.deepcopy(dict(parameters))
    if isinstance(parameters, list):
        properties: dict[str, Any] = {}
        required: list[str] = []
        for index, item in enumerate(parameters):
            if not isinstance(item, Mapping) or not str(item.get("name") or "").strip():
                raise ValueError(
                    f"{server}/{name}: parameter[{index}] must be an object with a name")
            field = str(item["name"]).strip()
            if field in properties:
                raise ValueError(f"{server}/{name}: duplicate parameter {field}")
            prop: dict[str, Any] = {"type": str(item.get("type") or "string")}
            description = str(item.get("description") or "").strip()
            if description:
                prop["description"] = description
            properties[field] = prop
            if item.get("required") is True:
                required.append(field)
        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema
    raise ValueError(f"{server}/{name}: parameters must be an object schema")


def compile_tool_info(path_or_raw: str | Path | Mapping[str, Any], *, domain: str | None = None,
                      servers: Sequence[str] | None = None) -> CompiledOpenDomain:
    domain = str(domain or _infer_domain(path_or_raw))
    if isinstance(path_or_raw, Mapping):
        raw = copy.deepcopy(dict(path_or_raw))
    else:
        raw = json.loads(Path(path_or_raw).read_text(encoding="utf-8"))
    records = list(_iter_source(raw, servers))
    if not records:
        raise ValueError("tool_info contains no selected tools")
    tools: list[CompiledTool] = []
    for server, fn in records:
        name = str(fn["name"])
        if name.casefold() in _NON_KNOWLEDGE_NAMES:
            continue
        parameters = _coerce_parameter_schema(server, name, fn.get("parameters"))
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        output_schema = copy.deepcopy(fn.get("output_schema"))
        public_name = f"{_slug(server)}__{_slug(name)}"
        description = str(fn.get("description") or "").strip()
        tools.append(CompiledTool(
            domain=domain, server=server, name=name, public_name=public_name,
            description=description, parameters=parameters, output_schema=output_schema,
            kind=_tool_kind(name, description, parameters),
        ))
    knowledge = [tool for tool in tools if tool.knowledge]
    if not knowledge:
        raise ValueError("domain has no knowledge tools after exclusions")
    registry_servers: dict[str, Any] = {}
    for server in dict.fromkeys(tool.server for tool in tools):
        server_tools = [tool.registry_record() for tool in tools if tool.server == server]
        registry_servers[server] = {"tools": server_tools, "fidelity": "schema_full",
                                    "role": "knowledge"}
    return CompiledOpenDomain(domain, tuple(tools), {"servers": registry_servers})


def write_compiled_registry(compiled: CompiledOpenDomain, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_compact(compiled.registry) + "\n", encoding="utf-8")
    return target


__all__ = ["CompiledTool", "CompiledOpenDomain", "compile_tool_info",
           "load_mcp_pool", "sample_pool_servers", "render_server_catalog",
           "count_knowledge_tools", "AnchorPick", "STALL_FACTOR",
           "pool_server_index", "sample_anchor_candidates",
           "write_compiled_registry", "dumps_compact"]
