"""由注册表驱动的受控 EvoSynth 工具执行层。

每个成功观测同 query digest、字段型 claim 和观测哈希一起保存。任务构造会
冻结一个基础世界；EvoSynth 推理先读取该基础世界，未命中的合法查询可以
通过 8822 按需材质化并缓存在当前 episode。显式 reference replay 仍只读取
已固定观测，不访问实时 API。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ast
import copy
import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

from openai import OpenAI

from evosynth.utils.paths import PROJECT_ROOT
from evosynth.task_gen.generation import prompt_loader
from evosynth.task_gen.generation.runtime import print_api_error


_MCP_WORLD_PROMPT = PROJECT_ROOT / "prompts" / "mcp" / "system_prompt.txt"
_SEARCH_WORLD_PROMPT = PROJECT_ROOT / "prompts" / "search" / "system_prompt.txt"
# 兼容离线注入 client 的旧测试；公开流水线只使用下方显式配置 owner。
_WORLD_MODEL = ""


@dataclass(frozen=True)
class WorldRuntimeConfig:
    base_url: str
    model: str
    api_key: str


_WORLD_RUNTIME: WorldRuntimeConfig | None = None


def configure_world_runtime(*, base_url: str, model: str,
                            api_key: str) -> WorldRuntimeConfig:
    values = (str(base_url).strip(), str(model).strip(), str(api_key).strip())
    if not all(values):
        raise ValueError("world base URL, model and API key must be non-empty")
    global _WORLD_RUNTIME
    _WORLD_RUNTIME = WorldRuntimeConfig(*values)
    return _WORLD_RUNTIME


def world_runtime_config() -> WorldRuntimeConfig:
    if _WORLD_RUNTIME is None:
        raise RuntimeError("EvoSynth runtime is not configured")
    return _WORLD_RUNTIME


_WORLD_MAX_TOKENS = int(os.environ.get("EVOSYNTH_WORLD_MAX_TOKENS", "16384"))
_WORLD_RATE_LIMIT_RETRIES = int(os.environ.get(
    "EVOSYNTH_WORLD_RATE_LIMIT_RETRIES", "3"))
# Retries for a completed agent response that is empty, malformed, or violates
# the frozen consistency contract. This is separate from transport retries.
_WORLD_MATERIALIZATION_RETRIES = int(os.environ.get(
    "EVOSYNTH_WORLD_MATERIALIZATION_RETRIES", "3"))
_WORLD_ENABLE_THINKING_RAW = os.environ.get(
    "EVOSYNTH_WORLD_ENABLE_THINKING", "").strip().casefold()
_WORLD_ENABLE_THINKING = (
    None if not _WORLD_ENABLE_THINKING_RAW
    else _WORLD_ENABLE_THINKING_RAW in {"1", "true", "yes", "on"}
)
_WORLD_CONCURRENCY = int(os.environ.get("EVOSYNTH_WORLD_CONCURRENCY", "1"))
_WORLD_TIMEOUT = float(os.environ.get("EVOSYNTH_WORLD_TIMEOUT_SECONDS", "240"))
_WORLD_QUEUE_TIMEOUT = float(os.environ.get(
    "EVOSYNTH_WORLD_QUEUE_TIMEOUT_SECONDS", "600"))
_WORLD_SEMAPHORE = threading.BoundedSemaphore(max(1, _WORLD_CONCURRENCY))


def configure_world_concurrency(workers: int) -> int:
    """把世界模型在途请求上限设为命令行给出的 worker 数。"""
    if workers <= 0:
        raise ValueError("world concurrency must be positive")
    global _WORLD_CONCURRENCY, _WORLD_SEMAPHORE
    _WORLD_CONCURRENCY = workers
    _WORLD_SEMAPHORE = threading.BoundedSemaphore(workers)
    return _WORLD_CONCURRENCY


class WorldInfrastructureError(RuntimeError):
    """8822 请求未完成；不得当成观测内容或语义失败继续返修。"""


class WorldMaterializationError(RuntimeError):
    """8822 已响应，但未产生符合注册工具输出合同的冻结观测。"""


def _registered_tools(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for server_name, server in (registry.get("servers") or {}).items():
        if not isinstance(server, Mapping):
            continue
        for raw in server.get("tools") or ():
            if not isinstance(raw, Mapping):
                continue
            result.append({
                "server": str(server_name),
                "name": str(raw.get("name") or ""),
                "description": str(raw.get("description") or ""),
                "parameters": raw.get("parameters") or {},
                "output_schema": raw.get("output_schema"),
            })
    return result


def world_prompt_path_for_registry(
        registry_or_path: Mapping[str, Any] | str | Path) -> Path:
    """Select the AgentWorld prompt family for this registry."""
    if isinstance(registry_or_path, Mapping):
        registry = registry_or_path
    else:
        registry = json.loads(
            Path(registry_or_path).read_text(encoding="utf-8"))
    names = {item["name"].casefold() for item in _registered_tools(registry)}
    return _SEARCH_WORLD_PROMPT if {"search", "fetch"} <= names else _MCP_WORLD_PROMPT


def render_world_system_prompt(registry: Mapping[str, Any]) -> str:
    prompt_path = world_prompt_path_for_registry(registry)
    tool_definitions = json.dumps(
        _registered_tools(registry), ensure_ascii=False, indent=2,
        sort_keys=True)
    base = prompt_path.read_text(encoding="utf-8")
    return (
        base.replace("{tool_definitions}", tool_definitions)
        .replace("{demonstrations}", "")
        .replace("{current_date}", datetime.date.today().isoformat())
    )


def _canon(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def query_digest(server: str, tool: str, arguments: Mapping[str, Any]) -> str:
    return sha256(_canon({"server": server, "tool": tool, "arguments": dict(arguments)}).encode()).hexdigest()


def versioned_query_digest(server: str, tool: str,
                           arguments: Mapping[str, Any],
                           state_version: str | None = None) -> str:
    """Bind observations to a generic world version when state can change."""
    if not state_version:
        return query_digest(server, tool, arguments)
    return sha256(_canon({"state_version": str(state_version),
                          "server": server, "tool": tool,
                          "arguments": dict(arguments)}).encode()).hexdigest()


def _declared_string_contract(schema: Any) -> bool:
    """注册表把输出声明为裸字符串 ``str``/``string``。"""
    return isinstance(schema, str) and schema.casefold().strip() in {"str", "string"}


_ERROR_ONLY_KEYS = {"error", "message", "detail", "errors"}


def _error_only_schema(schema: Any) -> bool:
    """注册表 schema 若只有 error/message 之类的键，说明它是从失败样本里抽出来的。

    真实成功返回里并没有这些字段；把它当显式合同会让每次成功观测都被判成
    "no declared output fields"，所以按未声明处理（首次返回什么形状就接受）。
    """
    return (isinstance(schema, Mapping) and bool(schema)
            and "properties" not in schema and "type" not in schema
            and {str(k).casefold() for k in schema} <= _ERROR_ONLY_KEYS)


def _split_top_level_union(text: str) -> list[str]:
    parts: list[str] = []
    depth, quote, start = 0, "", 0
    for index, char in enumerate(text):
        if quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char in "{[(":
            depth += 1
        elif char in "}])":
            depth -= 1
        elif char == "|" and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


def _union_string_fields(schema: Any) -> tuple[str, ...] | None:
    """解析 ``"{...} | str | {...}"`` 这种联合字符串合同，返回顶层字段并集。

    不是联合/字面量字符串时返回 None。含裸 ``str`` 分支（文本返回）时额外
    保留 ``value``，使纯文本观测仍能投影成 claim。
    """
    if not isinstance(schema, str):
        return None
    text = schema.strip()
    if not text or text[0] not in "{[" and "|" not in text:
        return None
    fields: list[str] = []
    has_scalar = False
    parsed_any = False
    for part in _split_top_level_union(text):
        try:
            value = ast.literal_eval(part)
        except (ValueError, SyntaxError):
            if part.casefold() in {"str", "string", "int", "float", "number",
                                   "bool", "boolean", "null", "none"}:
                has_scalar = True
            continue
        parsed_any = True
        for name in _fields(value):
            if name not in fields:
                fields.append(name)
    if not parsed_any:
        return None
    if has_scalar and "value" not in fields:
        fields.append("value")
    return tuple(fields)


def _explicit_output_schema(schema: Any) -> bool:
    """注册表里写明的输出合同。空值和裸 ``str`` 都不算显式合同。

    MCP 工具的文本输出经常本身就是 JSON 字符串，``str`` 只说明"返回文本"，
    不给出字段，因此与空合同一样：首次返回什么形状就接受什么形状。
    """
    if schema is None:
        return False
    if isinstance(schema, str) and not schema.strip():
        return False
    if _declared_string_contract(schema):
        return False
    if isinstance(schema, (Mapping, list, tuple)) and not schema:
        return False
    if _error_only_schema(schema):
        return False
    return True


def _infer_output_schema(observation: Any) -> Any | None:
    """从一次真实返回里抽出可复用的输出合同。

    对象取顶层键，对象列表取各条记录的键并集，并集里每条都出现的键为必填。
    空列表和空对象没有可固定的字段，返回 ``None``，留给下一次非空返回。
    """
    if isinstance(observation, list):
        if not observation:
            return None
        if all(isinstance(item, Mapping) for item in observation):
            properties: dict[str, Any] = {}
            for item in observation:
                for key, value in item.items():
                    name = str(key)
                    current = properties.get(name)
                    if current is None or current.get("type") == "null":
                        inferred = _infer_output_schema(value)
                        if inferred is not None:
                            properties[name] = inferred
            if not properties:
                return None
            required = [key for key in properties
                        if all(key in item for item in observation)]
            items: dict[str, Any] = {"type": "object", "properties": properties}
            if required:
                items["required"] = required
            return {"type": "array", "items": items}
        item_schema = _infer_output_schema(observation[0])
        if item_schema is None:
            return None
        return {"type": "array", "items": item_schema}
    if isinstance(observation, Mapping):
        properties = {}
        for key, value in observation.items():
            inferred = _infer_output_schema(value)
            if inferred is not None:
                properties[str(key)] = inferred
        if not properties:
            return None
        return {
            "type": "object",
            "properties": properties,
            "required": list(properties),
        }
    if isinstance(observation, bool):
        return {"type": "boolean"}
    if isinstance(observation, int):
        return {"type": "integer"}
    if isinstance(observation, float):
        return {"type": "number"}
    if isinstance(observation, str):
        return {"type": "string"}
    if observation is None:
        return {"type": "null"}
    return None


def _fields(schema: Any) -> tuple[str, ...]:
    if isinstance(schema, Mapping):
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            return tuple(str(k) for k in properties)
        if schema.get("type") == "array" and "items" in schema:
            return _fields(schema.get("items"))
        if schema.get("type") in {
                "string", "integer", "number", "boolean", "null"}:
            return ("value",)
        # 旧注册表也允许 ``{"field": "str"}`` 这样的字段示例；只有
        # 不带标准 JSON Schema 关键字时，顶层键才是业务字段。
        return tuple(str(k) for k in schema)
    if isinstance(schema, list) and schema and isinstance(schema[0], Mapping):
        return _fields(schema[0])
    if isinstance(schema, list) and schema:
        return ("value",)
    if isinstance(schema, (str, int, float, bool)):
        # 注册表允许标量/标量数组输出；统一投影为 value claim。
        return ("value",)
    return ()


def _field_key(value: Any) -> str:
    """Normalize a declared field name without any domain vocabulary."""
    return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())


def _materializable_descriptor(value: Any) -> bool:
    """Whether a compact example-schema leaf can produce an actual scalar."""
    if isinstance(value, Mapping):
        kinds = value.get("type")
        if isinstance(kinds, str):
            return kinds.casefold() not in {"null", "missing", "none"}
        return True
    if isinstance(value, str):
        lowered = value.casefold().strip()
        return lowered not in {"null", "missing", "none", "nonetype"}
    return value is not None


def _project_contract(schema: Any, requested_fields: tuple[str, ...], *,
                      salt: str, maximum_fields: int = 16) -> Any:
    """Build a bounded, schema-derived materialization view.

    Large provider contracts can contain hundreds of optional fields and do
    not fit one world-model response.  The projection is selected only from
    the supplied output contract: sampled information facets are retained
    first and remaining materializable fields are chosen deterministically.
    No server, tool, domain, or field recipe is embedded here.
    """
    if isinstance(schema, list):
        if not schema:
            return []
        return [_project_contract(schema[0], requested_fields,
                                  salt=salt + "/*", maximum_fields=maximum_fields)]
    if not isinstance(schema, Mapping):
        return schema
    properties = schema.get("properties")
    standard = isinstance(properties, Mapping)
    items = properties if standard else schema
    if not isinstance(items, Mapping):
        if schema.get("type") == "array" and "items" in schema:
            result = dict(schema)
            result["items"] = _project_contract(
                schema["items"], requested_fields, salt=salt + "/*",
                maximum_fields=maximum_fields)
            return result
        return schema
    # ``items`` is already the properties mapping for standard JSON Schema;
    # for legacy example contracts every top-level key is a business field.
    business = [(str(key), value) for key, value in items.items()]
    if len(business) <= maximum_fields:
        return schema
    wanted = {_field_key(value) for value in requested_fields if str(value).strip()}

    def requested(key: str, value: Any) -> bool:
        normalized = _field_key(key)
        declared = {_field_key(item) for item in _fields(value)}
        return (any(normalized == item or normalized.endswith(" " + item)
                    or item.endswith(" " + normalized) for item in wanted)
                or bool(declared & wanted))

    preferred = [(key, value) for key, value in business if requested(key, value)]
    preferred_keys = {key for key, _value in preferred}
    remaining = [(key, value) for key, value in business
                 if key not in preferred_keys and _materializable_descriptor(value)]
    remaining.sort(key=lambda item: sha256(
        (salt + "\0" + _field_key(item[0])).encode()).hexdigest())
    chosen = preferred + remaining[:max(0, maximum_fields - len(preferred))]
    if not chosen:
        # A contract containing only explicitly absent fields cannot support a
        # grounded task.  Keep a deterministic bounded subset so the lens will
        # fail closed if the model cannot produce a declared value.
        chosen = sorted(business, key=lambda item: item[0])[:maximum_fields]
    selected = {
        key: _project_contract(value, requested_fields,
                               salt=salt + "/" + key,
                               maximum_fields=maximum_fields)
        for key, value in chosen
    }
    if not standard:
        return selected
    result = {key: value for key, value in schema.items()
              if key not in {"properties", "required"}}
    result["properties"] = selected
    required = [str(item) for item in schema.get("required", ()) if str(item) in selected]
    if required:
        result["required"] = required
    return result


def _context_tokens(value: Any) -> set[str]:
    """提取参数中的稳定实体/模式标记，用于选择相关冻结观测。"""
    output: set[str] = set()
    if isinstance(value, Mapping):
        for item in value.values():
            output |= _context_tokens(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            output |= _context_tokens(item)
    elif isinstance(value, str):
        token = " ".join(value.casefold().strip().split())
        if len(token) >= 4:
            output.add(token)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if abs(float(value)) >= 20:
            output.add(str(value))
    return output


def _json_response(raw: str, *, allow_plain_string: bool = False) -> Any:
    """解析纯 JSON 或带模型思考前缀的单一 JSON 值；其余内容失效关闭。"""
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty materializer response")
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    decoder = json.JSONDecoder()
    # output_schema 允许字符串、数值、布尔和 null；旧实现只扫描对象/数组，
    # 导致 fetch 等合法标量工具永远无法材质化。
    starts = [i for i, char in enumerate(text)
              if char in "[{\"-0123456789" or char in "tfn"]
    for start in starts:
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        trailing = text[start + end:].strip()
        if not trailing or trailing == "```":
            return value
    if allow_plain_string:
        # A provider whose output contract is the scalar string often emits the
        # string value directly (for example a report or execution plan).  It is
        # already the complete observation, not commentary around structured
        # JSON.  Accepting it is a schema-level rule shared by every server.
        return text
    raise ValueError("materializer response contains no single JSON value")


@dataclass(frozen=True)
class FrozenObservation:
    server: str
    tool: str
    arguments: dict[str, Any]
    observation: Any
    claims: tuple[dict[str, Any], ...]
    query_digest: str
    observation_hash: str


class SchemaLens:
    """通用 observation lens：只把 schema 已声明的顶层字段变成 claim。"""
    def __init__(self, server: str, tool: Mapping[str, Any]):
        self.server, self.tool = server, dict(tool)
        self.name = str(tool["name"])
        schema = tool.get("output_schema")
        self.output_schema = schema
        # 没有显式合同时，第一次返回什么形状就接受什么形状，不套 ``value``。
        self.open_schema = not _explicit_output_schema(schema)
        self.require_declared_fields = (
            bool(tool.get("_require_declared_fields")) and not self.open_schema)
        union_fields = None if self.open_schema else _union_string_fields(schema)
        if self.open_schema:
            self.fields = ()
        elif union_fields:
            self.fields = union_fields
        else:
            self.fields = _fields(schema)
        if not self.open_schema and not self.fields:
            # 显式标量合同没有具名字段，观测按根标量 ``value`` 投影。
            self.fields = ("value",)

    def _check_output_shape(self, observation: Any) -> None:
        schema = self.output_schema
        if not isinstance(schema, Mapping):
            return
        kind = schema.get("type")
        if kind == "array" and not isinstance(observation, list):
            raise ValueError(f"{self.server}/{self.name}: output schema requires an array")
        if kind == "object" and not isinstance(observation, Mapping):
            raise ValueError(f"{self.server}/{self.name}: output schema requires an object")
        expected = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
        }.get(str(kind))
        if expected is None:
            return
        if kind == "null":
            if observation is not None:
                raise ValueError(f"{self.server}/{self.name}: output schema requires null")
            return
        if isinstance(observation, bool) and kind in {"integer", "number"}:
            raise ValueError(f"{self.server}/{self.name}: output schema requires {kind}")
        if not isinstance(observation, expected):
            raise ValueError(f"{self.server}/{self.name}: output schema requires {kind}")

    def extract(self, observation: Any, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
        if self.open_schema:
            inferred = _infer_output_schema(observation)
            self.fields = _fields(inferred) if inferred is not None else ()
            scalar_row = (
                not isinstance(observation, (Mapping, list))
                or (isinstance(observation, list) and observation
                    and not isinstance(observation[0], Mapping)))
            if not self.fields and scalar_row:
                self.fields = ("value",)
        elif self.require_declared_fields:
            self._check_output_shape(observation)
        if not isinstance(observation, (Mapping, list)):
            observation = {"value": observation}
        rows = observation if isinstance(observation, list) else [observation]
        rows = [item if isinstance(item, Mapping) else {"value": item} for item in rows]
        if not all(isinstance(row, Mapping) for row in rows):
            raise ValueError(f"{self.server}/{self.name}: observation must be object or object list")
        # 空记录集是合法的真实观测:某只股票若在观察区间内没有分红/拆股,
        # ``get_stock_actions`` 的真实返回就是 ``[]``;要求它至少含一个声明
        # 字段等于宣布"没有事件"这种事实非法。这里把"零条记录"与"有记录但
        # 不含任何声明字段"区分开:前者返回空 claim 集,后者仍是 schema 违约。
        if not rows:
            return ()
        if self.require_declared_fields:
            for row in rows:
                missing = [field for field in self.fields if field not in row]
                if missing:
                    raise ValueError(
                        f"{self.server}/{self.name}: observation missing declared "
                        f"output fields {missing}")
        subject = next((str(arguments[k]) for k in ("ticker", "place_id", "address", "origin") if k in arguments),
                       query_digest(self.server, self.name, arguments)[:16])
        claims = []
        for index, row in enumerate(rows):
            for field in self.fields:
                if field in row:
                    claims.append({"subject": subject, "predicate": f"{self.server}/{self.name}.{field}",
                                   "value": row[field], "row": index})
        if not claims:
            if self.open_schema:
                return ()
            shape = _canon(observation)
            raise ValueError(f"{self.server}/{self.name}: no declared output fields observed; shape={shape[:500]}")
        return tuple(claims)


class ControlledWorld:
    def __init__(self, registry_path: Path, world_path: Path, *, client: Any | None = None,
                 world_context: Mapping[str, Any] | None = None,
                 queue_wait_observer: Callable[[float], None] | None = None):
        self.registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        self.world_prompt_path = world_prompt_path_for_registry(self.registry)
        self.world_system_prompt = render_world_system_prompt(self.registry)
        self.world_path = Path(world_path)
        self.rows: dict[str, FrozenObservation] = {}
        self.inferred_output_schemas: dict[str, Any] = {}
        self._schema_lock = threading.Lock()
        self.client = client
        self._model_id: str | None = None
        self.world_context = copy.deepcopy(dict(world_context or {}))
        self.queue_wait_observer = queue_wait_observer
        # Per-call provenance for offline reliability analysis.  This is kept
        # separate from the frozen world itself and is safe to omit in replay.
        self.last_materialization_meta: dict[str, Any] = {}
        if self.world_path.exists():
            self._load()

    def _tool(self, server: str, name: str) -> dict[str, Any]:
        for item in self.registry["servers"][server]["tools"]:
            if item["name"] == name:
                return dict(item)
        raise KeyError(f"unregistered tool: {server}/{name}")

    def _schema_key(self, server: str, tool: str) -> str:
        return f"{server}/{tool}"

    def _effective_output_schema(self, server: str, tool: str,
                                 spec: Mapping[str, Any]) -> Any:
        declared = spec.get("output_schema")
        if _explicit_output_schema(declared):
            return declared
        inferred = self.inferred_output_schemas.get(self._schema_key(server, tool))
        if inferred is None and _declared_string_contract(declared):
            # 裸 ``str`` 仍作为提示词里的"返回文本"线索，但不约束字段。
            return declared
        return inferred

    def _lens_for(self, server: str, tool: str, spec: Mapping[str, Any]) -> SchemaLens:
        view = dict(spec)
        if not _explicit_output_schema(spec.get("output_schema")):
            inferred = self.inferred_output_schemas.get(self._schema_key(server, tool))
            if inferred is not None:
                view["output_schema"] = inferred
                view["_require_declared_fields"] = True
        return SchemaLens(server, view)

    def _remember_output_schema(self, server: str, tool: str,
                                spec: Mapping[str, Any], observation: Any) -> None:
        if _explicit_output_schema(spec.get("output_schema")):
            return
        key = self._schema_key(server, tool)
        with self._schema_lock:
            if key in self.inferred_output_schemas:
                return
            inferred = _infer_output_schema(observation)
            if inferred is not None:
                self.inferred_output_schemas[key] = inferred

    def _backfill_inferred_schemas(self) -> None:
        for row in self.rows.values():
            try:
                spec = self._tool(row.server, row.tool)
            except KeyError:
                continue
            self._remember_output_schema(row.server, row.tool, spec, row.observation)

    def _system_prompt(self) -> str:
        """后续调用把已抽出的输出合同写进工具定义，供世界模型对齐。"""
        if not self.inferred_output_schemas:
            return self.world_system_prompt
        registry = copy.deepcopy(self.registry)
        for server_name, server in (registry.get("servers") or {}).items():
            if not isinstance(server, Mapping):
                continue
            for tool in server.get("tools") or ():
                if not isinstance(tool, Mapping):
                    continue
                inferred = self.inferred_output_schemas.get(
                    self._schema_key(str(server_name), str(tool.get("name") or "")))
                if inferred is None or _explicit_output_schema(tool.get("output_schema")):
                    continue
                tool["output_schema"] = inferred
        return render_world_system_prompt(registry)

    def _related_frozen_context(self, arguments: Mapping[str, Any], *, limit: int = 6
                                ) -> list[dict[str, Any]]:
        """按参数值重合选择跨工具上下文，不推测领域或字段语义。"""
        query = _context_tokens(arguments)
        if not query:
            return []
        ranked = []
        for row in self.rows.values():
            overlap = query & (
                _context_tokens(row.arguments) | _context_tokens(row.observation))
            if not overlap:
                continue
            ranked.append((len(overlap), row.query_digest, {
                "server": row.server, "tool": row.tool,
                "arguments": row.arguments, "observation": row.observation,
            }))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        context: list[dict[str, Any]] = []
        used_chars = 0
        for _score, _digest, item in ranked[:limit]:
            size = len(_canon(item))
            if context and used_chars + size > 12000:
                break
            context.append(item); used_chars += size
        return context

    def pin_observation(self, server: str, tool: str,
                        arguments: Mapping[str, Any], observation: Any,
                        *, state_version: str | None = None
                        ) -> FrozenObservation:
        """固定一条由初始世界合同提供的观测。

        该入口与按需材质化使用同一个 ``SchemaLens``、query digest 和观测哈希
        协议。它不理解工具语义，也不调用 8811；不符合 ``tool_info`` 输出合同
        的初始观测会在世界进入执行阶段前失败。
        """
        spec = self._tool(server, tool)
        lens = self._lens_for(server, tool, spec)
        digest = versioned_query_digest(server, tool, arguments, state_version)
        claims = lens.extract(observation, arguments)
        candidate = FrozenObservation(
            server, tool, dict(arguments), copy.deepcopy(observation), claims,
            digest, sha256(_canon(observation).encode()).hexdigest())
        existing = self.rows.get(digest)
        if existing is not None:
            if existing.observation_hash != candidate.observation_hash:
                raise ValueError(
                    f"initial observation conflicts with pinned call: {server}/{tool}")
            return existing
        self.rows[digest] = candidate
        self._remember_output_schema(server, tool, spec, observation)
        return candidate

    def alias_observation(self, row: FrozenObservation, *,
                          state_version: str) -> FrozenObservation:
        """Expose one immutable observation under a new world version.

        Initial read observations remain true until a write occurs.  The alias is
        created only for the current explicit version and still participates in
        ordinary freezing and hashing; no domain-specific invalidation rule is used.
        """
        digest = versioned_query_digest(
            row.server, row.tool, row.arguments, state_version)
        alias = FrozenObservation(
            row.server, row.tool, dict(row.arguments),
            copy.deepcopy(row.observation), copy.deepcopy(row.claims), digest,
            row.observation_hash)
        existing = self.rows.get(digest)
        if existing is not None and existing.observation_hash != alias.observation_hash:
            raise ValueError("versioned observation alias conflicts with frozen world")
        self.rows[digest] = existing or alias
        return self.rows[digest]

    def materialize(self, server: str, tool: str, arguments: Mapping[str, Any],
                    *, generate: Callable[[str], str] | None = None,
                    state_version: str | None = None) -> FrozenObservation:
        spec = self._tool(server, tool)
        lens = self._lens_for(server, tool, spec)
        digest = versioned_query_digest(server, tool, arguments, state_version)
        if digest in self.rows:
            self.last_materialization_meta = {}
            return self.rows[digest]
        full_contract = self._effective_output_schema(server, tool, spec)
        scalar_string_contract = (
            _declared_string_contract(spec.get("output_schema"))
            or _declared_string_contract(full_contract)) or (
            isinstance(full_contract, Mapping)
            and full_contract.get("type") == "string")
        prompt_contract = _project_contract(full_contract, (), salt=f"{server}/{tool}")
        frozen_context = self._related_frozen_context(arguments)
        tool_envelope = {"server": server, "name": tool,
                         "parameters": spec.get("parameters"),
                         "output_schema": prompt_contract,
                         "arguments": dict(arguments)}
        if state_version:
            tool_envelope["world_state_version"] = str(state_version)
        if self.world_context:
            tool_envelope["frozen_initial_world"] = self.world_context
        if frozen_context:
            tool_envelope["frozen_related_observations"] = frozen_context
        prompt = prompt_loader.render(
            "world_materialize_user.txt", tool_envelope=_canon(tool_envelope))
        default_generate = generate is None
        if default_generate:
            # 由本层三次契约修正循环负责重试，关闭 SDK 隐式重试，避免一次
            # 失败被不可见地放大成多轮长等待。公开流水线的连接参数均来自
            # configure_world_runtime；注入 client 只用于离线测试。
            configured = _WORLD_RUNTIME
            if self.client is not None:
                client = self.client
            else:
                configured = world_runtime_config()
                client = OpenAI(
                    base_url=configured.base_url,
                    api_key=configured.api_key,
                    timeout=_WORLD_TIMEOUT,
                    max_retries=0,
                )
            if self._model_id is None:
                explicit_model = (
                    (_WORLD_MODEL or (configured.model if configured else ""))
                    if self.client is not None
                    else configured.model
                )
                if explicit_model:
                    self._model_id = explicit_model
                else:
                    queue_started = time.monotonic()
                    acquired = _WORLD_SEMAPHORE.acquire(timeout=_WORLD_QUEUE_TIMEOUT)
                    queue_waited = time.monotonic() - queue_started
                    if self.queue_wait_observer is not None:
                        self.queue_wait_observer(queue_waited)
                    if not acquired:
                        error = WorldInfrastructureError(
                            "world model scheduler queue timed out")
                        print_api_error("world", error)
                        raise error
                    try:
                        self._model_id = client.models.list().data[0].id
                    except Exception as exc:
                        print_api_error("world discovery", exc)
                        raise WorldInfrastructureError(
                            "world model discovery failed: "
                            f"{type(exc).__name__}: {exc}") from exc
                    finally:
                        _WORLD_SEMAPHORE.release()
            # A broad-schema prompt may spend most of its budget reasoning
            # before emitting the JSON value. Keep the materializer response
            # bounded so one tool observation cannot stall the batch.
            def generate(p: str) -> str:
                queue_started = time.monotonic()
                acquired = _WORLD_SEMAPHORE.acquire(timeout=_WORLD_QUEUE_TIMEOUT)
                queue_waited = time.monotonic() - queue_started
                if self.queue_wait_observer is not None:
                    self.queue_wait_observer(queue_waited)
                if not acquired:
                    error = WorldInfrastructureError(
                        "8822 scheduler queue timed out")
                    print_api_error("world", error)
                    raise error
                try:
                    # The caller can disable reasoning for compact schema
                    # responses through EVOSYNTH_WORLD_ENABLE_THINKING.
                    request = {
                        "model": self._model_id,
                        "temperature": 0,
                        "max_tokens": _WORLD_MAX_TOKENS,
                        "messages": [
                            {"role": "system", "content": self._system_prompt()},
                            {"role": "user", "content": p},
                        ],
                    }
                    if _WORLD_ENABLE_THINKING is not None:
                        request["extra_body"] = {
                            "enable_thinking": _WORLD_ENABLE_THINKING}
                    for request_attempt in range(
                            max(1, _WORLD_RATE_LIMIT_RETRIES + 1)):
                        try:
                            from evosynth.task_gen.generation.runtime import (
                                throttle_request,
                            )
                            throttle_request()
                            return client.chat.completions.create(
                                **request).choices[0].message.content or ""
                        except Exception as exc:
                            print_api_error(
                                f"world attempt={request_attempt + 1}", exc)
                            status = getattr(exc, "status_code", None)
                            error_text = (
                                f"{type(exc).__name__}: {exc}"
                            ).casefold()
                            transient = (
                                status in {408, 429, 500, 502, 503, 504}
                                or any(token in error_text for token in (
                                    "apiconnectionerror", "connection error",
                                    "timeout", "temporarily unavailable",
                                ))
                            )
                            if (not transient or request_attempt >=
                                    _WORLD_RATE_LIMIT_RETRIES):
                                raise
                            # The router's audit-capture outage is reported as
                            # 503. Treat it like a transient provider failure,
                            # with bounded exponential backoff, rather than
                            # misclassifying it as a bad world observation.
                            time.sleep(min(30, 5 * (2 ** request_attempt)))
                    raise AssertionError("unreachable world request retry state")
                except Exception as exc:
                    raise WorldInfrastructureError(
                        "world materialization request failed: "
                        f"{type(exc).__name__}: {exc}") from exc
                finally:
                    _WORLD_SEMAPHORE.release()
        assert generate is not None
        observation = None; claims = None; last_error: Exception | None = None
        consistency_feedback = ""
        initial_observation: Any = None
        repair_attempts = 0
        audit_triggered = bool(frozen_context and default_generate)
        # 首次响应若仅有 JSON/schema/一致性问题，允许三次重试。
        # 连接失败和超时由 ``WorldInfrastructureError`` 立即上抛，绝不进入返修。
        attempts = (1 + max(0, _WORLD_MATERIALIZATION_RETRIES)
                    if default_generate else 1)
        for attempt in range(attempts):
            repair_attempts = attempt
            suffix = ("" if attempt == 0 else
                      "\n" + prompt_loader.render(
                          "world_correction_suffix.txt",
                          consistency_feedback=consistency_feedback))
            raw = generate(prompt + suffix)
            try:
                candidate_observation = _json_response(
                    raw, allow_plain_string=scalar_string_contract)
                if attempt == 0:
                    initial_observation = copy.deepcopy(candidate_observation)
                candidate_claims = lens.extract(candidate_observation, arguments)
                if frozen_context and default_generate:
                    audit_prompt = prompt_loader.render(
                        "world_consistency_audit.txt",
                        frozen=_canon(frozen_context),
                        candidate=_canon(candidate_observation),
                    )
                    audit_payload = _json_response(generate(audit_prompt))
                    if not isinstance(audit_payload, Mapping) or not isinstance(
                            audit_payload.get("consistent"), bool):
                        raise ValueError("consistency audit returned invalid contract")
                    conflicts = list(audit_payload.get("conflicts") or [])
                    if not bool(audit_payload.get("consistent")) or conflicts:
                        consistency_feedback = (
                            "Frozen consistency conflicts: " + _canon(conflicts))
                        raise ValueError(
                            "materializer violated frozen consistency contract")
                observation, claims = candidate_observation, candidate_claims
                break
            except WorldInfrastructureError:
                raise
            except Exception as exc:
                last_error = exc
        if observation is None or claims is None:
            self.last_materialization_meta = {
                "audit_triggered": audit_triggered,
                "repair_attempts": repair_attempts,
                "repair_triggered": repair_attempts > 0,
                "initial_observation": initial_observation,
                "final_observation": None,
            }
            raise WorldMaterializationError(
                f"{server}/{tool}: materializer failed output or consistency contract "
                f"after retries: {last_error}") from last_error
        row = FrozenObservation(server, tool, dict(arguments), observation, claims, digest,
                                sha256(_canon(observation).encode()).hexdigest())
        self.rows[digest] = row
        self._remember_output_schema(server, tool, spec, observation)
        self.last_materialization_meta = {
            "audit_triggered": audit_triggered,
            "repair_attempts": repair_attempts,
            "repair_triggered": repair_attempts > 0,
            "initial_observation": copy.deepcopy(initial_observation),
            "final_observation": copy.deepcopy(observation),
        }
        return row

    def execute(self, server: str, tool: str, arguments: Mapping[str, Any], *,
                state_version: str | None = None) -> FrozenObservation:
        digest = versioned_query_digest(server, tool, arguments, state_version)
        if digest not in self.rows:
            raise RuntimeError(f"world is not pinned for {server}/{tool}")
        return self.rows[digest]

    def freeze(self) -> str:
        payload = {key: asdict(value) for key, value in sorted(self.rows.items())}
        envelope = {"version": 1, "observations": payload}
        if self.inferred_output_schemas:
            envelope["inferred_output_schemas"] = dict(sorted(
                self.inferred_output_schemas.items()))
        envelope["bundle_hash"] = sha256(_canon(envelope).encode()).hexdigest()
        self.world_path.parent.mkdir(parents=True, exist_ok=True)
        self.world_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        return envelope["bundle_hash"]

    def _load(self) -> None:
        data = json.loads(self.world_path.read_text(encoding="utf-8"))
        expected = data.pop("bundle_hash", "")
        if expected != sha256(_canon(data).encode()).hexdigest():
            raise ValueError("world bundle hash mismatch")
        self.inferred_output_schemas = dict(data.get("inferred_output_schemas") or {})
        self.rows = {key: FrozenObservation(**value) for key, value in data.get("observations", {}).items()}
        self._backfill_inferred_schemas()
