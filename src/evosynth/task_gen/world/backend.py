"""共享合成流水线使用的通用 EvoSynth 后端。

该后端负责 EvoSynth 特有的两级观测解析：优先读取已经固定的观测；对
``tool_info`` 中已注册工具的合法新参数查询，可以通过 8822 按需材质化并在
当前 episode 中缓存。显式 reference replay 仍保持只读。任务采样、agent
规划、反向提案和质量门禁继续使用根 ``synthesis``，本模块不认识
任何具体 domain。
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from jsonschema import validate as validate_json

from evosynth.task_gen.common.json_utils import canonical_json
from evosynth.task_gen.common.synthesis import CallRecord, ExecutionTrace
from evosynth.task_gen.world.controlled import (
    ControlledWorld, FrozenObservation, WorldInfrastructureError,
    WorldMaterializationError,
)
from evosynth.task_gen.world.frozen_knowledge import (
    FrozenEpisodeState, FrozenKnowledgeIndex,
)
from evosynth.task_gen.generation.tool_info import CompiledOpenDomain, CompiledTool
from evosynth.task_gen.world.universal import UniversalWorldContract


# 一次 fetch/工具材质化失败(8822 回了错误形式或不符合输出合同)是该次调用
# 失败,调查仍可继续:错误以普通工具观测文本返回给研究 agent,由其换参重试。
# 只有连续多次全部失败才升级为放弃该 seed 的 WorldMaterializationError。
_MATERIALIZATION_ESCALATION_THRESHOLD = max(1, int(os.environ.get(
    "EVOSYNTH_MATERIALIZATION_ESCALATION_THRESHOLD", "3")))

# 判定"某一列被研究真正消费过"所需的最短连续复述段。单个值反复出现不构成
# 证据。阈值取 max(_FACET_MIN_REPEATED_VALUES, len(values) // 3):
# Discovery-token handling is kept separate from ordinary input values.
# 被消费的列(longest_run/n)落在 0.32–1.00,同构未使用的列(High/Low/Open/
# Volume/Dividends/Stock Splits)全部 ≤ 0.27;1/3 恰好落在两簇之间,并对
# 噪声簇留有余量。绝对下限 8 防止短列被零散命中误判。
_FACET_MIN_REPEATED_VALUES = 8
_FACET_REPEATED_RUN_DENOMINATOR = 3


def _apply_defaults(schema: Mapping[str, Any], value: Any) -> Any:
    if not isinstance(value, Mapping):
        return copy.deepcopy(value)
    result = copy.deepcopy(dict(value))
    properties = schema.get("properties") or {}
    if not isinstance(properties, Mapping):
        return result
    for name, child in properties.items():
        if not isinstance(child, Mapping):
            continue
        if name not in result and "default" in child:
            result[name] = copy.deepcopy(child["default"])
        elif name in result and isinstance(result[name], Mapping):
            result[name] = _apply_defaults(child, result[name])
    return result


def normalize_arguments(tool: CompiledTool, arguments: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise TypeError(f"{tool.public_name}: arguments must be an object")
    normalized = _apply_defaults(tool.parameters, arguments)
    validate_json(normalized, tool.parameters)
    return dict(normalized)


def _primitive_values(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    output: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            output.extend(_primitive_values(item, prefix + "/" + escaped))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            output.extend(_primitive_values(item, prefix + f"/{index}"))
    elif value is None or isinstance(value, (str, int, float, bool)):
        output.append((prefix or "/", value))
    return output


def _dependency_token(value: Any) -> str | None:
    """无条件可信的"强"依赖 token(长字符串/大数值,碰撞率天然低)。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        token = value.strip()
        if len(token) < 4:
            return None
        return "s:" + token.casefold()
    if isinstance(value, (int, float)):
        # 小计数和页码碰撞率太高，无出处时不足以证明数据流。
        if abs(float(value)) < 20:
            return None
        return "n:" + repr(value)
    return None


def _weak_dependency_token(value: Any) -> str | None:
    """发现性豁免的"弱"token(当前版本 ②):短字符串与小数值。

    弱 token 本身不证明数据流;``infer_runtime_dependencies`` 只在它逐字等于
    某次先前观测的整字段值(字段级出处,非子串撞见——观测侧 token 按
    ``_primitive_values`` 的叶字段值整体计算)且不来自任务 known_inputs
    (是调查中发现的,不是题面给的)时才连边。碰撞控制靠出处判定,不靠
    长度阈值。此前 <4 字符字符串/<20 数值被一刀切排除,导致"线索→消解出
    2 字符 ticker→深查"的可证边永不成立。
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        return "s:" + token.casefold()
    if isinstance(value, (int, float)):
        return "n:" + repr(value)
    return None


def known_input_tokens(known_inputs: Iterable[Any]) -> set[str]:
    """任务 known_inputs 的 token 化(整句 + 词级 + 数值双 repr)。

    known_inputs 是自由文本线索/根输入;短标识符(ticker 等)常内嵌在句中,
    因此除整句 token 外还登记词级 token,使"题面直给的值"不能经发现性豁免
    连边(previously_supplied 语义的任务级前置)。
    """
    tokens: set[str] = set()
    for item in known_inputs or ():
        text = str(item).strip()
        if not text:
            continue
        tokens.add("s:" + text.casefold())
        for word in re.findall(r"[\w.'\-]+", text):
            word = word.strip(".'-")
            if not word:
                continue
            tokens.add("s:" + word.casefold())
            try:
                number = float(word)
            except ValueError:
                continue
            tokens.add("n:" + repr(number))
            if number.is_integer():
                tokens.add("n:" + repr(int(number)))
    return tokens


def infer_runtime_dependencies(*, current_arguments: Mapping[str, Any],
                               previous: Sequence[tuple[CallRecord, Any]],
                               previously_supplied: set[str],
                               known_inputs: Iterable[Any] = ()) -> tuple[str, ...]:
    """只依据真实值传递证明依赖，不根据调用先后臆造边。

    强 token(长字符串/大数值)行为与历史一致:novel(未被 agent 供给过)
    且逐字段命中先前观测即连边。弱 token(短字符串/小数值,当前版本 ②
    发现性豁免)额外要求不在任务 known_inputs 中——短值只有"先被观测字段
    披露、再被参数复用"且确属发现时才构成边。

    当前规则:known_inputs 阻断对所有 token(强+弱)生效——题面直给
    的值(无论长短)复用不构成发现边,previously_supplied 语义的任务级前置。
    """
    known = known_input_tokens(known_inputs)
    novel: set[str] = set()
    for _path, value in _primitive_values(current_arguments):
        strong = _dependency_token(value)
        if strong is not None:
            if strong not in previously_supplied and strong not in known:
                novel.add(strong)
            continue
        weak = _weak_dependency_token(value)
        if weak is not None and weak not in previously_supplied \
                and weak not in known:
            novel.add(weak)
    if not novel:
        return ()
    parents: list[str] = []
    for call, observation in previous:
        # 观测侧按叶字段整值计算(弱 token 是强 token 的超集):字段级出处。
        observed = {_weak_dependency_token(value)
                    for _path, value in _primitive_values(observation)}
        observed.discard(None)
        if novel & observed:
            parents.append(call.identity)
    return tuple(dict.fromkeys(parents))


def all_dependency_tokens(value: Any) -> set[str]:
    """强+弱 token 全集:用于研究循环的 previously_supplied 记账,使 agent
    已在参数中供给过的短值被观测回显后不会误判为新发现(当前版本 ②)。"""
    tokens = {_weak_dependency_token(item)
              for _path, item in _primitive_values(value)}
    tokens.discard(None)
    return {str(item) for item in tokens}


class GenericEvoSynthBackend:
    """``WorldBackend`` 的 schema 驱动 EvoSynth 实现。"""

    kind = "open"

    def __init__(self, compiled: CompiledOpenDomain, *, registry_path: Path,
                 world_path: Path, allow_materialize: bool = True,
                 materialize_generate: Callable[[str], str] | None = None,
                 replay_mode: str = "exact",
                 queue_wait_observer: Callable[[float], None] | None = None):
        self.compiled = compiled
        self.registry_path = Path(registry_path)
        self.world_path = Path(world_path)
        self.allow_materialize = allow_materialize
        self.materialize_generate = materialize_generate
        # Complexity's entity range describes the investigation/world as a
        # whole.  It must not be projected onto every array returned by every
        # search-like tool: a coherent multi-entity investigation can
        # legitimately receive one recommendation, one route, or one matching
        # record from an individual call.
        self.world = ControlledWorld(
            self.registry_path, self.world_path,
            queue_wait_observer=queue_wait_observer)
        self.base_digests = frozenset(self.world.rows)
        if replay_mode not in {"exact", "indexed"}:
            raise ValueError(f"unknown frozen replay mode: {replay_mode}")
        self.replay_mode = replay_mode
        self.episode_state = FrozenEpisodeState([], [], {})
        self.knowledge_index = (
            FrozenKnowledgeIndex(self.world.rows, compiled.tools)
            if not allow_materialize else None
        )
        self.last_row: FrozenObservation | None = None
        self.call_log: list[dict[str, Any]] = []
        self._materialization_failure_streak = 0
        self._initial: Any = None
        self.initial_world: UniversalWorldContract | None = None
        self.state_version = 0
        self.world_events: list[dict[str, Any]] = []

    def bootstrap(self, *, seed: int | None = None,
                  spec: Mapping[str, Any] | None = None) -> Any:
        del seed, spec
        self._initial = self.snapshot()
        return copy.deepcopy(self._initial)

    def pin_initial_world(self, contract: UniversalWorldContract) -> None:
        """把世界画像设为后续材质化的一致性锚点。

        画像只描述实体、关系和不变式，不预置工具观测。研究代理的调用一律
        按需材质化；未知 server 不需要注册 Python handler 或领域后端。
        """
        self.initial_world = contract
        self.state_version = 0
        self.world_events = []
        self.world.world_context = contract.to_dict()

    def checkpoint(self) -> dict[str, Any]:
        """捕获一次研究尝试前的开放世界状态。

        Closed World 会从同一初态重跑失败的整理执行。EvoSynth 也需要相同
        语义，但不能把失败尝试临时材质化的观测泄漏给下一次尝试。检查点只
        复制通用冻结观测和 episode 状态，不认识领域、工具或字段。
        """
        return {
            "rows": copy.deepcopy(self.world.rows),
            "episode_state": copy.deepcopy(self.episode_state),
            "call_log": copy.deepcopy(self.call_log),
            "state_version": self.state_version,
            "world_events": copy.deepcopy(self.world_events),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        """恢复研究尝试检查点；正式 ``freeze`` 只会看到保留的尝试。"""
        self.world.rows = copy.deepcopy(dict(checkpoint.get("rows") or {}))
        episode = checkpoint.get("episode_state")
        if isinstance(episode, FrozenEpisodeState):
            self.episode_state = copy.deepcopy(episode)
        else:
            self.episode_state = FrozenEpisodeState([], [], {})
        self.call_log = copy.deepcopy(list(checkpoint.get("call_log") or ()))
        self.state_version = int(checkpoint.get("state_version") or 0)
        self.world_events = copy.deepcopy(
            list(checkpoint.get("world_events") or ()))
        self.last_row = None
        self._materialization_failure_streak = 0

    def tool_schemas(self) -> list[dict[str, Any]]:
        return self.compiled.public_schemas()

    def call(self, tool: str, arguments: Mapping[str, Any], *,
             source: str = "execution") -> str:
        compiled_tool = self.compiled.by_public_name(tool)
        normalized = normalize_arguments(compiled_tool, arguments)
        state_key = str(self.state_version) if self.initial_world is not None else None
        if self.initial_world is not None:
            self.world.world_context = {
                "initial_world": self.initial_world.to_dict(),
                "successful_episode_events": copy.deepcopy(self.world_events),
            }
        try:
            self.world.last_materialization_meta = {}
            if self.allow_materialize:
                from evosynth.task_gen.world.controlled import versioned_query_digest
                digest = versioned_query_digest(
                    compiled_tool.server, compiled_tool.name, normalized, state_key)
                if digest in self.base_digests:
                    provenance = "pinned_controlled_world"
                elif digest in self.world.rows:
                    provenance = "episode_materialized_cache"
                else:
                    provenance = "qwen_agentworld_materialized"
                row = self.world.materialize(
                    compiled_tool.server, compiled_tool.name, normalized,
                    generate=self.materialize_generate,
                    state_version=state_key)
            else:
                try:
                    row = self.world.execute(
                        compiled_tool.server, compiled_tool.name, normalized,
                        state_version=state_key)
                except RuntimeError:
                    if self.replay_mode != "indexed" or self.knowledge_index is None:
                        raise
                    template = self.knowledge_index.template(
                        compiled_tool.server, compiled_tool.name)
                    if template is None:
                        raise
                    row = self.knowledge_index.resolve(
                        template, normalized, self.episode_state)
                    if row is None:
                        raise RuntimeError(
                            f"frozen index miss for {compiled_tool.server}/{compiled_tool.name}")
                provenance = "pinned_controlled_world"
            self.last_row = row
            observation = canonical_json(row.observation)
            success = True
            effect = "read"
            event = {"state_version_before": self.state_version,
                     "tool": tool, "arguments": copy.deepcopy(normalized),
                     "observation": copy.deepcopy(row.observation),
                     "effect": effect}
            if effect in {"write", "mixed"}:
                self.state_version += 1
            event["state_version_after"] = self.state_version
            self.world_events.append(event)
        except WorldInfrastructureError:
            # 基础设施故障不是一个可供研究代理修正的工具观测。若把它包装成
            # ``Error: ...``，外层会错误地重做调查并放大长尾等待。
            raise
        except WorldMaterializationError as exc:
            # 8822 已响应但输出不符合注册合同(如对合法 fetch 返回错误形式)。
            # 这是该次调用失败,不是整次调查失败:恢复为可重试的普通工具错误,
            # 让研究 agent 换参数继续;只有连续多次全部失败才升级放弃 seed。
            self._materialization_failure_streak += 1
            if self._materialization_failure_streak >= \
                    _MATERIALIZATION_ESCALATION_THRESHOLD:
                raise
            self.last_row = None
            observation = f"Error: WorldMaterializationError: {exc}"
            provenance = "materialization_error"
            success = False
        except Exception as exc:
            self.last_row = None
            observation = f"Error: {type(exc).__name__}: {exc}"
            provenance = "error"
            success = False
        else:
            self._materialization_failure_streak = 0
        materialization_meta = copy.deepcopy(self.world.last_materialization_meta)
        self.call_log.append({"step_index": len(self.call_log), "tool": tool, "tool_id": compiled_tool.tool_id,
                              "tool_schema": compiled_tool.registry_record(),
                              "args": normalized, "observation": observation,
                              "success": success, "source": source,
                              "provenance": provenance,
                              "is_frozen_hit": provenance == "pinned_controlled_world",
                              "is_dynamic_materialization": provenance == "qwen_agentworld_materialized",
                              **materialization_meta,
                              "state_version": self.state_version,
                              "episode_state": self.episode_state.snapshot()})
        return observation

    def snapshot(self) -> dict[str, Any]:
        return {"state_version": self.state_version,
                "world_events": copy.deepcopy(self.world_events),
                "observations": {digest: asdict(row)
                                 for digest, row in sorted(self.world.rows.items())}}

    def diff(self, before: Any, after: Any) -> Mapping[str, Any]:
        left_map = (before or {}).get("observations", before or {}) \
            if isinstance(before, Mapping) else {}
        right_map = (after or {}).get("observations", after or {}) \
            if isinstance(after, Mapping) else {}
        left = set(left_map)
        right = set(right_map)
        before_version = int((before or {}).get("state_version", 0)) \
            if isinstance(before, Mapping) else 0
        after_version = int((after or {}).get("state_version", 0)) \
            if isinstance(after, Mapping) else 0
        return {"changed": left != right or before_version != after_version,
                "state_version_before": before_version,
                "state_version_after": after_version,
                "observations_added": len(right - left),
                "observation_digests": sorted(right - left)}

    def replay(self, reference: Sequence[Mapping[str, Any]], *,
               source: str = "replay") -> ExecutionTrace:
        # freeze() 已在共享管线中先于 replay 调用；新实例保证重放不能读取
        # construction 期间的可变内存对象。
        replay_backend = GenericEvoSynthBackend(
            self.compiled, registry_path=self.registry_path, world_path=self.world_path,
            allow_materialize=False, replay_mode=self.replay_mode)
        if self.initial_world is not None:
            replay_backend.pin_initial_world(self.initial_world)
        calls: list[CallRecord] = []
        for index, step in enumerate(reference, 1):
            name = str(step.get("tool") or "")
            args = step.get("arguments") or {}
            observation = replay_backend.call(name, args, source=source)
            calls.append(CallRecord(
                index, name, dict(args), observation,
                not observation.startswith("Error:"), source, "pinned_controlled_world",
                call_id=(str(step["call_id"]) if step.get("call_id") else None),
                depends_on=tuple(str(item) for item in step.get("depends_on", ())),
                produces=tuple(str(item) for item in step.get("produces", ())),
                relevant=True, kind=str(step.get("kind", "observation")),
                state_changed=False))
        return ExecutionTrace(calls=calls, final_snapshot=replay_backend.snapshot())

    def freeze(self) -> Mapping[str, Any]:
        digest = self.world.freeze()
        return {"bundle_hash": digest, "world_path": str(self.world_path),
                "observations": len(self.world.rows),
                "replay_mode": self.replay_mode,
                "state_version": self.state_version,
                "world_events": len(self.world_events),
                "episode_state": self.episode_state.snapshot()}




__all__ = ["GenericEvoSynthBackend", "infer_runtime_dependencies",
           "all_dependency_tokens", "known_input_tokens", "normalize_arguments",
           "canonical_json"]
