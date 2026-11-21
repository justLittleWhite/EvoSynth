"""仅由 ``tool_info`` 驱动的通用初始世界合同。

初始世界只保存材质化所需的系统摘要和有限实体集合。研究代理调查时再按需
材质化工具观测；关系和不变量不属于当前 active pipeline 的合同。
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evosynth.task_gen.generation.tool_info import (
    CompiledOpenDomain, dumps_compact)
from evosynth.task_gen.generation import prompt_loader


WORLD_CONTRACT_VERSION = "universal_initial_world"


@dataclass(frozen=True)
class UniversalWorldContract:
    system_summary: str
    entities: tuple[Mapping[str, Any], ...]
    version: str = WORLD_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "system_summary": self.system_summary,
            "entities": [copy.deepcopy(dict(item)) for item in self.entities],
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
                          + "\n", encoding="utf-8")
        return target


def _validate_entities(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("initial world requires a non-empty entities array")
    # if len(raw) > 10:
    #     raise ValueError("initial world may contain at most 10 entities")
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"entity {index} must be an object")
        if set(item) != {"id", "type", "attributes"}:
            raise ValueError(
                f"entity {index} must contain exactly id, type, attributes")
        ident = str(item.get("id") or "").strip()
        kind = str(item.get("type") or "").strip()
        attributes = item.get("attributes")
        if not ident or not kind or not isinstance(attributes, Mapping):
            raise ValueError(f"entity {index} has invalid id/type/attributes")
        if ident in seen:
            raise ValueError(f"duplicate entity id: {ident}")
        seen.add(ident)
        entities.append({"id": ident, "type": kind,
                         "attributes": copy.deepcopy(dict(attributes))})
    return tuple(entities)


def parse_world_contract(payload: Mapping[str, Any],
                         compiled: CompiledOpenDomain
                         ) -> UniversalWorldContract:
    """验证一份世界画像。``compiled`` 保留给调用方，画像本身不再引用工具。"""
    del compiled
    body = dict(payload)
    given_version = body.pop("version", None)
    expected = {"system_summary", "entities"}
    if set(body) != expected:
        extra = sorted(set(body) - expected)
        missing = sorted(expected - set(body))
        raise ValueError(
            "initial world has invalid top-level fields: "
            f"extra={extra} missing={missing}")
    if given_version is not None and str(given_version) != WORLD_CONTRACT_VERSION:
        raise ValueError("unsupported initial-world contract version")
    summary = str(body.get("system_summary") or "").strip()
    if len(summary) < 40:
        raise ValueError("system_summary is too short to define one coherent system")
    entities = _validate_entities(body.get("entities"))
    return UniversalWorldContract(
        system_summary=summary, entities=entities)


def world_contract_prompt(*, compiled: CompiledOpenDomain,
                          persona: str,
                          archetype: str,
                          archetype_desc: str,
                          connected_tools: Sequence[str] | None = None) -> str:
    """渲染初始世界构造提示。

    只接收 Persona、archetype 名称与描述、以及 ``tool_info``。世界该有
    几个实体、几条初始观测,由模型根据 archetype 描述自行判断,不再由
    外部数量区间或约束列表注入。

    ``connected_tools`` 是这次任务研究代理实际会连接的工具子集(见
    ``task_space.OpenDistributionController._sample_connected_tools``)。
    传入时按 ``public_name`` 过滤,只把这些工具的声明交给世界构造——世界
    不应该造出只有未连接工具才能查到的实体;缺省(``None``)时退回全部
    知识工具,兼容测试/离线渲染等直接调用点。
    """
    knowledge = compiled.knowledge_tools
    if connected_tools is not None:
        allowed = set(connected_tools)
        knowledge = tuple(
            tool for tool in knowledge if tool.public_name in allowed) or knowledge
    tool_info = [{
        "capability": f"capability_{index}",
        "public_name": tool.public_name,
        "server": tool.server,
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "output_schema": tool.output_schema,
    } for index, tool in enumerate(knowledge, 1)]
    return prompt_loader.render(
        "world_contract.txt",
        persona=persona,
        archetype=archetype,
        archetype_desc=archetype_desc,
        tool_info=dumps_compact(tool_info),
    )


def generate_world_contract(*, compiled: CompiledOpenDomain, persona: str,
                            archetype: str,
                            archetype_desc: str,
                            generate_json: Callable[[str], Mapping[str, Any]],
                            connected_tools: Sequence[str] | None = None,
                            max_attempts: int = 3) -> UniversalWorldContract:
    """调用外部生成器，并用机械反馈重试同一个世界合同。"""
    prompt = world_contract_prompt(
        compiled=compiled, persona=persona, archetype=archetype,
        archetype_desc=archetype_desc, connected_tools=connected_tools)
    feedback = ""
    last: Exception | None = None
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        try:
            payload = generate_json(prompt + feedback)
            return parse_world_contract(payload, compiled)
        except Exception as exc:
            last = exc
            from evosynth.task_gen.generation import runtime
            runtime._log_event(
                "world_contract_retry",
                attempt=attempt + 1, max_attempts=attempts,
                error=f"{type(exc).__name__}: {exc}"[:2000])
            feedback = prompt_loader.render(
                "world_contract_retry.txt",
                error=f"{type(exc).__name__}: {exc}",
                strict=False)
    raise ValueError(
        f"world contract generation failed after {attempts} attempts: {last}"
    ) from last


__all__ = ["WORLD_CONTRACT_VERSION", "UniversalWorldContract",
           "parse_world_contract", "world_contract_prompt",
           "generate_world_contract"]
