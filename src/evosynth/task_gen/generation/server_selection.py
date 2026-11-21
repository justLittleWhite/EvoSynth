"""EvoSynth 的 MCP server 选择：锚点优先。

覆盖控制侧先按缺口选定一个锚点 server（见 ``tool_info.sample_anchor_candidates``），
LLM 只裁决两件事：锚点能否与当前 persona / archetype 成立，以及要不要从若干
可选同伴里再加几个。LLM 不再决定「做哪个 server」，因此覆盖频率控制不会被
LLM 的偏好稀释。锚点不兼容时返回 ``compatible=False``，由调用方重采 persona。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from openai import OpenAI

from evosynth.task_gen.generation import prompt_loader, runtime
from evosynth.task_gen.generation.tool_info import (
    AnchorPick, count_knowledge_tools, pool_server_index, render_server_catalog,
)

MIN_KNOWLEDGE_TOOLS = 2


class ServerSelectionError(RuntimeError):
    """LLM 没有给出可用的 server 选择。"""


@dataclass(frozen=True)
class ServerSelection:
    anchor: str
    candidates: dict[str, list[str]]
    selected: tuple[str, ...]
    compatible: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"anchor": self.anchor,
                "candidates": {domain: list(names)
                               for domain, names in self.candidates.items()},
                "compatible": self.compatible,
                "selected": list(self.selected), "reason": self.reason}


def select_servers(*, pool: Mapping[str, Mapping[str, Any]], persona: str,
                   archetype: str, archetype_desc: str,
                   pick: AnchorPick) -> ServerSelection:
    domain_of, _ = pool_server_index(pool)
    names = list(pick.candidates())
    candidates: dict[str, list[str]] = {}
    for server in names:
        candidates.setdefault(domain_of[server], []).append(server)
    prompt = prompt_loader.render(
        "server_selection.txt", persona=persona, archetype=archetype,
        archetype_desc=archetype_desc, anchor=pick.anchor,
        servers=render_server_catalog(pool, names))
    allowed = set(names)

    def validate(payload: dict[str, Any]) -> ServerSelection:
        compatible = payload.get("compatible")
        if not isinstance(compatible, bool):
            raise ValueError("compatible must be true or false")
        reason = str(payload.get("reason") or "").strip()
        if not reason:
            raise ValueError("reason must be a non-empty string")
        raw = payload.get("servers")
        if not isinstance(raw, list):
            raise ValueError("servers must be a list")
        if not compatible:
            return ServerSelection(pick.anchor, candidates, (), False, reason)
        chosen: list[str] = []
        for item in raw:
            name = str(item).strip()
            if name not in allowed:
                raise ValueError(f"server {name!r} is not in the server list")
            if name not in chosen:
                chosen.append(name)
        if pick.anchor not in chosen:
            raise ValueError(
                f"servers must include the anchor {pick.anchor!r} when compatible; "
                "set compatible to false if the anchor cannot be used")
        if count_knowledge_tools(pool, chosen) < MIN_KNOWLEDGE_TOOLS:
            raise ValueError(
                f"selected servers offer fewer than {MIN_KNOWLEDGE_TOOLS} functions; "
                "add companion servers")
        return ServerSelection(pick.anchor, candidates, tuple(chosen), True, reason)

    config = runtime.runtime_config()
    client = OpenAI(base_url=config.agent_base_url,
                    api_key=runtime._evosynth_agent_key(),
                    timeout=600, max_retries=0)
    state: dict[str, Any] = {}
    try:
        return runtime._run_json_stage(
            client=client, prompt=prompt, stage="server_selection",
            validator=validate, state=state, temperature=0.3, max_attempts=3)
    except (TimeoutError, runtime._ModelInfrastructureError):
        raise
    except ValueError as exc:
        raise ServerSelectionError(str(exc)) from exc


__all__ = ["ServerSelection", "ServerSelectionError", "select_servers"]
