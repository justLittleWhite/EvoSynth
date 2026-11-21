"""读取统一评测 record，筛出 acc==0 的题。

输入可以是 ``normalized/records.jsonl`` 文件，也可以是 run 目录（自动找 normalized/records.jsonl）。
eval_reason 是可选字段：有就带上，没有就走盲诊。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

REASON_KEYS = ("eval_reason", "reason", "explanation", "feedback", "message", "comment", "error")
TRAJECTORY_CHAR_LIMIT = 60_000
OBSERVATION_CHAR_LIMIT = 2_000


@dataclass
class BenchmarkCase:
    case_id: str
    benchmark: str
    question: str
    tools: list[str]
    tools_inferred: bool
    trajectory: str
    eval_reason: str | None
    meta: dict[str, Any] = field(default_factory=dict)


def _records_path(path: Path) -> Path:
    path = Path(path).expanduser()
    return path / "normalized" / "records.jsonl" if path.is_dir() else path


def iter_records(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for raw in paths:
        source = _records_path(raw)
        with open(source, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def is_failed(record: Mapping[str, Any]) -> bool:
    """acc == 0：官方评测明确失败。error/missing 属基础设施问题，不算模型错题。"""
    evaluation = record.get("evaluation") or {}
    if evaluation.get("status") == "failed":
        return True
    if evaluation.get("success") is False:
        return True
    score = evaluation.get("score")
    return evaluation.get("status") not in {"error", "missing"} and score == 0


def collect_reason(value: Any, *, depth: int = 0, out: list[str] | None = None) -> list[str]:
    """从评测详情里捞出人类可读的失败原因（嵌套 dict/list，深度 ≤ 4）。"""
    out = [] if out is None else out
    if depth > 4:
        return out
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in REASON_KEYS and isinstance(item, str) and item.strip():
                if item.strip() not in out:
                    out.append(item.strip())
            elif isinstance(item, (Mapping, list)):
                collect_reason(item, depth=depth + 1, out=out)
    elif isinstance(value, list):
        for item in value:
            collect_reason(item, depth=depth + 1, out=out)
    return out


def eval_reason_of(record: Mapping[str, Any]) -> str | None:
    evaluation = record.get("evaluation") or {}
    parts = collect_reason(evaluation.get("details"))
    parts += [str(e).strip() for e in evaluation.get("errors") or () if str(e).strip()]
    unique = list(dict.fromkeys(parts))
    return "\n".join(unique)[:2000] if unique else None


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 15] + "...[truncated]"


def render_trajectory(messages: list[Mapping[str, Any]]) -> str:
    """按轮次渲染：assistant 的文字/工具调用 + 紧随的 tool 观察。过长时保留头尾。"""
    blocks: list[str] = []
    turn = 0
    for message in messages:
        role = message.get("role")
        content = str(message.get("content") or "").strip()
        if role == "assistant":
            turn += 1
            lines = [f"[turn {turn}] ASSISTANT"]
            if content:
                lines.append(_clip(content, OBSERVATION_CHAR_LIMIT))
            for call in message.get("tool_calls") or ():
                lines.append(f"TOOL CALL: {call.get('name')}("
                             f"{json.dumps(call.get('arguments'), ensure_ascii=False)[:800]})")
            blocks.append("\n".join(lines))
        elif role == "tool":
            blocks.append(f"[turn {turn}] OBSERVATION: {_clip(content, OBSERVATION_CHAR_LIMIT)}")
        elif role == "user" and turn > 0:
            blocks.append(f"[turn {turn}] USER: {_clip(content, 800)}")
    text = "\n".join(blocks) or "(empty trajectory)"
    if len(text) > TRAJECTORY_CHAR_LIMIT:
        half = TRAJECTORY_CHAR_LIMIT // 2
        text = text[:half] + "\n...[middle turns omitted]...\n" + text[-half:]
    return text


def tool_names_from(record: Mapping[str, Any],
                    tools_by_benchmark: Mapping[str, list[str]] | None) -> tuple[list[str], bool]:
    """可用工具（完整挂载列表）：record.task.context.tools → --tools-file → 极端 fallback。

    fallback 只返回轨迹里调用过的工具，此时 inferred=True，诊断器禁止据此判断
    “漏调工具 / 过早停止”（见 diagnose.py），否则会幻觉归因。"""
    for source in ((record.get("task") or {}).get("context") or {}, record.get("metadata") or {}):
        for key in ("tools", "available_tools"):
            tools = source.get(key) if isinstance(source, Mapping) else None
            if isinstance(tools, list) and tools:
                return [_tool_name(t) for t in tools], False
    given = (tools_by_benchmark or {}).get(str(record.get("benchmark")))
    if isinstance(given, Mapping):  # {task_id: [...], "*": [...]}
        given = given.get(str(record.get("task_id"))) or given.get("*")
    if given:
        return [_tool_name(t) for t in given], False
    seen = [str(c.get("name")) for m in record.get("trajectory") or ()
            for c in m.get("tool_calls") or () if c.get("name")]
    return list(dict.fromkeys(seen)), True


def _tool_name(tool: Any) -> str:
    if isinstance(tool, Mapping):
        function = tool.get("function") if isinstance(tool.get("function"), Mapping) else tool
        name = str(function.get("name") or "")
        desc = str(function.get("description") or "").strip().split("\n")[0][:160]
        return f"{name}: {desc}" if desc else name
    return str(tool)


def load_failed_cases(paths: Iterable[Path], *, blind: bool = False,
                      tools_by_benchmark: Mapping[str, list[str]] | None = None
                      ) -> tuple[list[BenchmarkCase], dict[str, Any]]:
    cases: list[BenchmarkCase] = []
    totals: dict[str, dict[str, int]] = {}
    for record in iter_records(paths):
        benchmark = str(record.get("benchmark"))
        stat = totals.setdefault(benchmark, {"total": 0, "failed": 0, "with_reason": 0, "tools_inferred": 0})
        stat["total"] += 1
        if not is_failed(record):
            continue
        stat["failed"] += 1
        reason = None if blind else eval_reason_of(record)
        stat["with_reason"] += reason is not None
        tools, inferred = tool_names_from(record, tools_by_benchmark)
        stat["tools_inferred"] += inferred
        cases.append(BenchmarkCase(
            case_id=str(record["record_id"]), benchmark=benchmark,
            question=str((record.get("task") or {}).get("instruction") or ""),
            tools=tools, tools_inferred=inferred,
            trajectory=render_trajectory(record.get("trajectory") or []),
            eval_reason=reason, meta={"task_id": record.get("task_id"),
                                      "run_id": record.get("run_id")}))
    return cases, {"per_benchmark": totals,
                   "failed": sum(v["failed"] for v in totals.values()),
                   "total": sum(v["total"] for v in totals.values())}
