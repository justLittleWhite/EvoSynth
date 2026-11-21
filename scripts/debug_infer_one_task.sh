#!/usr/bin/env bash
# Run inference for one task. Edit the parameters and SYSTEM_PROMPT below.
# Usage: bash scripts/debug_infer_one_task.sh <task_id or task.json path>
# The script performs inference only; it does not run the judge.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

# ─────────────── configurable parameters ───────────────
TASK="${1:-uwg_mcp_pool_20261026_3718}"   # task_id，或一份 *.json 的路径
TASKS_DIR=""
OUTPUT_DIR=""
MAX_STEPS=50
TEMPERATURE=0.5
MAX_TOKENS=12000
MODEL="deepseek-v4.1-flash-1017"
BASE_URL=""
API_KEY_FILE=""
export EVOSYNTH_WORLD_MATERIALIZATION_RETRIES="${EVOSYNTH_WORLD_MATERIALIZATION_RETRIES:-3}"

# Keep this prompt aligned with the rollout solver prompt. This standalone
# script intentionally does not read the module-level prompt.
SYSTEM_PROMPT=$(cat <<'EOF'
You are a helpful assistant. Your goal is to fulfill the user's requests in an interactive environment by step-by-step use of available tools, while proactively communicating with the user when necessary, until the user ends the conversation.
At each step, you will receive either the user's task/reply or the environment's tool call result.
- If you lack essential information to complete the task or perform a tool call, and it cannot be obtained through the existing tool set, actively ask the user for specific details.
- If you can proceed with the current information, select one tool from the tool set and provide complete, valid parameters. Avoid making parallel tool calls or calling a tool while interacting with the user in one step.
- When you believe the task is completed, provide the answer directly without any explanation.
EOF
)
# The user message is the original task text without an added title.
# ────────────────────────────────────────

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '2,6p' "$0"
  exit 0
fi
if [[ -z "$TASK" ]]; then
  echo "请指定 task_id 或 task.json 路径" >&2
  exit 2
fi
if [[ ! -d "$TASKS_DIR" ]]; then
  echo "任务目录不存在: $TASKS_DIR" >&2
  exit 2
fi
if [[ ! -s "$API_KEY_FILE" ]]; then
  echo "密钥文件不存在或为空: $API_KEY_FILE" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export DEBUG_TASK="$TASK"
export DEBUG_TASKS_DIR="$TASKS_DIR"
export DEBUG_OUTPUT_DIR="$OUTPUT_DIR"
export DEBUG_MAX_STEPS="$MAX_STEPS"
export DEBUG_TEMPERATURE="$TEMPERATURE"
export DEBUG_MAX_TOKENS="$MAX_TOKENS"
export DEBUG_BASE_URL="$BASE_URL"
export DEBUG_MODEL="$MODEL"
export DEBUG_API_KEY_FILE="$API_KEY_FILE"
export DEBUG_SYSTEM_PROMPT="$SYSTEM_PROMPT"

exec python3 - <<'PY'
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from evosynth.task_gen.contracts.task_bundle import TaskBundle
from evosynth.task_gen.contracts.versioned_bundle import load_bundle
from evosynth.task_gen.generation.runtime import (
    _atomic_write_json,
    _chat,
    _message_dict,
    configure_request_concurrency,
)
from evosynth.task_gen.generation.tool_info import compile_tool_info
from evosynth.task_gen.world.backend import (
    GenericEvoSynthBackend,
    normalize_arguments,
)
from evosynth.task_gen.world.universal import parse_world_contract
from evosynth.task_infer.rollout_engine import (
    _artifact_for_task,
    _pinned_input,
    configure_runtime,
)
from evosynth.utils import read_api_key_file


def _clip(text: object, limit: int = 800) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def resolve_task(spec: str, tasks_dir: Path) -> Path:
    raw = Path(spec).expanduser()
    if raw.is_file():
        return raw.resolve()
    name = spec.strip()
    if not name.endswith(".json"):
        name = f"{name}.json"
    direct = tasks_dir / name
    if direct.is_file():
        return direct.resolve()
    stem = Path(name).stem
    matches = [
        path for path in sorted(tasks_dir.glob(f"*{stem}*.json"))
        if not path.name.endswith((".public.json", ".construction_trace.json"))
    ]
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise SystemExit(f"找不到任务: {spec} （目录 {tasks_dir}）")
    shown = "\n".join(f"  {path}" for path in matches[:20])
    raise SystemExit(f"任务标识匹配到 {len(matches)} 个文件，请写完整 task_id 或路径:\n{shown}")


def main() -> int:
    tasks_dir = Path(os.environ["DEBUG_TASKS_DIR"])
    task_path = resolve_task(os.environ["DEBUG_TASK"], tasks_dir)
    output_dir = Path(os.environ["DEBUG_OUTPUT_DIR"])
    max_steps = int(os.environ["DEBUG_MAX_STEPS"])
    temperature = float(os.environ["DEBUG_TEMPERATURE"])
    max_tokens = int(os.environ["DEBUG_MAX_TOKENS"])
    system_prompt = os.environ["DEBUG_SYSTEM_PROMPT"]
    base_url = os.environ["DEBUG_BASE_URL"].strip()
    model = os.environ["DEBUG_MODEL"].strip()
    api_key = read_api_key_file(os.environ["DEBUG_API_KEY_FILE"])

    configure_request_concurrency(1)
    configure_runtime(base_url=base_url, model=model, api_key=api_key)

    started = time.time()
    task = TaskBundle.load(task_path)
    # 与 run_infer 一致：artifact_rel 相对 tasks 目录的上一级解析。
    tasks_root = tasks_dir.parent
    artifact = _artifact_for_task(task_path, tasks_root, task.metadata.get("artifact_rel"))
    bundle = load_bundle(artifact / "world", check_runtime=False, check_framework=False)
    task.verify_manifest(bundle.manifest)
    compiled = compile_tool_info(bundle.registry_path, domain=str(task.metadata["domain"]))
    runtime = GenericEvoSynthBackend(
        compiled,
        registry_path=bundle.registry_path,
        world_path=_pinned_input(bundle, "controlled_world.json"),
        allow_materialize=True,
    )
    initial_world_path = _pinned_input(bundle, "initial_world.json", required=False)
    if initial_world_path is not None:
        initial_payload = json.loads(initial_world_path.read_text(encoding="utf-8"))
        runtime.pin_initial_world(parse_world_contract(initial_payload, compiled))
    schemas = compiled.public_schemas()
    tool_names = [
        str((item.get("function") or {}).get("name") or item.get("name") or "")
        for item in schemas
    ]

    user_text = f"{task.task_natural}"
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    print(f"[task] id={task.task_id}")
    print(f"[task] file={task_path}")
    print(f"[task] domain={task.metadata.get('domain')} tools={len(schemas)}")
    print(f"[task] tool_names={', '.join(name for name in tool_names if name)}")
    print(f"[task] model={model} base_url={base_url} steps<={max_steps} temperature={temperature}")
    print("----- system prompt -----")
    print(system_prompt)
    print("----- user -----")
    print(user_text)
    print("-------------------------")

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=600, max_retries=0)
    calls = []
    final = ""
    invalid = 0
    for step in range(max_steps):
        print(f"[step {step + 1}] request messages={len(messages)}", flush=True)
        response = _chat(
            client,
            model=model,
            messages=messages,
            tools=schemas,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message
        messages.append(_message_dict(message))
        reasoning = getattr(message, "reasoning_content", "") or ""
        if reasoning:
            print(f"[step {step + 1}] reasoning {_clip(reasoning)}")
        if message.content:
            print(f"[step {step + 1}] content {_clip(message.content)}")
        if not message.tool_calls:
            final = message.content or ""
            print(f"[step {step + 1}] final answer")
            print(final)
            break
        raw_call = message.tool_calls[0]
        print(f"[step {step + 1}] tool={raw_call.function.name} args={raw_call.function.arguments}")
        try:
            args = json.loads(raw_call.function.arguments or "{}")
            args = normalize_arguments(compiled.by_public_name(raw_call.function.name), args)
            observation = runtime.call(raw_call.function.name, args, source="solver")
            if str(observation).startswith("Error:"):
                invalid += 1
                print(f"[step {step + 1}] tool error {_clip(observation, 500)}")
            else:
                print(f"[step {step + 1}] observation {_clip(observation, 800)}")
        except Exception as exc:
            args = {}
            observation = f"Error: {exc}"
            invalid += 1
            print(f"[step {step + 1}] tool exception {type(exc).__name__}: {exc}", file=sys.stderr)
        calls.append({
            "tool": raw_call.function.name,
            "arguments": args,
            "observation": observation[:4000],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": raw_call.id,
            "content": observation[:16000],
        })
    else:
        print(f"[step] 达到 MAX_STEPS={max_steps}，没有最终回答")

    final_answer = (final or "").strip()
    parsed = {"answer": final_answer} if final_answer else None
    result = {
        "task_id": task.task_id,
        "status": "inferred",
        "task_text": task.task_natural,
        "system_prompt": system_prompt,
        "final_text": final,
        "answer": parsed,
        "answer_refs": parsed,
        "expected": task.metadata.get("correct_answer"),
        "tool_calls": len(calls),
        "invalid_tool_calls": invalid,
        "distinct_successful_tools": len({
            str(row.get("tool") or "") for row in runtime.call_log if row.get("success")
        }),
        "frozen_world_hits": sum(
            row.get("provenance") == "pinned_controlled_world" for row in runtime.call_log
        ),
        "qwen_agentworld_materializations": sum(
            row.get("provenance") == "qwen_agentworld_materialized" for row in runtime.call_log
        ),
        "episode_cache_hits": sum(
            row.get("provenance") == "episode_materialized_cache" for row in runtime.call_log
        ),
        "parse_error": None,
        "call_trace": calls,
        "wall_time_sec": round(time.time() - started, 3),
        "solver_model": model,
        "agent_url": base_url,
        "debug": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task.task_id}.json"
    _atomic_write_json(out_path, {**result, "messages": messages})
    print("----- expected -----")
    print(json.dumps(task.metadata.get("correct_answer"), ensure_ascii=False, indent=2))
    print(f"[done] tools={len(calls)} invalid={invalid} sec={result['wall_time_sec']} file={out_path}")
    return 0


raise SystemExit(main())
PY
