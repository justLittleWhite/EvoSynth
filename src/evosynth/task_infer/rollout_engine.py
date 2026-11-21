#!/usr/bin/env python3
"""Teacher inference and answer judging using one endpoint/model/key configuration."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import os
import re
import shutil
import sys
import threading
from collections.abc import Collection
from typing import Any, Mapping, TextIO

from openai import OpenAI

from evosynth.utils import extract_json
from evosynth.task_gen.generation.runtime import (
    _atomic_write_json, _chat, _message_dict,
)
from evosynth.task_gen.world.backend import (
    GenericEvoSynthBackend, normalize_arguments,
)
from evosynth.task_gen.contracts.task_bundle import TaskBundle
from evosynth.task_gen.generation.tool_info import compile_tool_info
from evosynth.task_gen.world.universal import parse_world_contract
from evosynth.task_gen.contracts.versioned_bundle import load_bundle

import json
from pathlib import Path
import time

AGENT_URL = ""
INFER_MODEL = ""
_API_KEY = ""
_PRINT_LOCK = threading.Lock()
_PROGRESS = {"stage": "", "done": 0, "total": 0, "inferred": 0,
             "pass": 0, "fail": 0, "error": 0, "started": 0.0}
# 进度条占着当前行且不换行。其它 stdout/stderr（API 报错、traceback）若直接写，
# 会接在进度条后面；超宽折行后下一次 ``\r`` 只能清掉最后一行，留下残段。
_RAW_STDOUT = sys.stdout
_RAW_STDERR = sys.stderr
_PROGRESS_HELD = False
_FOREIGN_PARTIAL = False
_GUARD_INSTALLED = False

_COLOR = {
    "purple": "\033[35m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "white": "\033[37m",
}


def _clip(text: Any, limit: int = 400) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


_CACHED_COLUMNS = 0


def _terminal_columns() -> int:
    global _CACHED_COLUMNS
    if _CACHED_COLUMNS > 0:
        return _CACHED_COLUMNS
    for stream in (_RAW_STDOUT, _RAW_STDERR):
        try:
            columns = os.get_terminal_size(stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            continue
        if columns > 0:
            _CACHED_COLUMNS = columns
            return columns
    try:
        with open("/dev/tty", "rb", buffering=0) as tty:
            columns = os.get_terminal_size(tty.fileno()).columns
            if columns > 0:
                _CACHED_COLUMNS = columns
                return columns
    except OSError:
        pass
    columns = shutil.get_terminal_size(fallback=(100, 24)).columns
    _CACHED_COLUMNS = columns if columns > 0 else 100
    return _CACHED_COLUMNS


def _progress_line() -> str:
    """拼一行进度。窄终端时先缩短条，再丢掉耗时和尾部计数，始终留下 done/total。

    这一行一旦超过终端宽度就会折行，下一次 ``\\r`` 只能擦掉最后一行，
    上面会留下 ``infer [----] 6`` 这样的残段。
    """
    total = max(0, int(_PROGRESS["total"]))
    done = min(int(_PROGRESS["done"]), total)
    elapsed = max(0.0, time.time() - float(_PROGRESS["started"] or time.time()))
    rate = f"{100.0 * done / total:.1f}%" if total else "0.0%"
    stage = str(_PROGRESS.get("stage") or "")
    if stage == "infer":
        tail = f"done={_PROGRESS['inferred']} ERR={_PROGRESS['error']}"
    else:
        tail = (
            f"PASS={_PROGRESS['pass']} FAIL={_PROGRESS['fail']} "
            f"ERR={_PROGRESS['error']}"
        )
    budget = max(16, _terminal_columns() - 1)

    def render(bar_width: int, *, with_tail: bool, with_elapsed: bool) -> str:
        if bar_width <= 0:
            head = f"{stage} {done}/{total} {rate}"
        else:
            filled = 0 if total == 0 else int(bar_width * done / total)
            if done > 0 and filled == 0:
                filled = 1
            filled = min(bar_width, filled)
            bar = "#" * filled + "-" * (bar_width - filled)
            head = f"{stage} [{bar}] {done}/{total} {rate}"
        parts = [head]
        if with_tail:
            parts.append(tail)
        if with_elapsed:
            parts.append(f"{elapsed:.0f}s")
        return "  ".join(parts)

    for bar_width in (28, 16, 8, 0):
        for with_elapsed in (True, False):
            for with_tail in (True, False):
                line = render(bar_width, with_tail=with_tail, with_elapsed=with_elapsed)
                if len(line) <= budget:
                    return line
    line = f"{stage} {done}/{total}"
    return line if len(line) <= budget else line[:budget]


def _suspend_progress() -> None:
    """调用方已持有 ``_PRINT_LOCK``。把占着当前行的进度条擦掉。"""
    global _PROGRESS_HELD
    if not _PROGRESS_HELD:
        return
    _RAW_STDOUT.write("\r\033[2K")
    _RAW_STDOUT.flush()
    _PROGRESS_HELD = False


def _write_progress() -> None:
    """调用方已持有 ``_PRINT_LOCK``。"""
    global _PROGRESS_HELD, _FOREIGN_PARTIAL
    if not _PROGRESS.get("stage"):
        return
    if _FOREIGN_PARTIAL:
        _RAW_STDOUT.write("\n")
        _RAW_STDOUT.flush()
        _FOREIGN_PARTIAL = False
    _RAW_STDOUT.write(f"\r\033[2K{_COLOR['white']}{_progress_line()}\033[0m")
    _RAW_STDOUT.flush()
    _PROGRESS_HELD = True


class _StreamGuard:
    """把进度条和其它输出隔开：先擦掉进度条，写完一行后再画回去。"""

    def __init__(self, raw: TextIO) -> None:
        self._raw = raw

    def write(self, data: str) -> int:
        global _FOREIGN_PARTIAL
        text = str(data)
        if not text:
            return 0
        with _PRINT_LOCK:
            _suspend_progress()
            self._raw.write(text)
            self._raw.flush()
            if text.endswith("\n"):
                _FOREIGN_PARTIAL = False
                _write_progress()
            else:
                _FOREIGN_PARTIAL = True
        return len(text)

    def flush(self) -> None:
        self._raw.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def _install_stream_guard() -> None:
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    sys.stdout = _StreamGuard(_RAW_STDOUT)
    sys.stderr = _StreamGuard(_RAW_STDERR)
    _GUARD_INSTALLED = True


def _restore_stream_guard() -> None:
    global _GUARD_INSTALLED, _PROGRESS_HELD, _FOREIGN_PARTIAL
    with _PRINT_LOCK:
        _PROGRESS["stage"] = ""
        if _FOREIGN_PARTIAL:
            _RAW_STDOUT.write("\n")
            _FOREIGN_PARTIAL = False
        if _PROGRESS_HELD:
            _RAW_STDOUT.write("\n")
            _RAW_STDOUT.flush()
            _PROGRESS_HELD = False
    sys.stdout = _RAW_STDOUT
    sys.stderr = _RAW_STDERR
    _GUARD_INSTALLED = False


def _print_color(text: str, color: str) -> None:
    code = _COLOR.get(color, "")
    flat = re.sub(r"\s+", " ", text).strip()
    with _PRINT_LOCK:
        global _FOREIGN_PARTIAL
        if _FOREIGN_PARTIAL:
            _RAW_STDOUT.write("\n")
            _RAW_STDOUT.flush()
            _FOREIGN_PARTIAL = False
        _suspend_progress()
        _RAW_STDOUT.write(f"{code}{flat}\033[0m\n")
        _RAW_STDOUT.flush()
        _write_progress()


def _mark_task_done(status: str) -> None:
    with _PRINT_LOCK:
        _PROGRESS["done"] += 1
        if _PROGRESS.get("stage") == "infer":
            if status == "inferred":
                _PROGRESS["inferred"] += 1
            else:
                _PROGRESS["error"] += 1
        elif status == "PASS":
            _PROGRESS["pass"] += 1
        elif status == "FAIL":
            _PROGRESS["fail"] += 1
        else:
            _PROGRESS["error"] += 1
        _write_progress()


def configure_runtime(*, base_url: str, model: str, api_key: str) -> None:
    """显式配置 Open rollout 的唯一 endpoint/model/key owner。"""
    values = (str(base_url).strip(), str(model).strip(), str(api_key).strip())
    if not all(values):
        raise ValueError("base URL, model and API key must be non-empty")
    global AGENT_URL, INFER_MODEL, _API_KEY
    AGENT_URL, INFER_MODEL, _API_KEY = values
    from evosynth.task_gen.world.controlled import (
        configure_world_runtime,
    )
    configure_world_runtime(
        base_url=AGENT_URL, model=INFER_MODEL, api_key=_API_KEY)


def _pinned_input(bundle: Any, source_name: str, *,
                  required: bool = True) -> Path | None:
    revision = json.loads(bundle.world_revision_path.read_text(encoding="utf-8"))
    matches = [(bundle.root / str(item["path"])).resolve()
               for item in revision.get("files", ())
               if item.get("kind") == "world" and item.get("source_name") == source_name]
    if not matches and not required:
        return None
    if len(matches) != 1 or not matches[0].is_file():
        raise RuntimeError(f"pinned input missing or ambiguous: {source_name}")
    return matches[0]


def _artifact_for_task(task_path: Path, tasks_root: Path, artifact_rel: Any) -> Path:
    """Resolve an attempt artifact without depending on the caller's cwd.

    ``artifact_rel`` is written relative to the generation directory.  Older
    task exports and manually copied attempts occasionally pass a tasks
    directory whose parent is not the original generation directory; try the
    task's own attempt root as a compatibility fallback, but require the
    immutable ``world`` bundle before accepting a candidate.
    """
    raw = Path(str(artifact_rel or ""))
    candidates = [raw] if raw.is_absolute() else [
        Path(tasks_root) / raw,
        Path(task_path).parent.parent / raw,
        Path(task_path).parent / raw,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_dir() and (resolved / "world").is_dir():
            return resolved
    raise RuntimeError(
        f"task artifact/world bundle not found: task={task_path} artifact_rel={artifact_rel}")


def _answers_equal(expected: Any, actual: Any) -> bool:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        return set(expected) == set(actual) and all(
            _answers_equal(expected[key], actual[key]) for key in expected)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            _answers_equal(left, right) for left, right in zip(expected, actual))
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)) \
            and not isinstance(expected, bool) and not isinstance(actual, bool):
        return abs(float(expected) - float(actual)) <= 1e-9
    return expected == actual


SYSTEM_PROMPT = "You are a strict evaluator for AI agent trajectories. Evaluate whether the agent's trajectory is correct based on the task and the evidence in the trajectory. Output ONLY a valid JSON object."

JUDGE_PROMPT_TRAJECTORY = (Path(__file__).resolve().parents[3] / "prompts/infer/trajectory_judge.txt").read_text(encoding="utf-8")


def _format_trajectory(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    step = 0
    for message in messages[2:]:
        role = message.get("role")
        if role == "assistant":
            step += 1
            block = [f"### Assistant step {step}"]
            reasoning = str(message.get("reasoning_content") or "").strip()
            content = str(message.get("content") or "").strip()
            if reasoning:
                block.append(f"Reasoning: {reasoning[:4000]}")
            if content:
                block.append(content[:4000])
            for raw_call in message.get("tool_calls") or ():
                function = raw_call.get("function") or {}
                block.append(
                    f"Tool call: {function.get('name')}({function.get('arguments')})")
            parts.append("\n".join(block))
        elif role == "tool":
            parts.append(
                "### Tool result\n" + str(message.get("content") or "")[:4000])
    return "\n\n".join(parts) or "(empty trajectory)"


def _llm_trajectory_judge(*, task_text: str, messages: list[dict[str, Any]],
                          agent_answer: str) -> tuple[int, str]:
    """Trace-level judge; does not compare against hidden ground truth."""
    client = OpenAI(base_url=AGENT_URL, api_key=_API_KEY, timeout=600,
                    max_retries=0)
    prompt = JUDGE_PROMPT_TRAJECTORY.format(
        question=task_text,
        trajectory=_format_trajectory(messages),
        agent_answer=agent_answer,
    )
    response = _chat(
        client, model=INFER_MODEL, temperature=0, max_tokens=8192,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    judged = extract_json(response.choices[0].message.content or "")
    if not isinstance(judged, Mapping):
        raise ValueError("trajectory_judge_invalid_json")
    try:
        score = int(judged.get("score"))
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory_judge_invalid_score") from exc
    if score not in {0, 1}:
        raise ValueError("trajectory_judge_invalid_score")
    return score, str(judged.get("reason") or "")[:800]


def _solve_one(task_path: Path, tasks_root: Path, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    task = TaskBundle.load(task_path)
    artifact = _artifact_for_task(task_path, tasks_root, task.metadata.get("artifact_rel"))
    # The bundle records the runtime configuration of the generation host.
    # Inference is intentionally allowed on another host/process; the task's
    # immutable manifest and task-to-world binding below remain authoritative.
    # Framework source drift is also skipped: later serialization-only edits
    # (unused task JSON fields) must not block solving a pinned world.
    bundle = load_bundle(
        artifact / "world", check_runtime=False, check_framework=False)
    task.verify_manifest(bundle.manifest)
    compiled = compile_tool_info(bundle.registry_path, domain=str(task.metadata["domain"]))
    runtime = GenericEvoSynthBackend(
        compiled, registry_path=bundle.registry_path,
        world_path=_pinned_input(bundle, "controlled_world.json"),
        # EvoSynth inference first reads the pinned base world.  A schema-valid
        # query that is absent there is materialized by Qwen-AgentWorld and kept
        # in this episode's in-memory overlay; it never mutates another task's
        # world bundle.
        allow_materialize=True)
    initial_world_path = _pinned_input(bundle, "initial_world.json", required=False)
    if initial_world_path is not None:
        initial_payload = json.loads(initial_world_path.read_text(encoding="utf-8"))
        runtime.pin_initial_world(parse_world_contract(initial_payload, compiled))

    schemas = compiled.public_schemas()

    # 1. 行为准则放在 System Prompt 中；最终回答直接写在 assistant 文本里。
    system_prompt = "You are a helpful assistant.\n## Guidelines\n- Carefully read tool descriptions and use correct argument names.\n- Only provide your final answer when you have sufficient information.\nWhen you have completed your analysis and gathered all necessary information, provide your brief final response directly without any explanation."

    # 2. User 消息保持纯净，只放任务本身（或保留轻量标题）
    user_text = f"""{task.task_natural}"""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    client = OpenAI(base_url=AGENT_URL, api_key=_API_KEY, timeout=600,
                    max_retries=0)
    calls = []
    final = ""
    invalid = 0
    # 发布包不再带 target_complexity；求解步数用固定安全上限。
    solver_step_limit = 50
    for step in range(solver_step_limit):
        try:
            response = _chat(client, model=INFER_MODEL, messages=messages,
                             tools=schemas,
                             tool_choice="auto",
                             parallel_tool_calls=False,
                             temperature=0.5, max_tokens=12000)
        except Exception as exc:
            _print_color(f"[error] {task.task_id} step={step + 1} {exc}", "red")
            raise
        message = response.choices[0].message
        messages.append(_message_dict(message))
        if not message.tool_calls:
            # Evidence coverage is checked by the publish gate rather than by
            # inserting intermediate system reminders into the trajectory.
            final = message.content or ""
            preview = _clip(final) or _clip(getattr(message, "reasoning_content", ""))
            _print_color(
                f"[model] {task.task_id} step={step + 1} final {preview}",
                "yellow")
            break
        raw_call = message.tool_calls[0]
        preview = _clip(message.content or getattr(message, "reasoning_content", ""))
        _print_color(
            f"[model] {task.task_id} step={step + 1} "
            f"tool={raw_call.function.name} {preview}",
            "yellow")
        try:
            args = json.loads(raw_call.function.arguments or "{}")
            args = normalize_arguments(compiled.by_public_name(raw_call.function.name), args)
            observation = runtime.call(raw_call.function.name, args, source="solver")
            if observation.startswith("Error:"):
                invalid += 1
                _print_color(
                    f"[error] {task.task_id} {raw_call.function.name} "
                    f"{_clip(observation, 300)}",
                    "red")
            else:
                _print_color(
                    f"[tool] {task.task_id} {raw_call.function.name}"
                    f"({_clip(json.dumps(args, ensure_ascii=False), 240)}) "
                    f"-> {_clip(observation, 240)}",
                    "purple")
        except Exception as exc:
            args = {}; observation = f"Error: {exc}"; invalid += 1
            _print_color(
                f"[error] {task.task_id} {raw_call.function.name} {exc}",
                "red")
        calls.append({"tool": raw_call.function.name, "arguments": args,
                      "observation": observation[:4000]})
        messages.append({"role": "tool", "tool_call_id": raw_call.id,
                         "content": observation[:16000]})
    final_answer = (final or "").strip()
    parsed = {"answer": final_answer} if final_answer else None
    result = {"task_id": task.task_id, "status": "inferred",
              "task_text": task.task_natural,
              "final_text": final,
              "answer": parsed, "answer_refs": parsed,
              "expected": task.metadata["correct_answer"],
              "tool_calls": len(calls), "invalid_tool_calls": invalid,
              # nudge 删除后用它归因覆盖不足的失败(实际成功覆盖的不同工具数)
              "distinct_successful_tools": len({
                  str(row.get("tool") or "") for row in runtime.call_log
                  if row.get("success")}),
              "frozen_world_hits": sum(
                  row.get("provenance") == "pinned_controlled_world"
                  for row in runtime.call_log),
              "qwen_agentworld_materializations": sum(
                  row.get("provenance") == "qwen_agentworld_materialized"
                  for row in runtime.call_log),
              "episode_cache_hits": sum(
                  row.get("provenance") == "episode_materialized_cache"
                  for row in runtime.call_log),
              "parse_error": None, "call_trace": calls,
              # Keep the backend's per-call provenance and materialization
              # candidates for independent offline consistency evaluation.
              "world_call_log": copy.deepcopy(runtime.call_log),
              "wall_time_sec": round(time.time() - started, 3),
              "solver_model": INFER_MODEL, "agent_url": AGENT_URL}
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_dir / f"{task.task_id}.json", {**result, "messages": messages})
    return result


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


_INFER_DONE_STATUSES = {"inferred", "PASS", "FAIL", "answer_judge_error"}
_JUDGE_DONE_STATUSES = {"PASS", "FAIL", "answer_judge_error"}


def _infer_done_payload(path: Path) -> dict[str, Any] | None:
    """已写出轨迹，或已经评完的结果，都不再重做 infer；若因 API 故障导致 materializer 报错则重做。"""
    payload = _load_json_object(path)
    if payload is None or payload.get("status") not in _INFER_DONE_STATUSES:
        return None
    if payload.get("status") == "inferred" and "messages" not in payload:
        return None
    for row in payload.get("call_trace") or ():
        obs = str((row or {}).get("observation") or "")
        if "WorldMaterializationError" in obs or "AuthenticationError" in obs:
            return None
    return payload


def _judge_done_payload(path: Path) -> dict[str, Any] | None:
    payload = _load_json_object(path)
    if payload is None:
        return None
    if payload.get("status") in _JUDGE_DONE_STATUSES:
        return payload
    return None


def _task_paths(tasks_dir: Path) -> list[Path]:
    return [path for path in sorted(Path(tasks_dir).glob("uwg_*.json"))
            if not path.name.endswith((".public.json", ".construction_trace.json"))]


def run_infer(*, tasks_dir: Path, output_dir: Path, workers: int = 20,
              limit: int | None = None, skip_completed: bool = True) -> dict[str, Any]:
    found = _task_paths(tasks_dir)
    output_root = Path(output_dir)
    skipped_rows: list[dict[str, Any]] = []
    pending = found
    if skip_completed:
        pending = []
        for path in found:
            existing = _infer_done_payload(output_root / f"{path.stem}.json")
            if existing is None:
                pending.append(path)
            else:
                skipped_rows.append({"task_id": path.stem,
                                     "status": existing.get("status"),
                                     "skipped": True})
    if limit is not None:
        pending = pending[:limit]
    _PROGRESS.update({
        "stage": "infer", "done": 0, "total": len(pending), "inferred": 0,
        "pass": 0, "fail": 0, "error": 0, "started": time.time(),
    })
    _install_stream_guard()
    try:
        return _run_infer_body(
            tasks_dir=tasks_dir, found=found, pending=pending,
            skipped_rows=skipped_rows, output_root=output_root, workers=workers)
    finally:
        _restore_stream_guard()


def _run_infer_body(*, tasks_dir: Path, found: list[Path], pending: list[Path],
                    skipped_rows: list[dict[str, Any]], output_root: Path,
                    workers: int) -> dict[str, Any]:
    _print_color(
        f"[infer] start found={len(found)} skipped={len(skipped_rows)} "
        f"pending={len(pending)} workers={workers} model={INFER_MODEL}",
        "white")
    results = list(skipped_rows)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_solve_one, path, Path(tasks_dir).parent,
                               output_root): path for path in pending}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                _mark_task_done(str(result.get("status") or "exception"))
            except Exception as exc:
                task_id = futures[future].stem
                _print_color(f"[error] {task_id} {type(exc).__name__}: {exc}", "red")
                results.append({"task_id": task_id, "status": "exception",
                                "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
                _mark_task_done("exception")
    counts = Counter(str(row["status"]) for row in results)
    report = {"stage": "infer", "found": len(found), "skipped": len(skipped_rows),
              "pending": len(pending), "tasks": len(found),
              "status_counts": dict(sorted(counts.items())),
              "solver_model": INFER_MODEL, "agent_url": AGENT_URL}
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_root / "infer_report.json", report)
    return report


def _infer_result_paths(output_dir: Path) -> list[Path]:
    skip = {"infer_report.json", "judge_report.json"}
    return sorted(
        path for path in Path(output_dir).glob("uwg_*.json")
        if path.name not in skip and not path.name.endswith(".public.json"))


def _judge_one(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid infer result: {path}")
    task_id = str(payload.get("task_id") or path.stem)
    if payload.get("status") == "exception" or "messages" not in payload:
        payload["status"] = "exception"
        payload["judge_score"] = None
        payload["judge_reason"] = ""
        payload["judge_error"] = payload.get("error") or "missing trajectory"
        _print_color(f"[error] {task_id} skip judge: {payload['judge_error']}", "red")
        _atomic_write_json(path, payload)
        return payload
    agent_answer = str(payload.get("final_text") or "").strip()
    if not agent_answer:
        answer = payload.get("answer") or {}
        if isinstance(answer, Mapping):
            agent_answer = str(answer.get("answer") or "").strip()
    messages = list(payload.get("messages") or ())
    task_text = str(payload.get("task_text") or "")
    if not task_text:
        for message in messages:
            if message.get("role") == "user":
                task_text = str(message.get("content") or "")
                break
    try:
        score, reason = _llm_trajectory_judge(
            task_text=task_text,
            messages=messages,
            agent_answer=agent_answer,
        )
        payload["judge_score"] = score
        payload["judge_reason"] = reason
        payload["judge_error"] = None
        payload["status"] = "PASS" if score == 1 else "FAIL"
        _print_color(
            f"[judge] {task_id} score={score} {_clip(reason, 240)}",
            "yellow" if score == 1 else "red")
    except Exception as exc:
        payload["judge_score"] = None
        payload["judge_reason"] = ""
        payload["judge_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        payload["status"] = "answer_judge_error"
        _print_color(f"[error] {task_id} judge {exc}", "red")
    _atomic_write_json(path, payload)
    return payload


def run_judge(*, output_dir: Path, workers: int = 20,
              skip_completed: bool = True,
              only_statuses: Collection[str] | None = None) -> dict[str, Any]:
    """Judge 已有轨迹。

    ``only_statuses`` 非空时只重跑这些状态（例如 ``answer_judge_error``），
    其余结果原样计入报告、不改文件。
    """
    found = _infer_result_paths(output_dir)
    skipped_rows: list[dict[str, Any]] = []
    pending = found
    wanted = {str(item) for item in only_statuses} if only_statuses else None
    if wanted is not None:
        pending = []
        for path in found:
            payload = _load_json_object(path)
            status = str((payload or {}).get("status") or "")
            if status in wanted:
                pending.append(path)
            elif payload is not None:
                skipped_rows.append({"task_id": path.stem, "status": status,
                                     "skipped": True})
    elif skip_completed:
        pending = []
        for path in found:
            existing = _judge_done_payload(path)
            if existing is None:
                pending.append(path)
            else:
                skipped_rows.append({"task_id": path.stem,
                                     "status": existing.get("status"),
                                     "skipped": True})
    _PROGRESS.update({
        "stage": "judge", "done": 0, "total": len(pending), "inferred": 0,
        "pass": 0, "fail": 0, "error": 0, "started": time.time(),
    })
    scope = f" only={','.join(sorted(wanted))}" if wanted else ""
    _install_stream_guard()
    try:
        return _run_judge_body(
            output_dir=output_dir, found=found, pending=pending,
            skipped_rows=skipped_rows, workers=workers, wanted=wanted, scope=scope)
    finally:
        _restore_stream_guard()


def _run_judge_body(*, output_dir: Path, found: list[Path], pending: list[Path],
                    skipped_rows: list[dict[str, Any]], workers: int,
                    wanted: set[str] | None, scope: str) -> dict[str, Any]:
    _print_color(
        f"[judge] start found={len(found)} skipped={len(skipped_rows)} "
        f"pending={len(pending)} workers={workers} model={INFER_MODEL}{scope}",
        "white")
    results = list(skipped_rows)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(_judge_one, path): path for path in pending}
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                _mark_task_done(str(result.get("status") or "exception"))
            except Exception as exc:
                path = futures[future]
                _print_color(f"[error] {path.stem} {type(exc).__name__}: {exc}", "red")
                results.append({"task_id": path.stem, "status": "exception",
                                "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
                _mark_task_done("exception")
    counts = Counter(str(row.get("status")) for row in results)
    passed = counts.get("PASS", 0)
    report = {"stage": "judge", "found": len(found), "skipped": len(skipped_rows),
              "pending": len(pending), "tasks": len(found), "passed": passed,
              "success_rate": passed / len(found) if found else 0.0,
              "status_counts": dict(sorted(counts.items())),
              "solver_model": INFER_MODEL, "agent_url": AGENT_URL}
    if wanted is not None:
        report["rejudge_statuses"] = sorted(wanted)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _atomic_write_json(Path(output_dir) / "judge_report.json", report)
    return report


__all__ = ["configure_runtime", "run_infer", "run_judge"]
