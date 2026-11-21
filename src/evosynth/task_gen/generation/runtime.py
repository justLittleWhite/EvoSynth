"""EvoSynth 生成运行时：凭据、截止时间、日志、模型请求与结构化阶段。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import os
import random
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

from openai import OpenAI

from evosynth.utils import extract_json
from evosynth.task_gen.common.llm_sampling import normalize_sampling_kwargs

MAX_AGENT_RETRIES = 5

MAX_TOOL_RECOVERY_ATTEMPTS = 1

MAX_CONSECUTIVE_TOOL_FAILURES = 2

GENERATION_MAX_TOKENS = 16384

@dataclass(frozen=True)
class OpenGenerationRuntimeConfig:
    agent_base_url: str
    agent_model: str
    agent_api_key: str
    world_base_url: str
    world_model: str
    world_api_key: str


_RUNTIME = OpenGenerationRuntimeConfig(
    "",
    "generator-model-not-configured",
    "",
    "",
    "world-model-not-configured",
    "",
)


def configure_runtime(*, agent_base_url: str, agent_model: str,
                      agent_api_key: str, world_base_url: str,
                      world_model: str,
                      world_api_key: str) -> OpenGenerationRuntimeConfig:
    values = (
        str(agent_base_url).strip(), str(agent_model).strip(),
        str(agent_api_key).strip(), str(world_base_url).strip(),
        str(world_model).strip(), str(world_api_key).strip(),
    )
    if not all(values):
        raise ValueError("EvoSynth endpoints, models and API keys must be non-empty")
    global _RUNTIME
    _RUNTIME = OpenGenerationRuntimeConfig(*values)
    from evosynth.task_gen.world.controlled import (
        configure_world_runtime,
    )
    configure_world_runtime(
        base_url=_RUNTIME.world_base_url,
        model=_RUNTIME.world_model,
        api_key=_RUNTIME.world_api_key,
    )
    return _RUNTIME


def runtime_config() -> OpenGenerationRuntimeConfig:
    return _RUNTIME


def configure_request_concurrency(workers: int) -> int:
    """agent 与世界模型的在途请求上限都跟随 --workers。"""
    if workers <= 0:
        raise ValueError("request concurrency must be positive")
    global MODEL_CONCURRENCY, _MODEL_SEMAPHORE
    MODEL_CONCURRENCY = workers
    _MODEL_SEMAPHORE = threading.BoundedSemaphore(workers)
    from evosynth.task_gen.world.controlled import (
        configure_world_concurrency,
    )
    configure_world_concurrency(workers)
    return MODEL_CONCURRENCY

TASK_DEADLINE_SECONDS = int(os.environ.get("EVOSYNTH_TASK_DEADLINE_SECONDS", "0"))

INFRA_RETRY_ATTEMPTS = int(os.environ.get("EVOSYNTH_INFRA_RETRY_ATTEMPTS", "2"))

MODEL_CONCURRENCY = int(os.environ.get("EVOSYNTH_MODEL_CONCURRENCY", "15"))

MODEL_QUEUE_TIMEOUT_SECONDS = int(os.environ.get(
    "EVOSYNTH_MODEL_QUEUE_TIMEOUT_SECONDS", "600"))

_MODEL_SEMAPHORE = threading.BoundedSemaphore(max(1, MODEL_CONCURRENCY))

# 错峰起步：服务端会限制请求速率的爬升速度（Throttling.BurstRate）。
# 进程启动后的前 2×并发 个请求按固定间隔依次发出，之后不再限速。
RAMP_INTERVAL_SECONDS = float(os.environ.get("EVOSYNTH_RAMP_INTERVAL_SECONDS", "0.2"))
# 爬升期结束后仍保持的最小请求间隔，避免任务同时完成后又同时发出一波请求。
STEADY_INTERVAL_SECONDS = float(os.environ.get("EVOSYNTH_STEADY_INTERVAL_SECONDS", "0.1"))
_RAMP_LOCK = threading.Lock()
_RAMP_STATE = {"sent": 0, "next_at": 0.0}


def throttle_request() -> None:
    """agent 与世界模型请求共用：爬升期慢发，之后也不允许同一瞬间齐发。"""
    with _RAMP_LOCK:
        ramping = _RAMP_STATE["sent"] < 2 * max(1, MODEL_CONCURRENCY)
        interval = RAMP_INTERVAL_SECONDS if ramping else STEADY_INTERVAL_SECONDS
        if interval <= 0:
            return
        _RAMP_STATE["sent"] += 1
        now = time.monotonic()
        start_at = max(now, _RAMP_STATE["next_at"])
        _RAMP_STATE["next_at"] = start_at + interval
    delay = start_at - now
    if delay > 0:
        time.sleep(delay)


_ramp_wait = throttle_request

_TASK_CONTEXT = threading.local()

_LOG_CONTEXT = threading.local()

_LOG_LOCK = threading.Lock()
_STDOUT_PROGRESS_EVENTS = frozenset({
    "generation_start", "world_generation_started", "world_generation_finished",
    "task_start", "initial_world_constructed",
    "research_completed", "research_attempt_discarded", "pipeline_result",
    "task_coherence_started", "task_coherence_result",
    "research_revision",
    "task_rejected", "task_publish_failed", "task_published",
    "seed_infrastructure_retry", "generation_end",
})
_MISSING_LOG_CONTEXT = object()

def print_api_error(scope: str, exc: BaseException) -> None:
    """API 失败立刻打到终端，不等任务结束或日志汇总。"""
    task_id = str(getattr(_LOG_CONTEXT, "task_id", "") or "")
    suffix = f" task={task_id}" if task_id else ""
    text = f"[evosynth] API error {scope}{suffix}: {type(exc).__name__}: {exc}"
    print(f"\033[31m{text}\033[0m", flush=True)


def _world_model_metadata() -> dict[str, str]:
    return {
        "world_observation_endpoint": _RUNTIME.world_base_url,
        "world_observation_model": _RUNTIME.world_model,
    }

def _start_task_deadline() -> None:
    _TASK_CONTEXT.deadline = (
        time.monotonic() + TASK_DEADLINE_SECONDS
        if TASK_DEADLINE_SECONDS > 0 else 0.0)

def _remaining_task_seconds() -> float:
    deadline = float(getattr(_TASK_CONTEXT, "deadline", 0.0) or 0.0)
    return deadline - time.monotonic() if deadline else 600.0

def _check_task_deadline(stage: str) -> None:
    deadline = float(getattr(_TASK_CONTEXT, "deadline", 0.0) or 0.0)
    if deadline and _remaining_task_seconds() <= 0:
        raise TimeoutError(f"task deadline exceeded during {stage}")

def _extend_task_deadline(waited_seconds: float) -> None:
    """资源槽位排队不消耗任务的实际执行预算。"""
    deadline = float(getattr(_TASK_CONTEXT, "deadline", 0.0) or 0.0)
    if deadline and waited_seconds > 0:
        _TASK_CONTEXT.deadline = deadline + waited_seconds

def _observe_world_queue_wait(waited_seconds: float) -> None:
    _extend_task_deadline(waited_seconds)
    _log_event("world_queue_wait", seconds=waited_seconds)

class _ModelInfrastructureError(RuntimeError):
    """生成模型调度或连接故障，不属于任务语义拒绝。"""

def _evosynth_agent_key() -> str:
    """返回显式配置的生成凭据；未配置时失败关闭。"""
    if not _RUNTIME.agent_api_key:
        raise RuntimeError("EvoSynth agent API key is not configured")
    return _RUNTIME.agent_api_key

class _ResearchAttemptIncomplete(RuntimeError):
    """当前真实执行不足以反写一个已经解决的任务。"""

def _recoverable_tool_failure(observation: str) -> bool:
    """仅重试可能随请求恢复的传输或世界材质化失败。"""
    lowered = str(observation).casefold()
    return any(token in lowered for token in (
        "timeout", "timed out", "connection", "temporarily unavailable",
        "rate limit", "429", "empty materializer response",
        "materializer failed output or consistency contract",
    ))

def _infrastructure_failure(reasons: Sequence[Any]) -> bool:
    text = " ".join(str(item) for item in reasons).casefold()
    return any(token in text for token in (
        "worldinfrastructureerror", "_modelinfrastructureerror",
        "model scheduler queue timed out", "8822 scheduler queue timed out",
        "apiconnectionerror", "connectionerror", "api connection",
        "apitimeouterror", "agent api failed after",
        "rate limit", "429", "selected model is at capacity",
    ))

def _bind_log(path: Path | None, *, domain: str | None = None,
              task_id: str | None = None) -> None:
    _LOG_CONTEXT.path = str(path) if path else ""
    _LOG_CONTEXT.domain = str(domain or "")
    _LOG_CONTEXT.task_id = str(task_id or "")


def set_task_log_task_id(task_id: str | None) -> None:
    """更新当前线程的任务日志标识。"""
    _LOG_CONTEXT.task_id = str(task_id or "")


@contextmanager
def task_log_context(
        task_id: str | None = None,
        ) -> Iterator[Callable[[str | None], None]]:
    """隔离任务日志上下文，并在正常返回或异常时恢复进入前状态。"""
    fields = ("path", "domain", "task_id")
    previous = {
        field: getattr(_LOG_CONTEXT, field, _MISSING_LOG_CONTEXT)
        for field in fields
    }
    if task_id is not None:
        set_task_log_task_id(task_id)
    try:
        yield set_task_log_task_id
    finally:
        for field, value in previous.items():
            if value is _MISSING_LOG_CONTEXT:
                if hasattr(_LOG_CONTEXT, field):
                    delattr(_LOG_CONTEXT, field)
            else:
                setattr(_LOG_CONTEXT, field, value)


def _log_event(event: str, **payload: Any) -> None:
    path = getattr(_LOG_CONTEXT, "path", "")
    context = {}
    if getattr(_LOG_CONTEXT, "domain", "") and "domain" not in payload:
        context["domain"] = _LOG_CONTEXT.domain
    if getattr(_LOG_CONTEXT, "task_id", "") and "task_id" not in payload:
        context["task_id"] = _LOG_CONTEXT.task_id
    row = {"time": datetime.now().isoformat(timespec="milliseconds"),
           "event": event, **context, **payload}
    if os.environ.get("EVOSYNTH_STDOUT_PROGRESS", "").casefold() in {
            "1", "true", "yes", "on"} and event in _STDOUT_PROGRESS_EVENTS:
        task_id = str(payload.get("task_id") or context.get("task_id") or "")
        suffix = f" task={task_id}" if task_id else ""
        details = []
        for key in ("attempt", "accepted", "status", "reason", "reasons",
                    "entities", "seed"):
            if key in payload and payload[key] not in (None, "", [], {}):
                value = payload[key]
                rendered = str(value).replace("\n", " ")
                details.append(f"{key}={rendered[:240]}")
        print(f"[evosynth] {event}{suffix} " + " ".join(details), flush=True)
    if not path:
        return
    try:
        text = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with _LOG_LOCK:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with Path(path).open("a", encoding="utf-8") as handle:
                handle.write(text)
    except Exception:
        # 日志失败不能改变合成语义；结构化 attempt 文件仍会保留错误。
        pass

def _normalize_sampling(kwargs: dict[str, Any]) -> dict[str, Any]:
    return normalize_sampling_kwargs(str(kwargs.get("model", "")), kwargs)

def _chat(client: OpenAI, **kwargs: Any) -> Any:
    kwargs = _normalize_sampling(kwargs)
    delay = 2
    last: Exception | None = None
    for attempt in range(MAX_AGENT_RETRIES):
        _check_task_deadline("model request")
        try:
            safe = {key: value for key, value in kwargs.items()
                    if key not in {"api_key", "client"}}
            _log_event("model_request", attempt=attempt + 1, request=safe)
            queue_started = time.monotonic()
            acquired = _MODEL_SEMAPHORE.acquire(
                timeout=max(1, MODEL_QUEUE_TIMEOUT_SECONDS))
            queue_waited = time.monotonic() - queue_started
            _extend_task_deadline(queue_waited)
            _log_event("model_queue_wait", seconds=queue_waited,
                       acquired=acquired)
            if not acquired:
                raise _ModelInfrastructureError(
                    "model scheduler queue timed out")
            try:
                # 槽位等待已经返还给任务预算；在真正发请求前再按新的剩余
                # 时间设置 SDK 超时，不能沿用排队前已经缩短的值。
                kwargs["timeout"] = max(
                    1.0, min(float(kwargs.get("timeout", 600)),
                             _remaining_task_seconds()))
                _ramp_wait()
                response = client.chat.completions.create(**kwargs)
            finally:
                _MODEL_SEMAPHORE.release()
            _log_event("model_response", attempt=attempt + 1,
                       response=response.model_dump(exclude_none=False)
                       if hasattr(response, "model_dump") else str(response))
            return response
        except Exception as exc:
            last = exc
            print_api_error(f"agent attempt={attempt + 1}", exc)
            _log_event("model_error", attempt=attempt + 1,
                       error=f"{type(exc).__name__}: {exc}")
            if attempt + 1 < MAX_AGENT_RETRIES:
                if isinstance(exc, (TimeoutError, _ModelInfrastructureError)):
                    raise
                # 加随机抖动，避免大量线程在同一时刻一起重试。
                jittered = delay * (0.5 + random.random())
                time.sleep(min(jittered, max(0.0, _remaining_task_seconds())))
                delay = min(30, delay * 2)
    raise RuntimeError(f"agent API failed after {MAX_AGENT_RETRIES} attempts: {last}")

def _message_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        value = message.model_dump(exclude_none=True)
    elif isinstance(message, Mapping):
        value = dict(message)
    else:
        value = {"role": "assistant", "content": str(message)}
    if getattr(message, "reasoning_content", None) is not None:
        value["reasoning_content"] = getattr(message, "reasoning_content") or ""
    if value.get("tool_calls"):
        value["tool_calls"] = value["tool_calls"][:1]
    return value

_OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.I | re.S)


def _unwrap_output_json(raw: str) -> str:
    """优先取 <output> 内文本，避免 extract_json 命中标签外的嵌套对象。"""
    match = _OUTPUT_RE.search(raw or "")
    return match.group(1).strip() if match else (raw or "")


def _run_json_stage(*, client: OpenAI, prompt: str, stage: str,
                    validator: Any, state: dict[str, Any],
                    temperature: float = 0.2,
                    max_attempts: int = 2,
                    enable_thinking: bool = False) -> Any:
    """在同一冻结证据上运行有界结构化生成与机械返修。"""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    messages = [{"role": "user", "content": prompt}]
    attempts = []
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        _check_task_deadline(stage)
        response = _chat(
            client, model=_RUNTIME.agent_model, temperature=temperature,
            max_tokens=GENERATION_MAX_TOKENS,
            extra_body={"enable_thinking": enable_thinking},
            messages=messages)
        raw = response.choices[0].message.content or ""
        reasoning = getattr(
            response.choices[0].message, "reasoning_content", "") or ""
        parsed = extract_json(_unwrap_output_json(raw))
        debug = {
            "attempt": attempt + 1,
            "finish_reason": str(getattr(
                response.choices[0], "finish_reason", "") or ""),
            "content_chars": len(raw), "content_tail": raw[-2000:],
            "reasoning_chars": len(reasoning),
            "reasoning_tail": reasoning[-1000:],
            "thinking_disabled": not enable_thinking,
        }
        attempts.append(debug)
        try:
            if not isinstance(parsed, Mapping):
                raise TypeError(f"{stage} must return one JSON object")
            result = validator(dict(parsed))
            debug["validation_status"] = "ok"
            state[f"{stage}_debug"] = {**debug, "attempts": attempts}
            return result
        except Exception as exc:
            last_error = exc
            debug["validation_status"] = "invalid"
            debug["validation_error"] = f"{type(exc).__name__}:{str(exc)[:800]}"
            if attempt + 1 < max_attempts:
                messages.extend([
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content":
                     f"The {stage} JSON failed mechanical validation: "
                     f"{type(exc).__name__}: {str(exc)[:800]}. Return the complete corrected "
                     "JSON object only. Use exactly the same frozen evidence; do not add calls, "
                     "claims, entities, values, tools, or task candidates."},
                ])
    state[f"{stage}_debug"] = {**attempts[-1], "attempts": attempts}
    raise ValueError(f"{stage}:{last_error}") from last_error

def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
