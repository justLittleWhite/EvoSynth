#!/usr/bin/env python3
"""用 LLM 判断求解答案和 ground truth 在语义、数值上是否是同一个结果。

只比较两个答案文本，不看轨迹、工具调用或推理。表述、单位和多余说明可以不同，
决定结果的实体和数字必须相同。score 为 1 或 0。

结果写回每条轨迹的 ``answer_equiv_score`` / ``answer_equiv_reason``，
不修改原来的 ``status`` 和轨迹裁判字段。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evosynth.utils import (  # noqa: E402
    chat_with_retry,
    extract_json,
    openai_client,
    read_api_key_file,
)

DEFAULT_ROLLOUT = None
DEFAULT_TASKS = None
DEFAULT_BASE_URL = ""
DEFAULT_MODEL = "qwen3.8-flash"
DEFAULT_KEY_FILE = Path("")

_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
_WS_RE = re.compile(r"\s+")
_PRINT_LOCK = threading.Lock()

SYSTEM_PROMPT = (
    "You compare a predicted answer with the ground truth. "
    "Decide whether they state the same result in meaning and in numbers. "
    "Output ONLY a valid JSON object."
)

USER_PROMPT = """Compare the predicted answer with the ground truth.

Score 1 only when both name the same result:
- The decisive entity, choice, or conclusion is the same.
- Every number that is part of the result is numerically the same.
  17, 17.0, and 17°C are the same quantity when they refer to the same value.
  Different quantities are not the same.
- Wording, word order, units written on only one side, and extra explanation
  do not matter when they do not change or contradict the result.
  Ground truth "G234" and prediction "Book train G234" are the same.

Score 0 when any of these holds:
- The decisive entity or choice differs.
- A number that belongs to the result differs.
- The prediction treats an additional, different result as also correct.
- The prediction is empty or does not commit to the ground-truth result.

Ground truth:
{gold}

Predicted answer:
{predicted}

Return only this JSON object:
{{"score": 0 or 1, "reason": "short justification"}}
"""


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict) and set(value) == {"answer"}:
        return _as_text(value.get("answer"))
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _prediction(payload: dict) -> str:
    answer = payload.get("answer")
    if isinstance(answer, dict):
        text = _as_text(answer.get("answer"))
        if text:
            return text
    elif isinstance(answer, str) and answer.strip():
        return answer.strip()
    final = str(payload.get("final_text") or "")
    match = _ANSWER_RE.search(final)
    if match:
        return match.group(1).strip()
    return final.strip()


def _ground_truth(payload: dict, task: dict | None) -> str:
    expected = payload.get("expected")
    if isinstance(expected, dict):
        text = _as_text(expected.get("answer"))
        if text:
            return text
    elif isinstance(expected, str) and expected.strip():
        return expected.strip()
    if not task:
        return ""
    correct = (task.get("metadata") or {}).get("correct_answer")
    if isinstance(correct, dict):
        return _as_text(correct.get("answer"))
    return _as_text(correct)


def _load_task(tasks_dir: Path, task_id: str) -> dict | None:
    path = tasks_dir / f"{task_id}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _message_text(message) -> str:
    content = str(getattr(message, "content", None) or "").strip()
    if content:
        return content
    return str(getattr(message, "reasoning_content", None) or "").strip()


def _judge_pair(client, model: str, gold: str, predicted: str) -> tuple[int, str]:
    response = chat_with_retry(
        client,
        model=model,
        temperature=0,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(gold=gold, predicted=predicted)},
        ],
    )
    raw = _message_text(response.choices[0].message)
    judged = extract_json(raw)
    if not isinstance(judged, dict):
        raise ValueError("answer_equiv_invalid_json")
    try:
        score = int(judged.get("score"))
    except (TypeError, ValueError) as exc:
        raise ValueError("answer_equiv_invalid_score") from exc
    if score not in {0, 1}:
        raise ValueError("answer_equiv_invalid_score")
    return score, str(judged.get("reason") or "")[:800]


def _atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    os.replace(temporary, path)


def _already_done(payload: dict) -> bool:
    return (
        payload.get("answer_equiv_score") in {0, 1}
        and payload.get("answer_correct") in {"PASS", "FAIL"}
        and not payload.get("answer_equiv_error")
    )


def _judge_file(path: Path, *, client, model: str, tasks_dir: Path,
                skip_completed: bool) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid rollout: {path}")
    task_id = str(payload.get("task_id") or path.stem)
    if skip_completed and _already_done(payload):
        return {"task_id": task_id, "status": "skipped",
                "score": payload.get("answer_equiv_score")}
    if (skip_completed
            and payload.get("answer_equiv_score") in {0, 1}
            and not payload.get("answer_equiv_error")):
        score = int(payload["answer_equiv_score"])
        payload["answer_correct"] = "PASS" if score == 1 else "FAIL"
        _atomic_write(path, payload)
        return {"task_id": task_id, "status": "skipped", "score": score}
    task = _load_task(tasks_dir, task_id)
    gold = _ground_truth(payload, task)
    predicted = _prediction(payload)
    if not gold:
        payload["answer_equiv_score"] = None
        payload["answer_equiv_reason"] = ""
        payload["answer_equiv_error"] = "missing_ground_truth"
        payload["answer_equiv_model"] = model
        payload["answer_correct"] = None
        _atomic_write(path, payload)
        return {"task_id": task_id, "status": "error", "error": "missing_ground_truth"}
    if not predicted:
        payload["answer_equiv_score"] = 0
        payload["answer_equiv_reason"] = "predicted answer is empty"
        payload["answer_equiv_error"] = None
        payload["answer_equiv_model"] = model
        payload["answer_correct"] = "FAIL"
        _atomic_write(path, payload)
        return {"task_id": task_id, "status": "judged", "score": 0}
    if _WS_RE.sub(" ", predicted).casefold() == _WS_RE.sub(" ", gold).casefold():
        score, reason = 1, "same text after whitespace and case normalization"
    else:
        score, reason = _judge_pair(client, model, gold, predicted)
    payload["answer_equiv_score"] = score
    payload["answer_equiv_reason"] = reason
    payload["answer_equiv_error"] = None
    payload["answer_equiv_model"] = model
    payload["answer_correct"] = "PASS" if score == 1 else "FAIL"
    _atomic_write(path, payload)
    return {"task_id": task_id, "status": "judged", "score": score, "reason": reason}


def _format_progress(done: int, total: int, counts: dict[str, int], started: float) -> str:
    elapsed = max(0.0, time.time() - started)
    rate = done / elapsed if elapsed > 0 and done else 0.0
    eta = (total - done) / rate if rate else 0.0
    pct = 100.0 * done / total if total else 0.0
    width = 28
    filled = 0 if total == 0 else int(width * done / total)
    if done > 0 and filled == 0:
        filled = 1
    filled = min(width, filled)
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{bar}] {done}/{total} {pct:5.1f}%  "
        f"same={counts['same']} diff={counts['different']} "
        f"skip={counts['skipped']} err={counts['error']}  "
        f"{elapsed:.0f}s eta {eta:.0f}s"
    )


def _show_progress(done: int, total: int, counts: dict[str, int], started: float) -> None:
    line = _format_progress(done, total, counts, started)
    columns = shutil.get_terminal_size(fallback=(100, 24)).columns
    if len(line) >= columns:
        line = line[: max(1, columns - 1)]
    sys.stderr.write("\r\033[2K" + line)
    sys.stderr.flush()


def _iter_rollouts(root: Path) -> list[Path]:
    return [path for path in sorted(root.glob("uwg_*.json"))
            if not path.name.endswith(".public.json")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT)
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS,
                        help="轨迹里没有 expected 时，从这里读 correct_answer")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-completed", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not args.rollout_dir.is_dir():
        parser.error(f"目录不存在: {args.rollout_dir}")
    try:
        api_key = read_api_key_file(args.api_key_file)
    except ValueError as exc:
        parser.error(str(exc))

    paths = _iter_rollouts(args.rollout_dir)
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        parser.error(f"未找到 uwg_*.json: {args.rollout_dir}")

    client = openai_client(base_url=args.base_url, api_key=api_key, timeout=600, max_retries=0)
    counts = {"judged": 0, "same": 0, "different": 0, "skipped": 0, "error": 0}
    started = time.time()
    total = len(paths)
    _show_progress(0, total, counts, started)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _judge_file, path, client=client, model=args.model,
                tasks_dir=args.tasks_dir, skip_completed=args.skip_completed,
            ): path
            for path in paths
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            path = futures[future]
            error_line = ""
            try:
                row = future.result()
            except Exception as exc:
                counts["error"] += 1
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        payload["answer_equiv_score"] = None
                        payload["answer_equiv_reason"] = ""
                        payload["answer_equiv_error"] = f"{type(exc).__name__}: {exc}"[:500]
                        payload["answer_equiv_model"] = args.model
                        payload["answer_correct"] = None
                        _atomic_write(path, payload)
                except (OSError, json.JSONDecodeError, ValueError):
                    pass
                error_line = f"[error] {path.stem} {type(exc).__name__}: {exc}"
            else:
                status = row["status"]
                if status == "skipped":
                    counts["skipped"] += 1
                elif status == "error":
                    counts["error"] += 1
                    error_line = f"[error] {row['task_id']} {row['error']}"
                else:
                    counts["judged"] += 1
                    counts["same" if row["score"] == 1 else "different"] += 1
            with _PRINT_LOCK:
                if error_line:
                    sys.stderr.write("\r\033[2K" + error_line + "\n")
                _show_progress(done, total, counts, started)
    sys.stderr.write("\n")
    sys.stderr.flush()

    same = counts["same"]
    different = counts["different"]
    judged = counts["judged"]
    summary = {
        "rollout_dir": str(args.rollout_dir),
        "model": args.model,
        "files": len(paths),
        **counts,
        "same_rate_this_run": (same / judged) if judged else None,
    }
    report_path = args.rollout_dir / "answer_equiv_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
