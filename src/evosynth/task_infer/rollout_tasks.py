"""Stable public entry point for EvoSynth teacher rollout."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from evosynth.utils import read_api_key_file
from evosynth.task_gen.generation.runtime import (
    configure_request_concurrency,
)

from .rollout_engine import configure_runtime, run_infer, run_judge

# API_KEY_FILE = Path("")
BASE_URL = ""
MODEL = "glm-5.2"
API_KEY_FILE = Path("")
PROCESS_REWARD = Path("")
from . import answer_equiv

ANSWER_EQUIV = Path(answer_equiv.__file__).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("infer", "judge", "all"), default="all",
        help="infer 只求解；judge 先算过程奖励再评已有结果；all 为 rollout → 过程奖励 → judge")
    parser.add_argument("--tasks-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument(
        "--base-url", default=BASE_URL,
        help="OpenAI 兼容接口，默认 %(default)s")
    parser.add_argument(
        "--model", default=MODEL,
        help="求解模型名，默认 %(default)s")
    parser.add_argument(
        "--api-key-file", type=Path, default=API_KEY_FILE,
        help="API key 文件，默认 %(default)s")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-completed", action=argparse.BooleanOptionalAction, default=True,
        help="跳过已完成的 infer/judge，断点续跑；默认开启，--no-skip-completed 全量重跑")
    parser.add_argument(
        "--rejudge-errors", action="store_true",
        help="只重新 judge 状态为 answer_judge_error 的结果；需配合 --stage judge")
    return parser


def _score_process_rewards(output_dir: Path) -> dict[str, Any]:
    """模型 rollout 之后、LLM judge 之前，把过程奖励写回每条轨迹。"""
    if not PROCESS_REWARD.is_file():
        raise FileNotFoundError(f"过程奖励模块不存在: {PROCESS_REWARD}")
    spec = importlib.util.spec_from_file_location("score_process_reward", PROCESS_REWARD)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载过程奖励模块: {PROCESS_REWARD}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_rollout_dir(output_dir)


def _score_answer_equiv(*, output_dir: Path, tasks_dir: Path | None,
                        base_url: str, model: str, api_key_file: Path,
                        workers: int, skip_completed: bool) -> dict[str, Any]:
    """调用答案等价性裁判，写入 answer_correct（PASS/FAIL），不改 status。"""
    if not ANSWER_EQUIV.is_file():
        raise FileNotFoundError(f"答案裁判脚本不存在: {ANSWER_EQUIV}")
    import sys
    spec = importlib.util.spec_from_file_location("judge_answer_equiv", ANSWER_EQUIV)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载答案裁判脚本: {ANSWER_EQUIV}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    resolved_tasks = tasks_dir if tasks_dir is not None else (output_dir.parent / "tasks")
    argv = [
        str(ANSWER_EQUIV),
        "--rollout-dir", str(output_dir),
        "--tasks-dir", str(resolved_tasks),
        "--base-url", base_url,
        "--model", model,
        "--api-key-file", str(api_key_file),
        "--workers", str(workers),
        "--skip-completed" if skip_completed else "--no-skip-completed",
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old_argv
    report_path = output_dir / "answer_equiv_report.json"
    return json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    model = str(args.model).strip()
    base_url = str(args.base_url).strip()
    if not model:
        parser.error("--model must not be empty")
    if not base_url:
        parser.error("--base-url must not be empty")
    if args.rejudge_errors and args.stage != "judge":
        parser.error("--rejudge-errors 需要 --stage judge")
    if args.stage in {"infer", "all"}:
        if args.tasks_dir is None or not args.tasks_dir.is_dir():
            parser.error("--tasks-dir is required and must be a directory")
    if args.stage == "judge" and not args.output_dir.is_dir():
        parser.error(f"--output-dir is not a directory: {args.output_dir}")
    configure_request_concurrency(args.workers)
    try:
        api_key = read_api_key_file(args.api_key_file)
    except ValueError as exc:
        parser.error(str(exc))
    configure_runtime(base_url=base_url, model=model, api_key=api_key)
    reports: dict[str, Any] = {}
    if args.stage in {"infer", "all"}:
        reports["infer"] = run_infer(
            tasks_dir=args.tasks_dir,
            output_dir=args.output_dir,
            workers=args.workers,
            limit=args.limit,
            skip_completed=args.skip_completed,
        )
    if args.stage in {"judge", "all"}:
        if not args.output_dir.is_dir():
            parser.error(f"--output-dir is not a directory: {args.output_dir}")
        if PROCESS_REWARD.is_file():
            reports["process_reward"] = _score_process_rewards(args.output_dir)
        else:
            reports["process_reward"] = {"status": "skipped", "reason": "过程奖励模块未配置"}
        reports["judge"] = run_judge(
            output_dir=args.output_dir, workers=args.workers,
            skip_completed=args.skip_completed,
            only_statuses=("answer_judge_error",) if args.rejudge_errors else None)
        reports["answer_judge"] = _score_answer_equiv(
            output_dir=args.output_dir, tasks_dir=args.tasks_dir,
            base_url=base_url, model=model, api_key_file=args.api_key_file,
            workers=args.workers, skip_completed=args.skip_completed)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if args.stage == "infer":
        infer = reports.get("infer") or {}
        return 0 if infer.get("found", infer.get("tasks", 0)) > 0 else 1
    judge = reports.get("judge") or {}
    return 0 if judge.get("found", judge.get("tasks", 0)) > 0 else 1


__all__ = ["build_parser", "main", "run_infer", "run_judge"]


if __name__ == "__main__":
    raise SystemExit(main())
