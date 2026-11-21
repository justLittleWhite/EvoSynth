"""CLI for splitting benchmark tasks into capability archetypes.

用法:
    python -m capability_split.run_split \\
        --benchmark external-benchmark \\
        --tasks <task-path> \\
        --out out/capabilities.json \\
        --model glm-5.2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from capability_split.splitter import (load_tasks, save, split_benchmark,
                                           stage1_cache_path, validate)
else:
    from .splitter import load_tasks, save, split_benchmark, stage1_cache_path, validate


def main() -> None:
    ap = argparse.ArgumentParser(description="benchmark → capability archetypes")
    ap.add_argument("--benchmark", required=True, help="benchmark 名(用于 LLM 提示)")
    ap.add_argument("--tasks", required=True, action="append",
                    help="任务目录或 JSON/JSONL/CSV 路径;可重复传多路合并,"
                         "支持 '前缀=路径' 给 task_id 加前缀防重名")
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-file", type=Path, default=None)
    ap.add_argument("--field", default=None,
                    help="json/jsonl 里题面字段名(默认自动探测常见键)")
    ap.add_argument("--glob", default=None,
                    help="显式文件模式,每个匹配文件=一个任务(如 '*.md')")
    args = ap.parse_args()

    import os
    model = args.model or os.environ.get("CAPABILITY_SPLIT_MODEL") or os.environ.get("SYNTHESIS_MODEL")
    base_url = args.base_url or os.environ.get("CAPABILITY_SPLIT_BASE_URL") or os.environ.get("LLMROUTER_BASE_URL")
    key_file = args.api_key_file or (
        Path(os.environ["CAPABILITY_SPLIT_API_KEY_FILE"])
        if os.environ.get("CAPABILITY_SPLIT_API_KEY_FILE") else None)
    api_key = os.environ.get("CAPABILITY_SPLIT_API_KEY") or os.environ.get("LLMROUTER_API_KEY")
    if not model:
        ap.error("--model or CAPABILITY_SPLIT_MODEL is required")
    if not base_url:
        ap.error("--base-url or CAPABILITY_SPLIT_BASE_URL is required")
    if not api_key and not key_file:
        ap.error("--api-key-file, CAPABILITY_SPLIT_API_KEY_FILE, or CAPABILITY_SPLIT_API_KEY is required")

    from openai import OpenAI
    if key_file:
        try:
            api_key = key_file.read_text(encoding="utf-8").strip()
        except OSError:
            sys.exit(f"cannot read API key file: {key_file}")
        if not api_key:
            sys.exit(f"API key file is empty: {key_file}")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=600,
    )

    tasks = []
    for entry in args.tasks:
        prefix, _, path = entry.rpartition("=")
        part = load_tasks(path, field=args.field, glob=args.glob)
        if prefix:
            for t in part:
                t["task_id"] = f"{prefix}/{t['task_id']}"
        print(f"[capability_split] {prefix or path}: {len(part)} tasks")
        tasks.extend(part)
    if not tasks:
        sys.exit(f"no tasks found under {args.tasks}")
    print(f"[capability_split] {args.benchmark}: loaded {len(tasks)} tasks total")

    cache = stage1_cache_path(args.out)
    print(f"[capability_split] stage 1 cache: {cache}", flush=True)
    result = split_benchmark(client, model, args.benchmark, tasks, cache_path=cache)
    out = save(result, args.out)
    print(f"[capability_split] {len(result)} archetypes → {out}")
    for w in validate(result):
        print(f"  ! {w}")


if __name__ == "__main__":
    main()
