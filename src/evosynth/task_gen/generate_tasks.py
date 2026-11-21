"""Stable public entry point for EvoSynth task generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evosynth.utils import read_api_key_file

from .generation_api import generate_domain
from .generation.runtime import configure_request_concurrency, configure_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        default=None,
        help="可选的任务标签，只用于 task id 与日志；省略时为 mcp_pool",
    )
    parser.add_argument("--tool-info", type=Path, required=True)
    parser.add_argument(
        "--capabilities",
        type=Path,
        required=True,
        help="archetype JSON 数组，字段为 archetype / archetype_desc / verfiy",
    )
    parser.add_argument(
        "--servers", nargs="*", default=None,
        help="固定使用的 MCP server，给出后跳过候选抽样与 LLM 选择。"
             "省略时每个 initial world 先按覆盖缺口选一个锚点 server，再配若干可选同伴，"
             "由 LLM 裁决锚点与 persona 是否兼容（不兼容则重采 persona）并选同伴",
    )
    parser.add_argument(
        "--companions", type=int, default=3,
        help="锚点 server 之外提供给 LLM 的可选同伴 server 数（默认 3，可跨 domain）",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--world-base-url", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--world-api-key-file", type=Path, required=True)
    parser.add_argument(
        "--count", type=int, default=40,
        help="要成功发布的任务数。未设置 --max-attempts 时会继续采样后续 seed，直到发布数达到该值",
    )
    parser.add_argument("--seed-start", type=int, default=20260806)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="同时运行的 seed 数，也是 agent 与世界模型各自的在途请求上限",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=None,
        help="最多尝试的 seed 数。省略时上限为 count 的 20 倍",
    )
    parser.add_argument(
        "--archetype-quota", type=Path, default=None,
        help="JSON 对象 {archetype: 发布数}：严格配额。给出后 --count 自动取配额总和，"
             "采样只在未满的桶里抽，满桶多出的成功任务不发布",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="可选的完整合成 JSONL 日志路径",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.tool_info.is_file():
        parser.error(f"--tool-info does not exist: {args.tool_info}")
    if not args.capabilities.is_file():
        parser.error(f"--capabilities does not exist: {args.capabilities}")
    if args.companions < 0:
        parser.error("--companions must be non-negative")
    archetype_quota = None
    if args.archetype_quota is not None:
        try:
            archetype_quota = {str(k): int(v) for k, v in json.loads(
                args.archetype_quota.read_text(encoding="utf-8")).items()}
        except (OSError, ValueError, AttributeError) as exc:
            parser.error(f"--archetype-quota unreadable: {exc}")
        args.count = sum(v for v in archetype_quota.values() if v > 0)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    configure_request_concurrency(args.workers)
    if args.count < 0:
        parser.error("--count must be non-negative")
    if args.max_attempts is not None and args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    try:
        agent_key = read_api_key_file(args.api_key_file)
        world_key = read_api_key_file(args.world_api_key_file)
    except ValueError as exc:
        parser.error(str(exc))
    configure_runtime(
        agent_base_url=args.base_url,
        agent_model=args.model,
        agent_api_key=agent_key,
        world_base_url=args.world_base_url,
        world_model=args.world_model,
        world_api_key=world_key,
    )
    report = generate_domain(
        domain=args.domain,
        tool_info=args.tool_info,
        capabilities=args.capabilities,
        output_dir=args.output_dir,
        count=args.count,
        seed_start=args.seed_start,
        servers=args.servers,
        companions=args.companions,
        workers=args.workers,
        max_attempts=args.max_attempts,
        log_path=args.log_path,
        archetype_quota=archetype_quota,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("published_complete") else 1


__all__ = ["build_parser", "generate_domain", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
