"""训练集清单 training_manifest.json：本轮哪些任务参训（总量固定，默认 10k）。

磁盘上的旧题文件全程只读，不删不改；本模块只读目录，写一个 manifest。
应在「下一轮合成跑完（并 rollout+judge）之后」运行：

    python -m evosynth.feedback.manifest \
        --feedback-dir <本轮 feedback 输出目录> \
        --history <旧 gen 目录>[:<旧 rollout 目录>] ... \
        --new-gen-dir <新 gen 目录> --new-rollout-dir <新 rollout 目录>

规则：

1. 旧题：已退役变体、不在当前 capabilities 里的题型 → 0%。其余题型一视同仁，
   按轨迹复杂度（跨 server → 调用步数 → 去重调用数）排序，去掉最难 15% 和最简单 15%
   （最难的常是 teacher 走弯路所致），保留中间约 70%。没有 rollout 的题无法排序，不剔除。
2. 旧题配额 = total × old_share（默认 10k × 65% = 6.5k）。保留的旧题超出配额时，
   各题型按比例再收缩（仍取复杂度居中的那一段）。
3. 新题配额 = total − 旧题保留数。新题不够配额就全留（总量可能不足 total，不强行补）；
   超出配额时才按题型比例裁剪。新题的 Base/Variant 构成由合成阶段决定，这里不再配比。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from . import supply

SCHEMA_VERSION = "feedback.training-manifest"


@dataclass
class ManifestConfig:
    total: int = 10000          # 训练集目标总量
    old_share: float = 0.65     # 旧题配额占 total 的比例（≈ 6~7k）
    trim_hard: float = 0.15     # 每个题型剔除最难的比例
    trim_easy: float = 0.15     # 每个题型剔除最简单的比例
    seed: int = 0               # 稳定哈希抽取的种子


def _order(task_id: str, seed: int) -> str:
    return hashlib.sha1(f"{seed}:{task_id}".encode()).hexdigest()


def collect_tasks(gen_dir: Path, rollout_dir: Path | None, source: str) -> list[dict[str, Any]]:
    """一个合成目录里的合格任务。有 rollout → 只取 PASS；否则取全部已发布。"""
    gen_dir = Path(gen_dir)
    index = supply.task_index(gen_dir)
    rows = supply.rollout_task_rows(rollout_dir) if rollout_dir else None
    tasks = []
    for task_id, info in index.items():
        row = rows.get(task_id) if rows is not None else None
        if rows is not None and not (row and row["status"] == "PASS"):
            continue
        tasks.append({
            "task_id": task_id, "archetype": info["archetype"], "source": source,
            "task_path": str(gen_dir / "tasks" / f"{task_id}.json"),
            "rollout_path": str(Path(rollout_dir) / f"{task_id}.json") if row else None,
            "qualified_basis": "rollout_pass" if rows is not None else "published",
            # 复杂度排序键（高 = 更难）；没有 rollout 的题为 None
            "complexity": ([int(row["cross_server"]), row["tool_calls"], row["distinct_calls"]]
                           if row else None)})
    return tasks


def trim_extremes(tasks: list[dict[str, Any]], cfg: ManifestConfig) -> list[dict[str, Any]]:
    """去掉复杂度最高 trim_hard、最低 trim_easy；没有复杂度的题无法判断，原样保留。"""
    known = [t for t in tasks if t["complexity"] is not None]
    unknown = [t for t in tasks if t["complexity"] is None]
    ordered = sorted(known, key=lambda t: (t["complexity"], _order(t["task_id"], cfg.seed)))
    low, high = int(len(ordered) * cfg.trim_easy), int(len(ordered) * cfg.trim_hard)
    return ordered[low:len(ordered) - high] + unknown


def take_middle(tasks: list[dict[str, Any]], k: int, seed: int) -> list[dict[str, Any]]:
    """取 k 条：复杂度都已知时取居中的一段（两端对称收缩），否则按哈希取前 k 条。"""
    if k >= len(tasks):
        return list(tasks)
    if all(t["complexity"] is not None for t in tasks):
        ordered = sorted(tasks, key=lambda t: (t["complexity"], _order(t["task_id"], seed)))
        start = (len(ordered) - k) // 2
        return ordered[start:start + k]
    return sorted(tasks, key=lambda t: _order(t["task_id"], seed))[:k]


def allocate(sizes: dict[str, int], budget: int) -> dict[str, int]:
    """把 budget 按各组规模等比例分配（最大余数法），每组不超过自身规模。"""
    total = sum(sizes.values())
    if total <= budget:
        return dict(sizes)
    raw = {k: v * budget / total for k, v in sizes.items()}
    out = {k: int(v) for k, v in raw.items()}
    left = budget - sum(out.values())
    for key in sorted(raw, key=lambda k: (raw[k] - out[k], k), reverse=True):
        if left <= 0:
            break
        if out[key] < sizes[key]:
            out[key] += 1
            left -= 1
    return out


def build_manifest(feedback_dir: Path, history: Sequence[tuple[Path, Path | None]],
                   new: tuple[Path, Path | None] | None, cfg: ManifestConfig | None = None
                   ) -> dict[str, Any]:
    cfg = cfg or ManifestConfig()
    feedback_dir = Path(feedback_dir)
    audit = json.loads((feedback_dir / "audit_report.json").read_text(encoding="utf-8"))
    capabilities = json.loads((feedback_dir / "capabilities.json").read_text(encoding="utf-8"))
    pool_of = {str(e["archetype"]): str(e.get("pool") or "base") for e in capabilities}
    retired = {str(r["archetype"]) for r in audit.get("retired_variants", [])}

    seen: set[str] = set()
    duplicates = 0

    def dedupe(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal duplicates
        out = []
        for task in tasks:
            if task["task_id"] in seen:
                duplicates += 1
                continue
            seen.add(task["task_id"])
            out.append(task)
        return out

    old: list[dict[str, Any]] = []
    for gen_dir, rollout_dir in history:
        old += dedupe(collect_tasks(gen_dir, rollout_dir, "history"))
    new_tasks = dedupe(collect_tasks(*new, "new")) if new else []

    excluded: Counter = Counter()
    by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in old:
        if task["archetype"] in retired or task["archetype"] not in pool_of:
            excluded["retired_or_not_in_pool"] += 1
        else:
            by_arch[task["archetype"]].append(task)

    # 1) 每个题型去掉最难/最简单两端
    trimmed = {name: trim_extremes(tasks, cfg) for name, tasks in by_arch.items()}
    excluded["extremes_trimmed"] = sum(len(by_arch[n]) - len(trimmed[n]) for n in trimmed)
    # 2) 旧题配额：超出则各题型等比例收缩
    old_cap = int(cfg.total * cfg.old_share)
    quota = allocate({n: len(t) for n, t in trimmed.items()}, old_cap)
    kept_old: list[dict[str, Any]] = []
    for name, tasks in trimmed.items():
        picked = take_middle(tasks, quota[name], cfg.seed)
        excluded["old_over_quota"] += len(tasks) - len(picked)
        kept_old += [{**t, "reason": "old_middle_complexity"} for t in picked]
    # 3) 新题配额 = total − 旧题保留数；超出才按题型比例裁剪
    new_in_pool = [t for t in new_tasks if t["archetype"] in pool_of]
    excluded["new_not_in_pool"] = len(new_tasks) - len(new_in_pool)
    new_by_arch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in new_in_pool:
        new_by_arch[task["archetype"]].append(task)
    new_quota = max(cfg.total - len(kept_old), 0)
    new_alloc = allocate({n: len(t) for n, t in new_by_arch.items()}, new_quota)
    kept_new: list[dict[str, Any]] = []
    for name, tasks in new_by_arch.items():
        ordered = sorted(tasks, key=lambda t: _order(t["task_id"], cfg.seed))
        kept_new += [{**t, "reason": "new"} for t in ordered[:new_alloc[name]]]
    excluded["new_over_quota"] = len(new_in_pool) - len(kept_new)

    chosen = [*kept_old, *kept_new]
    for task in chosen:
        task["pool"] = pool_of.get(task["archetype"], "base")
        task.pop("complexity", None)
    base_total = sum(t["pool"] == "base" for t in chosen)
    return {
        "schema_version": SCHEMA_VERSION, "round": audit.get("round"),
        "feedback_dir": str(feedback_dir), "config": asdict(cfg),
        "summary": {
            "total": len(chosen), "target_total": cfg.total,
            "old_kept": len(kept_old), "old_quota": old_cap,
            "new_kept": len(kept_new), "new_available": len(new_in_pool), "new_quota": new_quota,
            "base": base_total, "variant": len(chosen) - base_total,
            "base_share": round(base_total / len(chosen), 4) if chosen else None,
            "below_target": len(chosen) < cfg.total,
            "excluded": dict(excluded), "duplicate_task_ids": duplicates,
            "per_archetype": dict(Counter(t["archetype"] for t in chosen))},
        "tasks": [{k: t[k] for k in ("task_id", "archetype", "pool", "source", "reason",
                                     "task_path", "rollout_path", "qualified_basis")}
                  for t in chosen]}


def _parse_history(values: Sequence[str]) -> list[tuple[Path, Path | None]]:
    pairs = []
    for value in values:
        gen, _, rollout = value.partition(":")
        pairs.append((Path(gen), Path(rollout) if rollout else None))
    return pairs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feedback-dir", type=Path, required=True)
    p.add_argument("--history", nargs="*", default=[], metavar="GEN[:ROLLOUT]",
                   help="历史合成目录（只读），可带冒号接 rollout 目录；无 rollout 则取全部已发布")
    p.add_argument("--new-gen-dir", type=Path, default=None)
    p.add_argument("--new-rollout-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None, help="默认 <feedback-dir>/training_manifest.json")
    d = ManifestConfig()
    p.add_argument("--total", type=int, default=d.total)
    p.add_argument("--old-share", type=float, default=d.old_share)
    p.add_argument("--trim-hard", type=float, default=d.trim_hard)
    p.add_argument("--trim-easy", type=float, default=d.trim_easy)
    p.add_argument("--seed", type=int, default=d.seed)
    args = p.parse_args(argv)
    if args.total <= 0:
        p.error("--total must be > 0")
    for name in ("old_share", "trim_hard", "trim_easy"):
        if not 0.0 <= getattr(args, name) < 1.0:
            p.error(f"--{name.replace('_', '-')} must be in [0, 1)")
    if args.new_gen_dir is None and not args.history:
        p.error("need --history and/or --new-gen-dir")
    manifest = build_manifest(
        args.feedback_dir, _parse_history(args.history),
        (args.new_gen_dir, args.new_rollout_dir) if args.new_gen_dir else None,
        ManifestConfig(args.total, args.old_share, args.trim_hard, args.trim_easy, args.seed))
    out = args.out or args.feedback_dir / "training_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {out}")
    return 0 if manifest["tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
