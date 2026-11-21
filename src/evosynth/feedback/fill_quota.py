"""配额补缺循环：造题 → 教师 rollout+check → 数各桶通过数 → 只给有缺口的桶补造，直到凑满。

目标：新题总数 ``--target-total`` 指「已发布 且 rollout 判 PASS」的题，且各合成对象
（基础原型 / 变体）的数量严格等于 ``archetype_mix × target_total`` 换算的整数配额。

流程（每一轮一个子目录 ``<work-dir>/rounds/round_XX``，全部状态落盘，可随时中断重跑）：

1. 读 feedback 产物里的 ``archetype_mix``，用最大余数法换成总和恰好等于
   ``target_total`` 的整数配额 ``Q_b``。
2. 扫描已有各轮：每个桶「已发布数」「已通过 rollout 数」。
3. 缺口 ``D_b = Q_b - 通过数``。全部 ≤ 0 → 收尾。
4. 每个有缺口的桶按 ``ceil(D_b × margin / yield_b)`` 申请发布数。``yield_b`` 是该桶
   「发布后通过 rollout」的比例：先验用母家族的 trace_check_pass_rate，随着轮次累积
   用实测值逐步替换（伪计数平滑）。
5. 用 ``--archetype-quota`` 做严格配额造题（只抽未满的桶，满桶多出的题不发布），
   再对本轮新题跑教师 rollout。回到 2。
6. 收尾：每个桶从通过题里随机裁到恰好 ``Q_b`` 道，软链进 ``final/tasks`` 与
   ``final/rollout``，并写 ``final/manifest.json``。

用法见 scripts/fill_quota.sh。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import supply

PRIOR_WEIGHT = 10        # yield 先验的伪计数
DEFAULT_YIELD = 0.7      # 没有任何先验时的发布后通过率
MIN_YIELD = 0.25         # 估计通过率的下限，避免申请量爆炸
SEED_STRIDE = 10_000_000  # 各轮 seed 区间间隔，保证 task id 不撞


def largest_remainder(mix: Mapping[str, float], total: int) -> dict[str, int]:
    """把占比换成总和恰好为 total 的整数配额（最大余数法，平局按名字稳定排序）。"""
    weights = {k: float(v) for k, v in mix.items() if v > 0}
    norm = sum(weights.values())
    if not weights or norm <= 0:
        raise ValueError("archetype_mix is empty")
    exact = {k: total * v / norm for k, v in weights.items()}
    quota = {k: int(math.floor(v)) for k, v in exact.items()}
    left = total - sum(quota.values())
    order = sorted(exact, key=lambda k: (exact[k] - quota[k], k), reverse=True)
    for name in order[:left]:
        quota[name] += 1
    return quota


def yield_priors(audit_report: Mapping[str, Any] | None, capabilities: Sequence[Mapping[str, Any]],
                 buckets: Sequence[str]) -> dict[str, float]:
    """每个桶的发布后通过率先验：变体取母家族的 trace_check_pass_rate。"""
    parent_of = {str(e["archetype"]): str(e.get("parent") or e["archetype"]) for e in capabilities}
    families = (audit_report or {}).get("families") or {}
    priors = {}
    for name in buckets:
        rate = (families.get(parent_of.get(name, name)) or {}).get("trace_check_pass_rate")
        priors[name] = min(1.0, max(MIN_YIELD, float(rate))) if rate else DEFAULT_YIELD
    return priors


_ROLLOUT_FINAL_STATUSES = {"PASS", "FAIL", "answer_judge_error", "exception"}


def _rollout_statuses(round_dir: Path) -> dict[str, str]:
    status: dict[str, str] = {}
    for path in (round_dir / "rollout").glob("uwg_*.json"):
        if path.name.endswith(".public.json"):
            continue
        try:
            st = str(json.loads(path.read_text(encoding="utf-8")).get("status") or "")
        except (OSError, ValueError):
            continue
        if st in _ROLLOUT_FINAL_STATUSES:
            status[path.stem] = st
    return status


def scan_rounds(work_dir: Path) -> dict[str, Any]:
    """扫描所有轮：每个桶的已发布 / 已评测 / 通过数，以及通过题 id → 所在轮。"""
    published: Counter = Counter()
    evaluated: Counter = Counter()
    qualified: Counter = Counter()
    passed: dict[str, list[tuple[str, str]]] = {}   # bucket -> [(round_name, task_id)]
    unrolled = 0
    for round_dir in sorted((work_dir / "rounds").glob("round_*")):
        index = supply.task_archetypes(round_dir)
        status = _rollout_statuses(round_dir)
        for task_id, archetype in index.items():
            published[archetype] += 1
            if task_id not in status:
                unrolled += 1
            else:
                evaluated[archetype] += 1
                if status[task_id] == "PASS":
                    qualified[archetype] += 1
                    passed.setdefault(archetype, []).append((round_dir.name, task_id))
    return {"published": published, "evaluated": evaluated, "qualified": qualified,
            "passed": passed, "unrolled": unrolled}


def plan_round(quota: Mapping[str, int], state: Mapping[str, Any], priors: Mapping[str, float],
               margin: float) -> dict[str, int]:
    """本轮每个桶要申请的发布数；没有缺口的桶不申请。"""
    request: dict[str, int] = {}
    evaluated = state.get("evaluated") or state["published"]
    for name, target in quota.items():
        deficit = target - state["qualified"].get(name, 0)
        if deficit <= 0:
            continue
        ev, qual = evaluated.get(name, 0), state["qualified"].get(name, 0)
        est = (qual + PRIOR_WEIGHT * priors[name]) / (ev + PRIOR_WEIGHT)
        est = max(MIN_YIELD, min(1.0, est))
        request[name] = max(deficit, math.ceil(deficit * margin / est))
    return request


def _run(cmd: Sequence[str], env: Mapping[str, str], log_path: Path) -> int:
    """子进程输出同时打到终端和日志。按块转发，保留 rollout 进度条的 ``\\r`` 刷新。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[fill] $", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        list(cmd), env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.stdout is not None

    def _forward() -> None:
        with open(log_path, "ab") as log:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()

    copier = threading.Thread(target=_forward)
    copier.start()
    code = proc.wait()
    copier.join()
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()
    return code


def finalize(work_dir: Path, quota: Mapping[str, int], state: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """每个桶随机裁到恰好 Q_b 道通过题，软链出 final/tasks 与 final/rollout。"""
    final = work_dir / "final"
    tasks_out, rollout_out = final / "tasks", final / "rollout"
    for folder in (tasks_out, rollout_out):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.iterdir():
            old.unlink()
    rng = random.Random(seed)
    selected: dict[str, list[str]] = {}
    shortfall: dict[str, int] = {}
    for name, target in quota.items():
        pool = sorted(state["passed"].get(name, []))
        chosen = pool if len(pool) <= target else sorted(rng.sample(pool, target))
        selected[name] = [task_id for _, task_id in chosen]
        shortfall[name] = max(0, target - len(chosen))
        for round_name, task_id in chosen:
            round_dir = work_dir / "rounds" / round_name
            for suffix in (".json", ".public.json", ".construction_trace.json"):
                src = round_dir / "tasks" / f"{task_id}{suffix}"
                if src.exists():
                    (tasks_out / src.name).symlink_to(src.resolve())
            src = round_dir / "rollout" / f"{task_id}.json"
            if src.exists():
                (rollout_out / src.name).symlink_to(src.resolve())
    total = sum(len(v) for v in selected.values())
    manifest = {
        "target_total": sum(quota.values()), "selected_total": total,
        "complete": not any(shortfall.values()),
        "buckets": {name: {
            "quota": quota[name], "selected": len(selected[name]),
            "share": round(len(selected[name]) / total, 4) if total else 0.0,
            "target_share": round(quota[name] / max(1, sum(quota.values())), 4),
            "published": state["published"].get(name, 0),
            "qualified": state["qualified"].get(name, 0),
            "surplus_dropped": max(0, state["qualified"].get(name, 0) - quota[name]),
            "shortfall": shortfall[name]} for name in quota},
        "task_ids": {name: ids for name, ids in selected.items()},
    }
    (final / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feedback-dir", type=Path, required=True,
                   help="feedback 某一轮的输出目录（含 capabilities.json / curriculum.json / audit_report.json）")
    p.add_argument("--target-total", type=int, required=True, help="最终要凑够的新题数（发布且 rollout PASS）")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--tool-info", type=Path, required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--world-base-url", required=True)
    p.add_argument("--world-model", required=True)
    p.add_argument("--world-api-key-file", type=Path, required=True)
    p.add_argument("--rollout-model", default=None, help="教师 rollout 模型；省略则同 --model")
    p.add_argument("--workers", type=int, default=90, help="造题并发")
    p.add_argument("--rollout-workers", type=int, default=100)
    p.add_argument("--companions", type=int, default=8)
    p.add_argument("--world-samples-per-initial-world", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=20261020)
    p.add_argument("--margin", type=float, default=1.15, help="申请量的安全系数（>1 少补几轮）")
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--select-seed", type=int, default=0, help="收尾裁剪的随机种子")
    p.add_argument("--dry-run", action="store_true", help="只打印配额与首轮申请量，不调用任何 LLM")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fb = args.feedback_dir
    curriculum = json.loads((fb / "curriculum.json").read_text(encoding="utf-8"))
    mix = curriculum.get("archetype_mix") or {}
    if not mix:
        print("[fill] curriculum.json has no archetype_mix", file=sys.stderr)
        return 2
    capabilities = json.loads((fb / "capabilities.json").read_text(encoding="utf-8"))
    names = {str(e["archetype"]) for e in capabilities}
    missing = sorted(set(mix) - names)
    if missing:
        print(f"[fill] archetype_mix names not in capabilities.json: {missing}", file=sys.stderr)
        return 2
    audit_path = fb / "audit_report.json"
    audit_report = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else None

    quota = largest_remainder(mix, args.target_total)
    priors = yield_priors(audit_report, capabilities, list(quota))
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    (work / "quota_target.json").write_text(json.dumps(quota, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[fill] target {args.target_total}: {quota}", flush=True)

    state = scan_rounds(work)
    if args.dry_run:
        print(f"[fill] yield priors: {priors}")
        print(f"[fill] first-round request: {plan_round(quota, state, priors, args.margin)}")
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CURRICULUM_JSON"] = str(fb / "curriculum.json")   # anchor_hints 等仍生效；配额覆盖 archetype 抽取
    env["EVOSYNTH_WORLD_SAMPLES_PER_INITIAL_WORLD"] = str(args.world_samples_per_initial_world)
    # 与 generate_tasks.sh 相同：造题阶段事件打到终端。
    # 子进程 stdout 是管道，必须关缓冲，否则进度要攒满一块才出现。
    env["EVOSYNTH_STDOUT_PROGRESS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    # 直接跑本模块时也和 generate_tasks.sh 共用封禁名单。
    env.setdefault(
        "EVOSYNTH_BANNED_SEEDS",
        str(Path(__file__).resolve().parents[3] / "outputs/banned_seeds.json"))
    env.setdefault("EVOSYNTH_PUBLISH_REBUILD_ATTEMPTS", "2")
    py = sys.executable
    log_dir = work / "logs"

    def rollout(round_dir: Path) -> int:
        cmd = [py, "-m", "evosynth.task_infer.rollout_tasks",
               "--tasks-dir", str(round_dir / "tasks"), "--output-dir", str(round_dir / "rollout"),
               "--workers", str(args.rollout_workers),
               "--base-url", args.base_url, "--model", args.rollout_model or args.model,
               "--api-key-file", str(args.api_key_file)]
        return _run(cmd, env, log_dir / f"{round_dir.name}.rollout.log")

    for round_index in range(args.max_rounds):
        # 先把已有各轮没 rollout 完的补完（上次中断时可能留下）
        for round_dir in sorted((work / "rounds").glob("round_*")):
            if scan_rounds_dir_unrolled(round_dir):
                rollout(round_dir)
        state = scan_rounds(work)
        request = plan_round(quota, state, priors, args.margin)
        report = {name: {"quota": quota[name], "published": state["published"].get(name, 0),
                         "qualified": state["qualified"].get(name, 0)} for name in quota}
        print(f"[fill] round {round_index} status: {json.dumps(report, ensure_ascii=False)}", flush=True)
        if not request:
            print("[fill] all buckets filled", flush=True)
            break
        round_dir = work / "rounds" / f"round_{len(list((work / 'rounds').glob('round_*'))):02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[fill] {round_dir.name} request: {request}", flush=True)
        cmd = [py, "-m", "evosynth.task_gen.generate_tasks",
               "--tool-info", str(args.tool_info), "--capabilities", str(fb / "capabilities.json"),
               "--companions", str(args.companions), "--output-dir", str(round_dir),
               "--archetype-quota", str(round_dir / "request.json"),
               "--seed-start", str(args.seed_start + SEED_STRIDE * int(round_dir.name.split("_")[1])),
               "--workers", str(args.workers),
               "--base-url", args.base_url, "--model", args.model, "--api-key-file", str(args.api_key_file),
               "--world-base-url", args.world_base_url, "--world-model", args.world_model,
               "--world-api-key-file", str(args.world_api_key_file),
               "--log-path", str(round_dir / "synthesis.jsonl")]
        code = _run(cmd, env, log_dir / f"{round_dir.name}.generate.log")
        print(f"[fill] generate exit={code} (非 0 表示本轮没凑满申请量，缺口留给下一轮)", flush=True)
        rollout(round_dir)
        after = scan_rounds(work)
        gained = sum(after["qualified"].values()) - sum(state["qualified"].values())
        if gained <= 0:
            print("[fill] this round produced no qualified task; stopping to avoid a dead loop", flush=True)
            break

    state = scan_rounds(work)
    manifest = finalize(work, quota, state, args.select_seed)
    print(json.dumps({k: manifest[k] for k in ("target_total", "selected_total", "complete")}
                     | {"buckets": {n: {kk: b[kk] for kk in ("quota", "selected", "share", "target_share", "shortfall")}
                                    for n, b in manifest["buckets"].items()}},
                     ensure_ascii=False, indent=2))
    return 0 if manifest["complete"] else 1


def scan_rounds_dir_unrolled(round_dir: Path) -> bool:
    """该轮是否有已发布但还没走完 rollout+judge（含 answer_correct）的任务。"""
    index = supply.task_archetypes(round_dir)
    if not index:
        return False
    rollout_dir = round_dir / "rollout"
    for task_id in index:
        path = rollout_dir / f"{task_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return True
        st = str(payload.get("status") or "")
        if st not in _ROLLOUT_FINAL_STATUSES:
            return True
        if st != "exception" and payload.get("answer_correct") not in {"PASS", "FAIL"}:
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
