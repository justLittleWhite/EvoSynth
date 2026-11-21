"""合成训练集的供给侧画像（Synthetic Supply Profile）。纯脚本统计，不调用 LLM。

数据来源（均为 generate/rollout 脚本已有产物）：
- ``<gen_dir>/tasks/uwg_*.json``              已发布任务，metadata.archetype
- ``<gen_dir>/server_selection/*.json``       锚点/persona 兼容性裁决（Tool Planner 阶段）
- ``--log-path`` 完整合成 JSONL（可选）        task_start / task_published / task_rejected
- ``<rollout_dir>/uwg_*.json``                教师 rollout + judge（可选）

某个来源缺失时对应字段为 None，不猜测。
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

_SKIP_SUFFIX = (".public.json", ".construction_trace.json")


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(q * len(ordered)))])


def server_of(tool: str) -> str:
    return str(tool).split("__", 1)[0]


def task_index(gen_dir: Path) -> dict[str, dict[str, Any]]:
    """已发布任务 → archetype。"""
    index: dict[str, dict[str, Any]] = {}
    for path in sorted((Path(gen_dir) / "tasks").glob("uwg_*.json")):
        if path.name.endswith(_SKIP_SUFFIX):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        archetype = (payload.get("metadata") or {}).get("archetype")
        if archetype:
            index[path.stem] = {"archetype": str(archetype)}
    return index


def task_archetypes(gen_dir: Path) -> dict[str, str]:
    return {k: v["archetype"] for k, v in task_index(gen_dir).items()}


def progress_funnel(gen_dir: Path) -> dict[str, Counter]:
    """generation_progress.json 的 seeds 带 archetype 后，按 archetype 统计各状态（旧产物无此字段则为空）。"""
    path = Path(gen_dir) / "generation_progress.json"
    funnel: dict[str, Counter] = defaultdict(Counter)
    try:
        seeds = json.loads(path.read_text(encoding="utf-8")).get("seeds") or []
    except (OSError, json.JSONDecodeError):
        return funnel
    for row in seeds:
        if row.get("archetype"):
            funnel[str(row["archetype"])][str(row.get("status") or "unknown")] += 1
    return funnel


def planner_stats(gen_dir: Path) -> dict[str, dict[str, Any]]:
    """Tool Planner 阶段：每个 archetype 的 LLM 兼容裁决次数与锚点兼容率。"""
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"attempts": 0, "compatible": 0, "worlds": 0, "anchors": defaultdict(lambda: [0, 0])})
    for path in sorted((Path(gen_dir) / "server_selection").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        archetype = row.get("archetype")
        if not archetype:
            continue
        bucket = stats[str(archetype)]
        bucket["worlds"] += 1
        attempts = row.get("selection_attempts") or [
            {"anchor": row.get("anchor"), "compatible": bool(row.get("compatible"))}]
        for attempt in attempts:
            bucket["attempts"] += 1
            ok = bool(attempt.get("compatible"))
            bucket["compatible"] += ok
            anchor = str(attempt.get("anchor") or "")
            if anchor:
                bucket["anchors"][anchor][0] += 1
                bucket["anchors"][anchor][1] += ok
    return stats


def log_funnel(log_path: Path | None) -> dict[str, Counter]:
    """从完整合成日志统计每个 archetype 的 started / published / rejected。"""
    funnel: dict[str, Counter] = defaultdict(Counter)
    if not log_path or not Path(log_path).exists():
        return funnel
    archetype_of: dict[str, str] = {}
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            if '"task_start"' in line:
                kind = "task_start"
            elif '"task_published"' in line:
                kind = "task_published"
            elif '"task_rejected"' in line:
                kind = "task_rejected"
            else:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != kind:
                continue
            task_id = str(row.get("task_id") or "")
            if kind == "task_start":
                archetype = (row.get("spec") or {}).get("archetype")
                if archetype and task_id:
                    archetype_of[task_id] = str(archetype)
                    funnel[str(archetype)]["started"] += 1
            elif task_id in archetype_of:
                funnel[archetype_of[task_id]]["published" if kind == "task_published" else "rejected"] += 1
    return funnel


def rollout_stats(rollout_dir: Path | None, task_arch: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "pass": 0, "fail": 0, "calls": [], "distinct_tools": [], "cross_server": 0,
        "qualified": 0, "frozen_hits": 0, "cache_hits": 0, "total_calls": 0, "passed_ids": [],
        "distinct_calls": []})
    if not rollout_dir or not Path(rollout_dir).is_dir():
        return stats
    for path in sorted(Path(rollout_dir).glob("uwg_*.json")):
        if path.name.endswith(".public.json"):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        archetype = task_arch.get(str(row.get("task_id") or path.stem))
        status = row.get("status")
        if not archetype or status not in {"PASS", "FAIL"}:
            continue
        bucket = stats[archetype]
        bucket["pass" if status == "PASS" else "fail"] += 1
        calls = int(row.get("tool_calls") or 0)
        bucket["total_calls"] += calls
        bucket["frozen_hits"] += int(row.get("frozen_world_hits") or 0)
        bucket["cache_hits"] += int(row.get("episode_cache_hits") or 0)
        if status != "PASS":
            continue  # 复杂度指标只看合格轨迹
        tools = [str(c.get("tool")) for c in row.get("call_trace") or () if c.get("tool")]
        bucket["qualified"] += 1
        bucket["passed_ids"].append(str(row.get("task_id") or path.stem))
        bucket["calls"].append(calls)
        bucket["distinct_tools"].append(len(set(tools)))
        # 去重参数后的调用数：同一工具查多个不同实体算多次，区分单点查询与横向对比
        bucket["distinct_calls"].append(len({
            (str(c.get("tool")), json.dumps(c.get("arguments"), sort_keys=True, ensure_ascii=False,
                                            default=str)) for c in row.get("call_trace") or ()
            if c.get("tool")}))
        bucket["cross_server"] += len({server_of(t) for t in tools}) >= 2
    return stats


def rollout_task_rows(rollout_dir: Path | None) -> dict[str, dict[str, Any]]:
    """逐题的 rollout 摘要，供训练集清单按轨迹复杂度排序。目录不存在返回空 dict。"""
    rows: dict[str, dict[str, Any]] = {}
    if not rollout_dir or not Path(rollout_dir).is_dir():
        return rows
    for path in sorted(Path(rollout_dir).glob("uwg_*.json")):
        if path.name.endswith(".public.json"):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        trace = [c for c in row.get("call_trace") or () if c.get("tool")]
        rows[str(row.get("task_id") or path.stem)] = {
            "status": row.get("status"),
            "tool_calls": int(row.get("tool_calls") or 0),
            "cross_server": len({server_of(str(c["tool"])) for c in trace}) >= 2,
            "distinct_calls": len({(str(c["tool"]), json.dumps(
                c.get("arguments"), sort_keys=True, ensure_ascii=False, default=str)) for c in trace})}
    return rows


def load_train_ids(paths: Iterable[Path]) -> set[str]:
    """读训练集 jsonl（每行含 ``id``）。只取 id，逐行解析，不把整份数据留在内存。"""
    ids: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    task_id = json.loads(line).get("id")
                except json.JSONDecodeError:
                    continue
                if task_id:
                    ids.add(str(task_id))
    return ids


def build_supply_profile(gen_dir: Path, archetype_names: Iterable[str], *,
                         rollout_dir: Path | None = None,
                         log_path: Path | None = None,
                         extra_gen_dirs: Iterable[Path] = (),
                         train_ids: set[str] | None = None) -> dict[str, Any]:
    """``gen_dir`` 是主合成目录（完整题库、server_selection、generation_progress）。

    给了 ``train_ids`` 时，``final_training_count/share`` 只按这些 id 统计（实际参训的子集），
    ``extra_gen_dirs`` 只用来补全训练集里不在主目录的任务的 archetype；
    published_count、漏斗、trace_check_pass_rate 仍反映主目录完整题库。
    """
    names = list(dict.fromkeys(str(n) for n in archetype_names))
    task_arch = {k: v["archetype"] for k, v in task_index(gen_dir).items()}
    published = Counter(task_arch.values())
    train_counts: Counter | None = None
    train_unresolved: list[str] = []
    if train_ids is not None:
        for extra in extra_gen_dirs:
            for k, v in task_index(extra).items():
                if k in train_ids:
                    task_arch.setdefault(k, v["archetype"])
        train_counts = Counter(task_arch[i] for i in train_ids if i in task_arch)
        train_unresolved = sorted(i for i in train_ids if i not in task_arch)
    planner = planner_stats(gen_dir)
    funnel = log_funnel(log_path)
    progress = progress_funnel(gen_dir)
    rollout = rollout_stats(rollout_dir, task_arch)
    have_rollout = bool(rollout)

    rows: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys([*names, *published, *(train_counts or ()), *planner, *funnel,
                               *progress, *rollout]):
        p, f, r = planner.get(name), funnel.get(name, Counter()), rollout.get(name)
        g = progress.get(name, Counter())
        # 采样/质检漏斗优先用 generation_progress（不依赖日志），旧产物回退到日志
        if progress:
            started, audited = sum(g.values()), g.get("published", 0) + g.get("semantic_rejected", 0)
            audit_rate = _rate(g.get("published", 0), audited)
            source = "generation_progress"
        else:
            started = f.get("started") if funnel else None
            audit_rate = (_rate(f["published"], f["published"] + f["rejected"]) if funnel else None)
            source = "log" if started is not None else "published_tasks"
        judged = (r["pass"] + r["fail"]) if r else 0
        if train_counts is not None:
            final = train_counts.get(name, 0)  # 实际参训的子集
        else:
            final = ((r["qualified"] if r else 0) if have_rollout else published.get(name, 0))
        row: dict[str, Any] = {
            # 漏斗留存
            "sampled_count": started if started is not None else published.get(name, 0),
            "sampled_count_source": source,
            "status_counts": dict(g),
            "published_count": published.get(name, 0),
            "tool_compatible_rate": _rate(p["compatible"], p["attempts"]) if p else None,
            "task_audit_pass_rate": audit_rate,
            "trace_check_pass_rate": _rate(r["pass"], judged) if r else None,
            "final_training_count": final,
            # 轨迹复杂度（合格轨迹）
            "avg_tool_calls": round(statistics.fmean(r["calls"]), 3) if r and r["calls"] else None,
            "tool_calls_p50": _quantile(r["calls"], 0.5) if r else None,
            "tool_calls_p90": _quantile(r["calls"], 0.9) if r else None,
            "cross_server_rate": _rate(r["cross_server"], r["qualified"]) if r else None,
            "avg_distinct_tools": (round(statistics.fmean(r["distinct_tools"]), 3)
                                   if r and r["distinct_tools"] else None),
            "avg_distinct_calls": (round(statistics.fmean(r["distinct_calls"]), 3)
                                   if r and r["distinct_calls"] else None),
            # 模拟器：命中造题阶段冻结世界的调用比例 / 同一 episode 内缓存复用比例
            "cache_hit_rate": _rate(r["frozen_hits"], r["total_calls"]) if r else None,
            "episode_cache_hit_rate": _rate(r["cache_hits"], r["total_calls"]) if r else None,
            # 历史锚点兼容率，供「数量不足」时优先分配
            "anchor_compat": ({a: {"attempts": n, "compatible": c, "rate": _rate(c, n)}
                               for a, (n, c) in p["anchors"].items()} if p else {}),
        }
        rows[name] = row
    total_final = sum(r["final_training_count"] for r in rows.values())
    for row in rows.values():
        row["final_training_share"] = _rate(row["final_training_count"], total_final) or 0.0
    return {"archetypes": rows,
            "totals": {"published": sum(published.values()), "final_training": total_final,
                       "has_rollout": have_rollout, "has_log": bool(funnel), "has_progress_archetype": bool(progress),
                       "train_set_size": len(train_ids) if train_ids is not None else None,
                       "train_unresolved": len(train_unresolved),
                       "train_unresolved_examples": train_unresolved[:5]}}
