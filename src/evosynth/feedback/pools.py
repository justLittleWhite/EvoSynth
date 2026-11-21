"""Step 3：基础池 + 靶向变体池。基础池条目原样保留，变体只追加，不覆盖基础三元组。

产物（round 目录下）：
- ``capabilities.json``：base + variant 合并数组，可直接给 ``--capabilities``。
  每项除 archetype / archetype_desc / verify 外，带 pool / parent / created_round /
  initial_world（world_init_hint）。``load_archetypes`` 对额外字段兼容。
- ``curriculum.json``：``CURRICULUM_JSON`` 用，含 pool_mix / sampling_multipliers / anchor_hints。
- ``variants.json``：本轮变体池，下一轮 ``--prev-round-dir`` 读取。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

CURRICULUM_SCHEMA = "feedback.curriculum"
DEFAULT_MIN_CALLS = 5
DEFAULT_BASE_SHARE = 0.3   # 仅指「新合成题」中基础题（under_sampled 题型）的占比，不是训练集配比


def load_entries(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"archetype file must be a non-empty JSON array: {path}")
    entries = []
    for item in raw:
        entry = dict(item)
        if "verfiy" in entry and "verify" not in entry:  # 历史拼写
            entry["verify"] = entry.pop("verfiy")
        entries.append(entry)
    return entries


def split_pools(entries: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    base = [dict(e) for e in entries if (e.get("pool") or "base") == "base"]
    variants = [dict(e) for e in entries if (e.get("pool") or "base") == "variant"]
    return base, variants


def load_prev_state(round_dir: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """读上一轮 variants.json → (晋升为基础池的新维度, 仍挂在母家族下的变体)。

    new_dimension 是诊断空间的扩展：下一轮起视为基础池成员（会传给 Step 1）。
    parent 变体只是供给侧训练处方，永远不进 Step 1 的候选池。
    """
    if round_dir is None:
        return [], []
    path = Path(round_dir) / "variants.json"
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    promoted = [{**r, "pool": "base"} for r in rows if r.get("kind") == "new_dimension"]
    return promoted, [r for r in rows if r.get("kind") != "new_dimension"]


def family_map(base: Sequence[Mapping], variants: Sequence[Mapping]) -> dict[str, str]:
    """archetype 名 → 所属基础原型。新维度（无 parent）自成一族。"""
    mapping = {str(e["archetype"]): str(e["archetype"]) for e in base}
    for variant in variants:
        mapping[str(variant["archetype"])] = str(variant.get("parent") or variant["archetype"])
    return mapping


def load_prev_calls(round_dir: Path | None) -> dict[str, int]:
    """读上一轮 curriculum.json 的 min_distinct_calls（逐 archetype 的步数下限）。"""
    if round_dir is None:
        return {}
    path = Path(round_dir) / "curriculum.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8")).get("min_distinct_calls") or {}
    return {str(k): int(v) for k, v in raw.items() if isinstance(v, (int, float))}


def variant_entry(targeted: Mapping[str, Any], *, parent: str | None, kind: str,
                  round_id: int, world_init_hint: str, multiplier: float,
                  min_distinct_calls: int = DEFAULT_MIN_CALLS) -> dict[str, Any]:
    return {"archetype": str(targeted["archetype"]),
            "min_distinct_calls": int(min_distinct_calls),
            "archetype_desc": str(targeted["archetype_desc"]),
            "verify": str(targeted["verify"]),
            "pool": "variant", "parent": parent, "kind": kind,
            "created_round": round_id, "initial_world": world_init_hint,
            "sampling_multiplier": multiplier, "state": "active", "zero_rounds": 0}


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_round(out_dir: Path, *, round_id: int, base: Sequence[Mapping],
                variants: Sequence[Mapping], new_dimensions: Sequence[Mapping],
                anchor_hints: Mapping[str, Mapping[str, float]],
                archetype_mix: Mapping[str, float] | None = None,
                min_distinct_calls: Mapping[str, int] | None = None,
                multipliers: Mapping[str, float] | None = None,
                base_focus: Sequence[str] = (), base_share: float = DEFAULT_BASE_SHARE) -> dict[str, Path]:
    """``base`` 含原基础池 + 已晋升的新维度（pool=base）；``new_dimensions`` 是本轮新孵化的，
    本轮先放变体池做靶向采样，下一轮起晋升。

    ``archetype_mix``：逐 archetype 目标占比（和为 1），采样器存在时直接按它抽；
    ``min_distinct_calls``：各基础家族生效的 K，写进 capabilities 条目（造题读取）
    并存进 curriculum（下一轮 ``--prev-round-dir`` 继承）。
    """
    multipliers = dict(multipliers or {})
    archetype_mix = {str(k): float(v) for k, v in (archetype_mix or {}).items() if v > 0}
    calls = {str(k): int(v) for k, v in (min_distinct_calls or {}).items()}
    out_dir = Path(out_dir)
    base_entries = [{**dict(e), "pool": "base",
                     "min_distinct_calls": calls.get(
                         str(e["archetype"]),
                         int(e.get("min_distinct_calls") or DEFAULT_MIN_CALLS))}
                    for e in base]
    targeted = [dict(v) for v in [*variants, *new_dimensions]]
    names = [str(e["archetype"]) for e in [*base_entries, *targeted]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate archetype names across base and variant pools")
    # 新题的池子分配：基础题只给 base_focus（供给不足）的题型，其余全部合成变体
    base_names = {str(e["archetype"]) for e in base_entries}
    if archetype_mix:
        # pool_mix / base_focus 只作旧读取方的兼容摘要，真正的配比在 archetype_mix
        unknown = set(archetype_mix) - set(names)
        if unknown:
            raise ValueError(f"archetype_mix has unknown archetypes: {sorted(unknown)}")
        base_focus = [n for n in archetype_mix if n in base_names]
        variant_share = round(sum(w for n, w in archetype_mix.items() if n not in base_names), 4)
    focus = [n for n in base_focus if n in base_names]
    if archetype_mix:
        pass
    elif not targeted:
        variant_share = 0.0
    elif not focus:
        variant_share = 1.0
    else:
        variant_share = round(1.0 - base_share, 4)
    curriculum = {
        "schema_version": CURRICULUM_SCHEMA, "round": round_id,
        "pool_mix": {"base": round(1.0 - variant_share, 4), "variant": variant_share},
        "base_focus": focus,
        "archetype_mix": {k: round(v, 4) for k, v in archetype_mix.items()},
        "min_distinct_calls": {**calls, **{str(v["archetype"]): int(
            v.get("min_distinct_calls") or DEFAULT_MIN_CALLS) for v in targeted}},
        "sampling_multipliers": {k: round(float(v), 3) for k, v in multipliers.items()},
        "anchor_hints": {k: dict(v) for k, v in anchor_hints.items()},
        # 血缘：变体 → 母家族 / 创建轮次 / 状态，便于衰减、退役与以后回滚
        "lineage": {str(v["archetype"]): {
            "parent": v.get("parent"), "created_round": v.get("created_round"),
            "kind": v.get("kind"), "state": v.get("state", "active")} for v in targeted},
    }
    promoted = [dict(e) for e in base if e.get("kind") == "new_dimension"]
    paths = {"capabilities": out_dir / "capabilities.json",
             "curriculum": out_dir / "curriculum.json",
             "variants": out_dir / "variants.json"}
    _atomic_write(paths["capabilities"], base_entries + targeted)
    _atomic_write(paths["curriculum"], curriculum)
    # 持久化全部非原始条目：parent 变体 + 本轮新维度 + 已晋升的新维度
    _atomic_write(paths["variants"], [dict(v) for v in [*variants, *new_dimensions]] + promoted)
    return paths
