"""Step 2：供需交叉诊断（Supply-Demand Cross Audit）。

需求侧 = Step 1 的 Failure Cards（只针对 Base 池标注）；供给侧 = supply profile。
按基础 archetype 家族对表（变体的训练样本计入其 parent 家族的供给），自动分流：

* 能力盲区 uncovered：archetype_fit==missing 的占比 ≥ ``uncovered_ratio`` 且 ≥ ``min_cluster`` 条

家族处理规则（最终逻辑，无串联门槛）。需要处理的家族 = 需要干预，或满足变体条件。
每个处理家族拿到目标权重 M（∝ 失败占比，归一化到处理家族总和为 1）：

1. 调步数 K（``min_distinct_calls``，默认 5，下限 3）：只看该家族 Teacher Acc
   （trace_check_pass_rate），黄金区间 [0.60, 0.75]：> 0.75 → K+3；< 0.60 → K-1；区间内不变。
2. 分流：
   * exact 主导（未达变体条件）：不派生变体；K 按上条更新；基础原型拿 1.0×M。
   * 含显著 too_broad（too_broad ≥ 4，或需要干预且 too_broad ≥ 2）：用 too_broad 卡片
     派生变体 ``<base>__<tag>``。Teacher Acc 在区间内 → K 不动，变体拿 1.0×M；
     不在区间内 → K 按规则更新，基础原型（新 K）拿 0.3×M，变体拿 0.7×M。
   变体沿用旧 K（它承载的是 too_broad 的结构性指导，不承载步数调整）。
3. 单桶上限 ``max_bucket_share``（默认 0.35）：任一对象超过上限时压到上限，
   多出来的部分在其余变体间均分（基础原型不加）。均分后再超上限则继续压，直到没有变体可接收。

需要干预（需求高）：fail_count ≥ 2 且（failure_share ≥ 1/N_active 或 失败数排名 ≤ 3）。
N_active 是本轮至少有 1 条失败的家族数；排名兜底避免头部弱点虹吸掉二、三号弱点。

变体生命周期（绑定母家族，而不是看变体自己是否还被错题匹配到——
变体生效后错题自然消失，那恰恰说明它起作用了）：
* 母家族 failure_share 相对创建时的基准下降 > 30%，或低于 1/N_active → maintenance（权重 × ``maintenance_weight``=0.7，下限 1.0）
  基准 ``parent_failure_share`` 只在变体 active 且份额创新高时上调，进入 maintenance 后不被覆盖，
  否则下一轮份额不变就会对着新基准被误判为「没衰减」而重新激活
* 母家族本轮生成了新变体 → 旧变体被替换退役
* 母家族连续 ``retire_zero_rounds`` 轮零错题 → 退役
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from . import evolve, pools


@dataclass
class AuditConfig:
    uncovered_ratio: float = 0.15      # missing 卡占比阈值（分母：本次全部有效诊断）
    min_cluster: int = 2               # 一个簇至少需要的失败卡数；missing 总数同样至少这么多
    min_fail_count: int = 2            # 家族至少这么多错题才干预
    broad_min_count: int = 4           # too_broad ≥ 此数 → 派生变体
    broad_need_min_count: int = 2      # 或：需要干预 且 too_broad ≥ 此数 → 派生变体
    acc_low: float = 0.60              # Teacher Acc 黄金区间
    acc_high: float = 0.75
    k_default: int = 5                 # min_distinct_calls 初值
    k_floor: int = 3                   # 硬下限
    k_up_step: int = 3                 # Acc > acc_high（题太简单）→ K += 此值
    k_down_step: int = 1               # Acc < acc_low（题太难）→ K -= 此值
    base_share_when_split: float = 0.3  # Acc 不在区间时基础原型占 M 的比例，其余给变体
    new_dimension_share: float = 0.1   # 本轮新孵化维度合计占新题总量的比例
    max_bucket_share: float = 0.35     # 单个合成对象的新题占比上限；超出部分均分给其余变体
    variant_input: str = "all"         # too_broad 卡片投入 EVOLVE 的方式：all 一次性全投；clustered 先聚类再投
    max_variants_per_family: int = 3   # 一个家族最多派生几个变体（具体几个由模型按数据判断）
    cluster_max: int = 4               # variant_input=clustered 时的最大簇数
    top_k_demand: int = 3              # 失败数排名前 k 的家族即使份额不足 1/N_active 也干预
    low_supply_factor: float = 0.7     # 供给份额 < factor / N 视为「数量不足」
    high_pass_rate: float = 0.8        # trace check 通过率 ≥ 此值视为「题太容易」
    # 没有 rollout 时，用造题期/ rollout 复杂度作为「合成题偏简单」的辅助佐证
    simple_cache_hit: float = 0.85
    simple_cross_server: float = 0.3
    simple_tool_calls: float = 2.5
    decay_drop: float = 0.3            # 母家族失败份额下降超过此比例 → 维持态
    maintenance_weight: float = 0.7    # 维持态变体采样乘数的衰减系数（下限 1.0）
    retire_zero_rounds: int = 2
    max_new_archetypes: int = 2
    max_variants: int = 20
    anchor_min_attempts: int = 2
    multiplier_range: tuple[float, float] = (1.5, 3.0)


def _anchor_hints(row: Mapping[str, Any], min_attempts: int) -> dict[str, float]:
    hints = {}
    for anchor, info in (row.get("anchor_compat") or {}).items():
        if info["attempts"] >= min_attempts and info["rate"] is not None:
            hints[anchor] = round(0.5 + 1.5 * info["rate"], 3)
    return hints


def simplicity_evidence(row: Mapping[str, Any], cfg: AuditConfig) -> list[str]:
    """合成题偏简单的佐证，只看来自教师 rollout 的轨迹复杂度。"""
    evidence = []
    hit = row.get("cache_hit_rate")
    if hit is not None and hit >= cfg.simple_cache_hit:
        evidence.append(f"cache_hit_rate={hit}>={cfg.simple_cache_hit}")
    cross = row.get("cross_server_rate")
    if cross is not None and cross <= cfg.simple_cross_server:
        evidence.append(f"cross_server_rate={cross}<={cfg.simple_cross_server}")
    calls = row.get("avg_tool_calls")
    if calls is not None and calls <= cfg.simple_tool_calls:
        evidence.append(f"avg_tool_calls={calls}<={cfg.simple_tool_calls}")
    return evidence


def cap_bucket_share(mix: Mapping[str, float], variant_names: Sequence[str],
                     cap: float) -> dict[str, float]:
    """任一桶超过 cap 时压到 cap，超出部分在其余变体间均分。总和保持不变。"""
    mix = {str(k): float(v) for k, v in mix.items() if v > 0}
    if not mix or not 0 < cap < 1:
        return mix
    variants = {str(n) for n in variant_names}
    for _ in range(len(mix)):
        over = {n: w - cap for n, w in mix.items() if w > cap + 1e-12}
        if not over:
            break
        eligible = [n for n in mix if n in variants and n not in over and mix[n] < cap - 1e-12]
        if not eligible:
            break
        extra = sum(over.values()) / len(eligible)
        for name in over:
            mix[name] = cap
        for name in eligible:
            mix[name] += extra
    return mix


def in_golden_band(pass_rate: float | None, cfg: AuditConfig) -> bool:
    """Teacher Acc 是否在黄金区间；没有 rollout 数据时按「在区间内」处理（不动 K）。"""
    return pass_rate is None or cfg.acc_low <= pass_rate <= cfg.acc_high


def adjust_k(k: int, pass_rate: float | None, cfg: AuditConfig) -> int:
    """按 Teacher Acc 调 min_distinct_calls：太简单 +3，太难 -1，区间内不变；下限 k_floor。"""
    if pass_rate is None:
        return max(cfg.k_floor, k)
    if pass_rate > cfg.acc_high:
        return max(cfg.k_floor, k + cfg.k_up_step)
    if pass_rate < cfg.acc_low:
        return max(cfg.k_floor, k - cfg.k_down_step)
    return max(cfg.k_floor, k)


def run_audit(cards: Sequence[Mapping[str, Any]], supply: Mapping[str, Any],
              base: Sequence[Mapping[str, Any]], prev_variants: Sequence[Mapping[str, Any]],
              round_id: int, cfg: AuditConfig | None = None,
              prev_calls: Mapping[str, int] | None = None) -> dict[str, Any]:
    """``base`` = 原基础池 + 已晋升的新维度；``prev_variants`` 只含绑定母家族的变体；
    ``prev_calls`` = 上一轮各家族的 min_distinct_calls（缺省取 base 条目自带值，再缺省 k_default）。"""
    cfg = cfg or AuditConfig()
    prev_calls = dict(prev_calls or {})
    ok = [c for c in cards if c.get("status") == "ok"]
    total = len(ok)
    n_base = len(base)
    uniform = 1.0 / n_base
    fam = pools.family_map(base, prev_variants)
    rows = supply["archetypes"]
    total_final = max(1, supply["totals"]["final_training"])

    missing = [c for c in ok if c["archetype_fit"] == "missing"]
    by_family: dict[str, list[Mapping[str, Any]]] = {str(e["archetype"]): [] for e in base}
    for card in ok:
        if card["archetype_fit"] != "missing" and card["matched_archetype"] in by_family:
            by_family[card["matched_archetype"]].append(card)

    n_active = max(1, sum(1 for v in by_family.values() if v))
    ranked = sorted(by_family, key=lambda n: len(by_family[n]), reverse=True)
    rank_of = {name: i + 1 for i, name in enumerate(ranked)}

    taken = {str(e["archetype"]) for e in base} | {str(v["archetype"]) for v in prev_variants}
    new_variants: list[dict[str, Any]] = []     # 本轮新派生（挂母家族）
    new_dimensions: list[dict[str, Any]] = []   # 本轮新孵化（下一轮晋升基础池）
    min_calls: dict[str, int] = {}              # 各基础家族本轮生效的 K
    plans: dict[str, dict[str, Any]] = {}       # 处理家族 → {raw, base_frac, variant}
    anchor_hints: dict[str, dict[str, float]] = {}
    report_rows: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    # 预先并发派生所有满足 open_variant 条件的家族变体（各家族变体名以 <family>__ 为前缀，互不冲突）
    derive_jobs: dict[str, tuple[Mapping[str, Any], list[Mapping[str, Any]], list[dict[str, Any]]]] = {}
    for entry in base:
        name = str(entry["archetype"])
        family_cards = by_family[name]
        share = len(family_cards) / total if total else 0.0
        need = (len(family_cards) >= cfg.min_fail_count
                and (share >= 1.0 / n_active or rank_of[name] <= cfg.top_k_demand))
        too_broad = [c for c in family_cards if c["archetype_fit"] == "too_broad"]
        if len(too_broad) >= cfg.broad_min_count or (need and len(too_broad) >= cfg.broad_need_min_count):
            existing = [v for v in prev_variants if v.get("parent") == name]
            derive_jobs[name] = (entry, too_broad, existing)

    derive_results: dict[str, tuple[list[dict[str, Any]] | None, Exception | None]] = {}
    if derive_jobs:
        def _run_derive(item: tuple[str, tuple[Mapping[str, Any], list[Mapping[str, Any]], list[dict[str, Any]]]]):
            fam_name, (ent, tb, ex) = item
            local_taken = set(taken)
            try:
                res = evolve.derive_variants(
                    ent, tb, ex, local_taken, mode=cfg.variant_input,
                    max_variants=cfg.max_variants_per_family, cluster_max=cfg.cluster_max)
                return fam_name, res, None
            except Exception as exc:
                return fam_name, None, exc

        with ThreadPoolExecutor(max_workers=len(derive_jobs)) as ex_pool:
            for fam_name, res, exc in ex_pool.map(_run_derive, derive_jobs.items()):
                derive_results[fam_name] = (res, exc)
                if res:
                    for d in res:
                        taken.add(str(d["targeted"]["archetype"]))

    for entry in base:
        name = str(entry["archetype"])
        family_cards = by_family[name]
        share = len(family_cards) / total if total else 0.0
        row = rows.get(name, {})
        supply_share = sum(rows.get(n, {}).get("final_training_share", 0.0)
                           for n, f in fam.items() if f == name)
        pass_rate = row.get("trace_check_pass_rate")
        evidence = simplicity_evidence(row, cfg)
        too_broad = [c for c in family_cards if c["archetype_fit"] == "too_broad"]
        info: dict[str, Any] = {
            "failures": len(family_cards), "failure_share": round(share, 4),
            "rank": rank_of[name], "supply_share": round(supply_share, 4),
            "trace_check_pass_rate": pass_rate, "simplicity_evidence": evidence,
            "too_broad": len(too_broad),
            "stages": dict(Counter(c["failure_stage"] for c in family_cards)),
            "decision": "no_action", "reason": ""}
        report_rows[name] = info
        need = (len(family_cards) >= cfg.min_fail_count
                and (share >= 1.0 / n_active or rank_of[name] <= cfg.top_k_demand))
        broad = len(too_broad)
        k_cur = max(cfg.k_floor, int(prev_calls.get(name) or entry.get("min_distinct_calls")
                                     or cfg.k_default))
        k_new = adjust_k(k_cur, pass_rate, cfg)
        band_ok = in_golden_band(pass_rate, cfg)
        min_calls[name] = k_cur
        info.update(k_prev=k_cur, k_new=k_cur, acc_in_band=band_ok, base_weight=0.0,
                    variant_weight=0.0)
        open_variant = (broad >= cfg.broad_min_count
                        or (need and broad >= cfg.broad_need_min_count))
        if not need and not open_variant:
            info["reason"] = (f"{len(family_cards)} failures (share {share:.2f}, rank "
                              f"{rank_of[name]}) below intervention rule; "
                              f"too_broad {broad} below variant rule")
            continue

        created: list[dict[str, Any]] = []   # 本家族本轮新派生的变体（数量由模型按数据判断）
        reasons: list[str] = []
        if open_variant:
            derived_list, exc = derive_results[name]
            if exc is not None or derived_list is None:
                reasons.append(f"derive_failed: {type(exc).__name__}: {str(exc)[:200]}")
                errors.append(f"{name}: {reasons[-1]}")
            else:
                for derived in derived_list:
                    variant = pools.variant_entry(
                        derived["targeted"], parent=name, kind="difficulty_gap", round_id=round_id,
                        world_init_hint=derived["world_init_hint"], multiplier=derived["multiplier"],
                        min_distinct_calls=k_cur)  # 变体沿用旧 K
                    variant["parent_failure_share"] = round(share, 4)
                    created.append(variant)
                new_variants.extend(created)

        if not created and not need:
            info["decision"] = "derive_failed"
            info["reason"] = "; ".join(reasons)
            continue
        names_text = ", ".join(v["archetype"] for v in created)
        if created and band_ok:
            # Teacher Acc 在区间：K 不动，变体拿走全部 M（在多个变体间均分）
            base_frac, decision = 0.0, "variant_only"
            reasons.append(f"too_broad {broad}/{len(family_cards)} -> {len(created)} variant(s) {names_text}; "
                           f"acc {pass_rate} in [{cfg.acc_low},{cfg.acc_high}] -> K={k_cur}, "
                           "100% variant")
        elif created:
            # 区间外：更新 K；基础原型（新 K）0.3，变体合计 0.7
            min_calls[name] = k_new
            base_frac, decision = cfg.base_share_when_split, "variant_split"
            reasons.append(f"too_broad {broad}/{len(family_cards)} -> {len(created)} variant(s) {names_text}; "
                           f"acc {pass_rate} outside band -> K {k_cur}->{k_new}, "
                           f"{base_frac:.0%} base / {1 - base_frac:.0%} variant")
        else:
            # exact 主导（或派生失败回退）：不出变体，更新 K，基础原型拿全部 M
            min_calls[name] = k_new
            base_frac, decision = 1.0, "base_only"
            reasons.append(f"exact-dominated (too_broad {broad}/{len(family_cards)}); "
                           + (f"acc {pass_rate} outside band -> K {k_cur}->{k_new}; "
                              if not band_ok else f"acc {pass_rate} in band -> K={k_cur}; ")
                           + "100% base")
        if base_frac > 0:
            anchor_hints[name] = _anchor_hints(row, cfg.anchor_min_attempts)
        plans[name] = {"raw": share, "base_frac": base_frac,
                       "variants": [v["archetype"] for v in created]}
        info.update(decision=decision, k_new=min_calls[name],
                    variants=[v["archetype"] for v in created],
                    channels=[decision])
        info["reason"] = "; ".join(reasons)

    # ---- 变体生命周期：按母家族失败份额衰减，而不是看变体自己是否被错题匹配 ----
    replaced_parents = {v["parent"] for v in new_variants}
    kept: list[dict[str, Any]] = []
    retired: list[dict[str, Any]] = []
    for old in prev_variants:
        variant = dict(old)
        parent = str(variant.get("parent"))
        info = report_rows.get(parent)
        if info is None or parent in replaced_parents:
            retired.append({"archetype": variant["archetype"],
                            "reason": "replaced by new variant" if info else "parent not in base"})
            continue
        if info["failures"] == 0:
            variant["zero_rounds"] = int(variant.get("zero_rounds", 0)) + 1
            if variant["zero_rounds"] >= cfg.retire_zero_rounds:
                retired.append({"archetype": variant["archetype"],
                                "reason": f"parent had zero failures for {variant['zero_rounds']} rounds"})
                continue
            variant["state"] = "maintenance"
        else:
            variant["zero_rounds"] = 0
            baseline = float(variant.get("parent_failure_share") or 0.0)
            current = info["failure_share"]
            decayed = baseline > 0 and current <= baseline * (1 - cfg.decay_drop)
            variant["state"] = ("maintenance" if decayed or current < 1.0 / n_active
                                else "active")
            # 基准锚定在创建时；只有 active 且份额创新高才上调，maintenance 不覆盖
            if variant["state"] == "active" and current > baseline:
                variant["parent_failure_share"] = current
        kept.append(variant)

    uncovered_report: dict[str, Any] = {
        "missing_cards": len(missing), "ratio": round(len(missing) / total, 4) if total else 0.0,
        "triggered": False, "created": [], "dropped_clusters": []}
    if total and len(missing) / total >= cfg.uncovered_ratio and len(missing) >= cfg.min_cluster:
        uncovered_report["triggered"] = True
        pool_view = [*base, *kept, *new_variants]
        groups = sorted(evolve.cluster_missing(missing, max_clusters=cfg.max_new_archetypes + 1),
                        key=lambda g: len(g[1]), reverse=True)
        for label, members in groups:
            if len(uncovered_report["created"]) >= cfg.max_new_archetypes \
                    or len(members) < cfg.min_cluster:
                uncovered_report["dropped_clusters"].append({"label": label, "size": len(members)})
                continue
            try:
                made = evolve.incubate_archetype(label, members, pool_view, taken)
            except Exception as exc:
                errors.append(f"uncovered/{label}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            taken.add(made["targeted"]["archetype"])
            created = pools.variant_entry(
                made["targeted"], parent=None, kind="new_dimension", round_id=round_id,
                world_init_hint=made["world_init_hint"], multiplier=made["multiplier"])
            new_dimensions.append(created)
            pool_view.append(created)
            uncovered_report["created"].append(
                {"archetype": created["archetype"], "cluster": label, "size": len(members)})

    variants = (kept + new_variants)
    if cfg.max_variants and len(variants) > cfg.max_variants:
        variants.sort(key=lambda v: (v.get("state") == "active", v.get("created_round", 0)), reverse=True)
        for dropped in variants[cfg.max_variants:]:
            retired.append({"archetype": dropped["archetype"], "reason": "variant cap"})
        variants = variants[:cfg.max_variants]
        surviving = {str(v["archetype"]) for v in variants}
        for name, plan in plans.items():
            plan["variants"] = [v for v in plan["variants"] if v in surviving]
            report_rows[name]["variants"] = list(plan["variants"])

    # ---- 逐 archetype 目标占比：M_f = 失败占比（归一化到处理家族和为 1）× 基础/变体切分 ----
    # 母家族本轮没派生新变体、但有仍在役的旧变体：旧变体顶替变体份额，切分规则同上。
    kept_by_parent: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        if variant.get("parent") and variant not in new_variants:
            kept_by_parent.setdefault(str(variant["parent"]), []).append(variant)
    for name, plan in plans.items():
        carried = kept_by_parent.get(name)
        if not plan["variants"] and carried:
            band_ok = report_rows[name]["acc_in_band"]
            plan["base_frac"] = cfg.base_share_when_split if not band_ok else 0.0
            if not band_ok:
                min_calls[name] = report_rows[name]["k_new"]
            else:
                min_calls[name] = report_rows[name]["k_prev"]
            plan["carried"] = [str(v["archetype"]) for v in carried]
            report_rows[name].update(
                decision="variant_split" if not band_ok else "variant_only",
                k_new=min_calls[name], base_weight=0.0, variant_weight=0.0)
    raw_total = sum(p["raw"] for p in plans.values())
    mix: dict[str, float] = {}
    for name, plan in plans.items():
        weight = plan["raw"] / raw_total if raw_total else 0.0
        base_w = weight * plan["base_frac"]
        variant_w = weight - base_w
        if base_w > 0:
            mix[name] = mix.get(name, 0.0) + base_w
        targets = plan["variants"] or plan.get("carried", [])
        if targets:
            for target in targets:
                mix[target] = mix.get(target, 0.0) + variant_w / len(targets)
        else:
            mix[name] = mix.get(name, 0.0) + variant_w   # 无变体可用：回落基础原型
        report_rows[name].update(M=round(weight, 4), base_weight=round(base_w, 4),
                                 variant_weight=round(variant_w if targets else 0.0, 4))
    if new_dimensions:
        # 新孵化维度合计占 new_dimension_share，其余按比例缩放
        scale = 1.0 - cfg.new_dimension_share if mix else 0.0
        mix = {k: v * scale for k, v in mix.items()}
        for dim in new_dimensions:
            mix[str(dim["archetype"])] = cfg.new_dimension_share / len(new_dimensions)
    # 维持态变体份额 × maintenance_weight，随后整体重新归一化
    for variant in variants:
        key = str(variant["archetype"])
        if variant.get("state") == "maintenance" and key in mix:
            mix[key] *= cfg.maintenance_weight
    mix_total = sum(mix.values())
    mix = {k: v / mix_total for k, v in mix.items()} if mix_total else {}
    variant_names = [str(v["archetype"]) for v in [*variants, *new_dimensions]]
    mix = cap_bucket_share(mix, variant_names, cfg.max_bucket_share)
    mix = {k: round(v, 4) for k, v in mix.items()}
    drift = round(1.0 - sum(mix.values()), 4) if mix else 0.0
    if drift:
        # 四舍五入残差补给占比最高、且还没顶到上限的变体，避免把已封顶的桶再抬上去
        room = [n for n in variant_names if n in mix and mix[n] < cfg.max_bucket_share]
        if room:
            pick = max(room, key=lambda n: mix[n])
            mix[pick] = round(mix[pick] + drift, 4)
    for name, plan in plans.items():
        targets = plan["variants"] or plan.get("carried", [])
        report_rows[name]["base_weight"] = round(mix.get(name, 0.0), 4)
        report_rows[name]["variant_weight"] = round(sum(mix.get(t, 0.0) for t in targets), 4)
    return {
        "round": round_id, "thresholds": asdict(cfg),
        "totals": {"ok_cards": total, "error_cards": len(cards) - total,
                   "missing": len(missing), "n_active_families": n_active},
        "families": report_rows, "uncovered": uncovered_report,
        "variants": variants, "new_dimensions": new_dimensions,
        "retired_variants": retired, "archetype_mix": mix,
        "min_distinct_calls": min_calls,
        "anchor_hints": anchor_hints, "errors": errors}
