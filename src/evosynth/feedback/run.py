"""反馈闭环入口：诊断 → 供需审计 → 下一轮 capabilities / curriculum。

    python -m evosynth.feedback.run --records <records.jsonl|run_dir> ... \
        --capabilities <base_pool.json> --gen-dir <generate 输出目录> \
        [--rollout-dir <rollout 输出目录>] [--log-path <合成日志.jsonl>] \
        --out-dir <本轮反馈目录> --round 1 [--prev-round-dir <上一轮反馈目录>]

``--capabilities`` 始终传最初的基础池文件，它不会被修改；变体只写进 out-dir。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import diagnose, llm, pools, records, supply
from .audit import AuditConfig, run_audit


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("diagnose", "audit", "all"), default="all",
                   help="diagnose 只跑 Step 1；audit 用已有 failure_cards 跑 Step 2/3；all 全跑")
    p.add_argument("--records", type=Path, nargs="*", default=[],
                   help="评测生成的 records.jsonl 或 run 目录（含 normalized/records.jsonl）")
    p.add_argument("--blind", action="store_true", help="忽略 eval_reason，强制盲诊")
    p.add_argument("--tools-file", type=Path, default=None,
                   help='可选 JSON：{"<benchmark>": [工具名或 OpenAI tool schema]}')
    p.add_argument("--capabilities", type=Path, required=True, help="基础池 archetype JSON（不会被修改）")
    p.add_argument("--prev-round-dir", type=Path, default=None, help="上一轮反馈目录（读取其 variants.json）")
    p.add_argument("--gen-dir", type=Path, nargs="+", default=None,
                   help="本轮合成输出目录（含 tasks/ server_selection/）。可多个：第一个是主目录"
                        "（完整题库/漏斗），其余只用来补全训练集里不在主目录的任务")
    p.add_argument("--train-set", type=Path, nargs="*", default=[],
                   help="实际参训的数据 jsonl（每行含 id）。给了则 final_training_count/share "
                        "只按这些 id 统计；不给则按整个 gen-dir 题库算")
    p.add_argument("--rollout-dir", type=Path, default=None, help="对合成任务的 rollout+judge 输出目录")
    p.add_argument("--log-path", type=Path, default=None, help="generate_tasks --log-path 的完整合成日志")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--round", type=int, default=1, dest="round_id")
    p.add_argument("--base-share", type=float, default=pools.DEFAULT_BASE_SHARE,
                   help="新合成题中基础题（仅 under_sampled 题型）的占比，其余给变体池")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-error-ratio", type=float, default=0.1,
                   help="诊断卡失败比例超过该值就中止，不往下做审计（默认 10%%）")
    p.add_argument("--base-url", default=llm.DEFAULT_BASE_URL)
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--api-key-file", type=Path, default=llm.DEFAULT_API_KEY_FILE)
    d = AuditConfig()
    p.add_argument("--uncovered-ratio", type=float, default=d.uncovered_ratio)
    p.add_argument("--min-cluster", type=int, default=d.min_cluster)
    p.add_argument("--min-fail-count", type=int, default=d.min_fail_count)
    p.add_argument("--top-k-demand", type=int, default=d.top_k_demand)
    p.add_argument("--low-supply-factor", type=float, default=d.low_supply_factor)
    p.add_argument("--high-pass-rate", type=float, default=d.high_pass_rate)
    p.add_argument("--max-new-archetypes", type=int, default=d.max_new_archetypes)
    p.add_argument("--max-variants", type=int, default=d.max_variants)
    p.add_argument("--variant-input", choices=("all", "clustered"), default=d.variant_input,
                   help="too_broad 卡片投入变体派生的方式：all 一次性全投；clustered 先聚类再投")
    p.add_argument("--max-variants-per-family", type=int, default=d.max_variants_per_family,
                   help="一个家族最多派生几个变体（具体几个由模型按数据判断）")
    p.add_argument("--cluster-max", type=int, default=d.cluster_max,
                   help="variant-input=clustered 时的最大簇数")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.base_share <= 1.0:
        parser.error("--base-share must be in (0, 1]")
    if args.stage in {"diagnose", "all"} and not args.records:
        parser.error("--records is required for diagnose")
    if args.stage in {"audit", "all"} and args.gen_dir is None:
        parser.error("--gen-dir is required for audit")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cards_path = out_dir / "failure_cards.jsonl"

    original, variants_in_file = pools.split_pools(pools.load_entries(args.capabilities))
    if variants_in_file:
        parser.error("--capabilities must be the original base pool (found pool=variant entries)")
    promoted, prev_variants = pools.load_prev_state(args.prev_round_dir)
    base = [*original, *promoted]  # 扩展后的 Base：Step 1 只看它，不看 <base>__<tag> 变体
    pool = base
    llm.configure_from_file(base_url=args.base_url, model=args.model, api_key_file=args.api_key_file)

    if args.stage in {"diagnose", "all"}:
        tools = (json.loads(args.tools_file.read_text(encoding="utf-8"))
                 if args.tools_file else None)
        cases, case_stats = records.load_failed_cases(
            args.records, blind=args.blind, tools_by_benchmark=tools)
        print(f"[feedback] benchmark records: {case_stats}", flush=True)
        cards = diagnose.run_diagnosis(cases, pool, cards_path, workers=args.workers,
                                       resume=not args.no_resume)
        (out_dir / "case_stats.json").write_text(
            json.dumps(case_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        cards = diagnose.load_cards(cards_path)
    errored = sum(1 for c in cards if c.get("status") != "ok")
    if cards and errored / len(cards) > args.max_error_ratio:
        print(f"[feedback] ABORT: {errored}/{len(cards)} diagnosis cards failed "
              f"(> {args.max_error_ratio:.0%}); audit on partial cards would be biased. "
              "Re-run to retry failed cards (resume keeps the good ones).", flush=True)
        return 3
    if args.stage == "diagnose":
        return 0 if cards else 1

    train_ids = supply.load_train_ids(args.train_set) if args.train_set else None
    profile = supply.build_supply_profile(
        args.gen_dir[0], [e["archetype"] for e in [*pool, *prev_variants]],
        rollout_dir=args.rollout_dir, log_path=args.log_path,
        extra_gen_dirs=args.gen_dir[1:], train_ids=train_ids)
    if train_ids is not None:
        totals = profile["totals"]
        print(f"[feedback] train set: {totals['train_set_size']} ids, "
              f"unresolved {totals['train_unresolved']}", flush=True)
    (out_dir / "supply_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    cfg = AuditConfig(
        uncovered_ratio=args.uncovered_ratio, min_cluster=args.min_cluster,
        min_fail_count=args.min_fail_count, top_k_demand=args.top_k_demand,
        low_supply_factor=args.low_supply_factor,
        high_pass_rate=args.high_pass_rate, max_new_archetypes=args.max_new_archetypes,
        max_variants=args.max_variants, variant_input=args.variant_input,
        max_variants_per_family=args.max_variants_per_family, cluster_max=args.cluster_max)
    audit = run_audit(cards, profile, base, prev_variants, args.round_id, cfg,
                      prev_calls=pools.load_prev_calls(args.prev_round_dir))
    paths = pools.write_round(
        out_dir, round_id=args.round_id, base=base, variants=audit["variants"],
        new_dimensions=audit["new_dimensions"],
        anchor_hints=audit["anchor_hints"],
        archetype_mix=audit["archetype_mix"],
        min_distinct_calls=audit["min_distinct_calls"])
    (out_dir / "audit_report.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    decisions = {n: r["decision"] for n, r in audit["families"].items()}
    print(json.dumps({"decisions": decisions, "uncovered": audit["uncovered"],
                      "variants": [v["archetype"] for v in audit["variants"]],
                      "new_dimensions": [v["archetype"] for v in audit["new_dimensions"]],
                      "retired": audit["retired_variants"],
                      "outputs": {k: str(v) for k, v in paths.items()}},
                     ensure_ascii=False, indent=2))
    return 0 if not audit["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
