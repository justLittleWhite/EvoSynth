"""Step 1：单题无参考诊断（Trace-Level Post-Mortem）。

对每道 acc==0 的题生成一张 Failure Card。eval_reason 可选：有则锚定回溯，无则盲诊。
结果逐条追加到 JSONL，可断点续跑。
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import llm, prompts
from .records import BenchmarkCase

FITS = {"exact", "too_broad", "missing"}
STAGES = {"tool_routing", "argument_grounding", "incomplete_exploration",
          "state_recovery", "computation_and_synthesis"}
BLIND_SIGNALS = {"constraint_omission", "tool_anomaly", "premature_stop", "synthesis_error"}


def normalize_card(raw: Mapping[str, Any], pool_names: set[str]) -> dict[str, Any]:
    """校验并规整诊断输出；不合法返回 ValueError，由调用方记为 error 卡。"""
    fit = str(raw.get("archetype_fit") or "").strip()
    stage = str(raw.get("failure_stage") or "").strip()
    gap = str(raw.get("structural_gap") or "").strip()
    rule = str(raw.get("missing_verify_rule") or "").strip()
    if fit not in FITS or stage not in STAGES or not gap or not rule:
        raise ValueError(f"invalid diagnosis fields: fit={fit!r} stage={stage!r}")
    matched = raw.get("matched_archetype")
    matched = str(matched).strip() if matched not in (None, "", "null", "none") else None
    if matched is not None and matched not in pool_names:
        # 模型编了池里没有的名字：视为池里没有可匹配的原型
        matched, fit = None, "missing"
    if matched is None:
        fit = "missing"
    signal = raw.get("blind_signal")
    signal = signal if signal in BLIND_SIGNALS else None
    try:
        turn = int(raw.get("first_error_turn"))
    except (TypeError, ValueError):
        turn = None
    return {"matched_archetype": matched, "archetype_fit": fit, "failure_stage": stage,
            "blind_signal": signal, "first_error_turn": turn,
            "structural_gap": gap, "missing_verify_rule": rule}


def diagnose_case(case: BenchmarkCase, pool: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prompt = prompts.render(
        prompts.DIAGNOSE,
        benchmark_question=case.question,
        benchmark_tools=json.dumps(case.tools, ensure_ascii=False),
        model_trajectory=case.trajectory,
        eval_reason=case.eval_reason or "N/A",
        archetype_pool_json=json.dumps(prompts.pool_for_prompt(list(pool)),
                                       ensure_ascii=False, indent=1))
    card = {"case_id": case.case_id, "benchmark": case.benchmark,
            "mode": "anchored" if case.eval_reason else "blind", "status": "ok"}
    names = {str(e["archetype"]) for e in pool}
    last = ""
    for _ in range(2):  # 字段不合法时带原因重试一次
        try:
            raw = llm.call_json(prompt + (f"\n\nPREVIOUS ANSWER REJECTED: {last}" if last else ""),
                                max_tokens=12000, temperature=0.6)
            card.update(normalize_card(raw, names))
            return card
        except Exception as exc:  # 单题失败不拖垮整批，记 error 卡
            last = f"{type(exc).__name__}: {str(exc)[:300]}"
            if "no parsable JSON" in str(exc):
                break  # call_json 内部已带提示、加大预算重试过，外层不再整轮重发
    card.update(status="error", error=last)
    return card


def load_cards(path: Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def run_diagnosis(cases: Sequence[BenchmarkCase], pool: Sequence[Mapping[str, Any]],
                  out_path: Path, *, workers: int = 8, resume: bool = True) -> list[dict[str, Any]]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {c["case_id"]: c for c in load_cards(out_path)
            if resume and c.get("status") == "ok"}
    if not resume and out_path.exists():
        out_path.unlink()
    elif done:  # 重写文件，只保留成功卡，error 卡会被重跑
        out_path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n"
                                    for c in done.values()), encoding="utf-8")
    pending = [c for c in cases if c.case_id not in done]
    lock = threading.Lock()
    cards = list(done.values())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool_exec, \
            open(out_path, "a", encoding="utf-8") as handle:
        futures = [pool_exec.submit(diagnose_case, case, pool) for case in pending]
        for index, future in enumerate(as_completed(futures), 1):
            card = future.result()
            with lock:
                cards.append(card)
                handle.write(json.dumps(card, ensure_ascii=False) + "\n")
                handle.flush()
            print(f"[diagnose] {index}/{len(pending)} {card['case_id']} "
                  f"{card.get('archetype_fit') or card.get('error')}", flush=True)
    wanted = {c.case_id for c in cases}
    return [c for c in cards if c["case_id"] in wanted]
