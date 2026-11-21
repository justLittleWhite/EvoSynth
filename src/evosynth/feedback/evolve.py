"""弱点聚类与三元组演化：派生靶向 Sub-Archetype，或孵化全新 Archetype。"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from . import llm, prompts

MAX_GAP_LINES = 30
MULTIPLIER_RANGE = (1.0, 3.0)
_TAG_RE = re.compile(r"[^a-z0-9]+")


def _tag(text: str) -> str:
    return _TAG_RE.sub("_", str(text).lower()).strip("_")[:40] or "variant"


def _unique(name: str, taken: set[str]) -> str:
    candidate, index = name, 2
    while candidate in taken:
        candidate, index = f"{name}_{index}", index + 1
    return candidate


def _clamp_multiplier(value: Any, default: float = 2.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(MULTIPLIER_RANGE[0], min(MULTIPLIER_RANGE[1], number))


def gap_lines(cards: Sequence[Mapping[str, Any]]) -> str:
    seen: dict[str, None] = {}
    for card in cards:
        seen[f"- structural_gap: {card['structural_gap']}\n  missing_verify_rule: {card['missing_verify_rule']}"] = None
    return "\n".join(list(seen)[:MAX_GAP_LINES])


MAX_DESC_CHARS = 350
MAX_VERIFY_CHARS = 450
MAX_VERIFY_CLAUSES = 4
DEFAULT_MAX_VARIANTS = 3
MAX_CLUSTER_LINES = 10
_NUMBERED_RE = re.compile(r"(^|\s)\(?\d{1,2}[.)]\s")
_MEDIUM_RE = re.compile(
    r"\b(polyline|waypoint|waypoints|coordinate|coordinates|latitude|longitude|"
    r"webhook|webhooks|repo|repos|repository|repositories|branch|branches|commit|commits)\b",
    re.IGNORECASE,
)


def _field_problems(fields: Mapping[str, str]) -> list[str]:
    """变体 / 新维度必须和基础原型同风格：短、不带编号清单、不是解题规程、不带硬介质词。"""
    problems = []
    if len(fields["archetype_desc"]) > MAX_DESC_CHARS:
        problems.append(f"archetype_desc is {len(fields['archetype_desc'])} chars; limit {MAX_DESC_CHARS}")
    if len(fields["verify"]) > MAX_VERIFY_CHARS:
        problems.append(f"verify is {len(fields['verify'])} chars; limit {MAX_VERIFY_CHARS}")
    clauses = [c for c in re.split(r"[;；]", fields["verify"]) if c.strip()]
    if len(clauses) > MAX_VERIFY_CLAUSES:
        problems.append(f"verify has {len(clauses)} clauses; limit {MAX_VERIFY_CLAUSES}")
    if _NUMBERED_RE.search(fields["verify"]) or _NUMBERED_RE.search(fields["archetype_desc"]):
        problems.append("numbered rule lists are not allowed")
    blob = " ".join((fields["archetype"], fields["archetype_desc"], fields["verify"]))
    hit = _MEDIUM_RE.search(blob)
    if hit:
        problems.append(f"medium-specific term '{hit.group(0)}' is not allowed; use relational terms")
    return problems


def _clean_fields(raw: Mapping[str, Any], label: str = "") -> dict[str, str]:
    fields = {k: str(raw.get(k) or "").strip()
              for k in ("archetype", "archetype_desc", "verify")}
    if not all(fields.values()):
        raise ValueError(f"{label}needs archetype, archetype_desc and verify")
    problems = _field_problems(fields)
    if problems:
        raise ValueError(f"{label}" + "; ".join(problems))
    return fields


def _validate_targeted(output: Mapping[str, Any]) -> dict[str, str]:
    targeted = output.get("targeted_archetype")
    if not isinstance(targeted, Mapping):
        raise ValueError("missing targeted_archetype")
    return _clean_fields(targeted)


def cluster_text(groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]]) -> str:
    """把聚类结果排成 EVOLVE 的输入：每簇一段，簇内去重并限行。"""
    blocks = []
    for index, (label, members) in enumerate(groups, 1):
        seen: dict[str, None] = {}
        for card in members:
            seen[f"- structural_gap: {card['structural_gap']}\n  missing_verify_rule: {card['missing_verify_rule']}"] = None
        lines = "\n".join(list(seen)[:MAX_CLUSTER_LINES])
        blocks.append(f"CLUSTER {index} (label: {label}; {len(members)} observations):\n{lines}")
    return "\n\n".join(blocks)


def build_evolve_prompt(base: Mapping[str, Any], cards: Sequence[Mapping[str, Any]],
                        existing: Sequence[Mapping[str, Any]], *, mode: str = "all",
                        max_variants: int = DEFAULT_MAX_VARIANTS, cluster_max: int = 4,
                        meta: dict[str, Any] | None = None) -> str:
    """一次只放一个 base + 它自己的旧变体 + 它自己的 too_broad 卡片。

    mode="all"：卡片原样去重后一次性投入。
    mode="clustered"：先用 CLUSTER_GAPS 聚类（严格模式，聚类失败直接报错而不是悄悄退化成 all），
    再把聚类结果按簇排好一次性投入。
    """
    if mode == "all":
        observed = gap_lines(cards)
        groups: list[tuple[str, list[Mapping[str, Any]]]] = []
    elif mode == "clustered":
        groups = cluster_missing(cards, max_clusters=cluster_max, strict=True)
        observed = cluster_text(groups)
    else:
        raise ValueError(f"unknown variant input mode: {mode!r}")
    if meta is not None:
        meta["groups"] = [{"label": label, "size": len(members)} for label, members in groups]
        meta["observed_chars"] = len(observed)
    return prompts.render(
        prompts.EVOLVE,
        base_archetype_json=json.dumps(prompts.pool_for_prompt([base])[0], ensure_ascii=False, indent=1),
        existing_variants_json=json.dumps(
            [{"archetype": v["archetype"], "archetype_desc": v["archetype_desc"]} for v in existing],
            ensure_ascii=False) if existing else "[]",
        aggregated_structural_gaps_and_verify_rules=observed,
        max_variants=max_variants)


def _variant_items(output: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items = output.get("variants")
    if items is None and isinstance(output.get("targeted_archetype"), Mapping):   # 兼容旧输出
        items = [{**output["targeted_archetype"], "world_init_hint": output.get("world_init_hint")}]
    if not isinstance(items, list) or not items or not all(isinstance(i, Mapping) for i in items):
        raise ValueError("output needs a non-empty 'variants' list")
    return items


def derive_variants(base: Mapping[str, Any], cards: Sequence[Mapping[str, Any]],
                    existing: Sequence[Mapping[str, Any]], taken: set[str], *,
                    mode: str = "all", max_variants: int = DEFAULT_MAX_VARIANTS,
                    cluster_max: int = 4, retries: int = 3,
                    meta: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """由 too_broad 失败卡派生 1..max_variants 个 ``<base>__<tag>`` 变体，数量由模型按数据判断。

    任一变体不合格（超长、编号清单等）就整体带着原因重试。``taken`` 会被就地追加新名字。
    """
    base_name = str(base["archetype"])
    prompt = build_evolve_prompt(base, cards, existing, mode=mode, max_variants=max_variants,
                                 cluster_max=cluster_max, meta=meta)
    last_error: Exception | None = None
    attempts = 0
    for _ in range(max(1, retries)):
        attempts += 1
        note = "" if last_error is None else (
            f"\n\nYOUR PREVIOUS OUTPUT WAS REJECTED: {last_error}. Fix exactly this and answer again.")
        output = llm.call_json(prompt + note, max_tokens=12000, temperature=0.6)
        try:
            items = _variant_items(output)
            if len(items) > max_variants:
                raise ValueError(f"{len(items)} variants returned; limit {max_variants}")
            cleaned = [(_clean_fields(item, f"variant {i + 1}: "), item) for i, item in enumerate(items)]
            break
        except ValueError as exc:
            last_error = exc
    else:
        raise last_error  # type: ignore[misc]
    if meta is not None:
        meta["attempts"] = attempts
    results: list[dict[str, Any]] = []
    local_taken = set(taken)
    for fields, item in cleaned:
        raw_tag = fields["archetype"].split("__", 1)[1] if "__" in fields["archetype"] else fields["archetype"]
        fields["archetype"] = _unique(f"{base_name}__{_tag(raw_tag)}", local_taken)
        local_taken.add(fields["archetype"])
        results.append({"targeted": fields, "world_init_hint": str(item.get("world_init_hint") or "").strip(),
                        "multiplier": 2.0, "diagnosis_type": "synthesis_shortcut"})
    taken.update(local_taken)
    return results


def cluster_missing(cards: Sequence[Mapping[str, Any]], *, max_clusters: int = 3,
                    strict: bool = False) -> list[tuple[str, list[Mapping[str, Any]]]]:
    """能力盲区：把 archetype_fit==missing 的 gap 聚成若干簇。失败时退化为单簇。"""
    if len(cards) < 2:
        return [("uncovered", list(cards))] if cards else []
    items = [{"index": i, "structural_gap": c["structural_gap"],
              "missing_verify_rule": c["missing_verify_rule"]} for i, c in enumerate(cards)]
    try:
        output = llm.call_json(prompts.render(
            prompts.CLUSTER_GAPS, max_clusters=max_clusters,
            items_json=json.dumps(items, ensure_ascii=False, indent=1)), max_tokens=12000, temperature=0.6)
        assigned: dict[int, int] = {}
        clusters = output.get("clusters") or []
        for cid, cluster in enumerate(clusters):
            for member in cluster.get("members") or ():
                if isinstance(member, int) and 0 <= member < len(cards):
                    assigned.setdefault(member, cid)
        if not clusters or len(assigned) != len(cards):
            raise ValueError("clustering did not cover every item")
        groups = []
        for cid, cluster in enumerate(clusters):
            members = [cards[i] for i, c in assigned.items() if c == cid]
            if members:
                groups.append((_tag(cluster.get("label") or f"cluster_{cid}"), members))
        return groups
    except Exception:
        if strict:
            raise
        return [("uncovered", list(cards))]


def incubate_archetype(label: str, cluster: Sequence[Mapping[str, Any]],
                       pool: Sequence[Mapping[str, Any]], taken: set[str]) -> dict[str, Any]:
    """能力盲区：孵化全新 {archetype, archetype_desc, verify}。"""
    prompt = prompts.render(
        prompts.NEW_ARCHETYPE,
        archetype_pool_json=json.dumps(
            [{"archetype": e["archetype"], "archetype_desc": e["archetype_desc"]} for e in pool],
            ensure_ascii=False, indent=1),
        cluster_label=label, aggregated_structural_gaps_and_verify_rules=gap_lines(cluster))
    output = llm.call_json(prompt, max_tokens=12000, temperature=0.6)
    fields = _validate_targeted(output)
    name = _tag(fields["archetype"].replace("__", "_"))
    fields["archetype"] = _unique(name, taken)
    return {"targeted": fields, "world_init_hint": str(output.get("world_init_hint") or "").strip(),
            "multiplier": _clamp_multiplier(output.get("recommended_sampling_multiplier")),
            "diagnosis_type": "new_dimension"}
