"""benchmark → 能力 archetype 拆分器。

流程:load_tasks → 阶段 1 逐任务抽取结构 parts（成功一条写一条缓存，失败跳过）
→ 阶段 2a 按批归并候选 archetype → 阶段 2b 近重复合并
→ 覆盖率不足时补挂未覆盖 part → 去掉 member_part_ids 后落盘。

输出 JSON 是 archetype 列表,每项仅含 archetype、archetype_desc、verify。
"""

from __future__ import annotations

import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .prompts import (ANDES_FAILURE_SIGNALS, ANDES_PRINCIPLES, COVER_PROMPT,
                      MERGE_PROMPT, SIGNATURE_PROMPT, SPLIT_PROMPT)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Atlas/其他 benchmark 导出的 CSV 单行可能包含超长 trajectory 字段。
csv.field_size_limit(sys.maxsize)

# None = 不截断单个任务文本(避免长规格任务被误判)
TASK_TEXT_TRUNC: int | None = None
MAX_LLM_RETRIES = 3
SIGNATURE_WORKERS = 12
# 阶段 2a：每批签名数，约 8k tokens，落在注意力较稳的区间。
BATCH_SIZE = 60
BATCH_WORKERS = 10
# member_part_ids 并集至少盖住这么多输入 part 才算过关；低于 REPAIR 则补挂。
COVERAGE_OK = 0.90
COVERAGE_REPAIR = 0.80
MIN_MEMBERS = 2
STAGE1_SCHEMA_VERSION = 3

_PRINT_LOCK = threading.Lock()


def _log(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def stage1_cache_path(out_path: str | Path) -> Path:
    """与 --out 同目录的阶段 1 缓存，文件名带上输出 stem，避免不同 benchmark 互相覆盖。"""
    out = Path(out_path)
    return out.with_name(out.stem + ".stage1_signatures.cache.jsonl")


def stage1_key(index: int, task: dict) -> str:
    return f"{index}:{task.get('task_id', '')}"


# ── benchmark 任务装载 ─────────────────────────────────────────────────────

TEXT_KEYS = ("question", "description", "task", "instruction", "prompt",
             "query", "goal", "problem", "input")


def _truncate(text: str, trunc: int | None) -> str:
    return text if trunc is None else text[:trunc]


def load_tasks(path: str | Path, trunc: int | None = TASK_TEXT_TRUNC,
               field: str | None = None, glob: str | None = None) -> list[dict]:
    """按优先级尝试多种布局,返回 [{"task_id", "text"}]:

    1. jsonl 文件:每行一个对象,取 field(显式指定)或 TEXT_KEYS 首个命中;
    2. 单个 json 文件:顶层是任务数组,或 {"tasks"/"data": [...]};
    3. glob 显式指定:目录下匹配的每个文件 = 一个任务(整文件为题面);
    4. Markdown 任务目录:叶目录含 description.md → task_id = 相对路径;
    5. mcp-universe 式:目录下 *.json / *.yaml 含 TEXT_KEYS 字段;
    6. 兜底:目录下 *.md / *.txt,每文件一个任务(README/index 类文件跳过)。

    都不命中返回空;格式再特殊就绕过本函数,自组 [{"task_id","text"}]
    直接喂 split_benchmark(LLM 侧格式无关)。
    """
    root = Path(path)
    if root.is_file():
        return _load_file(root, trunc, field)
    if not root.is_dir():
        raise FileNotFoundError(f"benchmark path not found: {root}")
    if glob:
        return [{"task_id": str(f.relative_to(root)), "text": _truncate(f.read_text(), trunc)}
                for f in sorted(root.rglob(glob)) if f.is_file()]
    tasks = [{"task_id": str(md.parent.relative_to(root)),
              "text": _truncate(md.read_text(), trunc)}
             for md in sorted(root.rglob("description.md"))]
    if tasks:
        return tasks
    for pattern, loader in (("*.json", _load_structured), ("*.yaml", _load_structured),
                            ("*.yml", _load_structured)):
        for jf in sorted(root.rglob(pattern)):
            rec = loader(jf)
            text = _pick_text(rec, field) if isinstance(rec, dict) else None
            if text:
                tasks.append({"task_id": str(jf.relative_to(root).with_suffix("")),
                              "text": _truncate(text, trunc)})
        if tasks:
            return tasks
    skip = re.compile(r"^(readme|index|license|contributing)", re.I)
    return [{"task_id": str(f.relative_to(root).with_suffix("")),
             "text": _truncate(f.read_text(), trunc)}
            for ext in ("*.md", "*.txt") for f in sorted(root.rglob(ext))
            if not skip.match(f.name)]


def _load_file(fp: Path, trunc: int | None, field: str | None) -> list[dict]:
    tasks: list[dict] = []
    if fp.suffix.lower() == ".csv":
        with fp.open(newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))
    elif fp.suffix == ".jsonl":
        records = [json.loads(x) for x in fp.read_text().splitlines() if x.strip()]
    else:
        data = _load_structured(fp)
        if isinstance(data, dict):
            data = data.get("tasks") or data.get("data") or [data]
        records = data if isinstance(data, list) else []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        text = _pick_text(rec, field)
        if text:
            tasks.append({"task_id": str(rec.get("id") or rec.get("task_id")
                                              or rec.get("TASK") or rec.get("name")
                                              or f"item{i:04d}"),
                          "text": _truncate(text, trunc)})
    return tasks


def _load_structured(fp: Path):
    try:
        if fp.suffix in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(fp.read_text())
        return json.loads(fp.read_text())
    except Exception:
        return None


def _pick_text(rec: dict, field: str | None = None) -> str | None:
    keys = (field,) if field else TEXT_KEYS
    for key in keys:
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


# ── LLM 出口 ───────────────────────────────────────────────────────────────

def _apply_thinking(request: dict, model: str, enable_thinking: bool | None) -> None:
    """GLM 系列用 extra_body 显式开关 thinking。其他模型不传这个字段。"""
    if enable_thinking is None or not str(model).lower().startswith("glm"):
        return
    request["extra_body"] = {"enable_thinking": bool(enable_thinking)}


def _llm_json(client, model: str, prompt: str, *, max_tokens: int = 131072,
              temperature: float = 0.2, enable_thinking: bool | None = None):
    feedback = ""
    for attempt in range(MAX_LLM_RETRIES):
        try:
            request = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt + feedback}],
            }
            _apply_thinking(request, model, enable_thinking)
            r = client.chat.completions.create(**request)
        except Exception as e:
            if attempt == MAX_LLM_RETRIES - 1:
                raise
            print(f"[capability_split] LLM call failed ({type(e).__name__}: {e}); "
                  f"retry {attempt + 1}/{MAX_LLM_RETRIES - 1}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
            continue
        # Most gateways return an OpenAI response object; a few lightweight
        # compatibility clients return the response text directly.
        if isinstance(r, str):
            content = r
            reason = ""
            finish_reason = "stop"
        else:
            msg = r.choices[0].message
            content = msg.content or ""
            reason = getattr(msg, "reasoning_content", "") or ""
            finish_reason = r.choices[0].finish_reason
        # 推理模型(reasoning 通道)会吃掉大量输出预算;截断时必须让用户看到
        if finish_reason == "length":
            feedback = (f"\n\nPREVIOUS RESPONSE WAS TRUNCATED (finish_reason=length; "
                        f"content={len(content)} chars, reasoning={len(reason)} chars). "
                        "Return ONLY valid JSON.")
            time.sleep(1 + attempt)
            continue
        starts = [i for i, char in enumerate(content) if char in "[{"]
        for start in starts:
            try:
                parsed, _ = json.JSONDecoder().raw_decode(content[start:])
                if isinstance(parsed, (list, dict)):
                    return parsed
            except json.JSONDecodeError as e:
                feedback = (f"\n\nPREVIOUS RESPONSE WAS INVALID JSON ({e}). "
                            "Return ONLY valid JSON.")
        if starts:
            continue
        feedback = ("\n\nPREVIOUS RESPONSE HAD NO JSON "
                    f"(finish_reason={finish_reason}, "
                    f"content={len(content)} chars). Return ONLY valid JSON.")
        time.sleep(1 + attempt)
    raise RuntimeError(f"LLM failed to return valid JSON after {MAX_LLM_RETRIES} attempts")


# ── 拆分主流程：阶段 1 抽取 parts，阶段 2 归并 archetype ────────────────

def _parse_parts(parsed: object) -> list[dict]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("parts"), list):
        return []
    required = ("roles", "dependency", "operation", "decision_output",
                "external_effect", "shortcut")
    return [
        {key: str(part[key]).strip() for key in required}
        for part in parsed["parts"]
        if isinstance(part, dict) and all(str(part.get(key) or "").strip() for key in required)
    ]


def _extract_task_parts(client, model: str, task: dict) -> list[dict]:
    # 阶段 1 一律关闭 thinking，避免推理通道吃掉 JSON 预算；fallback 同样关闭。
    prompt = SIGNATURE_PROMPT.format(task=task["text"])
    parsed = _llm_json(client, model, prompt, max_tokens=7000, temperature=0.1,
                       enable_thinking=False)
    parts = _parse_parts(parsed)
    if parts:
        return parts
    # A short recovery instruction handles occasional truncated or overlong task signatures.
    recovery = (prompt + "\n\nRECOVERY: return exactly one concise part; each of the six fields "
                "must be one short sentence. Return complete valid JSON only.")
    parsed = _llm_json(client, model, recovery, max_tokens=2500, temperature=0.0,
                       enable_thinking=False)
    parts = _parse_parts(parsed)
    if not parts:
        raise ValueError(f"signature has no valid parts for task {task.get('task_id', '')!r}")
    return parts


def load_stage1_cache(path: str | Path | None) -> dict[str, list[dict]]:
    """读取已成功的阶段 1 记录。坏行跳过；同一 key 以最后一条为准。"""
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.is_file():
        return {}
    loaded: dict[str, list[dict]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = rec.get("key")
        parts = rec.get("parts")
        normalized = _parse_parts({"parts": parts})
        if (rec.get("schema_version") == STAGE1_SCHEMA_VERSION
                and isinstance(key, str) and normalized
                and len(normalized) == len(parts)):
            loaded[key] = normalized
    return loaded


def append_stage1_cache(path: Path, lock: threading.Lock, key: str,
                        task_id: str, parts: list[dict]) -> None:
    record = {"schema_version": STAGE1_SCHEMA_VERSION,
              "key": key, "task_id": task_id, "parts": parts}
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def _render_signatures(indexed: list[tuple[int, dict]]) -> str:
    keys = ("roles", "dependency", "operation", "decision_output", "shortcut")
    lines = []
    for part_id, item in indexed:
        signature = {key: item["signature"][key] for key in keys}
        rendered = json.dumps(signature, ensure_ascii=False, separators=(",", ": "))
        lines.append(f"[{part_id}] {rendered}")
    return "\n".join(lines)


def iter_part_batches(parts: list[dict], batch_size: int = BATCH_SIZE):
    """按全局 1-based id 切批，id 不随批次重编。"""
    for start in range(0, len(parts), batch_size):
        chunk = parts[start:start + batch_size]
        yield [(start + offset + 1, item) for offset, item in enumerate(chunk)]


def _parse_archetypes(parsed: object, *, with_members: bool) -> list[dict]:
    rows = parsed.get("archetypes") if isinstance(parsed, dict) else parsed
    return [
        _normalize_archetype(row, with_members=with_members)
        for row in (rows or [])
        if isinstance(row, dict)
    ]


def _split_batch(client, model: str, benchmark: str, indexed: list[tuple[int, dict]]) -> list[dict]:
    prompt = SPLIT_PROMPT.format(
        benchmark=benchmark,
        principles=ANDES_PRINCIPLES,
        failure_signals=ANDES_FAILURE_SIGNALS,
        n_parts=len(indexed),
        signatures=_render_signatures(indexed),
        scope_note=("This is one batch of a larger benchmark. The bracket numbers are global "
                    "part ids. member_part_ids must use only ids listed in this batch."),
    )
    parsed = _llm_json(client, model, prompt, max_tokens=16000, temperature=0.2,
                       enable_thinking=True)
    allowed = {part_id for part_id, _ in indexed}
    rows = _parse_archetypes(parsed, with_members=True)
    for row in rows:
        row["member_part_ids"] = [part_id for part_id in row["member_part_ids"] if part_id in allowed]
    return [row for row in rows if row["archetype"] and row["member_part_ids"]]


def _merge_candidates(client, model: str, benchmark: str, candidates: list[dict]) -> list[dict]:
    if len(candidates) <= 1:
        return candidates
    rendered = "\n".join(
        f"[{index + 1}] {json.dumps(row, ensure_ascii=False, separators=(',', ': '))}"
        for index, row in enumerate(candidates)
    )
    allowed = {part_id for row in candidates for part_id in row.get("member_part_ids") or []}
    prompt = MERGE_PROMPT.format(
        benchmark=benchmark,
        principles=ANDES_PRINCIPLES,
        failure_signals=ANDES_FAILURE_SIGNALS,
        n_candidates=len(candidates),
        candidates=rendered,
    )
    parsed = _llm_json(client, model, prompt, max_tokens=16000, temperature=0.2,
                       enable_thinking=True)
    merged = _parse_archetypes(parsed, with_members=True)
    for row in merged:
        row["member_part_ids"] = [part_id for part_id in row["member_part_ids"] if part_id in allowed]
    merged = [row for row in merged if row["archetype"] and row["member_part_ids"]]
    if not merged:
        raise RuntimeError("stage 2b returned no archetypes")
    return merged


def union_member_ids(rows: list[dict]) -> set[int]:
    covered: set[int] = set()
    for row in rows:
        covered.update(int(part_id) for part_id in row.get("member_part_ids") or [])
    return covered


def coverage_ratio(rows: list[dict], n_parts: int) -> float:
    if n_parts <= 0:
        return 1.0
    return len(union_member_ids(rows) & set(range(1, n_parts + 1))) / n_parts


def needs_coverage_repair(rows: list[dict], n_parts: int) -> bool:
    return coverage_ratio(rows, n_parts) < COVERAGE_REPAIR


def absorb_archetypes(base: list[dict], extra: list[dict]) -> list[dict]:
    """同名 archetype 合并 member_part_ids；新名字追加。"""
    by_name = {row["archetype"]: row for row in base}
    for row in extra:
        name = row.get("archetype") or ""
        if not name:
            continue
        if name in by_name:
            merged = set(by_name[name].get("member_part_ids") or [])
            merged.update(row.get("member_part_ids") or [])
            by_name[name]["member_part_ids"] = sorted(merged)
        else:
            base.append(row)
            by_name[name] = row
    return base


def audit_membership(rows: list[dict], n_parts: int) -> list[str]:
    warnings: list[str] = []
    covered = union_member_ids(rows) & set(range(1, n_parts + 1))
    ratio = coverage_ratio(rows, n_parts)
    if n_parts and ratio < COVERAGE_OK:
        warnings.append(
            f"member_part_ids coverage {len(covered)}/{n_parts} ({ratio:.1%}) < {COVERAGE_OK:.0%}"
        )
    for row in rows:
        ids = [part_id for part_id in row.get("member_part_ids") or [] if 1 <= part_id <= n_parts]
        if len(set(ids)) < MIN_MEMBERS:
            name = row.get("archetype") or "?"
            warnings.append(f"{name}: member_part_ids has {len(set(ids))} < {MIN_MEMBERS}")
    return warnings


def _repair_coverage(client, model: str, benchmark: str, parts: list[dict],
                     rows: list[dict]) -> list[dict]:
    n_parts = len(parts)
    missing = [part_id for part_id in range(1, n_parts + 1) if part_id not in union_member_ids(rows)]
    if not missing:
        return rows
    indexed = [(part_id, parts[part_id - 1]) for part_id in missing]
    existing = json.dumps([
        {"archetype": row["archetype"], "archetype_desc": row["archetype_desc"],
         "verify": row["verify"], "member_part_ids": row["member_part_ids"]}
        for row in rows
    ], ensure_ascii=False, indent=1)
    prompt = COVER_PROMPT.format(
        benchmark=benchmark,
        archetypes=existing,
        n_parts=len(indexed),
        signatures=_render_signatures(indexed),
    )
    parsed = _llm_json(client, model, prompt, max_tokens=16000, temperature=0.2,
                       enable_thinking=True)
    allowed = set(missing)
    extra = _parse_archetypes(parsed, with_members=True)
    for row in extra:
        row["member_part_ids"] = [part_id for part_id in row["member_part_ids"] if part_id in allowed]
    extra = [row for row in extra if row["archetype"] and row["member_part_ids"]]
    return absorb_archetypes(rows, extra)


def _strip_members(rows: list[dict]) -> list[dict]:
    return [
        {"archetype": row["archetype"], "archetype_desc": row["archetype_desc"],
         "verify": row["verify"]}
        for row in rows
    ]


def split_benchmark(client, model: str, benchmark: str, tasks: list[dict],
                    *, cache_path: str | Path | None = None) -> list[dict]:
    """阶段 1 抽 parts（可断点续跑）→ 分块归并 → 去重 → 覆盖率补挂。"""
    cached = load_stage1_cache(cache_path)
    parts_by_task: list[list[dict] | None] = [None] * len(tasks)
    pending: list[int] = []
    for index, task in enumerate(tasks):
        key = stage1_key(index, task)
        if key in cached:
            parts_by_task[index] = cached[key]
        else:
            pending.append(index)
    _log(f"[capability_split] stage 1: resumed {len(tasks) - len(pending)}/{len(tasks)} "
         f"from cache, running {len(pending)}")

    failures: list[str] = []
    cache_lock = threading.Lock()
    cache_file = Path(cache_path) if cache_path else None
    progress = {"done": len(tasks) - len(pending)}

    def _tick() -> int:
        with _PRINT_LOCK:
            progress["done"] += 1
            return progress["done"]

    def _run_one(index: int):
        task = tasks[index]
        parts = _extract_task_parts(client, model, task)
        if cache_file is not None:
            append_stage1_cache(cache_file, cache_lock, stage1_key(index, task),
                                str(task.get("task_id", "")), parts)
        return parts

    if pending:
        with ThreadPoolExecutor(max_workers=SIGNATURE_WORKERS) as pool:
            futures = {pool.submit(_run_one, index): index for index in pending}
            for future in as_completed(futures):
                index = futures[future]
                task_id = tasks[index].get("task_id", index)
                done = _tick()
                try:
                    parts_by_task[index] = future.result()
                    _log(f"[capability_split] stage 1: {done}/{len(tasks)} {task_id} "
                         f"({len(parts_by_task[index])} parts)")
                except Exception as exc:
                    failures.append(f"task {task_id!r}: {exc}")
                    _log(f"[capability_split] stage 1: skip {task_id!r} "
                         f"({done}/{len(tasks)}): {exc}")

    if failures:
        _log(f"[capability_split] stage 1: skipped {len(failures)} failed task(s); "
             "cached successes are kept")
    if not any(parts_by_task):
        raise RuntimeError("stage 1 produced no parts")

    parts = [
        {"task_index": task_index + 1, "part_index": part_index + 1, "signature": signature}
        for task_index, task_parts in enumerate(parts_by_task)
        for part_index, signature in enumerate(task_parts or [])
    ]
    batches = list(iter_part_batches(parts, BATCH_SIZE))
    _log(f"[capability_split] stage 2a: {len(parts)} parts in {len(batches)} batches "
         f"of up to {BATCH_SIZE}")

    candidates: list[dict] = []
    failed_batches = 0
    with ThreadPoolExecutor(max_workers=min(BATCH_WORKERS, max(len(batches), 1))) as pool:
        futures = {
            pool.submit(_split_batch, client, model, benchmark, batch): batch_index
            for batch_index, batch in enumerate(batches)
        }
        finished = 0
        for future in as_completed(futures):
            batch_index = futures[future]
            finished += 1
            try:
                rows = future.result()
                candidates.extend(rows)
                _log(f"[capability_split] stage 2a: batch {batch_index + 1}/{len(batches)} "
                     f"→ {len(rows)} archetypes ({finished}/{len(batches)} done)")
            except Exception as exc:
                failed_batches += 1
                _log(f"[capability_split] stage 2a: batch {batch_index + 1}/{len(batches)} "
                     f"failed: {exc}")
    if not candidates:
        raise RuntimeError("stage 2a produced no archetypes")

    try:
        merged = _merge_candidates(client, model, benchmark, candidates)
        _log(f"[capability_split] stage 2b: {len(candidates)} candidates → {len(merged)} archetypes")
    except Exception as exc:
        _log(f"[capability_split] stage 2b failed ({exc}); unioning batch candidates by name")
        merged = candidates
    merged = absorb_archetypes([], merged)

    ratio = coverage_ratio(merged, len(parts))
    if needs_coverage_repair(merged, len(parts)) or failed_batches:
        _log(f"[capability_split] stage 2: coverage {ratio:.1%}, repairing uncovered parts")
        try:
            merged = _repair_coverage(client, model, benchmark, parts, merged)
        except Exception as exc:
            _log(f"[capability_split] coverage repair failed: {exc}")
    for warning in audit_membership(merged, len(parts)):
        _log(f"[capability_split] ! {warning}")
    return _strip_members(merged)

def _oneline(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ── 输出规范化与校验 ─────────────────────────────────────────────────────

def _member_ids(arch: dict) -> list[int]:
    ids: list[int] = []
    for raw in arch.get("member_part_ids") or []:
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(set(ids))


def _normalize_archetype(arch: dict, *, with_members: bool = False) -> dict:
    """落盘只留三个键。归并过程中 with_members=True 时保留 member_part_ids。"""
    row = {
        "archetype": str(arch.get("archetype") or arch.get("name") or ""),
        "archetype_desc": str(arch.get("archetype_desc") or arch.get("desc") or ""),
        "verify": str(arch.get("verify") or ""),
    }
    if with_members:
        row["member_part_ids"] = _member_ids(arch)
    return row


def validate(result: list[dict]) -> list[str]:
    warnings: list[str] = []
    for i, arch in enumerate(result):
        name = arch.get("archetype") or ""
        if not _NAME_RE.match(name):
            warnings.append(f"invalid archetype name: {name!r}")
        if set(arch) != {"archetype", "archetype_desc", "verify"}:
            warnings.append(f"item {i}: unexpected output keys: {sorted(arch)}")
        if not arch.get("archetype_desc"):
            warnings.append(f"{name or f'item {i}'}: empty archetype_desc")
        if not arch.get("verify"):
            warnings.append(f"{name or f'item {i}'}: empty verify")
    return warnings


def save(result: list[dict], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return out
