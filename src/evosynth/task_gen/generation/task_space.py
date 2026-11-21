"""开放知识型域的任务原型格与分布控制。

题型只由 archetype 承担。原型不再写死在代码里,也不再按 domain 钉死:
调用方传入一份 archetype JSON(与 capability_split 产出同形),采样在这份
全集上按 coverage 加权抽取。EvoSynth 不再有复杂度档位,也不再另采一层
Constraint。

The sampler does not pre-generate a user-goal contract. Evidence topology is
computed from actual runtime dependencies after execution, rather than from
possible edges in a tool graph.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Mapping
from evosynth.task_gen.generation.tool_info import (
    AnchorPick,
    sample_anchor_candidates,
)
from evosynth.task_gen.common.distribution import (
    load_curriculum_biases,
    weighted_pick,
)

try:  # 并发批产时的跨进程写锁(见 world_graph.WorldGraph.save 同款设计)
    from filelock import FileLock
except ImportError:  # pragma: no cover
    FileLock = None

DEFAULT_MIN_DISTINCT_CALLS = 5
MIN_DISTINCT_CALLS_FLOOR = 3


def load_archetypes(path: Path) -> dict[str, dict]:
    """读取 capability_split 风格的 archetype JSON 数组。

    每项使用 ``archetype``、``archetype_desc``、``verfiy``（历史拼写，也接受
    ``verify``）。返回 ``{name: {"desc", "verify"}}``，供采样和世界构造查表。
    """
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read archetype file {source}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"archetype file must be a non-empty JSON array: {source}")
    catalog: dict[str, dict[str, str]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"archetype entry {index} is not an object: {source}")
        name = str(item.get("archetype") or "").strip()
        desc = str(item.get("archetype_desc") or item.get("desc") or "").strip()
        verify = str(item.get("verfiy") or item.get("verify") or "").strip()
        if not name or not desc:
            raise ValueError(
                f"archetype entry {index} missing archetype or archetype_desc: {source}")
        if name in catalog:
            raise ValueError(f"duplicate archetype {name!r}: {source}")
        # pool / initial_world 由 feedback 写入：pool=base|variant，initial_world 是给
        # World Initializer 的结构提示（世界构造经 archetype_initial_world 读取）。
        try:
            min_calls = int(item.get("min_distinct_calls") or DEFAULT_MIN_DISTINCT_CALLS)
        except (TypeError, ValueError):
            min_calls = DEFAULT_MIN_DISTINCT_CALLS
        catalog[name] = {
            "desc": desc, "verify": verify,
            "pool": str(item.get("pool") or "base").strip() or "base",
            "initial_world": str(
                item.get("initial_world") or item.get("world_init_hint") or "").strip(),
            # 研究 prompt 里「至少 K 次不同的成功工具调用」的 K，由 feedback 按
            # Teacher Acc 逐 archetype 调节；缺省 5，下限 3。
            "min_distinct_calls": max(MIN_DISTINCT_CALLS_FLOOR, min_calls)}
    return catalog


# coverage.json 的计数轴。
# 驱动采样的轴只记「已发布任务里真正用到」的量（被拒绝的任务不计）：
#   archetype / domain / server / tool_fn（tool_fn 每个任务内去重，记任务数）
#   server_anchored / server_anchor_realized：某 server 做过几次锚点、其中几次
#   真的被任务用上，用于识别「总被抽成锚点却从不产出」的 server。
# 以下三条只作诊断，不参与采样权重：
#   server_offered（曝光）/ server_picked（LLM 选中）/ server_vetoed（锚点与 persona
#   不兼容被否决）。
COUNT_AXES = ("archetype", "domain", "server", "tool_fn",
              "server_anchored", "server_anchor_realized",
              "server_offered", "server_picked", "server_vetoed")
PENDING_AXES = ("archetype", "domain", "server")
# 做过这么多次锚点却没有一次真正用上，就视为卡住，降权而不是继续占预算。
ANCHOR_STALL_LIMIT = 6
# archetype 轴的逆频率指数。一组 world 的多个任务共用一个 archetype，且各 archetype
# 通过率不同，alpha=1 时纠偏太软，已发布数差距会拉得很大；这里单独加大，
# 不影响 domain / server / 工具轴（它们仍用构造时的 alpha）。
ARCHETYPE_ALPHA = 3.0


def _load_pool_curriculum(path) -> tuple[dict, dict, dict, list]:
    """读取 feedback.curriculum 的三个字段；任何异常都静默降级为空。"""
    if not path or not os.path.exists(path):
        return {}, {}, {}, []
    try:
        data = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}, {}, {}, []
    if not isinstance(data, dict) or data.get("schema_version") != "feedback.curriculum":
        return {}, {}, {}, []
    mix = data.get("pool_mix") if isinstance(data.get("pool_mix"), dict) else {}
    multipliers = {str(k): float(v) for k, v in (data.get("sampling_multipliers") or {}).items()
                   if isinstance(v, (int, float)) and v > 0}
    hints = {str(a): {str(s): float(w) for s, w in h.items() if isinstance(w, (int, float)) and w > 0}
             for a, h in (data.get("anchor_hints") or {}).items() if isinstance(h, dict)}
    focus = [str(n) for n in (data.get("base_focus") or [])]
    return mix, multipliers, hints, focus


def _load_archetype_mix(path) -> dict[str, float]:
    """读取 curriculum 的逐 archetype 目标占比 ``archetype_mix``；缺失则为空（走旧逻辑）。"""
    if not path or not os.path.exists(path):
        return {}
    try:
        data = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("schema_version") != "feedback.curriculum":
        return {}
    raw = data.get("archetype_mix")
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items()
            if isinstance(v, (int, float)) and v > 0}


class OpenDistributionController:
    """按配额和 coverage 逆频率进行任务生成采样。

    当前版本覆盖控制按「已发布任务实际用到」计数，不再在抽样时就记账；
    server 侧改为「先按覆盖缺口选锚点，再由 LLM 裁决与 persona 是否兼容」。

    Curriculum bias 接线(当前版本,evosynth feedback 闭环):与
    CURRICULUM_JSON 环境变量指向
    feedback 编译产物,sampling_biases 逐条装载(同轴同值取 max),采样时
    λ=0.3 uniform floor(30% 概率忽略 bias)。零侵入:无 curriculum 时不多
    消耗任何 rng draw,采样流与历史 bit-identical;带 bias 的轴才多一次
    rng.random() 判定。EvoSynth 已无复杂度档位,curriculum 只装载 archetype。
    """

    def __init__(self, state_path: Path, archetypes: Mapping[str, Mapping[str, str]],
                 alpha: float = 1.0, archetype_alpha: float = ARCHETYPE_ALPHA):
        if not archetypes:
            raise ValueError("evosynth sampling requires a non-empty archetype catalog")
        self.archetypes = {
            str(name): {"desc": str(config.get("desc") or ""),
                        "verify": str(config.get("verify") or ""),
                        "pool": str(config.get("pool") or "base")}
            for name, config in archetypes.items()
        }
        self.path = Path(state_path)
        self.alpha = alpha
        self.archetype_alpha = archetype_alpha
        self.counts = {axis: {} for axis in COUNT_AXES}
        if self.path.exists():
            self.counts = json.load(open(self.path))
        self.counts.pop("tier", None)
        for axis in COUNT_AXES:
            self.counts.setdefault(axis, {})
        # 在途预留：已抽定但任务还没发布或被拒的量。只存内存，不落盘；
        # 采样时与已发布计数相加，避免并发批产时多组同时扎堆到同一个缺口。
        self.pending: dict[str, dict[str, int]] = {
            axis: {} for axis in PENDING_AXES}
        # 并发批产:记录加载时的基线,save 时只把增量合并进磁盘版本
        self._baseline = json.loads(json.dumps(self.counts))

        # Curriculum biases(缺文件/缺环境变量→空 bias,静默降级)。
        curriculum_path = os.environ.get("CURRICULUM_JSON")
        self.biases = load_curriculum_biases(
            curriculum_path,
            ("archetype",),
        )
        # feedback.curriculum：pool_mix / sampling_multipliers / anchor_hints。
        # 缺文件或旧 schema 时全为空，采样流与历史 bit-identical。
        self.pool_mix, self.multipliers, self.anchor_hints, self.base_focus = _load_pool_curriculum(
            curriculum_path)
        self.archetype_mix = _load_archetype_mix(curriculum_path)
        self.quota: dict[str, int] = {}

    def set_quota(self, quota: Mapping[str, int]) -> None:
        """设置本次运行每个 archetype 的发布数配额（>0 的才保留）。"""
        unknown = sorted(n for n in quota if n not in self.archetypes)
        if unknown:
            raise ValueError(f"archetype quota names not in capabilities: {unknown}")
        self.quota = {str(n): int(q) for n, q in quota.items() if int(q) > 0}

    def _effective(self, axis: str) -> dict[str, int]:
        """已发布计数 + 在途预留，用作采样时的 ``used``。"""
        merged = dict(self.counts.get(axis, {}))
        for key, value in self.pending.get(axis, {}).items():
            merged[key] = merged.get(key, 0) + value
        return merged

    def _reserve(self, axis: str, key: str, units: int) -> None:
        bucket = self.pending[axis]
        bucket[key] = bucket.get(key, 0) + units

    def _release(self, axis: str, key: str, units: int) -> None:
        bucket = self.pending[axis]
        left = bucket.get(key, 0) - units
        if left > 0:
            bucket[key] = left
        else:
            bucket.pop(key, None)

    def _bump(self, axis: str, key: str, units: int = 1) -> None:
        bucket = self.counts.setdefault(axis, {})
        bucket[key] = bucket.get(key, 0) + units

    def _pick(self, rng, axis, options: dict[str, float]) -> str:
        return weighted_pick(
            rng,
            options,
            used=self._effective(axis),
            biases=self.biases.get(axis),
            alpha=self.archetype_alpha if axis == "archetype" else self.alpha,
        )

    def _choose_archetype(self, rng) -> str:
        """基础池 / 变体池双轨：先按 pool_mix 抽池，再在池内按乘数 × 逆频率抽。

        没有变体或没有 pool_mix 时不多消耗 rng，行为与旧版一致。
        """
        names = list(self.archetypes)
        # 严格配额（fill_quota 补缺循环用）：只在「配额 - 已发布 - 在途」仍为正的桶里抽，
        # 按剩余量加权；在途预留已占满但还没发布时，退回「配额 - 已发布」，
        # 这样在途任务失败后缺口还会被重新补上。
        if self.quota:
            published = self.counts.get("archetype", {})
            pending = self.pending.get("archetype", {})
            open_net = {n: q - published.get(n, 0) - pending.get(n, 0)
                        for n, q in self.quota.items() if n in self.archetypes}
            cand = {n: v for n, v in open_net.items() if v > 0}
            if not cand:
                cand = {n: q - published.get(n, 0) for n, q in self.quota.items()
                        if n in self.archetypes and q - published.get(n, 0) > 0}
            if cand:
                point = rng.random() * sum(cand.values())
                acc = 0.0
                for name, weight in cand.items():
                    acc += weight
                    if point < acc:
                        return name
                return next(reversed(cand))
        # 逐 archetype 目标占比（feedback 的 M × 基础/变体切分）：存在时直接按占比抽，
        # 不再叠加逆频率，否则 alpha=3 的纠偏会把 30:70 这类配比拉平。
        mix = {n: w for n, w in self.archetype_mix.items() if n in self.archetypes}
        if mix:
            total = sum(mix.values())
            point = rng.random() * total
            acc = 0.0
            for name, weight in mix.items():
                acc += weight
                if point < acc:
                    return name
            return next(reversed(mix))
        variants = [n for n in names if self.archetypes[n].get("pool") == "variant"]
        bases = [n for n in names if n not in variants]
        # 基础池只给被点名「供给不足」的题型新合成（base_focus）；没有点名则不限制
        focus = [n for n in bases if n in set(self.base_focus)]
        if focus:
            bases = focus
        if variants and self.pool_mix:
            variant_share = float(self.pool_mix.get("variant", 0.0))
            if variant_share >= 1.0:
                names = variants
            elif variant_share <= 0.0:
                names = bases
            else:
                names = variants if rng.random() < variant_share else bases
        else:
            names = bases + variants if not focus else bases
        return self._pick(rng, "archetype",
                          {name: self.multipliers.get(name, 1.0) for name in names})

    def sample_archetype(self, seed: int, units: int = 0) -> str:
        """先于 server 选择抽 archetype。

        ``units > 0`` 时为这个 archetype 记 ``units`` 份在途预留（一份对应一个
        待产任务），之后用 ``settle`` 结算；任务发布才计入 coverage。
        ``units == 0`` 不预留也不计数，仅供试算。

        与 ``sample_tool_spec`` 内的抽取使用同一个 ``random.Random(seed)`` 首次
        抽取，因此同一 seed、同一 coverage 状态下结果一致。
        """
        rng = random.Random(seed)
        archetype = self._choose_archetype(rng)
        if units > 0:
            self._reserve("archetype", archetype, units)
        return archetype

    def stalled_servers(self) -> set[str]:
        """多次做锚点却几乎没产出任务的 server，采样时降权。"""
        anchored = self.counts.get("server_anchored", {})
        realized = self.counts.get("server_anchor_realized", {})
        return {server for server, n in anchored.items()
                if n - realized.get(server, 0) >= ANCHOR_STALL_LIMIT
                and realized.get(server, 0) * 4 < n}

    def sample_anchor(self, pool, seed: int, units: int = 1,
                      companions: int = 3, archetype: str | None = None) -> AnchorPick:
        """按覆盖缺口选锚点 server 与可选同伴，并预留 ``units`` 份在途配额。

        锚点的预留同时记在其 domain 与 server 上，使同一批里并发的其他组看见
        这些缺口「已有人在补」，转而去补别处。
        """
        pick = sample_anchor_candidates(
            pool, seed,
            tool_counts=self.counts.get("tool_fn", {}),
            domain_counts=self._effective("domain"),
            server_pending=self.pending["server"],
            stalled=self.stalled_servers(),
            companions=companions,
            alpha=self.alpha,
            server_bias=self.anchor_hints.get(archetype or ""),
        )
        self._reserve("domain", pick.domain, units)
        self._reserve("server", pick.anchor, units)
        for server in pick.candidates():
            self._bump("server_offered", server)
        return pick

    def release_anchor(self, pick: AnchorPick, units: int = 1) -> None:
        """放弃一个锚点预留（如锚点被 persona 否决后重抽），不计 coverage。"""
        self._release("domain", pick.domain, units)
        self._release("server", pick.anchor, units)

    def note_veto(self, anchor: str) -> None:
        self._bump("server_vetoed", anchor)

    def note_picked(self, servers) -> None:
        for server in dict.fromkeys(str(s) for s in servers):
            self._bump("server_picked", server)

    def settle(self, *, archetype: str | None, pick: AnchorPick | None,
               published: bool, tools=(), server_of_tool: Mapping[str, str] | None = None,
               domain_of_server: Mapping[str, str] | None = None) -> None:
        """结算一个任务的在途预留；仅 ``published`` 时把实际用到的量计入 coverage。

        ``tools`` 是任务参考调用里的知识工具 public_name，任务内去重后各记一次；
        server / domain 按这些工具反推，因此计的是「任务真正用上」而不是「被选中」。
        """
        if archetype is not None:
            self._release("archetype", archetype, 1)
        if pick is not None:
            self._release("domain", pick.domain, 1)
            self._release("server", pick.anchor, 1)
            self._bump("server_anchored", pick.anchor)
        if not published:
            return
        if archetype is not None:
            self._bump("archetype", archetype)
        used_tools = list(dict.fromkeys(str(t) for t in tools))
        used_servers: dict[str, None] = {}
        for tool in used_tools:
            self._bump("tool_fn", tool)
            server = (server_of_tool or {}).get(tool)
            if server:
                used_servers[server] = None
        for server in used_servers:
            self._bump("server", server)
        for domain in dict.fromkeys(
                (domain_of_server or {}).get(s) for s in used_servers):
            if domain:
                self._bump("domain", domain)
        if pick is not None and pick.anchor in used_servers:
            self._bump("server_anchor_realized", pick.anchor)

    def sample_tool_spec(self, seed: int, domain: str, capabilities,
                         archetype: str | None = None) -> dict:
        """按 Closed World 的同一机制采样开放域 ``SynthSpec``。

        ``capabilities`` 是可用的知识能力集合,只用来划定「起点可选哪些
        工具」与 archetype 可行性;终局字段、调用顺序一概不采样。
        ``archetype`` 由调用方先行抽定时(见 ``sample_archetype``)直接使用,
        并且不再重复记 coverage。
        """
        rng = random.Random(seed)
        # 工具声明只划定可用能力，不生成固定函数链，也不再套复杂度档位。
        knowledge = list(capabilities)
        if not knowledge:
            raise ValueError("open domain has no knowledge capabilities")
        if len(knowledge) < 2:
            raise ValueError(
                "highest-difficulty EvoSynth tasks require at least two knowledge tools")
        presampled = archetype is not None
        if presampled:
            if archetype not in self.archetypes:
                raise ValueError(f"unknown archetype: {archetype}")
        else:
            archetype = self._choose_archetype(rng)

        # 终局信息属性已不再采样(当前实现):原先「先采 answer_tool
        # 再采其叶字段」两级等权,等于把答案钉死在一个 (工具,字段) 上,任务
        # 随之退化成「去查那个字段」,跨条件/跨工具推理被挤掉。参照
        # 训练数据生成脚本的做法,种子只钉起点——这里就是
        # 「哪些能力可用」——哪条路径通向答案由研究代理在世界里自己走出。
        # 研究代理默认看到该域的全部知识工具,不再按子集抽样。
        connected_tools = [tool.public_name for tool in knowledge]

        spec = {
            "environment": domain,
            "archetype": archetype,
            "archetype_desc": self.archetypes[archetype]["desc"],
            # 不预选具体函数或调用顺序；只需保证至少两种知识能力可用。
            # calculate/calculator 等非知识工具在编译前已被剔除。
            "target_tools": (2, len(knowledge)),
            # 该域全部知识工具。研究代理和世界构造都看到这份全集。
            "connected_tools": connected_tools,
        }
        # 自行抽 archetype 的调用（dry-run 等）保持原行为：立即计数。
        # 由 sample_archetype 预抽的走 settle，发布后才计。
        if not presampled:
            self._bump("archetype", archetype)
        return spec

    def save(self) -> None:
        """并发安全落盘:filelock 互斥 + 增量合并(delta = 本进程新增计数)+ 原子替换。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if FileLock is not None:
            with FileLock(str(self.path) + ".lock", timeout=120):
                self._merge_and_write()
        else:
            self._merge_and_write()

    def _merge_and_write(self) -> None:
        merged = self.counts
        if self.path.exists():
            try:
                disk = json.load(open(self.path))
            except (json.JSONDecodeError, OSError):
                disk = None
            if disk is not None:
                merged = disk
                merged.pop("tier", None)
                for axis, cnt in self.counts.items():
                    base_axis = self._baseline.get(axis, {})
                    dst = merged.setdefault(axis, {})
                    for k, v in cnt.items():
                        delta = v - base_axis.get(k, 0)
                        if delta > 0:
                            dst[k] = dst.get(k, 0) + delta
        tmp = f"{self.path}.tmp{os.getpid()}"
        json.dump(merged, open(tmp, "w"), ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)
        # 重置基线:合并后的磁盘状态即新基线(同进程多次 save 不重复累加)
        self.counts = json.loads(json.dumps(merged))
        self._baseline = json.loads(json.dumps(merged))

    def report(self) -> str:
        lines = []
        for axis, cnt in self.counts.items():
            total = sum(cnt.values()) or 1
            lines.append(f"  {axis}: " + ", ".join(
                f"{k}:{v}({v/total:.0%})" for k, v in sorted(cnt.items(), key=lambda x: -x[1])))
        return "\n".join(lines)
