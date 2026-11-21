"""EvoSynth orchestration for single-task construction and domain generation."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

from openai import OpenAI

from evosynth.utils import sample_persona
from evosynth.task_gen.common.synthesis import (
    DerivationGraph, ExecutionFirstPipeline, ExecutionTrace, PersonaRef,
    QualityReport, SynthesisHooks, TaskPlan, measure_complexity,
)
from evosynth.task_gen.contracts.task_bundle import TaskBundle
from evosynth.task_gen.contracts.versioned_bundle import load_bundle, pin_bundle
from evosynth.task_gen.generation.task_space import (
    OpenDistributionController, load_archetypes,
)
from evosynth.task_gen.generation.tool_info import (
    CompiledOpenDomain, compile_tool_info, dumps_compact, load_mcp_pool,
    pool_server_index, write_compiled_registry,
)
from evosynth.task_gen.world.backend import GenericEvoSynthBackend
from evosynth.task_gen.world.controlled import (
    WorldInfrastructureError,
    world_prompt_path_for_registry,
)
from evosynth.task_gen.world.universal import (
    UniversalWorldContract, generate_world_contract, parse_world_contract)
from evosynth.task_gen.version import EVOSYNTH_PIPELINE_VERSION

from . import certificate as certificate_module
from . import proposal, research, runtime, server_selection

DEFAULT_COMPANIONS = 3
# 锚点与 persona 不兼容时，先重采 persona 的次数；超过则换锚点，最多换这么多次。
PERSONA_RESAMPLE_LIMIT = 5
ANCHOR_RETRY_LIMIT = 3
# 题目没有发布成功时，用同一 seed、同一初始世界重跑 research 的次数。
# 这两次仍然失败才封禁 seed。基础设施故障不计入，也不封禁。
PUBLISH_REBUILD_ATTEMPTS = max(0, int(os.environ.get(
    "EVOSYNTH_PUBLISH_REBUILD_ATTEMPTS", "2")))

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
EVOSYNTH_CONTRACT_FILES = (
    PACKAGE_ROOT / "task_gen/generation_api.py",
    PACKAGE_ROOT / "task_gen/version.py",
    PACKAGE_ROOT / "task_gen/common/synthesis.py",
    PACKAGE_ROOT / "task_gen/world/backend.py",
    PACKAGE_ROOT / "task_gen/world/controlled.py",
    PACKAGE_ROOT / "task_gen/generation/logic_graph.py",
    PACKAGE_ROOT / "task_gen/generation/tool_info.py",
    PACKAGE_ROOT / "task_gen/world/universal.py",
    PACKAGE_ROOT / "task_gen/generation/task_space.py",
    Path(__file__),
    Path(__file__).with_name("runtime.py"),
    Path(__file__).with_name("research.py"),
    Path(__file__).with_name("proposal.py"),
    Path(__file__).with_name("certificate.py"),
    Path(__file__).with_name("server_selection.py"),
)


def banned_seeds_path(output_dir: Path) -> Path:
    """封禁名单。``EVOSYNTH_BANNED_SEEDS`` 让两条造题入口共用一份。"""
    raw = os.environ.get("EVOSYNTH_BANNED_SEEDS", "").strip()
    return Path(raw) if raw else output_dir / "banned_seeds.json"


def load_seed_ledger(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"rebuild_limit": PUBLISH_REBUILD_ATTEMPTS, "seeds": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"rebuild_limit": PUBLISH_REBUILD_ATTEMPTS, "seeds": {}}
    seeds = payload.get("seeds") if isinstance(payload, dict) else None
    if not isinstance(seeds, dict):
        seeds = {}
    return {"rebuild_limit": PUBLISH_REBUILD_ATTEMPTS, "seeds": seeds}


def decide_publish_failure(record: Mapping[str, Any] | None, *,
                           rebuild_limit: int, reasons: Sequence[str],
                           task_id: str, now: str | None = None,
                           ) -> tuple[dict[str, Any], str]:
    """记一次发布失败。未满重造次数返回 rebuild，否则返回 ban。"""
    previous = dict(record or {})
    if previous.get("banned"):
        updated = dict(previous)
        updated["reasons"] = list(reasons)[:8]
        updated["task_id"] = task_id
        return updated, "ban"
    failures = int(previous.get("failures") or 0) + 1
    stamp = now or datetime.now().isoformat()
    updated = {
        **previous,
        "failures": failures,
        "task_id": task_id,
        "reasons": list(reasons)[:8],
        "updated_at": stamp,
        "banned": failures > rebuild_limit,
    }
    if updated["banned"]:
        updated["banned_at"] = stamp
        return updated, "ban"
    return updated, "rebuild"


class _MetricsOnlyQualityGate:
    """只统计轨迹复杂度，不用规则否决任务。"""

    def evaluate(self, trace: ExecutionTrace, **_kwargs: Any) -> QualityReport:
        return QualityReport(True, (), measure_complexity(trace))


def _execute_research(plan: TaskPlan, backend: GenericEvoSynthBackend,
                      compiled: CompiledOpenDomain, client: OpenAI,
                      state: dict[str, Any] | None = None) -> ExecutionTrace:
    """执行固定调查；失败或未完成时从同一初态有限重跑。"""
    state = state if state is not None else {}
    checkpoint = (backend.checkpoint()
                  if callable(getattr(backend, "checkpoint", None)) else None)
    discarded: list[dict[str, Any]] = []
    feedback = ""
    last_error = "research did not complete"
    # clued/named 身份披露轴已删除(当前实现):身份模式不再由 spec
    # 决定研究预算,统一使用默认次数(CLUED_MAX_RESEARCH_ATTEMPTS 常量已随轴
    # 一并从 research.py 删除)。
    max_attempts = research.MAX_RESEARCH_ATTEMPTS
    resumable = False
    for attempt in range(max_attempts):
        trace: ExecutionTrace | None = None
        # 上一次只是"收尾没写好"（没有结论/缺标签），调查本身已经完成：
        # 不还原世界、不重新调查，沿用已有对话与观测，只要求重新收尾。
        resume = (resumable and attempt
                  and state.get("research_messages") is not None
                  and state.get("research_trace") is not None
                  and state["research_trace"].relevant_successful_calls)
        if attempt and not resume and checkpoint is not None and callable(
                getattr(backend, "restore", None)):
            backend.restore(checkpoint)
        try:
            runtime._check_task_deadline("research attempt")
            if resume:
                trace = research._execute_research_once(
                    plan, backend, compiled, client, state,
                    resume_messages=state["research_messages"],
                    resume_trace=state["research_trace"],
                    resume_user=(
                        "Your previous reply did not end with a usable "
                        "<question>/<answer>/<evidence> block. Using the "
                        "observations already gathered, finish now in that "
                        "exact format. Call a tool only if one specific "
                        "observation is still missing. " + last_error[:400]))
            else:
                trace = research._execute_research_once(
                    plan, backend, compiled, client, state, retry_feedback=feedback)
            question, answer, evidence = proposal._task_and_answer(trace.conclusion)
            state["task_text"] = question
            state["answer"] = answer
            state["evidence"] = evidence
            state["discarded_research_attempts"] = discarded
            runtime._log_event("research_completed", attempt=attempt + 1,
                       task_text=question, answer=answer, trace=trace.to_dict())
            return trace
        except (TimeoutError, WorldInfrastructureError):
            raise
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:2000]}"
            resumable = ("no conclusion" in last_error
                         or "missing a non-empty" in last_error)
            discarded.append({
                "attempt": attempt + 1, "reason": last_error,
                "trace": trace.to_dict() if trace is not None else None,
            })
            runtime._log_event("research_attempt_discarded", attempt=attempt + 1,
                       reason=last_error,
                       trace=(trace.to_dict() if trace is not None else None))
            feedback = (
                "The previous attempt did not finish with both a non-empty <question> and "
                "<answer>: "
                + last_error
                + ". Run the investigation again and end with one concrete question and the "
                  "answer it asks for.")
    if checkpoint is not None and callable(getattr(backend, "restore", None)):
        backend.restore(checkpoint)
    state["discarded_research_attempts"] = discarded
    raise runtime._ResearchAttemptIncomplete(
        f"research incomplete after {max_attempts} attempts: {last_error}")


def _generate_initial_world(*, compiled: CompiledOpenDomain,
                            spec: Mapping[str, Any],
                            persona: Mapping[str, Any],
                            archetypes: Mapping[str, Mapping[str, str]]) -> UniversalWorldContract:
    """Generate one world contract before task workers are started."""
    config = runtime.runtime_config()
    client = OpenAI(base_url=config.agent_base_url,
                    api_key=runtime._evosynth_agent_key(),
                    timeout=600, max_retries=0)
    archetype = str(spec.get("archetype") or "")
    archetype_config = dict(archetypes.get(archetype) or {})
    state: dict[str, Any] = {"persona_selection": {
        "source": "sample_persona", "selected_line": persona.get("line")}}

    def generate_json(prompt: str) -> Mapping[str, Any]:
        return runtime._run_json_stage(
            client=client, prompt=prompt, stage="universal_initial_world",
            validator=lambda payload: dict(payload), state=state,
            temperature=0.4, max_attempts=2)

    world_archetype_desc = str(spec.get("archetype_desc")
                               or archetype_config.get("desc") or "")
    # feedback 变体携带的世界结构提示（基础池无此字段，描述保持不变）。
    world_hint = str(archetype_config.get("initial_world") or "").strip()
    if world_hint:
        world_archetype_desc += f"\nWorld structure hint: {world_hint}"
    return generate_world_contract(
        compiled=compiled,
        persona=str(persona.get("persona") or "research analyst"),
        archetype=archetype,
        archetype_desc=world_archetype_desc,
        connected_tools=spec.get("connected_tools") or None,
        generate_json=generate_json, max_attempts=3)

@runtime.task_log_context()
def _construct_one(*, compiled: CompiledOpenDomain, registry_path: Path,
                   artifact_root: Path, seed: int, spec: Mapping[str, Any],
                   persona: Mapping[str, Any],
                   archetypes: Mapping[str, Mapping[str, str]],
                   materialize_generate: Any = None,
                   log_path: Path | None = None,
                   initial_world: UniversalWorldContract | None = None,
                   seed_entity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    runtime._start_task_deadline()
    runtime._bind_log(log_path, domain=compiled.name)
    # SDK 隐式重试保持关闭，避免与 ``_chat`` 重试相乘。
    config = runtime.runtime_config()
    client = OpenAI(
        base_url=config.agent_base_url,
        api_key=runtime._evosynth_agent_key(),
        timeout=600,
        max_retries=0,
    )
    target = {
        "calls": list(spec.get("target_calls", spec.get("read_calls_range", (2, 4)))),
        # target_tools 是"倾向提示"而非下限：它告诉研究 agent 世界里有哪几
        # 类接口可用，但答案只需 1 个接口就能站稳时，绝不允许因为这里的
        # 数值而去补一次冗余调用。
        "tools": list(spec.get("target_tools", (1, max(1, len(compiled.knowledge_tools))))),
        "dependency_depth": 1,
        "constraints": 0,
    }
    # 当前实现:这里原本把 target_tools 下限抬到 2(max(2, ...))。
    # 取消该抬升。诊断:事后数工具个数来强制"多工具",产出的题只是看起来
    # 像多工具——实际一两次调用就能答完,而研究 agent 为了凑够计数被迫发
    # 冗余调用(用户举的例子:place_details 已经返回同样信息时,再调一次
    # detail 接口纯属浪费)。多工具应当是题面语义要求(条件取自 ≥2 个不同
    # 类目)的结果,而不是被计数逼出来的形状。难度要求已上移到 prompt 层。
    if len(compiled.knowledge_tools) < 2:
        # 域能力检查,与单题工具计数无关:世界至少要有两个知识工具才谈得上
        # 语义上的跨类目组合,否则这个域本身就不该走 EvoSynth 合成。保留。
        raise ValueError(
            "highest-difficulty EvoSynth synthesis requires at least two knowledge tools")
    # tool_diversity_policy 已删除(当前实现):它含
    # minimum_distinct_tools / mode=strict_multi_tool_outcome /
    # single_tool_allowed_for_seed 三个键,全部在描述一个"下限",而它自己
    # 没有任何消费者——只流向日志与任务元数据,却让读日志的人以为存在硬
    # 门禁。与其保留一个语义错误的记录,不如直接删掉。事后观测到的工具集合
    # 只留在构造审计里，不再写进发布任务。
    # 降级门禁的警告记录:不否决 seed,写入 progress 的 warnings 字段供分析。
    collected_gate_warnings: list[str] = []
    state: dict[str, Any] = {
        "persona_selection": {
            "source": "sample_persona",
            "selected_line": persona.get("line"),
        },
    }
    # 执行前目标合同已删除(当前实现):这里只保留构造世界所需的结构
    # 性状态,不再保存 goal_contract / terminal_property / evidence_topology。
    struct_spec = {**dict(spec), "target_tools": list(target["tools"])}
    if str(struct_spec.get("archetype") or "") not in archetypes:
        exc = ValueError(
            "sampled plan spec carries an unknown archetype: "
            + str(struct_spec.get("archetype") or ""))
        task_id = f"uwg_{compiled.name}_{seed}_purpose_unresolved"
        runtime.set_task_log_task_id(task_id)
        task_root = artifact_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        runtime._log_event("task_error", task_id=task_id, stage="structural_precondition",
                   error=f"{type(exc).__name__}: {exc}")
        (task_root / "construction_error.json").write_text(
            json.dumps({
                "task_id": task_id, "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"task_id": task_id, "accepted": False,
                "reasons": [f"{type(exc).__name__}:{str(exc)[:800]}"],
                "warnings": list(collected_gate_warnings),
                "artifact": str(task_root), "spec": dict(spec)}

    task_id = f"uwg_{compiled.name}_{seed}_{persona.get('line', seed)}"
    plan = TaskPlan(
        task_id=task_id, seed=seed, domain=compiled.name,
        persona=PersonaRef(
            str(persona.get("persona") or "research analyst"),
            persona.get("line"), dict(persona)),
        archetype=str(spec["archetype"]),
        archetype_desc=str(spec["archetype_desc"]),
        tier="", constraints=(),
        target_complexity=target, derivation=DerivationGraph(),
        payload={"seed_entity": dict(seed_entity)} if seed_entity else {},
        raw_spec=struct_spec)
    runtime.set_task_log_task_id(task_id)
    task_root = artifact_root / task_id
    task_root.mkdir(parents=True, exist_ok=True)
    world_path = task_root / "controlled_world.json"
    runtime._log_event("persona_selected", task_id=task_id, domain=compiled.name, seed=seed,
               persona_selection=state.get("persona_selection"))
    runtime._log_event("task_start", task_id=task_id, domain=compiled.name, seed=seed,
               spec=dict(spec), persona=dict(persona),
               seed_entity_id=str((seed_entity or {}).get("id") or ""))

    archetype_name = str(struct_spec.get("archetype") or "")
    archetype_config = dict(archetypes.get(archetype_name) or {})
    effective_spec = dict(struct_spec)
    effective_spec["target_tools"] = list(target["tools"])
    # 发布不再编译带 claim 的证书，只要求研究结论里有题和答案，
    # 再由题目一致性核查裁决。轨迹依赖、重复调用和题面泄漏不再机械否决。
    # 当前版本 语义变更:target_calls 从 [硬性发布下限, 上限] 变为
    # [软参考, 上限]。下限不再构成任何门禁(不得据此补调用或拒绝任务),只
    # 保留作复杂度统计与探索步数参考;键名与形状不变,消费方由 Stage 4 调整。
    # 研究代理默认加载该域全部知识工具。connected_tools 现在就是这份全集。
    effective_spec["connected_tools"] = [
        str(name) for name in (struct_spec.get("connected_tools") or ())]
    # 三条 archetype 语义字段由采样到的 archetype 查表补齐,供初始世界构造
    # 使用;世界构造不再认识任何领域配方,只按这些结构先验实例化。
    effective_spec["archetype_desc"] = str(
        archetype_config.get("desc") or struct_spec.get("archetype_desc") or "")
    effective_spec["archetype_initial_world"] = str(
        archetype_config.get("initial_world") or "")
    effective_spec["archetype_verify"] = str(archetype_config.get("verify") or "")
    effective_spec["archetype_min_distinct_calls"] = proposal.min_calls_of(
        archetype_config.get("min_distinct_calls"))
    # 采样器早期把同一字段写成 initial_world,与 archetype_initial_world 同义;
    # 统一到后者并去掉别名,避免两份同义语义各自漂移。
    effective_spec.pop("initial_world", None)
    plan = replace(plan, target_complexity=target, raw_spec=effective_spec)
    runtime._log_event("effective_spec", task_id=task_id, spec=dict(effective_spec))

    if initial_world is None:
        raise ValueError("task construction requires a pre-generated initial world")
    # Re-parse the serialized representation to ensure no dataclass-only state
    # bypasses the same contract later consumed from disk.
    initial_world = parse_world_contract(initial_world.to_dict(), compiled)
    initial_world_path = initial_world.save(task_root / "initial_world.json")
    state["initial_world"] = initial_world.to_dict()
    runtime._log_event("initial_world_constructed", task_id=task_id,
               entities=len(initial_world.entities),
               path=str(initial_world_path))
    # 执行前不存在真实的值传递；evidence topology 只在真实调用后计算。
    backend = GenericEvoSynthBackend(
        compiled, registry_path=registry_path, world_path=world_path,
        materialize_generate=materialize_generate,
        queue_wait_observer=runtime._observe_world_queue_wait)
    backend.pin_initial_world(initial_world)
    hooks = SynthesisHooks(
        execute=lambda p, b: _execute_research(p, b, compiled, client, state),
        propose=lambda p, t, o: proposal._propose(p, t, o, client, state),
        compile_certificate=lambda p, t, o, proposal: certificate_module._compile_certificate(
            p, t, o, proposal, state),
        naturalize=proposal._naturalize,
        public_values=lambda _p: state.get("public_values", ()),
        private_values=lambda _p, _t, _c: state.get("private_values", ()),
        replay=lambda _p, _b, _t: {
            "passed": True, "skipped": True,
            "reason": "evosynth_execution_witness_already_frozen",
        },
        compile_derivation=research._compile_derivation,
    )
    try:
        coherence_feedback = ""
        for synthesis_pass in range(2):
            active_hooks = hooks
            if synthesis_pass:
                active_hooks = replace(
                    hooks,
                    execute=lambda p, b, feedback=coherence_feedback: (
                        research.continue_research_after_review(
                            p, b, compiled, client, state, feedback)))
            result = ExecutionFirstPipeline(
                backend, _MetricsOnlyQualityGate()).construct(plan, active_hooks)
            # 参考轨迹就是研究期间的成功调用。不再按答案引用切子轨迹。
            trace = result["trace"]
            # evidence_topology 事后判定(当前实现,设计文档 §5.2/§7):
            # 执行前不再预置拓扑(见上方 topology = {}),这里才根据真实调用参数
            # 是否引用了前一步观察值(depends_on 的值传递边)判定。基础是真实
            # depends_on 而不是工具图的可能边——工具图只能说「可以传值」,旧实现
            # 却据此把任务标成 linked_retrieval 并逼出详情调用。只记进审计结果与
            # 任务元数据供分析,不作门禁(设计文档要的是不再自动加 handoff,不是
            # 再引入一条形态门禁)。
            evidence_topology = research.evidence_topology_for_trace(
                trace.reference_calls)
            result["evidence_topology"] = evidence_topology
            state["evidence_topology"] = evidence_topology
            # 证据工具下限门禁已删除(当前实现):它把"答案证据必须覆盖
            # ≥2 个不同工具接口"当作发布条件,且直接否掉短任务——而短任务未必简单,
            # 多工具本身也不是难度的来源。
            # 旧的多分支证据复杂度核算已删除；当前只保留实际调用轨迹，
            # 以及下方的轻量 evidence 来源校验。
            result["reference_solution"] = trace.reference_solution()
            answer = str(state.get("answer") or "").strip()
            task_text = str(result.get("task_natural") or "").strip()
            extra_reasons: list[str] = []
            evidence_reason = ""
            if not task_text or not answer:
                extra_reasons.append("missing_task_or_answer")
            else:
                evidence_ok, evidence_reason = proposal.validate_evidence(
                    trace=trace,
                    evidence=list(state.get("evidence") or ()),
                    min_calls=effective_spec["archetype_min_distinct_calls"],
                )
                if not evidence_ok:
                    extra_reasons.append("evidence_rejected:" + evidence_reason[:300])
            if task_text and answer and result.get("quality", {}).get("accepted"):
                # 题和答案都在才做一致性核查。
                runtime._log_event("task_coherence_started", task_id=task_id)
                coherent, coherence_reason, coherence_debug = proposal._audit_task_coherence(
                    task_text, answer, client, state)
                state["task_coherence_debug"] = coherence_debug
                runtime._log_event("task_coherence_result", task_id=task_id,
                           coherent=coherent, reason=coherence_reason)
                if not coherent:
                    extra_reasons.append(
                        "task_coherence_rejected:" + coherence_reason[:200])
            revise = (
                synthesis_pass == 0
                and any(str(item).startswith(("task_coherence_rejected:",
                                              "evidence_rejected:"))
                        for item in extra_reasons)
            )
            if revise:
                runtime._log_event(
                    "research_revision", task_id=task_id,
                    reason="; ".join(extra_reasons))
                coherence_feedback = "; ".join(extra_reasons)
                continue
            if extra_reasons:
                quality = result["quality"]
                quality["reasons"] = list(dict.fromkeys(
                    [*quality["reasons"], *extra_reasons]))
                quality["accepted"] = not quality["reasons"]
            runtime._log_event("pipeline_result", task_id=task_id,
                       accepted=bool(result.get("quality", {}).get("accepted")),
                       quality=result.get("quality"), trace=result.get("trace"),
                       task_natural=result.get("task_natural"),
                       certificate=result.get("certificate"),
                       reference_solution=result.get("reference_solution"))
            break
    except Exception as exc:
        runtime._log_event("task_error", task_id=task_id, stage="pipeline",
                   error=f"{type(exc).__name__}: {exc}",
                   world_snapshot=backend.snapshot())
        # 失败尝试也必须有结构化原因；否则批量 orchestrator 只能看到线程
        # future 异常，无法判断是工具材质化、agent 提案还是质量门禁失败。
        (task_root / "construction_error.json").write_text(
            json.dumps({"task_id": task_id, "error_type": type(exc).__name__,
                        "error": str(exc)[:2000],
                        "persona_selection": state.get("persona_selection"),
                        "property_ledger": state.get("property_ledger"),
                        "task_text": state.get("task_text"),
                        "answer": state.get("answer"),
                        "discarded_research_attempts": state.get(
                            "discarded_research_attempts"),
                        "task_coherence_debug": state.get("task_coherence_debug"),
                        "world_snapshot": backend.snapshot()},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {"task_id": task_id, "accepted": False,
                "reasons": [f"{type(exc).__name__}:{str(exc)[:800]}"],
                "warnings": list(collected_gate_warnings),
                "artifact": str(task_root), "spec": dict(plan.raw_spec)}
    result["execution_first_generation"] = {
        # 身份披露轴与执行前目标合同已删除(当前实现),这里不再记录
        # identity_disclosure / user_goal_contract / goal_* 系列调试字段。
        # 旧的证据链复杂度和探索轨迹字段已删除；参考见证就是
        # trace.reference_calls，发布侧只保留当前流程需要的字段。
        "persona_selection": state.get("persona_selection"),
        "property_ledger": state.get("property_ledger"),
        "task_text": state.get("task_text"),
        "answer": state.get("answer"),
        "discarded_research_attempts": state.get(
            "discarded_research_attempts"),
        "task_coherence_debug": state.get("task_coherence_debug"),
    }
    audit_path = task_root / "construction.json"
    audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                     default=lambda value: value.to_dict()), encoding="utf-8")
    if not result["quality"]["accepted"]:
        runtime._log_event("task_rejected", task_id=task_id,
                   reasons=result["quality"].get("reasons", []))
        return {"task_id": task_id, "accepted": False,
                "reasons": result["quality"]["reasons"],
                "warnings": list(collected_gate_warnings),
                "artifact": str(task_root), "spec": dict(plan.raw_spec)}

    effective_plan = result["plan"]
    certificate = result["certificate"]
    world_id = f"generic_open_{compiled.name}_{seed}"
    bundle = pin_bundle(
        root=task_root / "world", registry_path=registry_path,
        prompt_path=world_prompt_path_for_registry(registry_path),
        semantic_spec={"version": "tool_info_open", **compiled.to_dict()},
        world_files=[world_path, initial_world_path, audit_path],
        contract_files=EVOSYNTH_CONTRACT_FILES,
        world_id=world_id, world_revision=f"tool_info_open_{compiled.name}_{seed}",
        metadata={
            "domain": compiled.name,
            "pipeline": EVOSYNTH_PIPELINE_VERSION,
            "entrypoint_layout": "generate_tasks_and_private_generation",
            "servers": list(compiled.servers),
            "seed": seed,
        })
    loaded = load_bundle(task_root / "world")
    trace = result["trace"]
    knowledge_tools = [call.tool for call in trace.reference_calls]
    available_tools = list(dict.fromkeys([
        *[tool.public_name for tool in compiled.tools],
    ]))
    task = TaskBundle.create(
        task_id=task_id, task_natural=result["task_natural"],
        manifest=loaded.manifest,
        metadata={
            "domain": compiled.name, "artifact_rel": f"artifacts/{task_id}",
            "synthesis_models": [runtime.runtime_config().agent_model],
            **runtime._world_model_metadata(),
            "persona": effective_plan["persona"],
            "persona_line": effective_plan["persona_line"],
            "persona_selection": state.get("persona_selection"),
            "archetype": effective_plan["archetype"],
            # 执行前目标合同、shape 轴与身份披露轴已删除(当前实现):
            # 不发布合同/执行前拓扑/披露元数据,避免下游误以为存在执行前义务。
            # evidence_topology 是例外且方向相反:它由真实 depends_on 值传递
            # 边事后算出(见 research.evidence_topology_for_trace),是执行后
            # 事实,只供分析/筛选,不构成任何门禁。
            "evidence_topology": state.get("evidence_topology"),
            # 工具接入子集的采样结果,记进元数据仅供分析(消费者是
            # research._execute_research_once 与世界构造的 schema 过滤,
            # 不是门禁)。
            "connected_tools": list(
                plan.raw_spec.get("connected_tools") or ()),
            "available_tools": available_tools,
            "correct_answer": certificate["correct_answer"],
            "evidence": list(state.get("evidence") or ()),
            # derivation / quality / exploration_trace / evidence_trace /
            # target_complexity / actual_complexity / required_calls /
            # planned_capabilities / reference_witness 已删除:发布任务不再
            # 记录造题时调用了哪些工具。
            "replay": result["replay"],
        })
    # Construction success is not publication success.  The generation driver
    # emits ``task_published`` only after both public/private bundles are written
    # and read back successfully.
    runtime._log_event("task_constructed", task_id=task_id, task_natural=task.task_natural,
               metadata=task.metadata)
    return {"task_id": task_id, "accepted": True, "task": task,
            "artifact": str(task_root), "tools": knowledge_tools,
            "spec": dict(plan.raw_spec),
            "warnings": list(collected_gate_warnings),
            "bundle_hash": bundle.manifest.manifest_hash,
            "construction_trace": research.construction_trace_payload(
                task_id=task_id, state=state, trace=trace)}

def generate_domain(*, domain: str | None = None, tool_info: Path, output_dir: Path,
                    capabilities: Path, count: int, seed_start: int,
                    servers: Sequence[str] | None = None,
                    companions: int = DEFAULT_COMPANIONS,
                    workers: int = 1, max_attempts: int | None = None,
                    materialize_generate: Any = None,
                    log_path: Path | None = None,
                    archetype_quota: Mapping[str, int] | None = None) -> dict[str, Any]:
    """生成任务。

    ``archetype_quota``：每个 archetype 的发布数严格配额。采样只在未满的桶里抽，
    某桶已满后该桶多出来的成功任务不发布（surplus_success_not_published）。
    给了配额时 ``count`` 应等于配额总和，由调用方保证。

    默认从 ``tool_info`` 指向的 MCP 池里，对每个 initial world 组：先抽
    archetype，再按覆盖缺口选一个锚点 server 并配 ``companions`` 个可选同伴；LLM
    只裁决锚点与 persona 是否兼容、挑同伴。不兼容时重采 persona（锚点不变），
    连续 ``PERSONA_RESAMPLE_LIMIT`` 次不兼容才放弃该锚点并重抽。coverage 只计
    已发布任务实际用到的 archetype / domain / server / 工具。显式给出
    ``servers`` 时跳过抽样与选择，所有世界固定使用这些 server。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_path) if log_path else None
    domain_label = str(domain or "mcp_pool")
    runtime._bind_log(log_path, domain=domain_label)
    archetypes = load_archetypes(capabilities)
    fixed_servers = tuple(str(item) for item in servers) if servers else ()
    pool = None if fixed_servers else load_mcp_pool(tool_info)
    runtime._log_event("generation_start", domain=domain_label, tool_info=str(tool_info),
               capabilities=str(capabilities), archetype_count=len(archetypes),
               fixed_servers=list(fixed_servers), companions=companions,
               output_dir=str(output_dir), count=count, seed_start=seed_start,
               workers=workers, max_attempts=max_attempts)
    tasks_dir = output_dir / "tasks"; tasks_dir.mkdir(parents=True, exist_ok=True)
    artifacts = output_dir / "artifacts"; artifacts.mkdir(parents=True, exist_ok=True)
    selections_dir = output_dir / "server_selection"
    registries_dir = output_dir / "registries"
    controller = OpenDistributionController(
        output_dir / "coverage.json", archetypes=archetypes)
    quota = {str(k): int(v) for k, v in (archetype_quota or {}).items() if int(v) > 0}
    if quota:
        controller.set_quota(quota)
    accepted_by_arch: Counter = Counter()
    controller_lock = threading.Lock()
    domain_of_server = pool_server_index(pool)[0] if pool else None
    # 每个 initial world 组各自编译所选 server；seed 通过它找回所属组的工具表。
    seed_compiled: dict[int, CompiledOpenDomain] = {}
    seed_registry: dict[int, Path] = {}
    group_selections: dict[str, dict[str, Any]] = {}
    used_servers: dict[str, None] = {}
    # 每个 initial world 最多采样的任务数。实体更少时在世界生成后按实体数裁剪，
    # 因为每个任务要独占一个起点实体。
    world_samples = max(1, min(10, int(os.environ.get(
        "EVOSYNTH_WORLD_SAMPLES_PER_INITIAL_WORLD", "1"))))
    # count 是要成功发布的任务数。拒绝的 seed 不计入。未显式限制
    # max_attempts 时继续取后续 seed，直到发布数达到 count，或尝试次数
    # 达到安全上限。
    attempt_limit = max_attempts if max_attempts is not None else max(count * 20, count)
    scheduled = [0]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    lock = threading.Lock()
    progress_path = output_dir / "generation_progress.json"

    # 断点续跑：输出目录里已经完整发布的 task（私有包 + public 包都能读回且 id 一致）
    # 直接计入 accepted，其 seed 不再重新生成。尚未封禁的失败 seed 没有落盘，
    # 会照常重试，并沿用 banned_seeds.json 里已经用掉的重造次数。
    def _load_published() -> dict[int, dict[str, Any]]:
        found: dict[int, dict[str, Any]] = {}
        prefix = f"uwg_{domain_label}_"
        for task_path in sorted(tasks_dir.glob(f"{prefix}*.json")):
            if task_path.name.endswith((".public.json", ".construction_trace.json")):
                continue
            public_path = task_path.with_name(f"{task_path.stem}.public.json")
            seed_text = task_path.stem[len(prefix):].split("_", 1)[0]
            if not seed_text.isdigit() or not public_path.exists():
                continue
            try:
                task = TaskBundle.from_dict(json.loads(
                    task_path.read_text(encoding="utf-8")))
                public = json.loads(public_path.read_text(encoding="utf-8"))
                if task.task_id != task_path.stem or public.get("task_id") != task.task_id:
                    continue
            except Exception:
                continue
            found[int(seed_text)] = {
                "task_id": task.task_id, "accepted": True, "seed": int(seed_text),
                "status": "published", "resumed": True,
                "archetype": (task.metadata or {}).get("archetype"),
                "tools": list(dict.fromkeys(
                    (task.metadata or {}).get("available_tools") or [])),
                "reasons": [], "warnings": []}
        return found

    completed_seeds = _load_published()
    for seed_value in sorted(completed_seeds):
        if len(accepted) < count:
            accepted.append(completed_seeds[seed_value])
    accepted_by_arch.update(str(r.get("archetype")) for r in accepted if r.get("archetype"))

    def _bucket_full(seed: int) -> bool:
        """配额模式下，该 seed 所属 archetype 的桶是否已满。"""
        if not quota:
            return False
        arch = str((seed_reservation.get(seed) or {}).get("archetype") or "")
        return arch in quota and accepted_by_arch[arch] >= quota[arch]

    if completed_seeds:
        runtime._log_event("resume_loaded", published=len(completed_seeds),
                           counted=len(accepted))
        print(f"[evosynth] resume: found {len(completed_seeds)} published tasks, "
              f"skipping their seeds ({len(accepted)}/{count} already done)", flush=True)

    def row_status(row: Mapping[str, Any]) -> str:
        if row.get("accepted"):
            return "published"
        reason = " ".join(str(item) for item in row.get("reasons", ())).casefold()
        if "seed_banned" in reason:
            return "seed_banned"
        if "publish_failed" in reason:
            return "publication_failed"
        if "worldmaterializationerror" in reason:
            return "materialization_failed"
        if "worldinfrastructureerror" in reason or "8822 " in reason or any(
                token in reason for token in (
                    "connectionerror", "apiconnectionerror", "rate limit", "429",
                    "apitimeouterror", "agent api failed after",
                    "selected model is at capacity")):
            return "infra_failed"
        if "task deadline" in reason or "timeouterror" in reason:
            return "task_timeout"
        return "semantic_rejected"

    def write_progress(*, running: bool) -> None:
        rows = [*accepted, *rejected]
        counts = Counter(str(item.get("status") or row_status(item)) for item in rows)
        payload = {
            "domain": domain_label, "target_published": count,
            "requested_seeds": scheduled[0],
            "finished_seeds": len(rows), "accepted": len(accepted),
            "acceptance_rate": len(accepted) / len(rows) if rows else 0.0,
            "status_counts": dict(sorted(counts.items())),
            "running": running, "updated_at": datetime.now().isoformat(),
            "seeds": sorted([{
                "seed": item.get("seed"), "task_id": item.get("task_id"),
                # 供 feedback 按 archetype 统计漏斗，不依赖完整日志
                "archetype": item.get("archetype")
                or (item.get("spec") or {}).get("archetype"),
                "status": item.get("status") or row_status(item),
                "reasons": item.get("reasons", []),
                "warnings": item.get("warnings", []),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "generation_attempts": item.get("generation_attempts", []),
                "publish_attempts": item.get("publish_attempts"),
            } for item in rows], key=lambda item: int(item.get("seed") or -1)),
        }
        runtime._atomic_write_json(progress_path, payload)
        if os.environ.get("EVOSYNTH_STDOUT_PROGRESS", "").casefold() in {
                "1", "true", "yes", "on"}:
            print(
                f"[evosynth] progress domain={domain_label} "
                f"published={len(accepted)}/{count} finished={len(rows)} "
                f"rejected={len(rejected)} statuses={dict(sorted(counts.items()))}",
                flush=True)

    write_progress(running=True)

    def run(job):
        seed, spec, persona, initial_payload, seed_entity = job
        # 每次重造都从同一份世界快照重新解析，不把上一次研究的材质化带进来。
        initial_world = parse_world_contract(
            copy.deepcopy(initial_payload), seed_compiled[seed])
        seed_entity = copy.deepcopy(seed_entity)
        started = time.monotonic()
        attempts_log: list[dict[str, Any]] = []
        row: dict[str, Any] = {}
        for infra_attempt in range(max(1, runtime.INFRA_RETRY_ATTEMPTS)):
            attempt_started = time.monotonic()
            row = _construct_one(
                compiled=seed_compiled[seed], registry_path=seed_registry[seed],
                artifact_root=artifacts, seed=seed, spec=spec,
                persona=persona, archetypes=archetypes,
                materialize_generate=materialize_generate,
                log_path=log_path, initial_world=initial_world,
                seed_entity=seed_entity)
            reason_list = list(row.get("reasons") or ())
            infrastructure = runtime._infrastructure_failure(reason_list)
            attempts_log.append({
                "attempt": infra_attempt + 1,
                "elapsed_seconds": time.monotonic() - attempt_started,
                "accepted": bool(row.get("accepted")),
                "infrastructure_failure": infrastructure,
                "reasons": reason_list,
            })
            if row.get("accepted") or not infrastructure \
                    or infra_attempt + 1 >= runtime.INFRA_RETRY_ATTEMPTS:
                break
            runtime._log_event("seed_infrastructure_retry", domain=domain_label,
                       seed=seed, attempt=infra_attempt + 1,
                       reasons=reason_list)
        row["generation_attempts"] = attempts_log
        row["elapsed_seconds"] = time.monotonic() - started
        row.setdefault("archetype", (spec or {}).get("archetype"))
        return row

    # 世界按组现造现用。发布数不够时继续下一组 seed，不再把尝试次数当成目标。
    worlds: dict[str, UniversalWorldContract] = {}
    jobs: list[tuple] = []

    def _build_group(*, world_key: str, world_seed: int, archetype: str,
                     persona: Mapping[str, Any], holder: dict[str, Any], units: int):
        """选 server → 编译 → 采样 spec → 生成初始世界。运行在世界线程里。

        非固定 server 时：先按覆盖缺口抽锚点，LLM 判锚点与 persona 是否兼容；
        不兼容就重采 persona（锚点不变，因为它是覆盖缺口，不能因 persona 而丢）。
        连续多次都不兼容才放弃这个锚点，换一个再试。选定的锚点预留记在
        ``holder["pick"]``，由发布/拒绝时的 ``_settle`` 结算。
        """
        archetype_desc = archetypes[archetype]["desc"]
        if fixed_servers:
            selection: dict[str, Any] = {
                "candidates": {}, "selected": list(fixed_servers),
                "reason": "fixed by --servers", "archetype": archetype}
            chosen = fixed_servers
        else:
            picked = None
            attempt_log: list[dict[str, Any]] = []
            for anchor_try in range(ANCHOR_RETRY_LIMIT):
                with controller_lock:
                    pick = controller.sample_anchor(
                        pool, world_seed + anchor_try * 1_000_003, units=units,
                        companions=companions, archetype=archetype)
                holder["pick"] = pick
                for persona_try in range(PERSONA_RESAMPLE_LIMIT):
                    if persona_try:
                        persona = sample_persona(
                            world_seed + persona_try * 7_919
                            + anchor_try * 1_000_003)
                    verdict = server_selection.select_servers(
                        pool=pool,
                        persona=str(persona.get("persona") or "research analyst"),
                        archetype=archetype, archetype_desc=archetype_desc, pick=pick)
                    attempt_log.append({
                        "anchor": pick.anchor, "persona_line": persona.get("line"),
                        "compatible": verdict.compatible, "reason": verdict.reason})
                    if verdict.compatible:
                        picked = verdict
                        break
                    with controller_lock:
                        controller.note_veto(pick.anchor)
                if picked is not None:
                    break
                # 这个锚点在多个 persona 下都不成立：释放预留，换锚点。
                with controller_lock:
                    controller.release_anchor(pick, units)
                holder["pick"] = None
            if picked is None:
                raise server_selection.ServerSelectionError(
                    f"no compatible anchor after {ANCHOR_RETRY_LIMIT} anchors x "
                    f"{PERSONA_RESAMPLE_LIMIT} personas: {attempt_log[-1]['reason']}")
            selection = {**picked.to_dict(), "archetype": archetype,
                         "persona_line": persona.get("line"),
                         "selection_attempts": attempt_log}
            chosen = picked.selected
            with controller_lock:
                controller.note_picked(chosen)
        runtime._log_event("servers_selected", world=world_key, **selection)
        group_compiled = compile_tool_info(
            tool_info, domain=domain_label, servers=chosen)
        safe_key = world_key.replace(":", "_")
        group_registry = write_compiled_registry(
            group_compiled, registries_dir / f"{safe_key}.json")
        runtime._atomic_write_json(selections_dir / f"{safe_key}.json", selection)
        with controller_lock:
            spec = controller.sample_tool_spec(
                world_seed, domain_label, group_compiled.knowledge_tools,
                archetype=archetype)
        world = _generate_initial_world(
            compiled=group_compiled, spec=spec, persona=persona, archetypes=archetypes)
        with lock:
            group_selections[world_key] = selection
            for server in group_compiled.servers:
                used_servers.setdefault(server)
        return group_compiled, group_registry, spec, world, persona

    seed_reservation: dict[int, dict[str, Any]] = {}
    ledger_path = banned_seeds_path(output_dir)
    ledger = load_seed_ledger(ledger_path)
    banned: set[int] = set()
    for key, item in ledger["seeds"].items():
        if str(key).isdigit() and isinstance(item, dict) and item.get("banned"):
            banned.add(int(key))
    if banned:
        runtime._log_event("banned_seeds_loaded", path=str(ledger_path),
                           count=len(banned))
        print(f"[evosynth] skip {len(banned)} banned seeds from {ledger_path}",
              flush=True)

    def _save_ledger() -> None:
        runtime._atomic_write_json(ledger_path, {
            "rebuild_limit": PUBLISH_REBUILD_ATTEMPTS,
            "seeds": ledger["seeds"],
        })

    def _clear_publish_failure(seed: int) -> None:
        if ledger["seeds"].pop(str(seed), None) is not None:
            _save_ledger()

    def _settle(seed: int, *, published: bool, tools=()) -> None:
        """结算该 seed 的在途预留；仅发布成功才把实际用到的量计入 coverage。"""
        holder = seed_reservation.pop(seed, None)
        if holder is None:
            return
        compiled = seed_compiled.get(seed)
        with controller_lock:
            controller.settle(
                archetype=holder["archetype"], pick=holder.get("pick"),
                published=published, tools=tools,
                server_of_tool={t.public_name: t.server
                                for t in compiled.tools} if compiled else None,
                domain_of_server=domain_of_server)

    def _reject_seeds(persona: Mapping[str, Any], seeds: Sequence[int],
                      reason: str) -> None:
        for seed in seeds:
            holder = seed_reservation.get(seed)  # _settle 会弹出，先取 archetype
            _settle(seed, published=False)
            rejected.append({
                "seed": seed,
                "archetype": (holder or {}).get("archetype"),
                "task_id": f"uwg_{domain_label}_{seed}_{persona.get('line', seed)}",
                "accepted": False,
                "reasons": [reason[:800]],
                "warnings": [],
            })
        write_progress(running=True)

    def _top_up_jobs(next_job: int) -> None:
        """补足尚未提交的任务，使排队数能填满 worker。同一批世界并行生成。"""
        pending_groups = []
        while (len(jobs) - next_job + len(pending_groups) * world_samples
               < max(1, workers)
               and scheduled[0] < attempt_limit and len(accepted) < count):
            group = scheduled[0] // world_samples
            offset0 = group * world_samples
            span = min(world_samples, attempt_limit - offset0)
            if span <= 0:
                break
            seeds = [seed_start + offset0 + index for index in range(span)
                     if seed_start + offset0 + index not in completed_seeds
                     and seed_start + offset0 + index not in banned]
            scheduled[0] = offset0 + span
            if not seeds:
                continue
            world_key = f"{domain_label}:{group}"
            world_seed = seed_start + offset0
            persona = sample_persona(world_seed)
            # archetype 在 server 选择之前抽定，并为这组的每个待产任务记一份在途预留，
            # 使并发的后续各组避开已有人在补的缺口；发布后才正式计入 coverage。
            with controller_lock:
                archetype = controller.sample_archetype(world_seed, units=len(seeds))
            holder = {"archetype": archetype, "pick": None}
            for seed in seeds:
                seed_reservation[seed] = holder
            pending_groups.append(
                (world_key, world_seed, archetype, persona, seeds, holder))
        if not pending_groups:
            return
        runtime._log_event("world_generation_started", worlds=len(pending_groups))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as world_pool:
            futures = {
                world_pool.submit(
                    _build_group, world_key=world_key, world_seed=world_seed,
                    archetype=archetype, persona=persona, holder=holder,
                    units=len(seeds)): (world_key, persona, seeds)
                for (world_key, world_seed, archetype, persona, seeds,
                     holder) in pending_groups
            }
            built: dict[str, tuple[CompiledOpenDomain, Path, dict[str, Any],
                                   UniversalWorldContract, Mapping[str, Any]]] = {}
            for future in as_completed(futures):
                world_key, persona, seeds = futures[future]
                try:
                    built[world_key] = future.result()
                except Exception as exc:
                    stage = ("server_selection_failed"
                             if isinstance(exc, server_selection.ServerSelectionError)
                             else "world_contract_failed")
                    reason = f"{stage}:{type(exc).__name__}: {exc}"
                    runtime._log_event("world_generation_failed", world=world_key,
                                       error=reason[:2000])
                    _reject_seeds(persona, seeds, reason)
        for (world_key, _world_seed, _archetype, _persona, seeds,
             _holder) in pending_groups:
            if world_key not in built:
                continue
            group_compiled, group_registry, spec, world, persona = built[world_key]
            runnable = list(seeds)
            capacity = len(world.entities)
            if len(runnable) > capacity:
                dropped = runnable[capacity:]
                runnable = runnable[:capacity]
                # 这些 seed 还没开始研究。只退回采样时按任务数预留的配额，
                # 不记成拒绝，也不增加锚点的失败计数。
                for seed in dropped:
                    holder = seed_reservation.pop(seed, None)
                    if holder is None:
                        continue
                    with controller_lock:
                        controller._release("archetype", holder["archetype"], 1)
                        pick = holder.get("pick")
                        if pick is not None:
                            controller._release("domain", pick.domain, 1)
                            controller._release("server", pick.anchor, 1)
                runtime._log_event(
                    "world_samples_capped", world=world_key, entities=capacity,
                    requested=len(seeds), kept=len(runnable),
                    dropped_seeds=dropped)
            try:
                assigned = research.allocate_seed_entities(world, runnable)
            except Exception as exc:
                reason = f"world_contract_failed:{type(exc).__name__}: {exc}"
                runtime._log_event("world_generation_failed", world=world_key,
                                   error=reason[:2000])
                _reject_seeds(persona, runnable, reason)
                continue
            worlds[world_key] = world
            world_payload = copy.deepcopy(world.to_dict())
            for seed in runnable:
                seed_compiled[seed] = group_compiled
                seed_registry[seed] = group_registry
                jobs.append((
                    seed, spec, persona, world_payload,
                    copy.deepcopy(dict(assigned[seed])),
                ))
        runtime._log_event("world_generation_finished", worlds=len(worlds))

    # 名字不能用 pool：外层 pool 是 MCP 工具池，_build_group 靠闭包读取它。
    # 凑满发布数后不再等已提交的任务：那些线程是非守护线程，正常 shutdown
    # 会把进程留到它们跑完。取消未开始的任务后直接退出进程。
    task_pool = ThreadPoolExecutor(max_workers=max(1, workers))
    pending = {}
    next_job = 0
    quota_reached = False

    def _retry_or_ban(job, row) -> bool:
        """发布失败时用同一 seed 重跑 research。返回 True 表示已重新排队。

        基础设施故障不占重造次数，也不封禁。配额已满时只记下失败次数，
        留给下次运行继续用完剩余次数。
        """
        reasons = [str(item) for item in (row.get("reasons") or ())]
        if runtime._infrastructure_failure(reasons):
            return False
        seed = int(job[0])
        task_id = str(row.get("task_id") or f"uwg_{domain_label}_{seed}")
        record, decision = decide_publish_failure(
            ledger["seeds"].get(str(seed)),
            rebuild_limit=PUBLISH_REBUILD_ATTEMPTS,
            reasons=reasons, task_id=task_id)
        ledger["seeds"][str(seed)] = record
        _save_ledger()
        row["publish_attempts"] = record["failures"]
        if decision == "rebuild" and len(accepted) < count:
            jobs.append(job)
            runtime._log_event(
                "seed_research_rebuild", seed=seed, task_id=task_id,
                failures=record["failures"],
                rebuild_limit=PUBLISH_REBUILD_ATTEMPTS,
                reasons=reasons)
            print(
                f"[evosynth] seed {seed} unpublished "
                f"({record['failures']}/{PUBLISH_REBUILD_ATTEMPTS + 1}); "
                f"re-running research",
                flush=True)
            return True
        if decision == "ban":
            banned.add(seed)
            row["reasons"] = [*reasons, "seed_banned"]
            row["status"] = "seed_banned"
            runtime._log_event(
                "seed_banned", seed=seed, task_id=task_id,
                failures=record["failures"], reasons=row["reasons"])
            print(
                f"[evosynth] seed {seed} banned after "
                f"{record['failures']} unpublished research runs",
                flush=True)
        return False

    def launch() -> None:
        nonlocal next_job
        if len(accepted) >= count:
            return
        _top_up_jobs(next_job)
        while (len(pending) < max(1, workers) and next_job < len(jobs)
               and len(accepted) < count):
            job = jobs[next_job]
            next_job += 1
            pending[task_pool.submit(run, job)] = job
            if len(jobs) - next_job < max(1, workers):
                _top_up_jobs(next_job)

    try:
        launch()
        while pending and not quota_reached:
            for future in as_completed(tuple(pending)):
                job = pending.pop(future)
                seed = int(job[0])
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"task_id": f"attempt_{domain}_{seed}", "accepted": False,
                           "reasons": [f"exception:{type(exc).__name__}:{str(exc)[:500]}"],
                           "spec": {}}
                row["seed"] = seed
                with lock:
                    if (row.get("accepted") and len(accepted) < count
                            and not _bucket_full(seed)):
                        task = row.pop("task")
                        construction_trace = row.pop("construction_trace", None)
                        task_path = tasks_dir / f"{task.task_id}.json"
                        public_path = tasks_dir / f"{task.task_id}.public.json"
                        try:
                            task.save(task_path)
                            if construction_trace is not None:
                                runtime._atomic_write_json(
                                    tasks_dir / f"{task.task_id}.construction_trace.json",
                                    construction_trace)
                            public = task.solver_payload(
                                tool_schemas=seed_compiled[seed].public_schemas(
                                    task.metadata["available_tools"]),
                                answer_schema=certificate_module.ANSWER_ENVELOPE_SCHEMA)
                            # Publish the solver view atomically and verify both
                            # files before counting the task as accepted.
                            fd, temporary = tempfile.mkstemp(
                                prefix=f".{public_path.name}.", suffix=".tmp", dir=public_path.parent)
                            try:
                                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                                    json.dump(public, stream, ensure_ascii=False, indent=2)
                                    stream.write("\n")
                                    stream.flush(); os.fsync(stream.fileno())
                                os.replace(temporary, public_path)
                            except Exception:
                                try: os.unlink(temporary)
                                except OSError: pass
                                raise
                            loaded_task = TaskBundle.from_dict(json.loads(
                                task_path.read_text(encoding="utf-8")))
                            loaded_public = json.loads(public_path.read_text(encoding="utf-8"))
                            if loaded_task.task_id != task.task_id or loaded_public.get("task_id") != task.task_id:
                                raise ValueError("published task round-trip identity mismatch")
                        except Exception as exc:
                            row["accepted"] = False
                            row["reasons"] = [f"publish_failed:{type(exc).__name__}:{str(exc)[:500]}"]
                            runtime._log_event("task_publish_failed", task_id=task.task_id,
                                       reason=row["reasons"][0])
                            if not _retry_or_ban(job, row):
                                _settle(seed, published=False)
                                rejected.append(row)
                                row["status"] = row.get("status") or row_status(row)
                                write_progress(running=True)
                        else:
                            _clear_publish_failure(seed)
                            accepted.append(row)
                            published_arch = str(
                                (seed_reservation.get(seed) or {}).get("archetype") or "")
                            if published_arch:
                                accepted_by_arch[published_arch] += 1
                            row["status"] = "published"
                            # 发布后才记 coverage：archetype / domain / server /
                            # 工具（任务内去重）；落盘同样每发布一个一次。
                            _settle(seed, published=True, tools=row.get("tools") or [])
                            with controller_lock:
                                controller.save()
                            runtime._log_event("task_published", task_id=task.task_id,
                                       task_path=str(task_path), public_path=str(public_path))
                            row["status"] = row_status(row)
                            write_progress(running=True)
                    elif row.get("accepted"):
                        row.pop("task", None)
                        row.pop("construction_trace", None)
                        row["accepted"] = False
                        row["reasons"] = ["surplus_success_not_published"]
                        _settle(seed, published=False)
                        rejected.append(row)
                        row["status"] = row_status(row)
                        write_progress(running=True)
                    else:
                        if not _retry_or_ban(job, row):
                            _settle(seed, published=False)
                            rejected.append(row)
                            row["status"] = row.get("status") or row_status(row)
                            write_progress(running=True)
                if len(accepted) >= count:
                    quota_reached = True
                    runtime._log_event(
                        "publish_target_reached", published=len(accepted),
                        abandoned=len(pending))
                    print(
                        f"[evosynth] published {len(accepted)}/{count}; "
                        f"stopping {len(pending)} in-flight task(s)",
                        flush=True)
                    break
                launch()
                break
    finally:
        for future in pending:
            future.cancel()
        task_pool.shutdown(wait=not quota_reached, cancel_futures=True)
    controller.save()
    write_progress(running=False)
    function_counts = Counter(tool for row in accepted for tool in row.get("tools", ()))
    status_counts = Counter(str(row.get("status") or row_status(row))
                            for row in [*accepted, *rejected])
    report = {
        "domain": domain_label, "tool_info": str(tool_info),
        "servers": list(used_servers),
        "server_selections": dict(sorted(group_selections.items())),
        "requested": count, "attempted": len(accepted) + len(rejected),
        "world_samples_per_initial_world": world_samples,
        "initial_worlds_used": len(worlds),
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / (len(accepted) + len(rejected))
        if accepted or rejected else 0.0,
        "published_complete": len(accepted) == count,
        "attempt_limit": attempt_limit,
        "publish_rebuild_attempts": PUBLISH_REBUILD_ATTEMPTS,
        "banned_seeds": len(banned),
        "banned_seeds_path": str(ledger_path),
        "threshold_passed": len(accepted) >= math.ceil(scheduled[0] * 0.70),
        "status_counts": dict(sorted(status_counts.items())),
        "function_counts": dict(sorted(function_counts.items())),
        "accepted_tasks": accepted, "rejected_attempts": rejected,
        "generator_model": runtime.runtime_config().agent_model,
        "synthesis_models": [runtime.runtime_config().agent_model],
        "solver_model": runtime.runtime_config().agent_model,
        "agent_url": runtime.runtime_config().agent_base_url,
        "world_model_url": runtime._world_model_metadata()["world_observation_endpoint"],
        "world_model": runtime._world_model_metadata()["world_observation_model"],
    }
    (output_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    runtime._log_event("generation_end", domain=domain_label, report=report)
    if quota_reached:
        # 已提交的研究线程不会自己停。报告落盘后直接结束进程，
        # 不再等这些非守护线程把当前的模型请求跑完。
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        os._exit(0)
    return report
