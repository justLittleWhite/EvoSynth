"""EvoSynth 研究提示、真实执行与证据轨迹处理。"""
from __future__ import annotations

from hashlib import sha256
import json
import random
import time
from typing import Any, Mapping, Sequence

from openai import OpenAI

from evosynth.task_gen.generation import prompt_loader
from evosynth.task_gen.common.synthesis import (
    CallRecord, DerivationGraph, DerivationNode, ExecutionTrace, TaskPlan,
)
from evosynth.task_gen.world.backend import (
    GenericEvoSynthBackend, all_dependency_tokens, canonical_json,
    infer_runtime_dependencies, normalize_arguments,
)
from evosynth.task_gen.world.controlled import (
    WorldInfrastructureError, WorldMaterializationError,
)
from evosynth.task_gen.generation.tool_info import CompiledOpenDomain

from . import runtime

MAX_RESEARCH_ATTEMPTS = 2

def _initial_world_payload(initial_world: Any) -> dict[str, Any]:
    if initial_world is None:
        return {}
    if hasattr(initial_world, "to_dict"):
        payload = initial_world.to_dict()
    elif isinstance(initial_world, Mapping):
        payload = dict(initial_world)
    else:
        return {}
    payload.pop("version", None)
    return payload


def _world_entities(initial_world: Any) -> list[dict[str, Any]]:
    raw = list(_initial_world_payload(initial_world).get("entities") or [])
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def allocate_seed_entities(initial_world: Any,
                           task_seeds: Sequence[int]) -> dict[int, dict[str, Any]]:
    """同一初始世界内不放回地分配调查起点。

    任务并发启动，所以分配必须发生在研究开始之前。实体用过一次后，
    这个世界里的后续任务不能再抽到它。实体不够时直接失败，不再复用。
    """
    entities = _world_entities(initial_world)
    seeds = [int(seed) for seed in task_seeds]
    if len(seeds) > len({seed for seed in seeds}):
        raise ValueError("duplicate task seeds in one world entity allocation")
    if len(seeds) > len(entities):
        raise ValueError(
            f"initial world has {len(entities)} entities for {len(seeds)} tasks; "
            "refusing to sample an entity twice")
    rng = random.Random(min(seeds) if seeds else 0)
    order = list(range(len(entities)))
    rng.shuffle(order)
    chosen = [entities[index] for index in order[:len(seeds)]]
    return {seed: entity for seed, entity in zip(sorted(seeds), chosen)}


def _sample_seed_entity(initial_world: Any, rng_seed: int,
                        assigned: Mapping[str, Any] | None = None) -> str:
    """返回调查起点。同一世界的批量任务必须传入已分配实体。"""
    if assigned:
        return json.dumps(dict(assigned), ensure_ascii=False, indent=2)
    entities = _world_entities(initial_world)
    if not entities:
        return ""
    picked = random.Random(rng_seed).choice(entities)
    return json.dumps(picked, ensure_ascii=False, indent=2)


_RESEARCH_SYSTEM = "You are a helpful assistant."


def _research_user(plan: TaskPlan, compiled: CompiledOpenDomain,
                   initial_world: Any = None,
                   prompt_name: str = "research_agent.txt") -> str:
    """研究任务说明:以这个人的研究助理身份,从种子实体出发完成调查。

    不注入合成器元信息、规模区间或身份披露片段。循环里的步数上限只防
    死循环,不写进提示。研究代理看到的是该域全部知识工具。
    提示只放入一个实体,不放入整份世界。批量生成时该实体由世界内不放回
    分配决定;单任务研究仍按 plan.seed 抽取。
    """
    del compiled
    return prompt_loader.render(
        prompt_name,
        persona=plan.persona.text,
        seed=_sample_seed_entity(
            initial_world, plan.seed,
            assigned=plan.payload.get("seed_entity")
            if isinstance(plan.payload, Mapping) else None),
        archetype=plan.archetype,
        archetype_desc=plan.archetype_desc,
        verify=plan.raw_spec.get("archetype_verify", ""),
        min_calls=_plan_min_calls(plan),
    )


def _plan_min_calls(plan: TaskPlan) -> int:
    """该任务 archetype 的 min_distinct_calls（feedback 按 Teacher Acc 调节，默认 5）。"""
    from . import proposal
    return proposal.min_calls_of(plan.raw_spec.get("archetype_min_distinct_calls"))

def evidence_topology_for_trace(calls: Sequence[CallRecord]) -> dict[str, Any]:
    """事后判定证据子轨迹的 evidence_topology(执行后唯一诚实的基础)。

    Evidence topology is derived from actual runtime dependencies. A possible
    edge in the tool graph does not imply that a trajectory transferred a value;
    only dependencies proven from call arguments are recorded.

    判据:证据见证内存在任一调用的父节点也在见证内(即存在一条真实的
    handoff 边)就是 ``linked_retrieval``,否则 ``independent_observations``。
    只观测不裁决:该值只写进任务元数据与审计结果供分析/筛选,没有任何门禁
    核对它(设计文档 §5.2 的目标是「不因工具图有边就加详情调用」,不是重新
    引入一条形态门禁)。
    """
    by_id = {call.identity: call for call in calls}
    handoffs = [
        {"from_call": parent, "to_call": call.identity,
         "from_tool": by_id[parent].tool, "to_tool": call.tool}
        for call in calls for parent in call.depends_on if parent in by_id
    ]
    if not handoffs:
        return {"kind": "independent_observations", "handoffs": []}
    return {
        "kind": "linked_retrieval",
        "handoffs": sorted(handoffs, key=lambda item: (item["to_call"],
                                                       item["from_call"])),
    }

def _assistant_turn(message: Any) -> dict[str, Any]:
    """保留模型这一轮的正文和推理，不截断。"""
    content = getattr(message, "content", None)
    reasoning = getattr(message, "reasoning_content", None)
    extra = getattr(message, "model_extra", None) or {}
    if reasoning is None and isinstance(extra, dict):
        reasoning = extra.get("reasoning_content")
    if hasattr(message, "model_dump"):
        dumped = message.model_dump()
        if content is None:
            content = dumped.get("content")
        if reasoning is None:
            reasoning = dumped.get("reasoning_content")
    tool_calls = []
    for call in list(getattr(message, "tool_calls", None) or []):
        function = getattr(call, "function", None)
        tool_calls.append({
            "id": getattr(call, "id", None),
            "name": getattr(function, "name", None),
            "arguments": getattr(function, "arguments", None),
        })
    return {
        "content": content if content is not None else "",
        "reasoning_content": reasoning if reasoning is not None else "",
        "tool_calls": tool_calls,
    }


def _dependency_cursor(trace: ExecutionTrace) -> tuple[list[tuple[CallRecord, Any]], set[str]]:
    """从已有轨迹恢复值传递记账，便于同一段对话继续调用工具。"""
    previous: list[tuple[CallRecord, Any]] = []
    supplied: set[str] = set()
    for record in trace.calls:
        if not (record.success and record.relevant):
            continue
        try:
            parsed = json.loads(record.observation)
        except Exception:
            parsed = record.observation
        previous.append((record, parsed))
        supplied.update(all_dependency_tokens(record.arguments))
    return previous, supplied


def continue_research_after_review(
        plan: TaskPlan, backend: GenericEvoSynthBackend,
        compiled: CompiledOpenDomain, client: OpenAI,
        state: dict[str, Any], reason: str) -> ExecutionTrace:
    """在原研究对话末尾追加一轮评审意见，让助理重写题和答案。"""
    messages = state.get("research_messages")
    trace = state.get("research_trace")
    if not isinstance(messages, list) or not isinstance(trace, ExecutionTrace):
        raise runtime._ResearchAttemptIncomplete(
            "research dialogue is unavailable for a coherence revision")
    return _execute_research_once(
        plan, backend, compiled, client, state,
        resume_messages=messages,
        resume_trace=trace,
        resume_user=prompt_loader.render(
            "research_coherence_revise.txt", reason=str(reason)[:2000],
            min_calls=_plan_min_calls(plan)),
    )


def _conclusion_problem(conclusion: str, trace: ExecutionTrace) -> str:
    """结论能否直接被下游使用；可用返回空串，否则返回给 agent 看的具体问题。"""
    from . import proposal
    if not conclusion.strip():
        return "the reply was empty"
    try:
        _question, _answer, evidence = proposal._task_and_answer(conclusion)
    except runtime._ResearchAttemptIncomplete as exc:
        return str(exc)
    return proposal.evidence_format_problems(trace, evidence)


def _repair_conclusion(conclusion: str, *, trace: ExecutionTrace,
                       messages: list[dict[str, Any]], client: OpenAI,
                       assistant_turns: list[Any], enable_thinking: bool,
                       max_repairs: int = 2, min_calls: int = 5) -> str:
    """结论缺标签、evidence 格式错误或引用失败调用时，当场让 agent 修正。

    只改写收尾输出，不重跑调查、不调用工具；修不好就把原结论交给下游校验。
    """
    successful = ", ".join(call.identity for call in trace.calls if call.success) or "none"
    for _ in range(max_repairs):
        problem = _conclusion_problem(conclusion, trace)
        if not problem:
            return conclusion
        if conclusion.strip():
            messages_tail = {"role": "user", "content": prompt_loader.render(
                "research_finalize_repair.txt", problem=problem,
                successful=successful, min_calls=min_calls)}
        else:
            messages_tail = {"role": "user", "content": prompt_loader.render(
                "research_finalize_repair.txt", problem="the reply was empty",
                successful=successful, min_calls=min_calls)}
        messages.append(messages_tail)
        response = runtime._chat(
            client, model=runtime.runtime_config().agent_model,
            messages=messages, temperature=0.1,
            max_tokens=runtime.GENERATION_MAX_TOKENS,
            extra_body={"enable_thinking": enable_thinking})
        reply = response.choices[0].message
        assistant_turns.append(_assistant_turn(reply))
        messages.append(runtime._message_dict(reply))
        repaired = str(reply.content or "").strip()
        if repaired:
            conclusion = repaired
    return conclusion


def _execute_research_once(plan: TaskPlan, backend: GenericEvoSynthBackend,
                           compiled: CompiledOpenDomain, client: OpenAI,
                           state: dict[str, Any] | None = None,
                           retry_feedback: str = "",
                           user_prompt_name: str = "research_agent.txt",
                           enable_thinking: bool = True,
                           resume_messages: list[dict[str, Any]] | None = None,
                           resume_trace: ExecutionTrace | None = None,
                           resume_user: str = "") -> ExecutionTrace:
    state = state if state is not None else {}
    assistant_turns = state.setdefault("assistant_turns", [])
    if resume_messages is None:
        assistant_turns.clear()
    # 研究代理默认看到该域全部知识工具。raw_spec 里的 connected_tools
    # 现在就是这份全集；字段缺失时同样兜底为全部工具。
    connected = list(plan.raw_spec.get("connected_tools") or ())
    selected = connected or [tool.public_name for tool in compiled.tools]
    schemas = compiled.public_schemas(selected)
    if resume_messages is not None:
        messages = resume_messages
        trace = resume_trace if resume_trace is not None else ExecutionTrace()
        trace.conclusion = ""
        if resume_user:
            messages.append({"role": "user", "content": resume_user})
        previous, supplied = _dependency_cursor(trace)
    else:
        messages = [
            {"role": "system", "content": _RESEARCH_SYSTEM},
            {"role": "user", "content": _research_user(
                plan, compiled, backend.initial_world, prompt_name=user_prompt_name)},
        ]
        if retry_feedback:
            messages.append({"role": "user", "content": prompt_loader.render(
                "research_agent_retry.txt", feedback=retry_feedback[:4000])})
        trace = ExecutionTrace()
        previous = []
        supplied = set()
    final_conclusion = ""
    consecutive_failures = 0
    forced_continuations = 0
    force_tool_next = False

    # connected_tools 子集内的工具声明始终可见。执行代理只依据当前真实目的、
    # 工具声明和已经返回的观测选择动作；数据集覆盖率和跨轮历史绝不进入其
    # 上下文——分布层面的防塌缩只发生在采样 connected_tools 子集那一步。

    def coverage_gaps() -> list[str]:
        """返回「答案是否仍缺少事实」的缺口,而不是合同义务的缺口。

        当前实现:这里只保留一条机械缺口——完全没有成功观测。采样
        答案属性接口尚未成功的那条检查随 answer_property 轴一并删除:任务
        已经不再钉死终局 (工具,字段),没有哪个接口是「答案的真实来源」,
        强求某个工具必然成功只会把自由探索重新变成合同覆盖。其它(最低观测
        数、合同工具覆盖、链深度、导航 handoff)早先删除:它们是执行前合同
        残留,会为凑数逼出无价值调用。目标不是"满足合同",而是"答案已经可以
        确定"。

        提前收尾是可恢复的:返回非空时执行循环会强制再发一次合法工具调用。
        """
        successful = trace.relevant_successful_calls
        gaps: list[str] = []
        if not successful:
            gaps.append("no knowledge observation has been attempted")
        return gaps
    # 档位调用区间不再切断调查。24 只是进程安全上限,不写进提示,
    # 也不作为「预算用尽」的收束理由。
    step_limit = 28
    wrapup_at = step_limit - 8
    wrapup_sent = False
    for step_index in range(step_limit):
        runtime._check_task_deadline("research loop")
        if step_index >= wrapup_at and not wrapup_sent:
            wrapup_sent = True
            messages.append({"role": "user", "content": prompt_loader.render(
                "research_wrapup_reminder.txt",
                used=str(len(trace.calls)), limit=str(step_limit))})
        # 只暴露 connected_tools 子集内的声明；执行代理根据当前观测自主选择。
        active_schemas = schemas
        response = runtime._chat(
            client, model=runtime.runtime_config().agent_model,
            messages=messages, tools=active_schemas,
            tool_choice="required" if force_tool_next else "auto",
            parallel_tool_calls=False,
            temperature=0.35, max_tokens=runtime.GENERATION_MAX_TOKENS,
            extra_body={"enable_thinking": enable_thinking})
        force_tool_next = False
        message = response.choices[0].message
        assistant_turns.append(_assistant_turn(message))
        messages.append(runtime._message_dict(message))
        if not message.tool_calls:
            missing = coverage_gaps()
            if missing and forced_continuations < step_limit:
                messages.append({"role": "user", "content": prompt_loader.render(
                    "research_continue.txt", missing="; ".join(missing))})
                forced_continuations += 1
                force_tool_next = True
                continue
            final_conclusion = str(message.content or "").strip()
            break
        raw_call = message.tool_calls[0]
        public_name = str(raw_call.function.name)
        try:
            arguments = json.loads(raw_call.function.arguments or "{}")
            if not isinstance(arguments, Mapping):
                raise TypeError("tool arguments must be an object")
        except Exception as exc:
            observation = f"Error: invalid arguments: {exc}"
            trace.calls.append(CallRecord(
                len(trace.calls) + 1, public_name, {}, observation, False,
                "execution", "invalid_arguments", call_id=f"q{len(trace.calls) + 1}",
                relevant=True, kind="observation", state_changed=False))
            messages.append({"role": "tool", "tool_call_id": raw_call.id,
                             "content": observation})
            consecutive_failures += 1
            if consecutive_failures >= runtime.MAX_CONSECUTIVE_TOOL_FAILURES:
                raise runtime._ResearchAttemptIncomplete(
                    "consecutive invalid tool calls exhausted recovery budget")
            continue
        try:
            tool = compiled.by_public_name(public_name)
            arguments = normalize_arguments(tool, arguments)
        except Exception as exc:
            observation = f"Error: {type(exc).__name__}: {str(exc)[:1200]}"
            trace.calls.append(CallRecord(
                len(trace.calls) + 1, public_name, dict(arguments), observation, False,
                "execution", "invalid_tool_call", call_id=f"q{len(trace.calls) + 1}",
                relevant=True, kind="observation", state_changed=False))
            messages.append({"role": "tool", "tool_call_id": raw_call.id,
                             "content": observation})
            consecutive_failures += 1
            if consecutive_failures >= runtime.MAX_CONSECUTIVE_TOOL_FAILURES:
                raise runtime._ResearchAttemptIncomplete(
                    "consecutive invalid tool calls exhausted recovery budget")
            continue
        # ②known_inputs 传递(当前实现后简化):题面由研究代理自己形成,
        # 执行期不存在「题面已给定的输入」这一真值,因此这里传空表。发现性豁免
        # 由此整体生效:任何被观测披露的值都算真实值传递,不因它同时出现在
        # 用户题面中而失边(题面本来就在执行之后才写)。
        dependencies = infer_runtime_dependencies(
            current_arguments=arguments, previous=previous,
            previously_supplied=supplied, known_inputs=[])
        # 工具声明可能包含必填字段；模型第一次尝试时漏填字段
        # 不应让整个合成尝试崩溃。把后端校验异常转成普通工具错误回传给
        # agent，由下一轮自行修正参数。这一层不依赖任何领域或工具名称。
        observation = "Error: tool call did not run"
        row = None
        success = False
        for recovery_attempt in range(runtime.MAX_TOOL_RECOVERY_ATTEMPTS):
            runtime._check_task_deadline("tool materialization")
            try:
                observation = backend.call(public_name, arguments, source="execution")
                row = backend.last_row
                success = not observation.startswith("Error:") and row is not None
            except (WorldInfrastructureError, WorldMaterializationError):
                raise
            except Exception as exc:
                observation = f"Error: {type(exc).__name__}: {str(exc)[:1200]}"
                row = None
                success = False
            runtime._log_event("tool_call", phase="execution", tool=public_name,
                       arguments=dict(arguments), observation=observation,
                       success=success, recovery_attempt=recovery_attempt + 1,
                       backend_event=(backend.call_log[-1]
                                      if backend.call_log else None))
            if success:
                break
            if not runtime._recoverable_tool_failure(observation):
                break
            if recovery_attempt + 1 < runtime.MAX_TOOL_RECOVERY_ATTEMPTS:
                time.sleep(min(2, recovery_attempt + 1))
        relevant = tool.knowledge
        call_id = f"q{len(trace.calls) + 1}"
        produces = tuple(
            "claim:" + sha256(canonical_json([
                call_id, item.get("predicate"), item.get("row"), item.get("value")
            ]).encode()).hexdigest()[:20]
            for item in (row.claims if row is not None else ()))
        record = CallRecord(
            len(trace.calls) + 1, public_name, dict(arguments), observation, success,
            "execution", str(backend.call_log[-1].get("provenance", "error"))
            if backend.call_log else "error", call_id=call_id,
            depends_on=dependencies if relevant else (), produces=produces,
            relevant=relevant, kind="observation" if relevant else "reasoning",
            state_changed=False)
        trace.calls.append(record)
        if success and relevant:
            consecutive_failures = 0
            try:
                parsed_observation = json.loads(observation)
            except Exception:
                parsed_observation = observation
            previous.append((record, parsed_observation))
            # ②用全集记账(当前版本):agent 已供给过的短值被回显后不误判为新发现。
            supplied.update(all_dependency_tokens(arguments))
        elif not success:
            # 世界侧材质化失败不是 agent 的调用错误：它有独立的连续失败
            # 上限（backend 连续 ≥3 次升级 WorldMaterializationError 放弃
            # seed），不占用 agent 的连续失败预算，调查可换参数继续。
            last_provenance = str(
                backend.call_log[-1].get("provenance") or ""
            ) if backend.call_log else ""
            if last_provenance != "materialization_error":
                consecutive_failures += 1
        messages.append({"role": "tool", "tool_call_id": raw_call.id,
                         "content": observation[:16000]})
        if consecutive_failures >= runtime.MAX_CONSECUTIVE_TOOL_FAILURES:
            raise runtime._ResearchAttemptIncomplete(
                "consecutive tool failures exhausted recovery budget")
    # 与 Closed World 直接保留执行 summary 一样，EvoSynth 保留
    # 执行 Agent 基于观测给出的结论，不另起完成结果审计回路。
    if not final_conclusion:
        budget_user = {"role": "user", "content": prompt_loader.load_prompt(
            "research_budget_exhausted.txt")}
        messages.append(budget_user)
        for _closing_try in range(2):
            response = runtime._chat(
                client, model=runtime.runtime_config().agent_model,
                messages=messages,
                temperature=0.1, max_tokens=runtime.GENERATION_MAX_TOKENS,
                extra_body={"enable_thinking": enable_thinking})
            closing = response.choices[0].message
            assistant_turns.append(_assistant_turn(closing))
            messages.append(runtime._message_dict(closing))
            final_conclusion = str(closing.content or "").strip()
            if final_conclusion:
                break
            messages.append({"role": "user", "content": "Your reply was empty. "
                             + prompt_loader.load_prompt("research_budget_exhausted.txt")})
    final_conclusion = _repair_conclusion(
        final_conclusion, trace=trace, messages=messages, client=client,
        assistant_turns=assistant_turns, enable_thinking=enable_thinking,
        min_calls=_plan_min_calls(plan))
    if not final_conclusion:
        # 保留上下文，供下一次尝试从已有观测续跑，而不是整个重来。
        state["research_messages"] = messages
        state["research_trace"] = trace
        raise ValueError("research execution produced no conclusion")
    trace.conclusion = final_conclusion
    state["research_messages"] = messages
    state["research_trace"] = trace
    return trace


def construction_trace_payload(*, task_id: str, state: Mapping[str, Any],
                               trace: ExecutionTrace) -> dict[str, Any]:
    """研究代理完整对话，含每轮 reasoning_content，供发布旁路落盘。"""
    return {
        "task_id": task_id,
        "assistant_turns": list(state.get("assistant_turns") or []),
        "messages": list(state.get("research_messages") or []),
        "calls": [call.to_dict() for call in trace.calls],
        "conclusion": trace.conclusion,
        "evidence": list(state.get("evidence") or ()),
        "coherence_turns": list(state.get("coherence_turns") or []),
    }

def _compile_derivation(_plan: TaskPlan, trace: ExecutionTrace) -> DerivationGraph:
    nodes = tuple(DerivationNode(
        node_id=call.identity, operation=call.tool, inputs=call.depends_on,
        kind="observation", metadata={"arguments": dict(call.arguments)})
        for call in trace.relevant_successful_calls)
    return DerivationGraph(nodes)
