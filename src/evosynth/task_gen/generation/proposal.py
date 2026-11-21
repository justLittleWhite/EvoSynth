"""从研究结论取出题面和答案，并做一次题目一致性核查。"""
from __future__ import annotations

import re
from typing import Any, Mapping

from openai import OpenAI

from evosynth.utils import extract_json
from evosynth.task_gen.generation import prompt_loader
from evosynth.task_gen.common.synthesis import ExecutionTrace, TaskPlan

from . import runtime

_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.I | re.S)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
_EVIDENCE_RE = re.compile(r"<evidence>\s*(.*?)\s*</evidence>", re.I | re.S)
_OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.I | re.S)
_COHERENCE_RETRY = (
    "Return the verdict inside <output></output>. The tag must contain JSON "
    "with exactly two fields: \"coherent\" (boolean) and \"reason\" (string). "
    "No other fields."
)


def _task_and_answer(conclusion: str) -> tuple[str, str, list[str]]:
    """Extract the task plus the lightweight evidence block from the conclusion."""
    question_match = _QUESTION_RE.search(conclusion or "")
    answer_match = _ANSWER_RE.search(conclusion or "")
    question = " ".join(question_match.group(1).split()).strip() if question_match else ""
    answer = " ".join(answer_match.group(1).split()).strip() if answer_match else ""
    if not question or not answer:
        raise runtime._ResearchAttemptIncomplete(
            "research conclusion is missing a non-empty <question> or <answer>")
    evidence_match = _EVIDENCE_RE.search(conclusion or "")
    evidence = [line.strip() for line in (evidence_match.group(1).splitlines()
                                          if evidence_match else ())
                if line.strip()]
    return question, answer, evidence


# 宽松解析：箭头接受 ``<-`` / ``←`` / ``<=`` 等写法，``call_id=``、``field=`` 前缀可省，
# 分隔符接受逗号或分号。过严的格式只会把内容正确的 evidence 误杀。
_EVIDENCE_LINE_RE = re.compile(
    r"^[-*•]\s*(?P<value>.+?)\s*(?:<-+|<=+|←|⟵|⬅|\u27f5)\s*"
    r"(?:call[_ ]?id\s*[=:]\s*)?`?(?P<call>q\d+)`?\s*[,;]\s*"
    r"(?:field\s*[=:]\s*)?(?P<field>.+?)\s*$", re.I)


def evidence_format_problems(trace: ExecutionTrace,
                             evidence: list[str]) -> str:
    """只检查 evidence 的格式与调用引用（不检查数量）；无问题返回空串。

    供研究阶段在收尾时让 agent 当场修正，而不是等到入库才整题拒绝。
    """
    if not evidence:
        return "the <evidence> block is missing or empty"
    by_id = {call.identity: call for call in trace.calls}
    problems: list[str] = []
    for line in evidence:
        match = _EVIDENCE_LINE_RE.match(line)
        if match is None:
            problems.append(f"malformed evidence line: {line[:120]}")
            continue
        call_id = match.group("call").lower()
        call = by_id.get(call_id)
        if call is None:
            problems.append(f"{call_id} does not exist")
        elif not call.success:
            problems.append(f"{call_id} is a FAILED call and cannot be cited")
    return "; ".join(dict.fromkeys(problems))[:1500]


_MIN_EVIDENCE_CALLS = 5
_MIN_EVIDENCE_CALLS_FLOOR = 3


def min_calls_of(value: Any) -> int:
    """归一化 archetype 的 min_distinct_calls：缺省 5，下限 3。"""
    try:
        return max(_MIN_EVIDENCE_CALLS_FLOOR, int(value))
    except (TypeError, ValueError):
        return _MIN_EVIDENCE_CALLS


def validate_evidence(*, trace: ExecutionTrace, evidence: list[str],
                      min_calls: int = _MIN_EVIDENCE_CALLS) -> tuple[bool, str]:
    """校验 evidence 的引用结构：格式、成功调用、至少 min_calls 个不同 call_id。"""
    min_calls = min_calls_of(min_calls)
    if not evidence:
        return False, "missing evidence block"
    by_id = {call.identity: call for call in trace.calls}
    source_ids: set[str] = set()
    for line in evidence:
        match = _EVIDENCE_LINE_RE.match(line)
        if match is None:
            return False, f"malformed evidence line: {line[:200]}"
        value = match.group("value").strip()
        call_id = match.group("call").lower()
        field = match.group("field").strip()
        call = by_id.get(call_id)
        if call is None or not call.success:
            return False, f"evidence cites unsuccessful or unknown call {call_id}"
        if not value or not field:
            return False, "evidence value and field must be non-empty"
        source_ids.add(call_id)
    if len(source_ids) < min_calls:
        return False, (
            f"evidence cites {len(source_ids)} distinct successful "
            f"call(s); computing the answer must require observations from "
            f"at least {min_calls} distinct successful calls. "
            "Add a natural condition that needs an independent observation "
            "from another call, and cite that call.")
    return True, ""


def _coherence_verdict(raw: str) -> dict[str, Any] | None:
    """只接受 <output> 标签内、且恰好含 coherent/reason 的 JSON 对象。"""
    match = _OUTPUT_RE.search(raw or "")
    if match is None:
        return None
    parsed = extract_json(match.group(1))
    audit = dict(parsed) if isinstance(parsed, Mapping) else {}
    if (set(audit) == {"coherent", "reason"}
            and isinstance(audit.get("coherent"), bool)
            and isinstance(audit.get("reason"), str)):
        return audit
    return None


def _audit_task_coherence(
        task_text: str, answer: str, client: OpenAI,
        state: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    """Task Coherence / Semantic Consistency：一次模型核查内部一致性。

    只裁决三件事:条件之间是否矛盾、题面措辞是否自然明确、
    question→conditions→answer 是否语义指向同一件事。不评价题目好坏、
    难度或像不像 benchmark。判决只从 <output> 标签内的 JSON 读取。
    两轮都拿不到合法判决时失闭。
    """
    prompt = prompt_loader.render(
        "task_coherence_audit.txt",
        task_text=task_text,
        answer=answer,
    )
    messages = [{"role": "user", "content": prompt}]
    attempts: list[dict[str, Any]] = []
    for attempt in range(2):
        response = runtime._chat(
            client, model=runtime.runtime_config().agent_model,
            temperature=0, max_tokens=4096, messages=messages,
            extra_body={"enable_thinking": True})
        message = response.choices[0].message
        raw = message.content or ""
        audit = _coherence_verdict(raw)
        valid = audit is not None
        debug = {
            "attempt": attempt + 1,
            "finish_reason": str(getattr(response.choices[0], "finish_reason", "") or ""),
            "content": raw,
            "reasoning_content": getattr(message, "reasoning_content", None) or "",
            "validation_status": "ok" if valid else "invalid_contract",
        }
        attempts.append(debug)
        coherence_turns = state.setdefault("coherence_turns", [])
        coherence_turns.append({
            "content": raw,
            "reasoning_content": debug["reasoning_content"],
        })
        state["task_coherence_debug"] = {**debug, "attempts": attempts}
        if audit is not None:
            return (bool(audit["coherent"]), str(audit["reason"])[:500],
                    state["task_coherence_debug"])
        if attempt == 0:
            messages.extend([
                {"role": "assistant", "content": raw},
                {"role": "user", "content": _COHERENCE_RETRY},
            ])
    return (False, "task_coherence_check_returned_no_valid_verdict",
            state["task_coherence_debug"])


def _propose(plan: TaskPlan, trace: ExecutionTrace, _outcome: Mapping[str, Any],
             _client: OpenAI, state: dict[str, Any]) -> Mapping[str, Any]:
    """题面和答案都来自研究结论；这里不再重编译结果。"""
    del plan
    question, answer, evidence = _task_and_answer(trace.conclusion)
    state["task_text"] = question
    state["answer"] = answer
    state["evidence"] = evidence
    state["naturalized_task"] = question
    state["private_values"] = [answer]
    state["public_values"] = [
        dict(call.arguments) for call in trace.relevant_successful_calls]
    return {"task_instruct": question, "answer": answer,
            "evidence": evidence}


def _naturalize(_plan: TaskPlan, proposal: Mapping[str, Any]) -> str:
    return str(proposal.get("task_instruct") or "").strip()
