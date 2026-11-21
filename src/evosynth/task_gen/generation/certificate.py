"""发布所需的最小答案信封。

答案就是研究结论里的 ``<answer>`` 文本，写入任务元数据的 ``correct_answer``。
"""
from __future__ import annotations

from typing import Any, Mapping

from evosynth.task_gen.common.synthesis import ExecutionTrace, TaskPlan

ANSWER_ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _compile_certificate(_plan: TaskPlan, trace: ExecutionTrace,
                         _outcome: Mapping[str, Any], proposal: Mapping[str, Any],
                         state: Mapping[str, Any]) -> Mapping[str, Any]:
    """记录已经取出的答案。参考调用轨迹不再写入任务包。"""
    del _plan, trace, _outcome
    answer = str(state.get("answer") or proposal.get("answer") or "").strip()
    if not answer:
        raise ValueError("research answer is missing before publication")
    return {"correct_answer": {"answer": answer}}
