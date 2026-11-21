"""把 OpenAI 格式的轨迹渲染成 Qwen chat template(messages 形态)。

每个 assistant turn 都带 ``<think>`` 块(无 reasoning 时渲染空块,保持
Qwen3 no-think 格式统一),工具结果包在 ``<tool_response>``,调用包在
``<tool_call>``。
"""
from __future__ import annotations

import json
from pathlib import Path

QWEN_TOOLS_BLOCK = (Path(__file__).resolve().parents[3] / "prompts/messages/qwen_tools_block.txt").read_text(encoding="utf-8")


def to_qwen_messages(system_text: str, tools: list[dict], raw_log: list[dict]) -> list[dict]:
    """把 OpenAI 轨迹翻译成 Qwen chat template 文本格式的 messages。

    raw_log 条目:{"role": "user"/"assistant"/"tool", "content": str,
                   "tool_call": {"name","arguments"} | None,
                   "reasoning": str | None}
    """
    tool_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    out = [{"role": "system",
            "content": system_text + QWEN_TOOLS_BLOCK.format(tool_lines=tool_lines)}]
    for m in raw_log:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "tool":
            out.append({"role": "user",
                        "content": f"<tool_response>\n{m['content']}\n</tool_response>"})
        elif m["role"] == "assistant":
            # 每个 assistant turn 都带 <think> 块;无 reasoning 时渲染空块
            # (Qwen3 no-think 格式),保证训练数据格式统一
            parts = [f"<think>\n{m.get('reasoning') or ''}\n</think>"]
            if m.get("content"):
                parts.append(m["content"])
            if m.get("tool_call"):
                tc = json.dumps(m["tool_call"], ensure_ascii=False)
                parts.append(f"<tool_call>\n{tc}\n</tool_call>")
            out.append({"role": "assistant", "content": "\n".join(parts)})
    return out
