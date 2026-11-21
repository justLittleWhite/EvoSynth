"""从 LLM 回复里稳健地抽取 JSON 对象。

模型经常把 JSON 包在 ```json 代码块里,或在前面写一段 <think> 推理,
所以先剥 think 块,再按围栏优先、整段兜底,并在候选串里逐个 "{" 位置尝试解析。
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str | None) -> dict | None:
    """返回第一个可解析的 JSON 对象;全部失败返回 None。"""
    text = _THINK_RE.sub("", text or "")
    for cand in [m.group(1) for m in _FENCE_RE.finditer(text)] + [text]:
        idx = cand.find("{")
        while idx != -1:
            try:
                obj = json.JSONDecoder().raw_decode(cand[idx:])[0]
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            idx = cand.find("{", idx + 1)
    return None
