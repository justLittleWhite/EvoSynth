"""feedback 的 LLM 调用：与两个合成/rollout 脚本一致，显式 base-url / model / key。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from evosynth.utils import chat_with_retry, extract_json, openai_client, read_api_key_file

DEFAULT_BASE_URL = ""
DEFAULT_MODEL = "glm-5.2"
DEFAULT_API_KEY_FILE = Path("")

_SYSTEM = ("You follow the given protocol exactly. Reply with the requested "
           "<output>...</output> block containing a single JSON object and nothing else.")
_OUTPUT_RE = re.compile(r"<output>(.*?)</output>", re.DOTALL | re.IGNORECASE)

_CONFIG: dict[str, str] = {}
# 测试可注入：callable(prompt) -> dict
_OVERRIDE: Callable[[str], dict[str, Any]] | None = None


def configure(*, base_url: str, model: str, api_key: str) -> None:
    if not all(str(v).strip() for v in (base_url, model, api_key)):
        raise ValueError("base_url, model and api_key must be non-empty")
    _CONFIG.update(base_url=base_url, model=model, api_key=api_key)


def configure_from_file(*, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                        api_key_file: Path = DEFAULT_API_KEY_FILE) -> None:
    configure(base_url=base_url, model=model, api_key=read_api_key_file(api_key_file))


def set_override(fn: Callable[[str], dict[str, Any]] | None) -> None:
    global _OVERRIDE
    _OVERRIDE = fn


def parse_output(text: str) -> dict[str, Any] | None:
    match = _OUTPUT_RE.search(text or "")
    return extract_json(match.group(1) if match else text)


MAX_TOKENS_CAP = 32000


def call_json(prompt: str, *, max_tokens: int = 4096, temperature: float = 0.0,
              parse_retries: int = 2) -> dict[str, Any]:
    """一次协议式调用，返回 <output> 中的 JSON；解析失败重试，最终抛 ValueError。"""
    if _OVERRIDE is not None:
        return _OVERRIDE(prompt)
    if not _CONFIG:
        raise RuntimeError("feedback.llm not configured")
    client = openai_client(_CONFIG["base_url"], _CONFIG["api_key"], timeout=600)
    last = "empty response"
    budget = max_tokens
    nudge = ""
    for _ in range(parse_retries + 1):
        response = chat_with_retry(
            client, model=_CONFIG["model"], temperature=temperature, max_tokens=budget,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": prompt + nudge}])
        choice = response.choices[0]
        content = choice.message.content or ""
        reasoning = _reasoning_of(choice.message)
        # 思考模型：答案通常在 content；content 为空或被截断时，退回去 reasoning 里找 <output>。
        finish = getattr(choice, "finish_reason", None)
        parsed = parse_output(content)
        # 只有正常结束才去思考里找 <output>；被截断时思考里的 JSON 多半是示例片段，不能当答案。
        if parsed is None and reasoning and finish != "length":
            parsed = parse_output(reasoning)
        if parsed is not None:
            return parsed
        last = (f"no parsable JSON in <output> (finish={finish}, content_len={len(content)}, "
                f"reasoning_len={len(reasoning)}, head={content[:160]!r}, "
                f"reasoning_tail={reasoning[-300:]!r})")
        if finish == "length":
            budget = min(budget * 2, MAX_TOKENS_CAP)  # 被截断：下一次给更大预算
        nudge = ("\n\nYour previous reply had no valid <output> block. Think briefly, then reply "
                 "with only the <output>{...}</output> block.")
    raise ValueError(last)


def _reasoning_of(message: Any) -> str:
    value = getattr(message, "reasoning_content", None)
    if not value:
        extra = getattr(message, "model_extra", None) or {}
        value = extra.get("reasoning_content") or extra.get("reasoning")
    return value if isinstance(value, str) else ""
