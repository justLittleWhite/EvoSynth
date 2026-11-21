"""LLM 调用的重试策略与 Qwen chat template 渲染。

提供 LLM 请求的重试策略、错误分类和 Qwen 消息模板渲染。
"""
from __future__ import annotations

import time

from tenacity import (Retrying, retry_if_exception, stop_after_attempt,
                      wait_random_exponential)

MAX_RETRIES = 5


def _is_retryable_llm_error(exc: BaseException) -> bool:
    """只重试限流、超时、连接失败和服务端错误，不掩盖 4xx 请求错误。"""
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 425, 429} or isinstance(status, int) and status >= 500:
        return True
    name = type(exc).__name__
    if name in {"APITimeoutError", "APIConnectionError", "RateLimitError",
                "InternalServerError", "Timeout", "TimeoutError",
                "ConnectionError"}:
        return True
    message = str(exc).lower()
    return any(token in message for token in (
        "timed out", "timeout", "connection reset", "connection aborted",
        "temporarily unavailable", "rate limit", "too many requests", "429"))


def _log_retry(retry_state) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    delay = retry_state.next_action.sleep if retry_state.next_action else 0
    print(
        f"[llm-retry] attempt={retry_state.attempt_number}/{MAX_RETRIES} "
        f"wait={delay:.1f}s error={type(exc).__name__ if exc else 'unknown'}",
        flush=True,
    )


def chat_with_retry(client, **kwargs):
    """由 tenacity 统一承担请求级重试；每次调用独立退避，适合高并发。"""
    retrying = Retrying(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_random_exponential(multiplier=2, min=2, max=120),
        retry=retry_if_exception(_is_retryable_llm_error),
        before_sleep=_log_retry,
        sleep=time.sleep,
        reraise=True,
    )
    return retrying(client.chat.completions.create, **kwargs)


def openai_client(base_url: str, api_key: str,
                  timeout: int = 600, max_retries: int = 0):
    """建一个禁用了 SDK 内建重试的客户端(重试由 chat_with_retry 负责,
    避免两层退避相乘)。"""
    from openai import OpenAI

    if not base_url or not api_key:
        raise ValueError("base_url and api_key must be explicitly configured")
    return OpenAI(base_url=base_url, api_key=api_key,
                  timeout=timeout, max_retries=max_retries)
