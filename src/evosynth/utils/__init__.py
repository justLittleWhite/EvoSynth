"""EvoSynth shared utility layer and integration boundary.

为任务生成、rollout 和反馈模块提供统一的 JSON、LLM、路径与消息工具。
"""
from .json_utils import extract_json
from .llm_client import chat_with_retry, openai_client
from .llm_config import read_api_key_file, resolve_api_key
from .messages import to_qwen_messages
from .paths import (OUTPUT_ROOT, PERSONA_FILE, PROJECT_ROOT,
                    WORLD_POOL_SOURCES, output_path, project_path, prompts_dir,
                    repo_path, tool_info_dir)
from .persona import sample_persona

__all__ = [
    "extract_json",
    "chat_with_retry", "openai_client",
    "read_api_key_file", "resolve_api_key",
    "to_qwen_messages",
    "OUTPUT_ROOT", "PERSONA_FILE", "PROJECT_ROOT",
    "WORLD_POOL_SOURCES", "output_path", "project_path", "prompts_dir",
    "repo_path", "tool_info_dir",
    "sample_persona",
]
