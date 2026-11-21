"""开放域运行期常量:世界模型端点与 simulator prompt 路径。

集中定义世界模型运行时使用的端点配置和 simulator prompt 路径。
"""
from __future__ import annotations

from pathlib import Path

from evosynth.utils import prompts_dir

__all__ = ["WM_PROMPT_PATH"]

# 世界模型(Qwen-AgentWorld)simulator 的 system prompt,mcp 域版本。
WM_PROMPT_PATH: Path = prompts_dir() / "mcp" / "system_prompt.txt"
