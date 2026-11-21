"""集中管理项目路径、外部数据目录和输出目录。

所有路径都基于项目位置或显式环境变量解析，不依赖当前工作目录，也不包含
机器专属的绝对路径。
"""
from __future__ import annotations

import os
from pathlib import Path

# ─── 项目根 ──────────────────────────────────────────────────────────────────

# src/evosynth/utils/paths.py → 项目根 evosynth/
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(*parts: str) -> Path:
    """项目内路径,不依赖 cwd。"""
    return PROJECT_ROOT.joinpath(*parts)


# ─── 输出目录 ────────────────────────────────────────────────────────────────

OUTPUT_ROOT = Path(os.environ.get("EVOSYNTH_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))


def output_path(*parts: str) -> Path:
    return OUTPUT_ROOT.joinpath(*parts)


# ─── 外部数据 ────────────────────────────────────────────────────────────────

# PersonHub 采样语料(20k jsonl + 同目录偏移索引),默认在项目 data/personhub/。
PERSONA_FILE = Path(os.environ.get(
    "EVOSYNTH_PERSONA_FILE",
    str(project_path("data", "personhub", "elite_personas_20k.jsonl"))))

# 世界池种子目录:index_*.jsonl 的 source_file 字段按这个根解析。
# 池种子目录体量大,不随仓库分发,默认指回原工作区。
WORLD_POOL_SOURCES = Path(os.environ.get(
    "EVOSYNTH_WORLD_POOL_SOURCES",
    ">"))

def tool_info_dir() -> Path:
    """域工具 schema 的快照目录(项目内自带,可被 env 覆盖)。"""
    override = os.environ.get("EVOSYNTH_TOOL_INFO_DIR")
    return Path(override) if override else project_path("data", "tool_info")


def prompts_dir() -> Path:
    override = os.environ.get("EVOSYNTH_PROMPTS_DIR")
    return Path(override) if override else project_path("prompts")


def repo_path(relative: str) -> Path:
    """把仓内相对路径(如 index_fs.jsonl 里的 source_file)解析成绝对路径。"""
    return WORLD_POOL_SOURCES / relative
