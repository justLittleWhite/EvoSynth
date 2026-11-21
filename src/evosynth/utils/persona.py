"""persona 语料采样(带字节偏移索引)。

从 JSONL persona 语料中按字节偏移索引进行可复现采样。仓库通常只分发
采样后的语料文件，完整语料可通过配置指向外部位置。
索引文件是语料同目录的 ``.offsets.npy``。
"""
from __future__ import annotations

import json
import random
import threading
from pathlib import Path

from .paths import PERSONA_FILE

_PERSONA_SAMPLE_LINES = 20_000
_persona_index = None
_persona_index_lock = threading.Lock()


def _index_path() -> Path:
    return Path(str(PERSONA_FILE) + ".offsets.npy")


def _load_persona_index():
    """前 1M 行的字节偏移索引(8MB),首次构建后落盘复用。"""
    global _persona_index
    if _persona_index is not None:
        return _persona_index
    with _persona_index_lock:
        if _persona_index is not None:
            return _persona_index
        import numpy as np
        index_path = _index_path()
        if index_path.exists():
            _persona_index = np.load(index_path, mmap_mode="r")
            return _persona_index
        if not PERSONA_FILE.is_file():
            raise FileNotFoundError(
                f"persona 语料不存在:{PERSONA_FILE}。"
                " 请把采样后的 PersonHub jsonl 放到 data/personhub/,"
                " 或设置 EVOSYNTH_PERSONA_FILE。")
        offsets = np.empty(_PERSONA_SAMPLE_LINES, dtype=np.int64)
        pos = 0
        with open(PERSONA_FILE, "rb") as f:
            for i in range(_PERSONA_SAMPLE_LINES):
                offsets[i] = pos
                line = f.readline()
                if not line:
                    offsets = offsets[:i]
                    break
                pos += len(line)
        try:
            np.save(index_path, offsets)
        except OSError:
            pass  # 目录只读时退化为内存索引
        _persona_index = offsets
        return _persona_index


def sample_persona(seed: int, line_no: int | None = None) -> dict:
    """按 seed 采样一条 persona;``line_no`` 给定时直接取该行。"""
    rng = random.Random(seed)
    target = line_no if line_no is not None else rng.randrange(_PERSONA_SAMPLE_LINES)
    index = _load_persona_index()
    if target >= len(index):
        raise ValueError("persona line out of range")
    with open(PERSONA_FILE, "rb") as f:
        f.seek(int(index[target]))
        line = f.readline()
    return {"line": target, **json.loads(line)}
