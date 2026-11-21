"""公共 LLM endpoint 兼容常量与显式密钥文件读取。"""
from __future__ import annotations

import os
from pathlib import Path

def resolve_api_key(name: str, *, file: str | Path | None = None,
                    default: str | None = None) -> str:
    """按环境变量、对应的 *_FILE 或显式文件读取 API key。"""
    value = os.environ.get(name)
    if value:
        return value
    key_file = os.environ.get(f"{name}_FILE") or (str(file) if file else None)
    if key_file:
        path = Path(key_file).expanduser()
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(f"API key file is not readable: {path}") from exc
        if not value:
            raise RuntimeError(f"API key file is empty: {path}")
        return value
    if default is not None:
        return default
    raise RuntimeError(f"缺少 API key:{name} 未设置,且 {name}_FILE 指向的密钥文件不可读")


def read_api_key_file(file: str | Path) -> str:
    """读取显式密钥文件；不可读或空文件一律失败。"""
    path = Path(file).expanduser()
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"cannot read API key file: {path}") from exc
    if not value:
        raise ValueError(f"API key file is empty: {path}")
    return value
