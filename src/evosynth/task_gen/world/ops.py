"""材质化层世界组成算子(world composition operators)注册表。

算子只改变"初始世界与后续观测里存在什么内容"(经 8822 材质化 prompt 实现),
不改变任务语义、发布门禁或答案证明——干扰实体只是世界内容,发布门禁不因
其存在而变。所有算子与领域无关:指令只使用通用语言,不点名任何 server、
工具或字段;注册制,默认全部关闭,通过环境变量 ``EVOSYNTH_WORLD_OPS``
(逗号分隔算子名)开启。

当前算子:
- ``near_miss_entities``: an optional distractor operator currently disabled
  in ``DISABLED_WORLD_OPS``.
  世界时注入 1-2 个与调查焦点实体高相似的干扰实体(近似名称/重叠属性),
  并要求 8822 在后续观测中保持这些干扰实体存活且不与真实体事实合并。
  干扰实体记录进 manifest(构造产物与任务 metadata)。
"""
from __future__ import annotations

import copy
import logging
import os
from typing import Any, Mapping


_LOGGER = logging.getLogger(__name__)


# 初始世界实体 attributes 中的算子标记键/值。构造侧在固定世界前剥离该标记
# (它是构造记账,不是世界事实),同时把被标记实体写入 manifest。
WORLD_OP_MARKER_KEY = "world_op"
NEAR_MISS_MARKER = "near_miss_distractor"


class NearMissEntitiesOp:
    """向初始世界注入 1-2 个高相似干扰实体(近似名称/重叠属性)。"""

    name = "near_miss_entities"

    def initial_world_directive(self) -> str:
        """拼入初始世界构造 PURPOSE 的通用指令(不点名工具或领域)。"""
        return (
            "World composition operator (internal): additionally include one or "
            "two plausible near-miss entities — records that closely resemble the "
            "investigation's focal subject through similar names, overlapping "
            "attributes, nearby identifiers or adjacent categories, yet differ in "
            "at least one decisive fact so careful evidence still distinguishes "
            "them. They are ordinary world content, not the answer and not extra "
            "task requirements. Mark each such entity by adding the attribute "
            f"\"{WORLD_OP_MARKER_KEY}\": \"{NEAR_MISS_MARKER}\" inside its "
            "attributes object; do not mark the true focal subject.")

    def validate_payload(self, payload: Mapping[str, Any]) -> None:
        """机械校验:算子开启时初始世界必须真的带 1+ 个被标记干扰实体。"""
        entities = payload.get("entities")
        marked = 0
        if isinstance(entities, list):
            for item in entities:
                if not isinstance(item, Mapping):
                    continue
                attributes = item.get("attributes")
                if (isinstance(attributes, Mapping) and
                        str(attributes.get(WORLD_OP_MARKER_KEY) or "")
                        == NEAR_MISS_MARKER):
                    marked += 1
        if marked < 1:
            raise ValueError(
                "world op near_miss_entities is active but no entity carries "
                f"attributes.{WORLD_OP_MARKER_KEY}={NEAR_MISS_MARKER!r}")

    def extract_manifest(self, payload: Mapping[str, Any]
                         ) -> tuple[dict[str, Any], dict[str, Any]]:
        """剥离标记并返回 (干净世界 payload, 算子 manifest)。"""
        cleaned = copy.deepcopy(dict(payload))
        distractors: list[dict[str, Any]] = []
        for item in cleaned.get("entities") or ():
            if not isinstance(item, Mapping):
                continue
            attributes = item.get("attributes")
            if not (isinstance(attributes, Mapping) and
                    str(attributes.get(WORLD_OP_MARKER_KEY) or "")
                    == NEAR_MISS_MARKER):
                continue
            attributes = dict(attributes)
            attributes.pop(WORLD_OP_MARKER_KEY, None)
            if isinstance(item, dict):
                item["attributes"] = attributes
            distractors.append({
                "id": str(item.get("id") or ""),
                "type": str(item.get("type") or ""),
                "attributes": copy.deepcopy(attributes),
            })
        manifest = {"op": self.name, "distractor_entities": distractors}
        return cleaned, manifest

WORLD_OPERATORS: dict[str, Any] = {
    NearMissEntitiesOp.name: NearMissEntitiesOp,
}

# Registered operators that are intentionally disabled. Configuration values
# naming these operators are ignored safely; remove a name from this set to
# enable its registered implementation.
DISABLED_WORLD_OPS: frozenset[str] = frozenset({NearMissEntitiesOp.name})


def _curriculum_world_op_names() -> list[str]:
    """从 CURRICULUM_JSON 的 world_ops 段取算子名(feedback 闭环消费点)。

    与 closed world 共享同一份 curriculum:closed 专用算子名(如
    inject_payload/scale_files)不在 evosynth 注册表内,静默忽略;只有
    注册表命中且不在 DISABLED_WORLD_OPS 的名字生效(停用名记日志后忽略)。
    缺文件/坏 JSON 静默降级为空(零侵入)。
    """
    path = os.environ.get("CURRICULUM_JSON", "")
    if not path or not os.path.exists(path):
        return []
    try:
        import json
        with open(path, encoding="utf-8") as handle:
            curriculum = json.load(handle)
    except (OSError, ValueError):
        return []
    names: list[str] = []
    for entry in curriculum.get("world_ops") or ():
        if isinstance(entry, Mapping):
            name = str(entry.get("op") or "").strip()
            if name not in WORLD_OPERATORS:
                continue
            if name in DISABLED_WORLD_OPS:
                _LOGGER.warning(
                    "world op %r requested by CURRICULUM_JSON is registered "
                    "but disabled (user directive 当前版本); ignoring", name)
                continue
            names.append(name)
    return names


def active_world_ops() -> list[Any]:
    """实例化开启的算子:``EVOSYNTH_WORLD_OPS`` 环境变量(显式开关,未知名
    快速失败)∪ CURRICULUM_JSON world_ops 段(feedback 闭环,未注册名忽略)。
    两条通道均受 DISABLED_WORLD_OPS 过滤:已注册但停用的名字跳过并记日志。"""
    raw = os.environ.get("EVOSYNTH_WORLD_OPS", "")
    names = [item.strip() for item in raw.split(",") if item.strip()]
    for name in names:
        if name not in WORLD_OPERATORS:
            raise ValueError(
                f"unknown world op {name!r}; registered: "
                + ",".join(sorted(WORLD_OPERATORS)))
    ops: list[Any] = []
    for name in dict.fromkeys([*names, *_curriculum_world_op_names()]):
        if name in DISABLED_WORLD_OPS:
            _LOGGER.warning(
                "world op %r is registered but disabled (user directive "
                "当前版本); ignoring", name)
            continue
        ops.append(WORLD_OPERATORS[name]())
    return ops


__all__ = ["WORLD_OP_MARKER_KEY", "NEAR_MISS_MARKER", "NearMissEntitiesOp",
           "WORLD_OPERATORS", "DISABLED_WORLD_OPS", "active_world_ops"]
