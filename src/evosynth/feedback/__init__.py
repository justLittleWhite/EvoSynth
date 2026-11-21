"""Benchmark 驱动的反馈闭环。

三步：单题无参考诊断（diagnose）→ 供需交叉审计（supply + audit）→
基础池 + 靶向变体池（pools / evolve）。入口见 ``python -m evosynth.feedback.run``。
"""
