"""analysis 对评估实验上下文的只读访问边界。"""

from __future__ import annotations

from pathlib import Path

from ramanv2.evaluation.context import EvaluationContext, load_evaluation_context, resolve_level_name


def load_analysis_context(source_dir: Path | str) -> EvaluationContext:
    """读取固定实验快照、数据索引和 train/val 切分。"""
    return load_evaluation_context(source_dir)


def resolve_analysis_level(context: EvaluationContext, level_value: str) -> str:
    """按数据索引规范并校验待分析层级名称。"""
    return resolve_level_name(context, level_value)
