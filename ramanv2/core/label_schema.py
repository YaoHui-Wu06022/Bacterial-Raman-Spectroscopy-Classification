"""实验与数据集共用的标签方案。"""

from collections.abc import Mapping
from typing import Any


FLAT_LABEL_SCHEMA = "flat_label_v1"
HIERARCHY_LABEL_SCHEMA = "hierarchy_v1"
SUPPORTED_LABEL_SCHEMAS = frozenset({FLAT_LABEL_SCHEMA, HIERARCHY_LABEL_SCHEMA})


def validate_label_schema(values: Mapping[str, Any]) -> None:
    """校验实验产物使用当前的扁平或层级标签方案。"""
    schema = values.get("label_schema")
    if schema not in SUPPORTED_LABEL_SCHEMAS:
        supported = "、".join(sorted(SUPPORTED_LABEL_SCHEMAS))
        raise ValueError(
            f"label_schema 不兼容：{schema!r}；"
            f"当前仅支持 {supported}，请使用新实验目录重新训练。"
        )
