"""项目级配置、路径与运行时语义。"""

from .paths import (
    DATASET_ROOT,
    PROJECT_ROOT,
    resolve_path,
    stanford_reference_wavenumbers_path,
)

__all__ = [
    "DATASET_ROOT",
    "PROJECT_ROOT",
    "resolve_path",
    "stanford_reference_wavenumbers_path",
]
