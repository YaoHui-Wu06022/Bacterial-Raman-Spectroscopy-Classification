"""源谱变化只标记 init 待重建；保留审核结论及已有 Profile 产物、记录。"""

from __future__ import annotations

from ramanv2.pipeline.state import PipelineState


def invalidate_downstream(
    state: PipelineState,
    date_names: list[str] | tuple[str, ...] | set[str],
) -> dict[str, object]:
    """保留旧 init 的来源清单，避免后续全量重建静默遗漏日期。"""
    dates = {str(name) for name in date_names}
    if dates:
        state.init_generated_dates.difference_update(dates)
        report = state.dataset_builds.get("full_init")
        if isinstance(report, dict):
            report["status"] = "pending"
            report["pending_dates"] = sorted(set(report.get("pending_dates", [])) | dates)
        state.dataset_builds.pop("latest", None)
    return {"invalidated_dates": sorted(dates)}


__all__ = ["invalidate_downstream"]

