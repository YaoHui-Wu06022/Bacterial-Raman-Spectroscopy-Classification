"""Raman Web 前置流水线的纯 Python 编排接口。"""

from ramanv2.pipeline.state import (
    PIPELINE_STATE_VERSION,
    CleaningState,
    PipelineState,
    load_pipeline_state,
    save_pipeline_state,
)
__all__ = [
    "PIPELINE_STATE_VERSION",
    "CleaningState",
    "PipelineState",
    "load_pipeline_state",
    "save_pipeline_state",
    "run_affine_calibration",
    "run_shift_folder_audit",
    "DeletedSpectrum",
    "scan_deleted_spectra",
    "restore_deleted_spectra",
    "restore_deleted_folder",
    "initialize_folder_lineage",
    "load_folder_lineage",
    "save_folder_lineage",
]


def __getattr__(name: str):
    if name == "run_affine_calibration":
        from ramanv2.pipeline.calibration import run_affine_calibration

        return run_affine_calibration
    if name == "run_shift_folder_audit":
        from ramanv2.pipeline.audit import run_shift_folder_audit

        return run_shift_folder_audit
    if name in {"initialize_folder_lineage", "load_folder_lineage", "save_folder_lineage"}:
        from ramanv2.pipeline.lineage import initialize_folder_lineage, load_folder_lineage, save_folder_lineage
        return {
            "initialize_folder_lineage": initialize_folder_lineage,
            "load_folder_lineage": load_folder_lineage,
            "save_folder_lineage": save_folder_lineage,
        }[name]
    if name in {"DeletedSpectrum", "scan_deleted_spectra", "restore_deleted_spectra", "restore_deleted_folder"}:
        from ramanv2.pipeline.restoration import (
            DeletedSpectrum,
            restore_deleted_folder,
            restore_deleted_spectra,
            scan_deleted_spectra,
        )
        return {
            "DeletedSpectrum": DeletedSpectrum,
            "scan_deleted_spectra": scan_deleted_spectra,
            "restore_deleted_spectra": restore_deleted_spectra,
            "restore_deleted_folder": restore_deleted_folder,
        }[name]
    raise AttributeError(name)
