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
    "DeletedSpectrum",
    "DeletedFolder",
    "scan_deleted_folders",
    "scan_deleted_spectra",
    "restore_deleted_spectra",
    "restore_deleted_folder",
    "initialize_folder_lineage",
    "load_folder_lineage",
    "save_folder_lineage",
    "invalidate_downstream",
    "rebuild_shift_data",
    "load_date_affine_parameters",
    "rebuild_pipeline_records",
]


def __getattr__(name: str):
    if name == "run_affine_calibration":
        from ramanv2.pipeline.calibration import run_affine_calibration

        return run_affine_calibration
    if name in {"initialize_folder_lineage", "load_folder_lineage", "save_folder_lineage"}:
        from ramanv2.pipeline.lineage import initialize_folder_lineage, load_folder_lineage, save_folder_lineage
        return {
            "initialize_folder_lineage": initialize_folder_lineage,
            "load_folder_lineage": load_folder_lineage,
            "save_folder_lineage": save_folder_lineage,
        }[name]
    if name in {"DeletedSpectrum", "DeletedFolder", "scan_deleted_folders", "scan_deleted_spectra", "restore_deleted_spectra", "restore_deleted_folder"}:
        from ramanv2.pipeline.restoration import (
            DeletedFolder,
            DeletedSpectrum,
            restore_deleted_folder,
            restore_deleted_spectra,
            scan_deleted_folders,
            scan_deleted_spectra,
        )
        return {
            "DeletedFolder": DeletedFolder,
            "DeletedSpectrum": DeletedSpectrum,
            "scan_deleted_folders": scan_deleted_folders,
            "scan_deleted_spectra": scan_deleted_spectra,
            "restore_deleted_spectra": restore_deleted_spectra,
            "restore_deleted_folder": restore_deleted_folder,
        }[name]
    if name == "invalidate_downstream":
        from ramanv2.pipeline.invalidation import invalidate_downstream

        return invalidate_downstream
    if name in {"rebuild_shift_data", "load_date_affine_parameters"}:
        from ramanv2.pipeline.shift import load_date_affine_parameters, rebuild_shift_data

        return {
            "rebuild_shift_data": rebuild_shift_data,
            "load_date_affine_parameters": load_date_affine_parameters,
        }[name]
    if name == "rebuild_pipeline_records":
        from ramanv2.pipeline.records import rebuild_pipeline_records

        return rebuild_pipeline_records
    raise AttributeError(name)
