_EXPORTS = {
    "RamanDataset": ("raman.data.loader", "RamanDataset"),
    "resolve_dataset_stage": ("raman.data.paths", "resolve_dataset_stage"),
    "InputPreprocessor": ("raman.data.input", "InputPreprocessor"),
    "build_input_channels": ("raman.data.input", "build_input_channels"),
    "build_model_input": ("raman.data.input", "build_model_input"),
    "build_sg_kernels": ("raman.data.input", "build_sg_kernels"),
    "normalize_spectrum": ("raman.data.input", "normalize_spectrum"),
    "load_arc_intensity": ("raman.data.spectrum", "load_arc_intensity"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'raman.data' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
