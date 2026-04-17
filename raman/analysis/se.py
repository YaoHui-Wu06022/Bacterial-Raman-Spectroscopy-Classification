from raman.model import SEBlock1D


def log_seblock_summary(model, log):
    log("")
    log("===== SE Module Summary (Compact) =====")
    for name, module in model.named_modules():
        if isinstance(module, SEBlock1D):
            if module.latest_scale is None:
                log(f"{name}: scale not computed")
                continue

            s = module.latest_scale.mean(dim=0).detach().cpu().numpy()
            log(
                f"{name}: "
                f"mean={s.mean():.4f}, "
                f"std={s.std():.4f}, "
                f"min={s.min():.4f}, "
                f"max={s.max():.4f}"
            )
