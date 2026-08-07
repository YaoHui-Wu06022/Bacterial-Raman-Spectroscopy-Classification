def log_seblock_summary(runtime, model_path, log):
    log("")
    log("===== SE Module Summary (Compact) =====")

    se_stats = runtime.load_model_se_stats(model_path)
    if not se_stats:
        log("SE stats not found for this model")
        return

    for name, stats in se_stats.items():
        channel_mean = stats["channel_mean"].detach().cpu().numpy()
        channel_std = stats["channel_std"].detach().cpu().numpy()
        channel_min = stats["channel_min"].detach().cpu().numpy()
        channel_max = stats["channel_max"].detach().cpu().numpy()
        sample_count = int(stats.get("sample_count", 0))
        log(
            f"{name}: "
            f"mean={channel_mean.mean():.4f}, "
            f"std={channel_std.mean():.4f}, "
            f"min={channel_min.min():.4f}, "
            f"max={channel_max.max():.4f}, "
            f"samples={sample_count}"
        )
