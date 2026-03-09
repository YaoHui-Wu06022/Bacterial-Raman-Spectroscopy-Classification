import os
import re
import shutil
from pathlib import Path

import numpy as np

from dataset_process.common import (
    build_wn_ref,
    preprocess_single_spectrum,
    read_arc_data,
    save_mean_plot,
)

PACK_EXT = ".npz"

CUT_MIN = 600
CUT_MAX = 1800
TARGET_POINTS = 896
WN_REF = build_wn_ref(CUT_MIN, CUT_MAX, TARGET_POINTS)

ASLS_LAM = 1e5
ASLS_P = 0.01
ASLS_MAX_ITER = 10

MIN_SAMPLES_PER_CLASS = 8
NORM_METHOD = "snv"

PCA_ENABLED = True
PCA_COMPONENTS = 0.95
PCA_CENTER = True
PCA_OUTLIER_RATIO = 0.05


def resolve_path(base_dir, path_value):
    return (Path(base_dir) / path_value).resolve()


def iter_arc_dirs(root_dir):
    root_dir = os.fspath(root_dir)
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        arc_files = [name for name in files if name.lower().endswith(".arc_data")]
        if arc_files:
            yield Path(root), arc_files


def get_prefix(name, prefix_mode):
    if prefix_mode == "letters_sign":
        matched = re.match(r"([A-Za-z]+)([+-])?", name)
        if not matched:
            return None
        return f"{matched.group(1)}{matched.group(2) or ''}"
    if prefix_mode == "letters":
        matched = re.match(r"([A-Za-z]+)", name)
        return matched.group(1) if matched else None
    raise ValueError(f"Unknown prefix mode: {prefix_mode}")


def is_packed_path(path):
    return os.path.isfile(path) and str(path).lower().endswith(PACK_EXT)


def write_arc_data(path, wn, sp, fmt="%.8f"):
    arr = np.column_stack([wn, sp])
    np.savetxt(path, arr, fmt=[fmt, fmt])


class PackedArcDataset:
    def __init__(self, npz_path):
        if not is_packed_path(npz_path):
            raise FileNotFoundError(f"Missing packed file: {npz_path}")
        data = np.load(npz_path)
        self.root_name = (
            str(data["root_name"][0]) if "root_name" in data else "dataset_init"
        )
        self.paths = data["paths"].tolist()
        self.offsets = data["offsets"]
        self.lengths = data["lengths"]
        self.wn_all = data["wn_all"]
        self.sp_all = data["sp_all"]

    def __len__(self):
        return len(self.paths)

    def get(self, index):
        start = int(self.offsets[index])
        length = int(self.lengths[index])
        end = start + length
        rel_path = self.paths[index]
        wn = self.wn_all[start:end]
        sp = self.sp_all[start:end]
        return rel_path, wn, sp

    def iter_samples(self):
        for index in range(len(self.paths)):
            yield self.get(index)


def resolve_init_input(base_dir, profile):
    root_init = resolve_path(base_dir, profile.root_init)
    root_init_pack = resolve_path(base_dir, profile.root_init_pack)

    if root_init.is_dir():
        return root_init
    if is_packed_path(root_init):
        return root_init
    if is_packed_path(root_init_pack):
        return root_init_pack

    raise FileNotFoundError(f"Missing input dir/file: {root_init}")


def iter_init_groups(input_path):
    input_path = Path(input_path)

    if input_path.is_dir():
        for leaf_dir, arc_files in iter_arc_dirs(input_path):
            rel_dir = leaf_dir.relative_to(input_path)
            samples = []
            for fname in arc_files:
                wn, sp = read_arc_data(leaf_dir / fname)
                samples.append((fname, wn, sp))
            yield rel_dir, leaf_dir.name, samples
        return

    packed = PackedArcDataset(input_path)
    grouped = {}

    for rel_path, wn, sp in packed.iter_samples():
        normalized_rel_path = rel_path.replace("\\", "/")
        rel_dir = Path(os.path.dirname(normalized_rel_path) or ".")
        group_key = rel_dir.as_posix()
        if group_key not in grouped:
            grouped[group_key] = {
                "rel_dir": rel_dir,
                "leaf_name": packed.root_name if rel_dir == Path(".") else rel_dir.name,
                "samples": [],
            }
        grouped[group_key]["samples"].append(
            (os.path.basename(normalized_rel_path), wn, sp)
        )

    for group in grouped.values():
        yield group["rel_dir"], group["leaf_name"], group["samples"]


def pack_dataset_init(input_dir, output_path, verbose=True):
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    root_name = input_dir.resolve().name
    paths = []
    offsets = [0]
    wn_chunks = []
    sp_chunks = []

    for root, arc_files in iter_arc_dirs(input_dir):
        for fname in arc_files:
            full_path = root / fname
            wn, sp = read_arc_data(full_path)
            if wn.size == 0 or sp.size == 0:
                continue

            rel_path = full_path.relative_to(input_dir).as_posix()
            paths.append(rel_path)

            wn = wn.astype(np.float32)
            sp = sp.astype(np.float32)
            wn_chunks.append(wn)
            sp_chunks.append(sp)
            offsets.append(offsets[-1] + wn.size)

    if not paths:
        raise RuntimeError(f"No .arc_data files found under {input_dir}")

    wn_all = np.concatenate(wn_chunks, axis=0)
    sp_all = np.concatenate(sp_chunks, axis=0)
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    offsets = offsets[:-1]

    np.savez_compressed(
        output_path,
        root_name=np.asarray([root_name]),
        paths=np.asarray(paths),
        offsets=offsets,
        lengths=lengths,
        wn_all=wn_all,
        sp_all=sp_all,
    )

    if verbose:
        total = int(wn_all.size)
        print(f"[Pack] samples={len(paths)}, points={total}, saved={output_path}")


def unpack_dataset_init(npz_path, output_dir, verbose=True):
    npz_path = Path(npz_path)
    output_dir = Path(output_dir)
    packed = PackedArcDataset(npz_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    restored = 0
    for rel_path, wn, sp in packed.iter_samples():
        out_path = output_dir / Path(rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_arc_data(out_path, wn, sp)
        restored += 1

    if verbose:
        print(f"[Unpack] samples={restored}, restored={output_dir}")


def classify_dataset(profile, base_dir):
    base_dir = Path(base_dir)
    root_process_raw = resolve_path(base_dir, profile.root_process_raw)
    root_process_raw.mkdir(parents=True, exist_ok=True)

    input_path = resolve_init_input(base_dir, profile)

    copied = 0
    if Path(input_path).is_dir():
        for leaf_dir, arc_files in iter_arc_dirs(input_path):
            rel_dir = leaf_dir.relative_to(input_path)
            rel_parent = rel_dir.parent
            leaf_name = leaf_dir.name

            prefix = get_prefix(leaf_name, profile.prefix_mode)
            target_cls = prefix if prefix else leaf_name
            target_dir = (
                root_process_raw / target_cls
                if rel_parent == Path(".")
                else root_process_raw / rel_parent / target_cls
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            for fname in arc_files:
                src = leaf_dir / fname
                dst = target_dir / f"{leaf_name}_{fname}"
                shutil.copy(src, dst)
                copied += 1
    else:
        packed = PackedArcDataset(input_path)
        for rel_path, wn, sp in packed.iter_samples():
            normalized_rel_path = rel_path.replace("\\", "/")
            rel_dir = Path(os.path.dirname(normalized_rel_path) or ".")
            rel_parent = rel_dir.parent
            leaf_name = packed.root_name if rel_dir == Path(".") else rel_dir.name

            prefix = get_prefix(leaf_name, profile.prefix_mode)
            target_cls = prefix if prefix else leaf_name
            target_dir = (
                root_process_raw / target_cls
                if rel_parent in (Path("."), Path(""))
                else root_process_raw / rel_parent / target_cls
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            fname = os.path.basename(normalized_rel_path)
            dst = target_dir / f"{leaf_name}_{fname}"
            write_arc_data(dst, wn, sp)
            copied += 1

    print(f"Stage 1 complete: copied {copied} files into {root_process_raw}")


def pca_reconstruct_and_error(spectra, n_components=0.95, center=True):
    spectra = np.asarray(spectra, dtype=np.float32)
    if spectra.ndim != 2 or spectra.shape[0] < 2:
        return spectra, 0, np.zeros((spectra.shape[0],), dtype=np.float32)

    mean = spectra.mean(axis=0, keepdims=True) if center else 0.0
    spectra_centered = spectra - mean

    u_matrix, singular_values, vt_matrix = np.linalg.svd(
        spectra_centered, full_matrices=False
    )
    if singular_values.size == 0:
        return spectra, 0, np.zeros((spectra.shape[0],), dtype=np.float32)

    variance = (singular_values ** 2) / max(spectra_centered.shape[0] - 1, 1)
    total_var = variance.sum()
    if total_var <= 0:
        return spectra, 0, np.zeros((spectra.shape[0],), dtype=np.float32)

    if isinstance(n_components, float) and 0 < n_components <= 1:
        ratio_cum = np.cumsum(variance) / total_var
        components = int(np.searchsorted(ratio_cum, n_components) + 1)
    else:
        try:
            components = int(n_components)
        except Exception:
            components = 1

    components = max(1, min(components, vt_matrix.shape[0]))
    spectra_rec = (u_matrix[:, :components] * singular_values[:components]) @ vt_matrix[
        :components, :
    ]
    if center:
        spectra_rec = spectra_rec + mean

    errors = np.mean((spectra - spectra_rec) ** 2, axis=1)
    return spectra_rec, components, errors


def log_removed_samples(label, filenames, errors, threshold, log_path):
    if not filenames:
        return
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(
            f"[{label}] removed {len(filenames)} samples, "
            f"threshold={threshold:.6f}\n"
        )
        for fname, err in zip(filenames, errors):
            file.write(f"  {fname}\t{float(err):.6f}\n")


def preprocess_group_samples(
    samples,
    bad_bands,
    label_display,
    min_samples,
    log_path=None,
    apply_pca=True,
):
    spectra = []
    wn_list = []
    filenames = []

    for fname, wn, sp in samples:
        if wn.size == 0 or sp.size == 0:
            continue

        wn_u, sp_u = preprocess_single_spectrum(
            wn,
            sp,
            cut_min=CUT_MIN,
            cut_max=CUT_MAX,
            wn_ref=WN_REF,
            bad_bands=bad_bands,
            asls_lam=ASLS_LAM,
            asls_p=ASLS_P,
            asls_max_iter=ASLS_MAX_ITER,
        )
        if wn_u is None:
            continue

        spectra.append(sp_u)
        wn_list.append(wn_u)
        filenames.append(fname)

    stats = {
        "input": len(samples),
        "valid_before_pca": len(spectra),
        "kept": len(spectra),
        "removed": 0,
        "pca_components": 0,
        "threshold": None,
        "skip_reason": None,
    }

    if len(spectra) < min_samples:
        stats["skip_reason"] = "too_few_preprocessed"
        return None, stats

    spectra_arr = np.vstack(spectra)
    if apply_pca and PCA_ENABLED and spectra_arr.shape[0] > 1:
        _, pca_components, errors = pca_reconstruct_and_error(
            spectra_arr,
            n_components=PCA_COMPONENTS,
            center=PCA_CENTER,
        )
        if pca_components > 0:
            ratio = float(PCA_OUTLIER_RATIO)
            ratio = max(0.0, min(ratio, 1.0))
            if ratio <= 0.0:
                threshold = float("inf")
                keep_mask = np.ones_like(errors, dtype=bool)
            else:
                threshold = float(np.quantile(errors, 1.0 - ratio))
                keep_mask = errors <= threshold

            removed = int((~keep_mask).sum())
            stats["removed"] = removed
            stats["pca_components"] = pca_components
            stats["threshold"] = threshold

            if removed > 0:
                removed_mask = ~keep_mask
                removed_files = [
                    name for name, is_removed in zip(filenames, removed_mask) if is_removed
                ]
                removed_errors = errors[removed_mask]
                if log_path is not None:
                    log_removed_samples(
                        label_display,
                        removed_files,
                        removed_errors,
                        threshold,
                        log_path,
                    )

                spectra_arr = spectra_arr[keep_mask]
                filenames = [
                    name for name, is_kept in zip(filenames, keep_mask) if is_kept
                ]
                wn_list = [wn for wn, is_kept in zip(wn_list, keep_mask) if is_kept]

    stats["kept"] = len(filenames)
    if len(filenames) < min_samples:
        stats["skip_reason"] = "too_few_after_pca"
        return None, stats

    return {
        "wn": wn_list[0],
        "spectra": spectra_arr,
        "filenames": filenames,
        "wn_list": wn_list,
    }, stats


def preprocess_train_dataset(profile, base_dir):
    base_dir = Path(base_dir)
    root_process_raw = resolve_path(base_dir, profile.root_process_raw)
    root_process_clean = resolve_path(base_dir, profile.root_train_clean)
    root_figure = resolve_path(base_dir, profile.root_train_fig)
    log_path = resolve_path(base_dir, profile.log_name)

    if not root_process_raw.is_dir():
        raise FileNotFoundError(f"Missing input dir: {root_process_raw}")

    root_process_clean.mkdir(parents=True, exist_ok=True)
    root_figure.mkdir(parents=True, exist_ok=True)

    for cls_raw_dir, arc_files in iter_arc_dirs(root_process_raw):
        rel_dir = cls_raw_dir.relative_to(root_process_raw)
        label = rel_dir.as_posix() if rel_dir != Path(".") else root_process_raw.name
        label_display = label.replace("\\", "/")

        print(f"\n=== Processing: {label_display} ===")

        samples = []
        for fname in arc_files:
            wn, sp = read_arc_data(cls_raw_dir / fname)
            samples.append((fname, wn, sp))

        processed_group, stats = preprocess_group_samples(
            samples=samples,
            bad_bands=profile.train_bad_bands,
            label_display=label_display,
            min_samples=MIN_SAMPLES_PER_CLASS,
            log_path=log_path,
        )

        if stats["skip_reason"] == "too_few_preprocessed":
            print(
                f"  Skip: too few samples ({stats['valid_before_pca']}) in {label_display}"
            )
            continue

        if stats["pca_components"] > 0:
            print(
                f"  PCA outlier removal: k={stats['pca_components']}, "
                f"threshold={stats['threshold']:.6f}, removed={stats['removed']}"
            )

        if stats["skip_reason"] == "too_few_after_pca":
            print(f"  Skip: too few samples ({stats['kept']}) in {label_display}")
            continue

        spectra_arr = processed_group["spectra"]
        filenames = processed_group["filenames"]
        wn_list = processed_group["wn_list"]

        save_dir = root_process_clean / rel_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        for fname, wn_u, sp_u in zip(filenames, wn_list, spectra_arr):
            out_path = save_dir / fname
            with open(out_path, "w", encoding="utf-8") as file:
                for wavenumber, value in zip(wn_u, sp_u):
                    file.write(f"{wavenumber:.3f} {value:.3f}\n")

        rel_parent = rel_dir.parent
        fig_dir = root_figure if rel_parent == Path(".") else root_figure / rel_parent
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_save_path = fig_dir / f"{cls_raw_dir.name}.png"
        title = " - ".join(rel_dir.parts) + " (mean +/- std)"
        save_mean_plot(
            wn=wn_list[0],
            spectra=spectra_arr,
            out_path=fig_save_path,
            norm_method=NORM_METHOD,
            bad_bands=profile.train_bad_bands,
            title=title,
        )

        print(f"  Mean spectrum saved: {fig_save_path}")

    print("\nTraining dataset preprocessing finished:")
    print(f"- Clean spectra: {root_process_clean}")
    print(f"- Mean plots: {root_figure}")


def preview_init_dataset(profile, base_dir):
    base_dir = Path(base_dir)
    input_path = resolve_init_input(base_dir, profile)
    root_init_fig = resolve_path(base_dir, profile.root_init_fig)
    root_init_fig.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0

    for rel_dir, leaf_name, samples in iter_init_groups(input_path):
        label = rel_dir.as_posix() if rel_dir != Path(".") else leaf_name
        label_display = label.replace("\\", "/")

        print(f"\n=== Preview: {label_display} ===")

        processed_group, stats = preprocess_group_samples(
            samples=samples,
            bad_bands=profile.train_bad_bands,
            label_display=label_display,
            min_samples=1,
            log_path=None,
            apply_pca=False,
        )

        if stats["skip_reason"] is not None:
            print(
                f"  Skip: no valid spectra after preprocessing "
                f"({stats['valid_before_pca']}/{stats['input']})"
            )
            skipped += 1
            continue

        if stats["pca_components"] > 0:
            print(
                f"  PCA outlier removal: k={stats['pca_components']}, "
                f"threshold={stats['threshold']:.6f}, removed={stats['removed']}"
            )

        rel_parent = rel_dir.parent
        fig_dir = root_init_fig if rel_parent == Path(".") else root_init_fig / rel_parent
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_save_path = fig_dir / f"{leaf_name}.png"

        title = " - ".join(rel_dir.parts) if rel_dir != Path(".") else leaf_name
        title = (
            f"{title} (mean +/- std, kept {stats['kept']}/{stats['input']})"
        )

        save_mean_plot(
            wn=processed_group["wn"],
            spectra=processed_group["spectra"],
            out_path=fig_save_path,
            norm_method=NORM_METHOD,
            bad_bands=profile.train_bad_bands,
            title=title,
        )

        print(f"  Mean spectrum saved: {fig_save_path}")
        generated += 1

    print("\nDataset init preview finished:")
    print(f"- Mean plots: {root_init_fig}")
    print(f"- Generated={generated}, Skipped={skipped}")


def preprocess_test_dataset(profile, base_dir, input_dir=None, output_dir=None):
    base_dir = Path(base_dir)
    input_dir = (
        Path(input_dir)
        if input_dir is not None
        else resolve_path(base_dir, profile.root_test_raw)
    )
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else resolve_path(base_dir, profile.root_test_clean)
    )
    root_test_fig = resolve_path(base_dir, profile.root_test_fig)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input dir: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    root_test_fig.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    errored = 0

    for root, arc_files in iter_arc_dirs(input_dir):
        spectra = []
        wn_list = []

        for fname in arc_files:
            in_path = root / fname
            rel_dir = in_path.parent.relative_to(input_dir)
            out_dir = output_dir / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / fname

            try:
                wn, sp = read_arc_data(in_path)
                wn_u, sp_u = preprocess_single_spectrum(
                    wn,
                    sp,
                    cut_min=CUT_MIN,
                    cut_max=CUT_MAX,
                    wn_ref=WN_REF,
                    bad_bands=profile.test_bad_bands,
                    asls_lam=ASLS_LAM,
                    asls_p=ASLS_P,
                    asls_max_iter=ASLS_MAX_ITER,
                )
                if wn_u is None:
                    rel_path = in_path.relative_to(input_dir).as_posix()
                    print(f"[SKIP] {rel_path} (empty after cut)")
                    skipped += 1
                    continue

                with open(out_path, "w", encoding="utf-8") as file:
                    for wavenumber, signal in zip(wn_u, sp_u):
                        file.write(f"{wavenumber:.3f} {signal:.3f}\n")

                spectra.append(sp_u)
                wn_list.append(wn_u)
                processed += 1
            except Exception as exc:
                rel_path = in_path.relative_to(input_dir).as_posix()
                print(f"[ERROR] {rel_path}: {exc}")
                errored += 1

        if spectra:
            spectra_arr = np.vstack(spectra)
            wn_ref = wn_list[0]

            rel_dir = root.relative_to(input_dir)
            rel_parent = rel_dir.parent
            fig_dir = (
                root_test_fig if rel_parent == Path(".") else root_test_fig / rel_parent
            )
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / f"{root.name}.png"

            save_mean_plot(
                wn=wn_ref,
                spectra=spectra_arr,
                out_path=fig_path,
                norm_method=NORM_METHOD,
                bad_bands=profile.test_bad_bands,
                title=f"{rel_dir.as_posix()} (mean +/- std)",
            )

            print(f"  Mean spectrum saved: {fig_path}")

    print(
        "Test dataset preprocessing finished. "
        f"Processed={processed}, Skipped={skipped}, Error={errored}"
    )


def compute_totals(node):
    total = node.get("__count__", 0)
    for name, child in node.items():
        if name.startswith("__"):
            continue
        total += compute_totals(child)
    node["__total__"] = total
    return total


def build_tree(root_dir):
    tree = {}
    for leaf_dir, arc_files in iter_arc_dirs(root_dir):
        rel_dir = Path(leaf_dir).relative_to(root_dir)
        parts = [] if rel_dir == Path(".") else rel_dir.parts

        node = tree
        for part in parts:
            node = node.setdefault(part, {})

        node["__count__"] = node.get("__count__", 0) + len(arc_files)

    compute_totals(tree)
    return tree


def count_dataset(root_dir):
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Missing input dir: {root_dir}")

    tree = build_tree(root_dir)
    total_files = tree.get("__total__", 0)
    return tree, total_files


def print_tree(node, level=0, name=None):
    indent = "  " * level
    if name is not None:
        count = node.get("__count__", 0)
        total = node.get("__total__", 0)
        children = [key for key in node.keys() if not key.startswith("__")]
        if children:
            if count > 0:
                print(f"{indent}{name}: {count} 个文件 (含子目录总计 {total})")
            else:
                print(f"{indent}{name}: 总计 {total} 个文件")
        else:
            print(f"{indent}{name}: {count} 个文件")

    for child_name in sorted(key for key in node.keys() if not key.startswith("__")):
        print_tree(node[child_name], level + 1, child_name)


def print_results(tree, total_files):
    print("\n================ 数据集统计 ================\n")
    print(f"总文件数: {total_files}\n")

    root_count = tree.get("__count__", 0)
    if root_count:
        root_total = tree.get("__total__", 0)
        print(f"[根目录] {root_count} 个文件 (含子目录总计 {root_total})\n")

    for top_name in sorted(key for key in tree.keys() if not key.startswith("__")):
        print_tree(tree[top_name], 0, top_name)
        print("")

    print("============================================\n")
