import argparse
import json
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from raman.config import config as raman_default_config
from raman.data import RamanDataset
from raman.eval.common import compute_classification_metrics
from raman.tool.hierarchy import resolve_level_order
from raman.training.losses import FocalLoss, build_class_weights
from raman.training.split import save_split_files, split_by_lowest_level_ratio
from ramf.config import RaMFConfig
from ramf.model import RaMFNet


@dataclass
class RaMFTrainConfig:
    dataset_name: str = "GN"
    level: str = "level_1"
    output_root: str = "output/ramf"
    epochs: int = 65
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    train_split: float = 0.8
    seed: int = 42
    patience: int = 20
    gamma: float = 0.8
    num_workers: int = 0
    use_gpu: bool = True
    split_by_source_prefix: bool = False
    show_progress: bool = True


def _set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _run_dir(train_config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(train_config.output_root) / train_config.dataset_name / train_config.level / timestamp


def _build_raman_config(train_config):
    cfg = deepcopy(raman_default_config)
    cfg.dataset_name = train_config.dataset_name
    cfg.use_gpu = train_config.use_gpu
    cfg.train_split = train_config.train_split
    cfg.seed = train_config.seed
    return cfg


def _build_loader(dataset, indices, train_config, train):
    num_workers = max(int(train_config.num_workers), 0)
    kwargs = {
        "batch_size": int(train_config.batch_size),
        "shuffle": bool(train),
        "num_workers": num_workers,
    }
    if bool(train_config.use_gpu) and torch.cuda.is_available():
        kwargs["pin_memory"] = True
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(Subset(dataset, indices), **kwargs)


def _valid_level_indices(dataset, indices, level_idx):
    indices = np.asarray(indices, dtype=np.int64)
    labels = dataset.level_labels[indices, level_idx]
    return indices[labels >= 0]


def _train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    level_idx,
    epoch,
    total_epochs,
    show_progress=True,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    loader_iter = loader
    if show_progress:
        loader_iter = tqdm(
            loader,
            desc=f"Epoch {epoch}/{total_epochs}",
            leave=True,
        )

    for x, y, _ in loader_iter:
        x = x.to(device)
        y = y.to(device)
        y_level = y[:, level_idx] if y.ndim == 2 else y
        valid = y_level >= 0
        if not valid.any():
            continue

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss_each = criterion(logits[valid], y_level[valid])
        loss = loss_each.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = int(valid.sum().item())
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits[valid].argmax(1) == y_level[valid]).sum().item())
        total += batch_size
        if show_progress:
            loader_iter.set_postfix(
                {
                    "loss": f"{total_loss / max(total, 1):.4f}",
                    "acc": f"{100 * total_correct / max(total, 1):.2f}%",
                }
            )

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": total_correct / max(total, 1),
    }


def _evaluate(model, loader, device, level_idx, num_classes):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            y_level = y[:, level_idx] if y.ndim == 2 else y
            valid = y_level >= 0
            if not valid.any():
                continue
            logits = model(x)[valid]
            targets = y_level[valid]
            loss = criterion(logits, targets)
            batch_size = int(targets.size(0))
            total_loss += float(loss.item()) * batch_size
            total += batch_size
            all_targets.append(targets.detach().cpu().numpy())
            all_preds.append(logits.argmax(1).detach().cpu().numpy())

    if not all_targets:
        metrics = {"accuracy": 0.0, "macro_f1": 0.0, "macro_recall": 0.0}
    else:
        metrics = compute_classification_metrics(
            np.concatenate(all_targets),
            np.concatenate(all_preds),
            labels=range(num_classes),
        )
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def train_ramf(train_config=None, model_config=None):
    train_config = train_config or RaMFTrainConfig()
    model_config = model_config or RaMFConfig()
    _set_seed(train_config.seed)

    raman_config = _build_raman_config(train_config)
    run_dir = _run_dir(train_config)
    run_dir.mkdir(parents=True, exist_ok=False)

    full_dataset = RamanDataset(raman_config.dataset_root, augment=False, config=raman_config)
    level_name, _ = resolve_level_order(full_dataset, train_config.level)
    level_idx = full_dataset.head_name_to_idx[level_name]
    num_classes = full_dataset.num_classes_by_level[level_name]

    train_idx, val_idx = split_by_lowest_level_ratio(
        full_dataset,
        lowest_level="leaf",
        train_ratio=float(train_config.train_split),
        seed=int(train_config.seed),
        split_by_source_prefix=bool(train_config.split_by_source_prefix),
    )
    train_idx = _valid_level_indices(full_dataset, train_idx, level_idx)
    val_idx = _valid_level_indices(full_dataset, val_idx, level_idx)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("RaMF training requires non-empty train and validation splits.")

    save_split_files(full_dataset, train_idx, val_idx, run_dir)
    train_dataset = RamanDataset(raman_config.dataset_root, augment=True, config=raman_config)
    val_dataset = RamanDataset(raman_config.dataset_root, augment=False, config=raman_config)
    train_loader = _build_loader(train_dataset, train_idx, train_config, train=True)
    val_loader = _build_loader(val_dataset, val_idx, train_config, train=False)

    model_config.in_channels = int(raman_config.in_channels)
    device = torch.device(
        "cuda" if train_config.use_gpu and torch.cuda.is_available() else "cpu"
    )
    model = RaMFNet(num_classes=num_classes, config=model_config).to(device)
    labels_for_weights = full_dataset.level_labels[train_idx, level_idx]
    class_weights = torch.tensor(
        build_class_weights(labels_for_weights, num_classes),
        dtype=torch.float32,
        device=device,
    )
    criterion = FocalLoss(
        gamma=float(train_config.gamma),
        weight=class_weights,
        ignore_index=-1,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.learning_rate),
        weight_decay=float(train_config.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(train_config.epochs), 1),
        eta_min=float(train_config.learning_rate) * 0.05,
    )

    class_names = full_dataset.get_class_names(level_name)
    _write_json(run_dir / "train_config.json", asdict(train_config))
    _write_json(run_dir / "ramf_config.json", model_config.to_dict())
    _write_json(run_dir / "input_config.json", raman_config.to_dict())
    _write_json(
        run_dir / "classes.json",
        {"level": level_name, "class_names": class_names},
    )

    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    best_model_path = run_dir / "best_model.pt"

    print(f"[RaMF] run_dir={run_dir}")
    print(f"[RaMF] dataset={train_config.dataset_name} level={level_name}")
    print(f"[RaMF] train={len(train_idx)} val={len(val_idx)} classes={num_classes}")
    print(f"[RaMF] device={device}")

    for epoch in range(1, int(train_config.epochs) + 1):
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            level_idx,
            epoch,
            int(train_config.epochs),
            show_progress=bool(train_config.show_progress),
        )
        val_metrics = _evaluate(model, val_loader, device, level_idx, num_classes)
        scheduler.step()

        score = 0.6 * val_metrics["macro_f1"] + 0.4 * val_metrics["accuracy"]
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "score": score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy'] * 100:.2f}% "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy'] * 100:.2f}% "
            f"val_f1={val_metrics['macro_f1'] * 100:.2f}%"
        )

        if score >= best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            stale_epochs += 1

        _write_json(
            run_dir / "metrics.json",
            {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "history": history,
            },
        )
        if stale_epochs >= int(train_config.patience):
            print(f"[RaMF] early stop at epoch={epoch}")
            break

    torch.save(model.state_dict(), run_dir / "last_model.pt")
    print(f"[RaMF] best_epoch={best_epoch} best_model={best_model_path}")
    return {
        "run_dir": os.fspath(run_dir),
        "best_model_path": os.fspath(best_model_path),
        "best_epoch": best_epoch,
        "best_score": best_score,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the standalone RaMF model.")
    parser.add_argument("--dataset", default="GN", help="ASCII dataset profile id, e.g. GN/GP/MICRO.")
    parser.add_argument("--level", default="level_1", help="Target hierarchy level.")
    parser.add_argument("--output-root", default="output/ramf")
    parser.add_argument("--epochs", type=int, default=65)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--branch-channels", type=int, default=32)
    return parser.parse_args()


def main():
    args = _parse_args()
    train_config = RaMFTrainConfig(
        dataset_name=args.dataset,
        level=args.level,
        output_root=args.output_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
        use_gpu=not args.cpu,
    )
    model_config = RaMFConfig(
        transformer_dim=args.dim,
        transformer_heads=args.heads,
        transformer_layers=args.layers,
        image_size=args.image_size,
        branch_channels=args.branch_channels,
    )
    train_ramf(train_config, model_config)


if __name__ == "__main__":
    main()
