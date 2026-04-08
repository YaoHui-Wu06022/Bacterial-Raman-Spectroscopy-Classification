import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def resolve_prototype_path(model_path, prototype_path=None):
    """根据模型文件路径推导 prototype 文件路径。"""
    if prototype_path:
        return prototype_path
    if model_path.endswith("_model.pt"):
        return model_path.replace("_model.pt", "_prototypes.pt")
    root, _ = os.path.splitext(model_path)
    return f"{root}_prototypes.pt"


def save_prototype_bundle(bundle, path):
    """保存类别原型向量、样本数和有效掩码。"""
    torch.save(
        {
            "prototypes": bundle["prototypes"].cpu(),
            "counts": bundle["counts"].cpu(),
            "valid_mask": bundle["valid_mask"].cpu(),
        },
        path,
    )


def load_prototype_bundle(path, device):
    """读取 prototype 文件；缺失时返回 None。"""
    if not path or not os.path.exists(path):
        return None

    bundle = torch.load(path, map_location=device)
    prototypes = bundle.get("prototypes")
    counts = bundle.get("counts")
    valid_mask = bundle.get("valid_mask")
    if prototypes is None or counts is None or valid_mask is None:
        return None

    return {
        "prototypes": prototypes.to(device=device, dtype=torch.float32),
        "counts": counts.to(device=device),
        "valid_mask": valid_mask.to(device=device, dtype=torch.bool),
    }


def compute_class_prototypes(
    model,
    dataset,
    indices,
    level_idx,
    num_classes,
    device,
    loader_kwargs,
    label_map_tensor=None,
):
    """用训练集 clean view 的 embedding 均值构建类别 prototype。"""
    if len(indices) == 0:
        return None

    kwargs = dict(loader_kwargs)
    kwargs["shuffle"] = False
    loader = DataLoader(Subset(dataset, list(indices)), **kwargs)

    sums = None
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            _, feat = model(x, return_feat=True)
            feat = F.normalize(feat, p=2, dim=1)

            if y.ndim == 2:
                labels = y[:, level_idx]
            else:
                labels = y

            if label_map_tensor is not None:
                invalid = labels < 0
                labels = label_map_tensor[labels.clamp_min(0)]
                labels[invalid] = -1

            valid = (labels >= 0) & (labels < num_classes)
            if not valid.any():
                continue

            feat_valid = feat[valid]
            label_valid = labels[valid].long()

            if sums is None:
                sums = torch.zeros(
                    num_classes,
                    feat_valid.size(1),
                    dtype=feat_valid.dtype,
                    device=device,
                )

            sums.index_add_(0, label_valid, feat_valid)
            counts.index_add_(
                0,
                label_valid,
                torch.ones_like(label_valid, dtype=torch.long, device=device),
            )

    if sums is None:
        return None

    prototypes = torch.zeros_like(sums)
    valid_mask = counts > 0
    if valid_mask.any():
        prototypes[valid_mask] = sums[valid_mask] / counts[valid_mask].unsqueeze(1).to(
            sums.dtype
        )
        prototypes[valid_mask] = F.normalize(prototypes[valid_mask], p=2, dim=1)

    return {
        "prototypes": prototypes.detach().cpu(),
        "counts": counts.detach().cpu(),
        "valid_mask": valid_mask.detach().cpu(),
    }


def compute_fused_probs(logits, feat, prototype_bundle, config):
    """融合分类头概率和 prototype 相似度概率。"""
    classifier_probs = F.softmax(logits, dim=1)

    mode = str(getattr(config, "prototype_fusion_mode", "fusion")).lower()
    if mode == "classifier" or prototype_bundle is None:
        return classifier_probs

    prototypes = prototype_bundle["prototypes"]
    valid_mask = prototype_bundle["valid_mask"]
    if prototypes.numel() == 0 or not valid_mask.any():
        return classifier_probs

    feat = F.normalize(feat, p=2, dim=1)
    proto_scores = torch.matmul(feat, prototypes.t())
    proto_scores = proto_scores * float(
        getattr(config, "prototype_similarity_scale", 20.0)
    )

    invalid_logits = ~torch.isfinite(logits)
    if invalid_logits.any():
        proto_scores = proto_scores.masked_fill(invalid_logits, float("-inf"))

    proto_scores = proto_scores.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))
    valid_rows = torch.isfinite(proto_scores).any(dim=1)
    if not valid_rows.any():
        return classifier_probs

    prototype_probs = torch.zeros_like(classifier_probs)
    prototype_probs[valid_rows] = F.softmax(proto_scores[valid_rows], dim=1)

    if mode == "prototype":
        fused = classifier_probs.clone()
        fused[valid_rows] = prototype_probs[valid_rows]
        return fused

    weight = float(getattr(config, "prototype_fusion_weight", 0.35))
    weight = min(max(weight, 0.0), 1.0)
    fused = classifier_probs.clone()
    fused[valid_rows] = (
        (1.0 - weight) * classifier_probs[valid_rows]
        + weight * prototype_probs[valid_rows]
    )
    return fused
