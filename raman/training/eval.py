import numpy as np
import torch


def classification_metrics(y_true, y_pred, num_classes):
    """基于混淆矩阵计算训练中使用的宏平均 F1 和 balanced accuracy。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, pred in zip(y_true, y_pred):
        if target < 0 or pred < 0:
            continue
        cm[target, pred] += 1

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    support = tp + fn
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(support, 1)
    denom = precision + recall
    f1 = np.zeros_like(denom, dtype=float)
    np.divide(2 * precision * recall, denom, out=f1, where=denom > 0)

    valid = support > 0
    if valid.any():
        macro_f1 = float(np.mean(f1[valid]))
        balanced_acc = float(np.mean(recall[valid]))
    else:
        macro_f1 = 0.0
        balanced_acc = 0.0

    return {"macro_f1": macro_f1, "balanced_acc": balanced_acc}


def _select_logits(pred, head_name=None):
    """从模型输出中取出当前需要评估的 logits。"""
    if isinstance(pred, dict):
        if head_name is None:
            head_name = list(pred.keys())[-1]
        return pred[head_name]
    if isinstance(pred, (tuple, list)):
        return pred[0]
    return pred


def mask_logits_by_parent(logits, parent_labels, parent_to_children):
    """按父类标签对 logits 做遮罩，只保留该父类允许出现的子类。"""
    if parent_labels is None or parent_to_children is None:
        valid = torch.ones(logits.size(0), dtype=torch.bool, device=logits.device)
        return logits, valid

    device = logits.device
    batch = logits.size(0)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    valid = torch.zeros(batch, dtype=torch.bool, device=device)

    for i, parent_id in enumerate(parent_labels.tolist()):
        if parent_id < 0:
            continue

        child_idx = parent_to_children.get(parent_id)
        if child_idx is None:
            child_idx = parent_to_children.get(str(parent_id))
        if not child_idx:
            continue

        num_classes = logits.size(1)
        child_list = list(child_idx)
        invalid = [c for c in child_list if c < 0 or c >= num_classes]
        if invalid:
            raise ValueError(
                f"parent_to_children 索引越界：parent={parent_id}, "
                f"invalid={invalid}, num_classes={num_classes}"
            )

        mask[i, child_list] = True
        valid[i] = True

    masked_logits = logits.masked_fill(~mask, float("-inf"))
    if (~valid).any():
        masked_logits[~valid] = 0.0

    return masked_logits, valid


def _select_level_targets(y, head_index=None):
    """从多层标签中取出当前层级的目标标签。"""
    if y.ndim != 2:
        return y
    if head_index is None:
        head_index = y.size(1) - 1
    return y[:, head_index]


def _finalize_eval(total_loss, total_correct, total, all_targets, all_preds, num_classes):
    """汇总批次级结果，输出训练期统一指标。"""
    if num_classes is None or not all_targets:
        return 0.0, 0.0, {"macro_f1": 0.0, "balanced_acc": 0.0}

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    metrics = classification_metrics(y_true, y_pred, num_classes)
    return total_loss / max(total, 1), total_correct / max(total, 1), metrics


def _evaluate_loader(
    model,
    loader,
    device,
    target_builder,
    head_name=None,
    parent_index=None,
    parent_to_children=None,
):
    """训练期评估公共循环。"""
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds = []
    all_targets = []
    num_classes = None
    criterion_eval = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = _select_logits(model(x), head_name=head_name)
            y_full = y
            y = target_builder(y)

            if num_classes is None:
                num_classes = logits.size(1)

            if parent_index is not None and parent_to_children is not None:
                if y_full.ndim != 2:
                    raise ValueError("parent_index 需要输入完整的多层标签。")
                parent_labels = y_full[:, parent_index]
                logits, valid_parent = mask_logits_by_parent(
                    logits,
                    parent_labels,
                    parent_to_children,
                )
            else:
                valid_parent = torch.ones_like(y, dtype=torch.bool)

            valid = (y >= 0) & valid_parent
            if not valid.any():
                continue

            logits = logits[valid]
            y = y[valid]

            loss = criterion_eval(logits, y)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(1) == y).sum().item()
            total += batch_size

            all_preds.append(logits.argmax(1).detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    return _finalize_eval(
        total_loss,
        total_correct,
        total,
        all_targets,
        all_preds,
        num_classes,
    )


def evaluate_file_level(
    model,
    loader,
    device,
    head_index=None,
    head_name=None,
    parent_index=None,
    parent_to_children=None,
):
    """评估一个层级模型在文件级样本上的损失、准确率和摘要指标。"""
    return _evaluate_loader(
        model,
        loader,
        device,
        target_builder=lambda y: _select_level_targets(y, head_index),
        head_name=head_name,
        parent_index=parent_index,
        parent_to_children=parent_to_children,
    )


def evaluate_file_level_local(model, loader, device, head_index, label_map_tensor):
    """评估父类内子模型，并把全局标签映射为局部标签。"""

    def build_local_targets(y):
        y = _select_level_targets(y, head_index)
        if label_map_tensor is None:
            return y
        invalid = y < 0
        y = label_map_tensor[y.clamp_min(0)]
        y[invalid] = -1
        return y

    return _evaluate_loader(
        model,
        loader,
        device,
        target_builder=build_local_targets,
    )
