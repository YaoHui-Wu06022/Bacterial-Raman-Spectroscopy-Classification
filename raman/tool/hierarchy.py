"""层级 key、level 名称和层级元信息工具"""

import json
from pathlib import Path


ROOT_TAG = "__root__"
MISSING_TAG = "__missing__"


def parts_to_key(parts):
    """把路径层级片段转换成稳定层级 key"""
    if not parts:
        return ROOT_TAG
    return "/".join(str(part) for part in parts)


def level_number(level_name):
    """读取 level_1 里的数字"""
    text = str(level_name)
    if not text.startswith("level_") or "_" not in text:
        raise ValueError(f"Invalid level name: {level_name}")
    return int(text.split("_", 1)[1])


def resolve_level_order(dataset, target_level, field_name="target_level"):
    """解析目标业务层级，并返回从 level_1 到目标层的顺序"""
    target_level = dataset._resolve_level_name(target_level, field_name=field_name)
    if target_level not in dataset.level_names:
        raise ValueError(
            f"未知 {field_name}: {target_level}，可选值：{dataset.level_names}"
        )
    stop_idx = dataset.level_names.index(target_level) + 1
    return target_level, list(dataset.level_names[:stop_idx])


def label_from_parts(parts, level_name):
    """按路径层级推断某个业务 level 的标签"""
    if not parts:
        return None
    level_no = level_number(level_name)
    if len(parts) < level_no:
        return None
    return "/".join(str(part) for part in parts[:level_no])


def iter_ancestor_level_keys(rel_dir):
    """生成非叶子祖先层级 key，用于高层级聚合统计或绘图"""
    parts = tuple(Path(rel_dir).parts)
    if len(parts) <= 1:
        return
    for level_idx in range(1, len(parts)):
        yield level_idx, parts[:level_idx]


def safe_key_name(parts):
    """把层级路径片段转换成稳定文件名"""
    return "__".join(str(part) for part in parts)


def _normalize_parent_to_children(parent_to_children):
    """把 parent_to_children 的 key 和 child id 统一转成 int"""
    normalized = {}
    for level_name, mapping in (parent_to_children or {}).items():
        normalized[level_name] = {
            int(parent_idx): [int(child_id) for child_id in child_ids]
            for parent_idx, child_ids in mapping.items()
        }
    return normalized


def _normalize_parent_models(parent_models):
    """把 hierarchy_meta 里的 parent model 条目规整成统一字典"""
    normalized = {}
    for level_name, mapping in (parent_models or {}).items():
        normalized[level_name] = {}
        for parent_idx, entry in mapping.items():
            item = dict(entry) if isinstance(entry, dict) else {"model_path": entry}
            item["child_ids"] = [int(child_id) for child_id in item.get("child_ids", [])]
            normalized[level_name][int(parent_idx)] = item
    return normalized


def load_hierarchy_meta(exp_dir):
    """读取并规范化 hierarchy_meta.json"""
    meta_path = Path(exp_dir) / "hierarchy_meta.json"
    if not meta_path.exists():
        return None

    with open(meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)

    meta["parent_to_children"] = _normalize_parent_to_children(meta.get("parent_to_children", {}))
    meta["parent_models"] = _normalize_parent_models(meta.get("parent_models", {}))
    meta["level_models"] = dict(meta.get("level_models", {}))
    meta["class_names_by_level"] = dict(meta.get("class_names_by_level", {}))
    meta["head_names"] = list(meta.get("head_names", []))
    return meta


def build_hierarchy_meta(
    *,
    head_names,
    class_names_by_level,
    parent_to_children,
    parent_level_name,
    current_train_level,
    level_models,
    parent_models,
    runs,
):
    """按统一结构组装 hierarchy_meta.json 内容"""
    return {
        "head_names": list(head_names),
        "level_names": list(head_names),
        "class_names_by_level": class_names_by_level,
        "parent_to_children": parent_to_children,
        "parent_level_name": parent_level_name,
        "current_train_level": current_train_level,
        "level_models": level_models,
        "parent_models": parent_models,
        "runs": runs,
    }


def save_json(path, payload):
    """按 UTF-8 写出层级相关 JSON"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
