"""数据集阶段中 `.arc_data` 文件的层级统计。"""

from __future__ import annotations

from pathlib import Path

from ramanv2.common.naming import build_natural_key, parse_folder_prefix
from ramanv2.data.dataset_io import iter_arc_dirs
from ramanv2.data.hierarchy_spec import HierarchySpec


def _compute_totals(node: dict) -> int:
    """递归回填当前目录及所有子目录包含的光谱数。"""
    total = node.get("__count__", 0)
    for name, child in node.items():
        if not name.startswith("__"):
            total += _compute_totals(child)
    node["__total__"] = total
    return total


def build_count_tree(root_dir: Path | str) -> dict:
    """按目录层级构建光谱文件计数树。"""
    root_path = Path(root_dir)
    tree: dict = {}
    for leaf_dir, filenames in iter_arc_dirs(root_path):
        relative_dir = leaf_dir.relative_to(root_path)
        node = tree
        for name in (() if relative_dir == Path(".") else relative_dir.parts):
            node = node.setdefault(name, {})
        node["__count__"] = node.get("__count__", 0) + len(filenames)
    _compute_totals(tree)
    return tree


def count_dataset(root_dir: Path | str) -> tuple[dict, int]:
    """统计一个数据阶段的 `.arc_data` 文件数和目录分布。"""
    root_path = Path(root_dir)
    if not root_path.is_dir():
        raise FileNotFoundError(f"缺少数据目录：{root_path}")
    tree = build_count_tree(root_path)
    return tree, tree.get("__total__", 0)


def summarize_dataset(root_dir: Path | str) -> dict[str, int | bool]:
    """返回页面和构建报告共用的数据集统计摘要。"""
    root_path = Path(root_dir)
    if not root_path.is_dir():
        return {"exists": False, "genus_count": 0, "species_count": 0, "folder_count": 0, "file_count": 0}

    tree, total_files = count_dataset(root_path)

    def folder_count(node: dict) -> int:
        count = 1 if node.get("__count__", 0) else 0
        return count + sum(
            folder_count(child)
            for name, child in node.items()
            if not name.startswith("__")
        )

    species_keys = {
        (genus, parse_folder_prefix(folder, uppercase_enable=True))
        for genus, genus_tree in tree.items()
        if not genus.startswith("__")
        for folder, group in genus_tree.items()
        if not folder.startswith("__") and group.get("__total__", 0)
    }
    return {
        "exists": True,
        "genus_count": sum(1 for name in tree if not name.startswith("__")),
        "species_count": len(species_keys),
        "folder_count": folder_count(tree),
        "file_count": total_files,
    }


def _collect_leaf_dirs(node: dict, parts: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], int]]:
    """递归收集含光谱的叶子目录及其相对路径层级。"""
    leaves: list[tuple[tuple[str, ...], int]] = []
    if node.get("__count__", 0):
        leaves.append((parts, int(node["__count__"])))
    for name, child in node.items():
        if not name.startswith("__"):
            leaves.extend(_collect_leaf_dirs(child, (*parts, name)))
    return leaves


def summarize_dataset_hierarchy(root_dir: Path | str, hierarchy: HierarchySpec) -> dict[str, object]:
    """按层级配置统计每个业务层级的类别数量、文件夹数与光谱数。"""
    root_path = Path(root_dir)
    if not root_path.is_dir():
        return {"exists": False, "levels": [], "folder_count": 0, "file_count": 0}
    tree = build_count_tree(root_path)
    leaves = _collect_leaf_dirs(tree)
    levels: list[dict[str, object]] = []
    for depth, level in enumerate(hierarchy.levels):
        if level.directory:
            values = {parts[depth] for parts, _ in leaves if len(parts) > depth}
        else:
            mapping = hierarchy.mappings.get(level.name, {})
            prefixes = {
                parse_folder_prefix(parts[-1], uppercase_enable=True)
                for parts, _ in leaves
                if parts
            }
            values = {mapping.get(prefix, prefix) for prefix in prefixes}
        levels.append({"name": level.display_name, "count": len(values)})
    return {
        "exists": True,
        "levels": levels,
        "folder_count": len(leaves),
        "file_count": sum(count for _, count in leaves),
    }


def collect_taxon_counts(
    root_dir: Path | str,
    hierarchy: HierarchySpec | None = None,
) -> dict[str, object]:
    """按 init 层级统计类别、来源文件夹与光谱数，不读取谱内容。

    叶子目录的目录层级取自相对路径，非目录层级按文件夹前缀映射；同一类别
    跨日期合并。返回每个层级的显示名、各类别明细，以及按父层级汇总的行。
    """
    hierarchy = hierarchy or HierarchySpec.flat()
    root_path = Path(root_dir)
    levels = list(hierarchy.levels)
    depth = hierarchy.directory_depth
    grouped: dict[tuple[str, ...], dict[str, object]] = {}
    for leaf_dir, filenames in iter_arc_dirs(root_path):
        relative = leaf_dir.relative_to(root_path)
        if len(relative.parts) != depth + 1:
            continue
        prefix = parse_folder_prefix(leaf_dir.name, uppercase_enable=True)
        values = list(relative.parts[:depth])
        for level in levels[depth:]:
            mapping = hierarchy.mappings.get(level.name, {})
            values.append(mapping.get(prefix, prefix))
        key = tuple(values)
        row = grouped.setdefault(
            key,
            {"values": key, "folder_count": 0, "file_count": 0},
        )
        row["folder_count"] = int(row["folder_count"]) + 1
        row["file_count"] = int(row["file_count"]) + len(filenames)

    def sort_key(item: dict[str, object]) -> tuple[str, ...]:
        return tuple(build_natural_key(value) for value in item["values"])

    rows = sorted(grouped.values(), key=sort_key)
    summaries: dict[tuple[str, ...], dict[str, object]] = {}
    for row in rows:
        parent = tuple(row["values"][:-1])
        summary = summaries.setdefault(
            parent,
            {"values": parent, "species_count": 0, "folder_count": 0, "file_count": 0},
        )
        summary["species_count"] = int(summary["species_count"]) + 1
        summary["folder_count"] = int(summary["folder_count"]) + int(row["folder_count"])
        summary["file_count"] = int(summary["file_count"]) + int(row["file_count"])
    return {
        "levels": [level.display_name for level in levels],
        "rows": rows,
        "summaries": sorted(summaries.values(), key=sort_key),
    }


def _print_tree(node: dict, level: int = 0, name: str | None = None) -> None:
    """以缩进树形格式输出一个目录节点。"""
    if name is not None:
        indent = "  " * level
        count = node.get("__count__", 0)
        total = node.get("__total__", 0)
        child_names = [key for key in node if not key.startswith("__")]
        if child_names and count:
            print(f"{indent}{name}: {count} 个文件（含子目录共 {total}）")
        elif child_names:
            print(f"{indent}{name}: 共 {total} 个文件")
        else:
            print(f"{indent}{name}: {count} 个文件")
    for child_name in sorted(key for key in node if not key.startswith("__")):
        _print_tree(node[child_name], level + 1, child_name)


def print_count_results(tree: dict, total_files: int) -> None:
    """输出数据集统计摘要和目录树。"""
    print(f"总文件数：{total_files}")
    root_count = tree.get("__count__", 0)
    if root_count:
        print(f"根目录：{root_count} 个文件（含子目录共 {tree['__total__']}）")
    for name in sorted(key for key in tree if not key.startswith("__")):
        _print_tree(tree[name], name=name)
