"""数据集阶段中 `.arc_data` 文件的层级统计。"""

from __future__ import annotations

from pathlib import Path

from .io import iter_arc_dirs


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
