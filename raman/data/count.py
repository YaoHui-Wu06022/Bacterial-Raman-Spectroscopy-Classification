from pathlib import Path

from raman.data.archive import iter_arc_dirs

def compute_totals(node):
    """递归回填每个目录节点的总样本数"""
    total = node.get("__count__", 0)
    for name, child in node.items():
        if name.startswith("__"):
            continue
        total += compute_totals(child)
    node["__total__"] = total
    return total

def build_tree(root_dir):
    """把目录树转成带计数的嵌套字典，供 count 子命令打印"""
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
    """统计一个数据目录下各层文件数，并返回树形结构"""
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Missing input dir: {root_dir}")

    tree = build_tree(root_dir)
    total_files = tree.get("__total__", 0)
    return tree, total_files

def print_tree(node, level=0, name=None):
    """按缩进样式打印统计树，便于终端查看目录层级分布"""
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
    """统一打印 count 子命令的统计结果摘要"""
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

