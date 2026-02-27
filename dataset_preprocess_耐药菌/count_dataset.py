import os

# DATASET_ROOT = "../dataset_train_耐药菌"  # 改成你的数据集根目录
DATASET_ROOT = "dataset_raw"  # 改成你的数据集根目录


def iter_class_dirs(root_dir):
    # 遍历所有包含 .arc_data 的目录
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        arc_files = [f for f in files if f.lower().endswith(".arc_data")]
        if arc_files:
            yield root, arc_files


def compute_totals(node):
    # 递归累计子目录的文件总数
    total = node.get("__count__", 0)
    for name, child in node.items():
        if name.startswith("__"):
            continue
        total += compute_totals(child)
    node["__total__"] = total
    return total


def build_tree(root_dir):
    tree = {}

    for leaf_dir, arc_files in iter_class_dirs(root_dir):
        rel_dir = os.path.relpath(leaf_dir, root_dir)
        parts = [] if rel_dir == "." else rel_dir.split(os.sep)

        node = tree
        for part in parts:
            node = node.setdefault(part, {})

        node["__count__"] = node.get("__count__", 0) + len(arc_files)

    compute_totals(tree)
    return tree


def count_dataset(root_dir):
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Missing input dir: {root_dir}")

    tree = build_tree(root_dir)
    total_files = tree.get("__total__", 0)
    return tree, total_files


def print_tree(node, level=0, name=None):
    indent = "  " * level
    if name is not None:
        count = node.get("__count__", 0)
        total = node.get("__total__", 0)
        children = [k for k in node.keys() if not k.startswith("__")]
        if children:
            if count > 0:
                print(f"{indent}{name}: {count} 个文件 (含子目录总计 {total})")
            else:
                print(f"{indent}{name}: 总计 {total} 个文件")
        else:
            print(f"{indent}{name}: {count} 个文件")

    for child_name in sorted([k for k in node.keys() if not k.startswith("__")]):
        print_tree(node[child_name], level + 1, child_name)


def print_results(tree, total_files):
    print("\n================ 数据集统计 ================\n")
    print(f"总文件数: {total_files}\n")

    root_count = tree.get("__count__", 0)
    if root_count:
        root_total = tree.get("__total__", 0)
        print(f"[根目录] {root_count} 个文件 (含子目录总计 {root_total})\n")

    for top in sorted([k for k in tree.keys() if not k.startswith("__")]):
        print_tree(tree[top], 0, top)
        print("")

    print("============================================\n")


if __name__ == "__main__":
    results, total_files = count_dataset(DATASET_ROOT)
    print_results(results, total_files)
