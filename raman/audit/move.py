"""按路径或 audit 清单移动异常数据到 delete"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from raman.audit.config import DEFAULT_AUDIT_CONFIG
from raman.tool.dataset import resolve_dataset
from raman.tool.path import PROJECT_ROOT, is_relative_to


@dataclass(frozen=True)
class MoveItem:
    """待移动文件和目标信息"""

    source: Path
    destination: Path
    relative: Path
    reason: str
    category: str = ""


def normalize_category(category: str | None) -> str:
    """校验 delete 分类目录名称"""
    category = str(category or "").strip().strip("/\\")
    if not category:
        return ""
    if category not in DEFAULT_AUDIT_CONFIG.delete_categories:
        allowed = "、".join(DEFAULT_AUDIT_CONFIG.delete_categories)
        raise ValueError(f"Unsupported delete category: {category}. Allowed: {allowed}")
    return category


def find_unique_folder_by_name(init_root: Path, name: str) -> Path | None:
    """按末级文件夹名查找唯一 init 子文件夹"""
    matches = sorted(path for path in init_root.glob(f"*/{name}") if path.is_dir())
    if not matches:
        return None
    if len(matches) > 1:
        joined = "\n".join(f"  - {path}" for path in matches)
        raise ValueError(f"Folder name is ambiguous: {name}\n{joined}")
    return matches[0]


def source_from_input(raw: Path, dataset_dir: Path, init_root: Path, allow_genus: bool) -> tuple[Path, Path]:
    """解析待移动源路径及其相对 init 路径"""
    if raw.is_absolute():
        source = raw.resolve(strict=True)
    else:
        parts = raw.parts
        candidates = []
        if parts and parts[0] == "dataset":
            candidates.append(PROJECT_ROOT / raw)
        if parts and parts[0] == init_root.name:
            candidates.append(dataset_dir / raw)
            if len(parts) > 1:
                candidates.append(init_root / Path(*parts[1:]))
        else:
            candidates.append(init_root / raw)
            candidates.append(dataset_dir / raw)

        source = None
        for candidate in candidates:
            if candidate.exists():
                source = candidate.resolve(strict=True)
                break
        if source is None and len(parts) == 1:
            matched = find_unique_folder_by_name(init_root, parts[0])
            if matched is not None:
                source = matched.resolve(strict=True)
        if source is None:
            raise FileNotFoundError(f"Cannot find source under {init_root}: {raw}")

    init_root_resolved = init_root.resolve()
    dataset_dir_resolved = dataset_dir.resolve()
    if is_relative_to(source, init_root_resolved):
        rel = source.relative_to(init_root_resolved)
    elif is_relative_to(source, dataset_dir_resolved):
        rel = source.relative_to(dataset_dir_resolved)
        if not rel.parts or rel.parts[0] != init_root.name:
            raise ValueError(f"Source is under dataset but not under init: {source}")
        rel = Path(*rel.parts[1:])
    else:
        raise ValueError(f"Source must be inside {init_root_resolved}: {source}")

    if len(rel.parts) < 2 and not allow_genus:
        raise ValueError(f"Refusing to move a whole genus folder by default: {source}\nPass --allow-genus if this is intentional")
    return source, rel


def normalize_reason(reason: str) -> str:
    """校验并规范化移除原因标签"""
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError("Move reason is required")
    labels = [label.strip() for label in reason.replace(",", ";").split(";") if label.strip()]
    bad = [label for label in labels if label not in DEFAULT_AUDIT_CONFIG.delete_categories]
    if bad:
        allowed = "、".join(DEFAULT_AUDIT_CONFIG.delete_categories)
        raise ValueError(f"Unsupported reason label: {';'.join(bad)}. Allowed: {allowed}")
    return "；".join(dict.fromkeys(labels))


def resolve_candidate_list_path(text: str, dataset_dir: Path) -> Path:
    """把清单文件、扫描目录或时间戳目录名解析成候选 CSV"""
    raw = Path(str(text).strip().strip('"').strip("'"))
    candidates = [raw] if raw.is_absolute() else [
        Path.cwd() / raw,
        PROJECT_ROOT / raw,
        dataset_dir / raw,
        dataset_dir / "audit_full_scan" / raw,
    ]

    seen = set()
    tried = []
    missing_csv_dirs = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        tried.append(str(candidate))
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            csv_path = candidate / "delete_candidates.csv"
            if csv_path.is_file():
                return csv_path
            missing_csv_dirs.append(candidate)

    joined = "\n".join(f"  - {path}" for path in tried)
    if missing_csv_dirs:
        missing = "\n".join(f"  - {path}" for path in missing_csv_dirs)
        raise FileNotFoundError(f"Missing delete_candidates.csv under matched audit output folder:\n{missing}")
    raise FileNotFoundError(f"Cannot find candidate list or audit output folder:\n{joined}")


def build_item(
    text: str,
    reason: str,
    category: str,
    dataset_dir: Path,
    init_root: Path,
    delete_root: Path,
    allow_genus: bool,
) -> MoveItem:
    """根据输入路径构造一次移动任务"""
    raw = Path(str(text).strip().strip('"').strip("'"))
    source, rel = source_from_input(raw, dataset_dir, init_root, allow_genus)
    category = normalize_category(category)
    destination_root = (delete_root / category).resolve() if category else delete_root.resolve()
    destination = (destination_root / rel).resolve()
    if not is_relative_to(destination, destination_root):
        raise ValueError(f"Destination escapes delete root: {destination}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists, refusing to overwrite: {destination}")
    return MoveItem(source=source, destination=destination, relative=rel, reason=normalize_reason(reason), category=category)


def read_items_from_list(path: Path, fallback_reason: str | None = None, fallback_category: str | None = None) -> list[tuple[str, str, str]]:
    """从 CSV 或 TXT 清单读取移动条目"""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing list file: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            items = []
            for row in reader:
                rel_path = row.get("rel_path") or row.get("path")
                if not rel_path:
                    continue
                reason = row.get("reason_labels") or fallback_reason or ""
                category = row.get("delete_category") or fallback_category or ""
                items.append((rel_path, reason, category))
            return items
    if fallback_reason is None:
        raise ValueError("TXT manifest requires --reason because it has no reason_labels column")
    return [(line.strip(), fallback_reason, fallback_category or "") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_category_record(delete_root: Path, items: list[MoveItem]):
    """写入分类 delete 目录的移除记录"""
    by_category: dict[str, list[MoveItem]] = {}
    for item in items:
        if item.category:
            by_category.setdefault(item.category, []).append(item)
    for category, category_items in by_category.items():
        record_path = delete_root / category / "移除记录.txt"
        existing = record_path.read_text(encoding="utf-8") if record_path.exists() else ""
        lines = [existing.rstrip(), ""] if existing.strip() else []
        for item in category_items:
            lines.append(f"{item.relative.as_posix()}\t{item.reason}")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _append_legacy_records(delete_root: Path, items: list[MoveItem]):
    """兼容旧结构写入属目录移除记录"""
    grouped: dict[Path, list[tuple[str, str, str]]] = {}
    for item in items:
        if item.category or len(item.relative.parts) < 2:
            continue
        genus = item.relative.parts[0]
        folder = item.relative.parts[1]
        record_path = delete_root / genus / "移除记录.txt"
        grouped.setdefault(record_path, []).append((folder, item.destination.name, item.reason))

    for record_path, rows in grouped.items():
        existing = record_path.read_text(encoding="utf-8") if record_path.exists() else ""
        chunks = [existing.rstrip(), ""] if existing.strip() else []
        sections: dict[str, list[tuple[str, str]]] = {}
        for folder, filename, reason in rows:
            sections.setdefault(folder, []).append((filename, reason))
        for folder in sorted(sections):
            chunks.append(folder)
            for filename, reason in sections[folder]:
                chunks.append(f"{filename}\t{reason}")
            chunks.append("")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def execute_items(delete_root: Path, items: list[MoveItem], dry_run: bool):
    """打印并执行移动任务"""
    print("Dry-run move plan:" if dry_run else "Move plan:")
    for item in items:
        category = f" | category={item.category}" if item.category else ""
        print(f"{item.source} -> {item.destination} | {item.reason}{category}")
    if dry_run:
        print("Dry-run only. No files were moved.")
        return
    for item in items:
        item.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item.source), str(item.destination))
    _append_category_record(delete_root, items)
    _append_legacy_records(delete_root, items)
    print(f"Moved {len(items)} item(s).")


def move_items(dataset, paths=None, from_list=None, reason=None, dry_run=False, allow_genus=False, category=None):
    """移动指定路径或清单中的 init 数据到 delete"""
    profile, dataset_dir = resolve_dataset(dataset, PROJECT_ROOT)
    init_root = dataset_dir / profile.root_init
    delete_root = dataset_dir / "delete"
    if not init_root.is_dir():
        raise FileNotFoundError(f"Missing init root: {init_root}")

    raw_items: list[tuple[str, str, str]] = []
    if from_list:
        list_path = resolve_candidate_list_path(from_list, dataset_dir)
        raw_items.extend(read_items_from_list(list_path, reason, category))
    for path in paths or []:
        raw_items.append((path, reason or "", category or ""))
    if not raw_items and from_list:
        print("No paths to move. The candidate list is empty.")
        return []
    if not raw_items:
        raise ValueError("No paths to move. Use --path or --from-list")

    items = [
        build_item(path, item_reason, item_category, dataset_dir, init_root, delete_root, allow_genus)
        for path, item_reason, item_category in raw_items
    ]
    execute_items(delete_root, items, dry_run)
    return items


def build_parser():
    """构建 move 子命令参数解析器"""
    parser = argparse.ArgumentParser(description="把 init 下的数据移动到 dataset/<数据集>/delete")
    parser.add_argument("dataset", nargs="?", default="细菌", help="数据集名或 profile id")
    parser.add_argument("paths", nargs="*", help="要移动的文件夹或文件路径")
    parser.add_argument("--path", action="append", default=[], help="要移动的文件夹或文件路径，可重复")
    parser.add_argument("--from-list", default=None, help="audit 输出目录名、输出目录路径或 delete_candidates.csv")
    parser.add_argument("--reason", default=None, help="手工移动原因；CSV 清单优先使用 reason_labels 列")
    parser.add_argument("--category", default=None, choices=DEFAULT_AUDIT_CONFIG.delete_categories, help="移动到 delete 下的分类目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印移动计划，不移动文件")
    parser.add_argument("--allow-genus", action="store_true", help="允许移动整属目录")
    return parser


def main(argv=None):
    """执行 move 子命令"""
    args = build_parser().parse_args(argv)
    paths = list(args.paths) + list(args.path)
    move_items(
        args.dataset,
        paths=paths,
        from_list=args.from_list,
        reason=args.reason,
        dry_run=args.dry_run,
        allow_genus=args.allow_genus,
        category=args.category,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
