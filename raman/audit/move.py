"""按路径或审核清单移动异常数据到数据集 delete 目录"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from raman.audit.common import PROJECT_ROOT, resolve_dataset
from raman.audit.config import DEFAULT_AUDIT_CONFIG


@dataclass(frozen=True)
class MoveItem:
    source: Path
    destination: Path
    relative: Path
    reason: str


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def normalize_input(text: str) -> Path:
    return Path(str(text).strip().strip('"').strip("'"))


def find_unique_folder_by_name(init_root: Path, name: str) -> Path | None:
    matches = sorted(path for path in init_root.glob(f"*/{name}") if path.is_dir())
    if not matches:
        return None
    if len(matches) > 1:
        joined = "\n".join(f"  - {path}" for path in matches)
        raise ValueError(f"Folder name is ambiguous: {name}\n{joined}")
    return matches[0]


def source_from_input(raw: Path, dataset_dir: Path, init_root: Path, allow_genus: bool) -> tuple[Path, Path]:
    """解析输入路径并返回源路径和去掉 init 后的相对路径"""
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
    reason = str(reason or "").strip()
    if not reason:
        raise ValueError("Move reason is required")
    labels = [label.strip() for label in reason.replace(",", ";").split(";") if label.strip()]
    bad = [label for label in labels if label not in DEFAULT_AUDIT_CONFIG.delete_reason_labels]
    if bad:
        allowed = "、".join(DEFAULT_AUDIT_CONFIG.delete_reason_labels)
        raise ValueError(f"Unsupported reason label: {';'.join(bad)}. Allowed: {allowed}")
    return "；".join(dict.fromkeys(labels))


def build_item(text: str, reason: str, dataset_dir: Path, init_root: Path, delete_root: Path, allow_genus: bool) -> MoveItem:
    source, rel = source_from_input(normalize_input(text), dataset_dir, init_root, allow_genus)
    destination = (delete_root / rel).resolve()
    if not is_relative_to(destination, delete_root.resolve()):
        raise ValueError(f"Destination escapes delete root: {destination}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists, refusing to overwrite: {destination}")
    return MoveItem(source=source, destination=destination, relative=rel, reason=normalize_reason(reason))


def read_items_from_list(path: Path, fallback_reason: str | None = None) -> list[tuple[str, str]]:
    """读取 full_scan 生成的 delete_candidates.csv 或手工 txt 清单"""
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
                reason = row.get("reason_labels") or fallback_reason
                items.append((rel_path, reason or ""))
            return items
    if fallback_reason is None:
        raise ValueError("TXT manifest requires --reason because it has no reason_labels column")
    return [(line.strip(), fallback_reason) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_record_lines(delete_root: Path, destination: Path, rel: Path, reason: str) -> dict[Path, list[tuple[str, str, str]]]:
    """生成按属写入的移除记录内容"""
    if len(rel.parts) < 2:
        return {}
    genus = rel.parts[0]
    folder = rel.parts[1]
    records: dict[Path, list[tuple[str, str, str]]] = {}
    genus_record = delete_root / genus / "移除记录.txt"
    if destination.is_dir():
        files = sorted(path for path in destination.rglob("*.arc_data") if path.is_file())
        for file in files:
            file_rel = file.relative_to(delete_root)
            file_folder = file_rel.parts[1] if len(file_rel.parts) > 1 else folder
            file_record = delete_root / file_rel.parts[0] / "移除记录.txt"
            records.setdefault(file_record, []).append((file_folder, file.name, reason))
    else:
        records.setdefault(genus_record, []).append((folder, destination.name, reason))
    return records


def append_delete_records(delete_root: Path, items: list[MoveItem]):
    grouped: dict[Path, list[tuple[str, str, str]]] = {}
    for item in items:
        for record_path, rows in collect_record_lines(delete_root, item.destination, item.relative, item.reason).items():
            grouped.setdefault(record_path, []).extend(rows)

    for record_path, rows in grouped.items():
        existing = record_path.read_text(encoding="utf-8") if record_path.exists() else ""
        sections: dict[str, list[tuple[str, str]]] = {}
        for folder, filename, reason in rows:
            sections.setdefault(folder, []).append((filename, reason))
        chunks = [existing.rstrip(), ""] if existing.strip() else []
        for folder in sorted(sections):
            chunks.append(folder)
            for filename, reason in sections[folder]:
                chunks.append(f"{filename}\t{reason}")
            chunks.append("")
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")


def execute_items(delete_root: Path, items: list[MoveItem], dry_run: bool):
    print("Dry-run move plan:" if dry_run else "Move plan:")
    for item in items:
        print(f"{item.source} -> {item.destination} | {item.reason}")
    if dry_run:
        print("Dry-run only. No files were moved.")
        return
    for item in items:
        item.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item.source), str(item.destination))
    append_delete_records(delete_root, items)
    print(f"Moved {len(items)} item(s).")


def move_items(dataset, paths=None, from_list=None, reason=None, dry_run=False, allow_genus=False):
    """移动显式路径或 full_scan 清单中的候选数据"""
    profile, dataset_dir = resolve_dataset(dataset, PROJECT_ROOT)
    init_root = dataset_dir / profile.root_init
    delete_root = dataset_dir / "delete"
    if not init_root.is_dir():
        raise FileNotFoundError(f"Missing init root: {init_root}")

    raw_items: list[tuple[str, str]] = []
    if from_list:
        raw_items.extend(read_items_from_list(Path(from_list), reason))
    for path in paths or []:
        raw_items.append((path, reason or ""))
    if not raw_items and from_list:
        print("No paths to move. The candidate list is empty.")
        return []
    if not raw_items:
        raise ValueError("No paths to move. Use --path or --from-list")

    items = [build_item(path, item_reason, dataset_dir, init_root, delete_root, allow_genus) for path, item_reason in raw_items]
    execute_items(delete_root, items, dry_run)
    return items


def build_parser():
    parser = argparse.ArgumentParser(description="把 init 下的异常数据移动到 dataset/<数据集名>/delete")
    parser.add_argument("dataset", nargs="?", default="细菌", help="数据集名或 profile id")
    parser.add_argument("paths", nargs="*", help="要移动的文件夹或文件路径")
    parser.add_argument("--path", action="append", default=[], help="要移动的文件夹或文件路径，可重复")
    parser.add_argument("--from-list", default=None, help="full_scan 生成的 delete_candidates.csv")
    parser.add_argument("--reason", default=None, help="手动移动原因：残留宇宙射线 / 阶梯谱 / 粗糙噪声 / 参考组离群 / 组内离群，可用分号连接")
    parser.add_argument("--dry-run", action="store_true", help="只打印移动计划，不移动文件")
    parser.add_argument("--allow-genus", action="store_true", help="允许移动整属目录")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = list(args.paths) + list(args.path)
    move_items(
        args.dataset,
        paths=paths,
        from_list=args.from_list,
        reason=args.reason,
        dry_run=args.dry_run,
        allow_genus=args.allow_genus,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
