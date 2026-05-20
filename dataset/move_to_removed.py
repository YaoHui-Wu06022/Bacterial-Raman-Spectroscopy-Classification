"""Move init spectra/folders from a dataset into dataset/移除数据.

Examples:
    python dataset/move_to_removed.py Burkholderia/BCC01 --dry-run
    python dataset/move_to_removed.py Burkholderia/BCC01
    python dataset/move_to_removed.py Burkholderia/BCC01/CELL8_Area01_000_shift.arc_data
    python dataset/move_to_removed.py BCC01 --dry-run

The destination preserves the path under dataset/<dataset>/init:
    dataset/细菌/init/Burkholderia/BCC01
        -> dataset/移除数据/Burkholderia/BCC01
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "细菌"
REMOVED_DIR_NAME = "移除数据"


@dataclass(frozen=True)
class MovePlan:
    source: Path
    destination: Path
    relative: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Folder or file path under dataset/<dataset>/init.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset folder name. Default: 细菌.")
    parser.add_argument("--dry-run", action="store_true", help="Only print move plan, do not move files.")
    parser.add_argument("--allow-genus", action="store_true", help="Allow moving a whole genus folder such as Burkholderia.")
    return parser.parse_args()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_existing(path: Path) -> Path:
    return path.resolve(strict=True)


def normalize_input(text: str) -> Path:
    text = text.strip().strip('"').strip("'")
    return Path(text)


def find_unique_folder_by_name(init_root: Path, name: str) -> Path | None:
    matches = sorted(path for path in init_root.glob(f"*/{name}") if path.is_dir())
    if not matches:
        return None
    if len(matches) > 1:
        joined = "\n".join(f"  - {path}" for path in matches)
        raise ValueError(f"Folder name is ambiguous: {name}\n{joined}")
    return matches[0]


def source_from_input(raw: Path, dataset_dir: Path, init_root: Path, allow_genus: bool) -> Path:
    if raw.is_absolute():
        source = resolve_existing(raw)
    else:
        candidates = []
        parts = raw.parts
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
                source = resolve_existing(candidate)
                break
        if source is None and len(parts) == 1:
            matched = find_unique_folder_by_name(init_root, parts[0])
            if matched is not None:
                source = resolve_existing(matched)
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
        raise ValueError(
            f"Refusing to move a whole genus folder by default: {source}\n"
            "Pass --allow-genus if this is intentional."
        )
    return source


def build_plan(text: str, dataset_dir: Path, init_root: Path, removed_root: Path, allow_genus: bool) -> MovePlan:
    source = source_from_input(normalize_input(text), dataset_dir, init_root, allow_genus)
    rel = source.relative_to(init_root.resolve())
    destination = removed_root.resolve() / rel

    if not is_relative_to(destination, removed_root.resolve()):
        raise ValueError(f"Destination escapes removed root: {destination}")
    if destination.exists():
        raise FileExistsError(f"Destination already exists, refusing to overwrite: {destination}")
    return MovePlan(source=source, destination=destination, relative=rel)


def execute_plan(plan: MovePlan, dry_run: bool) -> None:
    print(f"{plan.source} -> {plan.destination}")
    if dry_run:
        return
    plan.destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(plan.source), str(plan.destination))


def main() -> int:
    args = parse_args()
    dataset_dir = PROJECT_ROOT / "dataset" / args.dataset
    init_root = dataset_dir / "init"
    removed_root = PROJECT_ROOT / "dataset" / REMOVED_DIR_NAME

    if not init_root.is_dir():
        raise FileNotFoundError(f"Missing init root: {init_root}")
    removed_root.mkdir(parents=True, exist_ok=True)

    plans = [
        build_plan(text, dataset_dir, init_root, removed_root, args.allow_genus)
        for text in args.paths
    ]

    print("Move plan:" if not args.dry_run else "Dry-run move plan:")
    for plan in plans:
        execute_plan(plan, args.dry_run)
    if args.dry_run:
        print("Dry-run only. No files were moved.")
    else:
        print(f"Moved {len(plans)} item(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
