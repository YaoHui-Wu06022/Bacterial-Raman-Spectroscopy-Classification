"""流水线元数据的 SQLite 存储、迁移与对账。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ramanv2.common.arc_data import list_arc_data_paths


DATABASE_FILE_NAME = "pipeline.db"
SCHEMA_VERSION = 1


def database_path(metadata_dir: Path | str) -> Path:
    return Path(metadata_dir).resolve() / DATABASE_FILE_NAME


@contextmanager
def connect(path: Path | str) -> Iterator[sqlite3.Connection]:
    """打开单次操作连接；DELETE 日志便于 preserve_paths 直接快照 DB 文件。"""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA busy_timeout = 5000")
    try:
        initialize_schema(connection)
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS lineage (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pipeline_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS folders (
            source_id TEXT PRIMARY KEY,
            date TEXT NOT NULL,
            folder TEXT NOT NULL,
            prefix TEXT NOT NULL,
            genus TEXT NOT NULL,
            folder_status TEXT NOT NULL,
            active_file_count INTEGER NOT NULL DEFAULT 0,
            deleted_file_count INTEGER NOT NULL DEFAULT 0,
            init_pending INTEGER NOT NULL DEFAULT 0,
            input_fingerprint TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS files (
            source_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            status TEXT NOT NULL,
            source_path TEXT,
            deleted_path TEXT,
            reason TEXT,
            recorded_at TEXT,
            PRIMARY KEY (source_id, file_name),
            FOREIGN KEY (source_id) REFERENCES folders(source_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS outputs (
            source_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            path TEXT,
            file_count INTEGER,
            output_fingerprint TEXT,
            PRIMARY KEY (source_id, kind),
            FOREIGN KEY (source_id) REFERENCES folders(source_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS test_defaults (
            source_id TEXT PRIMARY KEY,
            rank INTEGER NOT NULL DEFAULT 0,
            reason TEXT,
            updated_at TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_lineage_payload(path: Path | str, payload: dict[str, object]) -> None:
    """保存完整 lineage 载荷，只重建内容有变化的文件夹及其文件/产物索引。"""
    text = json.dumps(payload, ensure_ascii=False) + "\n"
    with connect(path) as connection:
        row = connection.execute("SELECT payload_json FROM lineage WHERE id = 1").fetchone()
        previous_text = str(row["payload_json"]) if row is not None else None
        if previous_text == text:
            return
        previous_folders = _previous_folders(previous_text)
        connection.execute(
            "INSERT INTO lineage(id, payload_json, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at",
            (text, _now()),
        )
        folders = payload.get("folders", {})
        if not isinstance(folders, dict):
            return
        for source_id in set(previous_folders) - set(folders):
            connection.execute("DELETE FROM folders WHERE source_id = ?", (str(source_id),))
        for source_id, record in folders.items():
            if not isinstance(record, dict) or previous_folders.get(source_id) == record:
                continue
            _write_folder_index(connection, str(source_id), record)


def _previous_folders(payload_text: str | None) -> dict[str, object]:
    if payload_text is None:
        return {}
    payload = json.loads(payload_text)
    folders = payload.get("folders", {}) if isinstance(payload, dict) else {}
    return folders if isinstance(folders, dict) else {}


def _write_folder_index(
    connection: sqlite3.Connection,
    source_id: str,
    record: dict[str, object],
) -> None:
    """写入单个文件夹的行；保留已存的输入指纹，由 db-check 判断是否过期。"""
    connection.execute(
        "INSERT INTO folders(source_id, date, folder, prefix, genus, folder_status, "
        "active_file_count, deleted_file_count, init_pending, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(source_id) DO UPDATE SET date=excluded.date, folder=excluded.folder, "
        "prefix=excluded.prefix, genus=excluded.genus, folder_status=excluded.folder_status, "
        "active_file_count=excluded.active_file_count, deleted_file_count=excluded.deleted_file_count, "
        "init_pending=excluded.init_pending, updated_at=excluded.updated_at",
        (
            source_id,
            str(record.get("date", "")),
            str(record.get("folder", "")),
            str(record.get("prefix", "")),
            str(record.get("genus", "")),
            str(record.get("folder_status", "")),
            int(record.get("active_file_count") or 0),
            int(record.get("deleted_file_count") or 0),
            int(bool(record.get("init_pending"))),
            str(record.get("updated_at") or _now()),
        ),
    )
    connection.execute("DELETE FROM files WHERE source_id = ?", (source_id,))
    connection.execute("DELETE FROM outputs WHERE source_id = ?", (source_id,))
    files = record.get("files", {})
    if isinstance(files, dict):
        for file_name, entry in files.items():
            if not isinstance(entry, dict):
                continue
            connection.execute(
                "INSERT OR REPLACE INTO files(source_id, file_name, status, source_path, "
                "deleted_path, reason, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    source_id,
                    str(file_name),
                    str(entry.get("status", "")),
                    _none_string(entry.get("source_path")),
                    _none_string(entry.get("deleted_path")),
                    _none_string(entry.get("reason")),
                    _none_string(entry.get("recorded_at")),
                ),
            )
    outputs = record.get("outputs", {})
    if isinstance(outputs, dict):
        for kind, output_path in outputs.items():
            if kind in {"profiles", "prefix_groups"} or kind.endswith("_updated_at") or output_path is None:
                continue
            if not isinstance(output_path, (str, Path)):
                continue
            connection.execute(
                "INSERT OR REPLACE INTO outputs(source_id, kind, path) VALUES (?, ?, ?)",
                (source_id, str(kind), str(output_path)),
            )


def load_lineage_payload(path: Path | str) -> dict[str, object] | None:
    with connect(path) as connection:
        row = connection.execute("SELECT payload_json FROM lineage WHERE id = 1").fetchone()
    if row is None:
        return None
    payload = json.loads(str(row["payload_json"]))
    return payload if isinstance(payload, dict) else None


def save_state_payload(path: Path | str, payload: dict[str, object]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    with connect(path) as connection:
        connection.execute(
            "INSERT INTO pipeline_state(id, payload_json, updated_at) VALUES (1, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json, updated_at=excluded.updated_at",
            (text, _now()),
        )


def load_state_payload(path: Path | str) -> dict[str, object] | None:
    with connect(path) as connection:
        row = connection.execute("SELECT payload_json FROM pipeline_state WHERE id = 1").fetchone()
    if row is None:
        return None
    payload = json.loads(str(row["payload_json"]))
    return payload if isinstance(payload, dict) else None






def save_test_defaults(path: Path | str, folders: list[str], reason: str | None = None) -> None:
    with connect(path) as connection:
        connection.execute("DELETE FROM test_defaults")
        timestamp = _now()
        for rank, source_id in enumerate(folders):
            connection.execute(
                "INSERT INTO test_defaults(source_id, rank, reason, updated_at) VALUES (?, ?, ?, ?)",
                (str(source_id), rank, reason, timestamp),
            )


def load_test_defaults(path: Path | str) -> list[str]:
    with connect(path) as connection:
        rows = connection.execute(
            "SELECT source_id FROM test_defaults ORDER BY rank, source_id"
        ).fetchall()
    return [str(row["source_id"]) for row in rows]






def fingerprint_paths(paths: list[Path], root: Path) -> str:
    """按相对名、大小和 mtime 生成稳定指纹，不读取光谱内容。"""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def refresh_fingerprints(metadata_dir: Path | str, cosmic_root: Path | str) -> dict[str, int]:
    """重算来源文件夹和已有产物的指纹，供 db-check 比较。"""
    directory = Path(metadata_dir).resolve()
    db_path = database_path(directory)
    cosmic = Path(cosmic_root).resolve()
    folders_updated = 0
    outputs_updated = 0
    with connect(db_path) as connection:
        for row in connection.execute("SELECT source_id, date, folder FROM folders").fetchall():
            source_id = str(row["source_id"])
            source = cosmic / str(row["date"]) / str(row["folder"])
            paths = list_arc_data_paths(source)
            fingerprint = fingerprint_paths(paths, source) if paths else None
            connection.execute(
                "UPDATE folders SET input_fingerprint = ? WHERE source_id = ?",
                (fingerprint, source_id),
            )
            folders_updated += 1
        for row in connection.execute("SELECT source_id, kind, path FROM outputs").fetchall():
            path = row["path"]
            if not path:
                continue
            target = Path(str(path))
            if not target.is_dir():
                connection.execute(
                    "UPDATE outputs SET file_count = 0, output_fingerprint = NULL "
                    "WHERE source_id = ? AND kind = ?",
                    (str(row["source_id"]), str(row["kind"])),
                )
                continue
            paths = list_arc_data_paths(target, recursive=True)
            fingerprint = fingerprint_paths(paths, target) if paths else None
            connection.execute(
                "UPDATE outputs SET file_count = ?, output_fingerprint = ? "
                "WHERE source_id = ? AND kind = ?",
                (len(paths), fingerprint, str(row["source_id"]), str(row["kind"])),
            )
            outputs_updated += 1
    return {"folders_updated": folders_updated, "outputs_updated": outputs_updated}


def check_database(metadata_dir: Path | str, cosmic_root: Path | str) -> dict[str, object]:
    """报告 DB 与磁盘的文件夹/产物指纹差异，不修改光谱。"""
    directory = Path(metadata_dir).resolve()
    db_path = database_path(directory)
    cosmic = Path(cosmic_root).resolve()
    stale_folders: list[str] = []
    stale_outputs: list[str] = []
    missing_outputs: list[str] = []
    with connect(db_path) as connection:
        for row in connection.execute("SELECT source_id, date, folder, input_fingerprint FROM folders").fetchall():
            source = cosmic / str(row["date"]) / str(row["folder"])
            paths = list_arc_data_paths(source)
            current = fingerprint_paths(paths, source) if paths else None
            if current != row["input_fingerprint"]:
                stale_folders.append(str(row["source_id"]))
        for row in connection.execute("SELECT source_id, kind, path, output_fingerprint FROM outputs").fetchall():
            target = Path(str(row["path"])) if row["path"] else None
            if target is None or not target.is_dir():
                missing_outputs.append(f"{row['source_id']}:{row['kind']}")
                continue
            paths = list_arc_data_paths(target, recursive=True)
            current = fingerprint_paths(paths, target) if paths else None
            if current != row["output_fingerprint"]:
                stale_outputs.append(f"{row['source_id']}:{row['kind']}")
    return {
        "database": str(db_path),
        "stale_folders": stale_folders,
        "stale_outputs": stale_outputs,
        "missing_outputs": missing_outputs,
    }


def _none_string(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "DATABASE_FILE_NAME",
    "check_database",
    "connect",
    "database_path",
    "load_lineage_payload",
    "load_state_payload",
    "load_test_defaults",
    "refresh_fingerprints",
    "save_lineage_payload",
    "save_state_payload",
    "save_test_defaults",
]