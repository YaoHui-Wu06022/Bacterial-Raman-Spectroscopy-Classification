"""流水线元数据的 SQLite 存储、迁移与对账。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import shutil
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

from ramanv2.common.file_ops import create_temporary_path, publish_file
from ramanv2.data.arc_data import list_arc_data_paths


DATABASE_FILE_NAME = "pipeline.db"
SCHEMA_VERSION = 2


def database_path(metadata_dir: Path | str) -> Path:
    return Path(metadata_dir).resolve() / DATABASE_FILE_NAME


@contextmanager
def connect(path: Path | str) -> Iterator[sqlite3.Connection]:
    """打开单次操作连接；DELETE 日志便于 preserve_paths 直接快照 DB 文件。"""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    try:
        version = _schema_version(connection)
        if version is None and not _has_tables(connection):
            initialize_schema(connection)
        elif version != SCHEMA_VERSION:
            raise ValueError(
                f"流水线数据库版本不受支持：{target}（{version}）；"
                "v1 数据集请运行 pipeline upgrade-db --dataset <id>"
            )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA busy_timeout = 5000")
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


def _has_tables(connection: sqlite3.Connection) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1"
    ).fetchone() is not None


def _schema_version(connection: sqlite3.Connection) -> int | None:
    has_meta = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
    ).fetchone()
    if not has_meta:
        return None
    row = connection.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    return int(row[0]) if row is not None else None


def upgrade_database(path: Path | str) -> dict[str, object]:
    """备份 v1 元数据，在副本中移除逐谱投影，验证后发布 v2。"""
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(f"流水线数据库不存在：{target}")
    before = target.stat()
    staged = create_temporary_path(target)
    backup_stage = create_temporary_path(target)
    backup = target.with_suffix(".v1.bak")
    try:
        with closing(sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)) as source:
            version = _schema_version(source)
            if version == SCHEMA_VERSION:
                return {"database": str(target), "schema_version": version, "status": "already_current"}
            if version != 1:
                raise ValueError(f"只支持将流水线数据库从 v1 升到 v2，当前版本为 {version}")
            with closing(sqlite3.connect(backup_stage)) as snapshot:
                source.backup(snapshot)
        shutil.copy2(backup_stage, staged)
        with closing(sqlite3.connect(backup_stage)) as previous, closing(sqlite3.connect(staged)) as updated:
            with updated:
                updated.execute("DROP TABLE files")
                updated.execute(
                    "UPDATE meta SET value = ? WHERE key = 'schema_version'", (str(SCHEMA_VERSION),)
                )
                _validate_database_upgrade(previous, updated)
        current = target.stat()
        if (current.st_mtime_ns, current.st_size) != (before.st_mtime_ns, before.st_size):
            raise RuntimeError("升级期间数据库发生变化，请停止流水线写入后重试")
        # 上次发布若失败，保留原备份，并为新的本次快照分配独立路径。
        if backup.exists():
            backup = create_temporary_path(backup)
        publish_file(backup_stage, backup)
        publish_file(staged, target)
    finally:
        staged.unlink(missing_ok=True)
        backup_stage.unlink(missing_ok=True)
    return {"database": str(target), "backup": str(backup), "schema_version": SCHEMA_VERSION, "status": "upgraded"}


def _validate_database_upgrade(previous: sqlite3.Connection, updated: sqlite3.Connection) -> None:
    """确认数据库完整且除派生投影和版本号外的记录一致。"""
    if _schema_version(updated) != SCHEMA_VERSION:
        raise RuntimeError("数据库升级后版本检查失败")
    if updated.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
        raise RuntimeError("数据库升级后完整性检查失败")
    if updated.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise RuntimeError("数据库升级后外键检查失败")
    for table in ("meta", "lineage", "pipeline_state", "folders", "outputs", "test_defaults"):
        where = " WHERE key != 'schema_version'" if table == "meta" else ""
        query = f"SELECT * FROM {table}{where} ORDER BY rowid"
        if any(old != new for old, new in zip_longest(previous.execute(query), updated.execute(query))):
            raise RuntimeError(f"数据库升级后记录不一致：{table}")


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
    connection.execute("DELETE FROM outputs WHERE source_id = ?", (source_id,))
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
    "upgrade_database",
]