"""Consistent, single-session SQLite exports with mandatory ZIP compression."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import zipfile

from .sessiondb import SessionDB


_TABLES = (
    "sessions",
    "runs",
    "messages",
    "events",
    "artifacts",
    "documents",
    "tool_outcomes",
    "final_answers",
    "run_checkpoints",
    "task_plan_items",
    "subagent_runs",
    "subagent_messages",
    "subagent_events",
    "subagent_tool_outcomes",
)
_MIN_FREE_BYTES = 64 * 1024 * 1024
_OUTPUT_RESERVE_BYTES = 512 * 1024 * 1024
_SOURCE_RESERVE_BYTES = 512 * 1024 * 1024


class SessionExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class SessionExport:
    path: Path
    filename: str
    inner_filename: str
    captured_at: float
    session_status: str
    sqlite_bytes: int
    archive_bytes: int
    estimated_logical_bytes: int
    row_counts: dict[str, int]


def _safe_stem(session_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_-]+", "-", session_id).strip("-_")[:64]
    if not readable:
        readable = hashlib.sha256(session_id.encode()).hexdigest()[:24]
    return "aurex-session-" + readable


def _copy_table(source: sqlite3.Connection, destination: sqlite3.Connection,
                table: str, session_id: str, *, include_private: bool,
                child_messages: dict[int, dict], child_documents: dict[str, dict]) -> int:
    where = "id=?" if table == "sessions" else "session_id=?"
    if table == "subagent_messages" and not include_private:
        where += (" AND EXISTS (SELECT 1 FROM subagent_tool_outcomes t "
                  "WHERE t.session_id=subagent_messages.session_id "
                  "AND t.subagent_id=subagent_messages.subagent_id "
                  "AND t.message_id=subagent_messages.id)")
    cursor = source.execute(f'SELECT * FROM "{table}" WHERE {where}', (session_id,))
    columns = [description[0] for description in cursor.description or ()]
    source_columns = len(columns)
    destination_columns = len(destination.execute(f'PRAGMA table_info("{table}")').fetchall())
    if source_columns != destination_columns:
        raise SessionExportError(f"unsupported {table} schema while exporting session")
    statement = f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in range(source_columns))})'
    count = 0

    def rows():
        # Some archived tool documents are hundreds of MiB.  Feed SQLite one
        # source row at a time so an export cannot materialize a batch of giant
        # strings and starve the running model process.
        nonlocal count
        for row in cursor:
            values = list(row)
            if not include_private:
                if table == "sessions":
                    values[columns.index("owner_id")] = ""
                elif table == "subagent_runs":
                    values[columns.index("context")] = json.dumps({
                        "redacted": True,
                        "reason": "private subagent handoff is not part of an owner export",
                    }, separators=(",", ":"))
                elif table == "subagent_messages":
                    message_id = int(values[columns.index("id")])
                    evidence = child_messages[message_id]
                    values[columns.index("role")] = "tool"
                    values[columns.index("data")] = json.dumps({
                        "role": "tool",
                        "tool_call_id": evidence["call_id"],
                        "content": json.dumps({
                            "ok": bool(evidence["ok"]),
                            "document_id": evidence["document_id"],
                            "message": "private subagent message omitted from owner export",
                        }, separators=(",", ":")),
                    }, separators=(",", ":"))
                elif table == "documents":
                    document_id = str(values[columns.index("id")])
                    evidence = child_documents.get(document_id)
                    if evidence is not None:
                        values[columns.index("content")] = json.dumps({
                            "ok": bool(evidence["ok"]),
                            "redacted": True,
                            "tool": evidence["name"],
                            "message": "private subagent tool payload omitted from owner export",
                        }, separators=(",", ":"))
            count += 1
            yield values

    destination.executemany(statement, rows())
    return count


def _orphan_count(db: sqlite3.Connection, sql: str) -> int:
    return int(db.execute(sql).fetchone()[0])


def _estimate_session_bytes(db: sqlite3.Connection, session_id: str) -> int:
    """Conservative logical payload estimate computed inside the read snapshot."""
    total = 0
    for table in _TABLES:
        columns = [str(row[1]) for row in db.execute(f'PRAGMA table_info("{table}")')]
        lengths = "+".join(
            f'COALESCE(length(CAST("{column}" AS BLOB)),0)' for column in columns)
        where = "id=?" if table == "sessions" else "session_id=?"
        # Account for records, varints, indexes and page fragmentation in
        # addition to the logical cell values. Final preflight below keeps a
        # further 3x raw+ZIP margin.
        overhead = 32 + len(columns) * 8
        value = db.execute(
            f'SELECT COALESCE(SUM(({lengths})+{overhead}),0) '
            f'FROM "{table}" WHERE {where}', (session_id,)).fetchone()[0]
        total += max(0, int(value or 0))
    return total


def _check_capacity(source_path: Path, output_dir: Path, estimated_bytes: int) -> None:
    output_reserve = max(
        _MIN_FREE_BYTES, min(_OUTPUT_RESERVE_BYTES, estimated_bytes))
    source_reserve = max(
        _MIN_FREE_BYTES, min(_SOURCE_RESERVE_BYTES, estimated_bytes))
    output_needed = estimated_bytes * 3 + output_reserve
    same_device = source_path.stat().st_dev == output_dir.stat().st_dev
    if same_device:
        if shutil.disk_usage(output_dir).free < output_needed + source_reserve:
            raise SessionExportError(
                "not enough free space for SQLite, compressed output, and active WAL growth")
        return
    if shutil.disk_usage(output_dir).free < output_needed:
        raise SessionExportError("not enough free space for the compressed session export")
    if shutil.disk_usage(source_path.parent).free < source_reserve:
        raise SessionExportError("not enough source-disk reserve for active WAL growth")


def _validate_relations(db: sqlite3.Connection, session_id: str) -> None:
    for table in _TABLES:
        column = "id" if table == "sessions" else "session_id"
        if db.execute(
                f'SELECT 1 FROM "{table}" WHERE "{column}"<>? LIMIT 1',
                (session_id,)).fetchone():
            raise SessionExportError(f"cross-session row detected in exported {table}")

    checks = {
        "message run": "SELECT COUNT(*) FROM messages m WHERE m.run_id<>'' AND NOT EXISTS (SELECT 1 FROM runs r WHERE r.id=m.run_id AND r.session_id=m.session_id)",
        "event run": "SELECT COUNT(*) FROM events e WHERE e.run_id<>'' AND NOT EXISTS (SELECT 1 FROM runs r WHERE r.id=e.run_id AND r.session_id=e.session_id)",
        "tool run": "SELECT COUNT(*) FROM tool_outcomes t WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.id=t.run_id AND r.session_id=t.session_id)",
        "tool document": "SELECT COUNT(*) FROM tool_outcomes t WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id=t.document_id AND d.session_id=t.session_id)",
        "tool message": "SELECT COUNT(*) FROM tool_outcomes t WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.id=t.message_id AND m.session_id=t.session_id)",
        "final run": "SELECT COUNT(*) FROM final_answers f WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.id=f.run_id AND r.session_id=f.session_id)",
        "final message": "SELECT COUNT(*) FROM final_answers f WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.id=f.message_id AND m.session_id=f.session_id)",
        "checkpoint run": "SELECT COUNT(*) FROM run_checkpoints c WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.id=c.run_id AND r.session_id=c.session_id)",
        "plan run": "SELECT COUNT(*) FROM task_plan_items p WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.id=p.run_id AND r.session_id=p.session_id)",
        "subagent parent": "SELECT COUNT(*) FROM subagent_runs s WHERE NOT EXISTS (SELECT 1 FROM runs r WHERE r.id=s.parent_run_id AND r.session_id=s.session_id)",
        "subagent message": "SELECT COUNT(*) FROM subagent_messages m WHERE NOT EXISTS (SELECT 1 FROM subagent_runs s WHERE s.id=m.subagent_id AND s.parent_run_id=m.parent_run_id AND s.session_id=m.session_id)",
        "subagent event": "SELECT COUNT(*) FROM subagent_events e WHERE NOT EXISTS (SELECT 1 FROM subagent_runs s WHERE s.id=e.subagent_id AND s.parent_run_id=e.parent_run_id AND s.session_id=e.session_id)",
        "subagent tool": "SELECT COUNT(*) FROM subagent_tool_outcomes t WHERE NOT EXISTS (SELECT 1 FROM subagent_runs s WHERE s.id=t.subagent_id AND s.parent_run_id=t.parent_run_id AND s.session_id=t.session_id)",
        "subagent tool document": "SELECT COUNT(*) FROM subagent_tool_outcomes t WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id=t.document_id AND d.session_id=t.session_id)",
        "subagent tool message": "SELECT COUNT(*) FROM subagent_tool_outcomes t WHERE NOT EXISTS (SELECT 1 FROM subagent_messages m WHERE m.id=t.message_id AND m.subagent_id=t.subagent_id AND m.session_id=t.session_id)",
    }
    broken = [name for name, sql in checks.items() if _orphan_count(db, sql)]
    if broken:
        raise SessionExportError("incomplete session relationships: " + ", ".join(broken))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def export_session_archive(database_path: str, session_id: str, output_dir: str, *,
                           include_private: bool = True) -> SessionExport:
    """Export one committed point-in-time view without pausing active writers."""
    if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
        raise SessionExportError("invalid session ID")
    if type(include_private) is not bool:
        raise SessionExportError("include_private must be a boolean")
    source_path = Path(database_path).expanduser().resolve()
    if not source_path.is_file():
        raise SessionExportError("session database is unavailable")
    requested_root = Path(output_dir).expanduser()
    if requested_root.is_symlink():
        raise SessionExportError("session export directory must not be a symlink")
    requested_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = requested_root.resolve()
    if not root.is_dir():
        raise SessionExportError("session export directory must be a real directory")
    os.chmod(root, 0o700)
    if shutil.disk_usage(root).free < _MIN_FREE_BYTES:
        raise SessionExportError("not enough free space to create a compressed session export")

    stem = _safe_stem(session_id)
    sqlite_path = root / (stem + ".sqlite3")
    archive_path = root / (stem + ".sqlite3.zip")
    inner_filename = stem + ".sqlite3"
    if sqlite_path.exists() or archive_path.exists():
        raise SessionExportError("session export destination already exists")

    source = destination = None
    captured_at = 0.0
    estimated_logical_bytes = 0
    row_counts: dict[str, int] = {}
    session_status = ""
    try:
        source_uri = source_path.as_uri() + "?mode=ro"
        source = sqlite3.connect(source_uri, uri=True, timeout=30)
        source.execute("PRAGMA query_only=ON")
        source.execute("PRAGMA busy_timeout=30000")
        source.execute("BEGIN")
        session = source.execute(
            "SELECT status FROM sessions WHERE id=?", (session_id,)).fetchone()
        if session is None:
            raise SessionExportError("session not found in export snapshot")
        captured_at = time.time()
        session_status = str(session[0])
        tables = {row[0] for row in source.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
        unknown = tables - set(_TABLES)
        missing = set(_TABLES) - tables
        if unknown or missing:
            detail = []
            if unknown:
                detail.append("unknown=" + ",".join(sorted(unknown)))
            if missing:
                detail.append("missing=" + ",".join(sorted(missing)))
            raise SessionExportError("unsupported session database schema (" + "; ".join(detail) + ")")

        estimated_logical_bytes = _estimate_session_bytes(source, session_id)
        _check_capacity(source_path, root, estimated_logical_bytes)

        child_messages: dict[int, dict] = {}
        child_documents: dict[str, dict] = {}
        if not include_private:
            for row in source.execute("""SELECT message_id,call_id,document_id,name,ok
                    FROM subagent_tool_outcomes WHERE session_id=?""", (session_id,)):
                evidence = {
                    "call_id": str(row[1]), "document_id": str(row[2]),
                    "name": str(row[3]), "ok": bool(row[4]),
                }
                child_messages[int(row[0])] = evidence
                child_documents[str(row[2])] = evidence

        # Build a new canonical database instead of deleting rows from a full
        # backup: no freelist page can retain another session's text.
        SessionDB(str(sqlite_path))
        os.chmod(sqlite_path, 0o600)
        destination = sqlite3.connect(sqlite_path, timeout=30)
        destination.execute("PRAGMA busy_timeout=30000")
        destination.execute("BEGIN IMMEDIATE")
        for table in _TABLES:
            row_counts[table] = _copy_table(
                source, destination, table, session_id,
                include_private=include_private,
                child_messages=child_messages, child_documents=child_documents)
        destination.execute("""CREATE TABLE export_metadata (
            key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
        metadata = {
            "schema": "aurex.session-export.v1",
            "session_id": session_id,
            "captured_at": repr(captured_at),
            "session_status": session_status,
            "point_in_time": "true",
            "compression_required": "true",
            "external_artifacts_included": "false",
            "private_subagent_history_included": str(include_private).lower(),
            "estimated_logical_bytes": str(estimated_logical_bytes),
            "row_counts": json.dumps(row_counts, ensure_ascii=False, sort_keys=True),
        }
        destination.executemany(
            "INSERT INTO export_metadata(key,value) VALUES(?,?)", metadata.items())
        destination.commit()
        source.rollback()
        source.close()
        source = None

        _validate_relations(destination, session_id)
        if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise SessionExportError("SQLite quick_check failed before compression")
        destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        destination.execute("PRAGMA journal_mode=DELETE")
        destination.execute("VACUUM")
        if destination.execute("PRAGMA freelist_count").fetchone()[0] != 0:
            raise SessionExportError("exported SQLite contains free pages")
        if destination.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise SessionExportError("SQLite quick_check failed after compaction")
        destination.close()
        destination = None

        with sqlite_path.open("rb") as exported_sqlite:
            header = exported_sqlite.read(16)
        if header != b"SQLite format 3\x00":
            raise SessionExportError("exported file is not SQLite")
        sqlite_bytes = sqlite_path.stat().st_size
        sqlite_digest = _sha256_file(sqlite_path)
        with zipfile.ZipFile(
                archive_path, "x", compression=zipfile.ZIP_DEFLATED,
                compresslevel=6, allowZip64=True) as archive:
            archive.write(sqlite_path, arcname=inner_filename)
        os.chmod(archive_path, 0o600)
        with archive_path.open("r+b") as output:
            os.fsync(output.fileno())
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            if (len(members) != 1 or members[0].filename != inner_filename
                    or members[0].compress_type != zipfile.ZIP_DEFLATED
                    or archive.testzip() is not None):
                raise SessionExportError("session export is not one valid deflated SQLite archive")
            digest = hashlib.sha256()
            with archive.open(members[0], "r") as compressed:
                while block := compressed.read(1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != sqlite_digest:
                raise SessionExportError("compressed SQLite does not match its verified source")
        sqlite_path.unlink()
        for suffix in ("-wal", "-shm", "-journal"):
            Path(str(sqlite_path) + suffix).unlink(missing_ok=True)
        _fsync_directory(root)
        return SessionExport(
            path=archive_path, filename=archive_path.name,
            inner_filename=inner_filename, captured_at=captured_at,
            session_status=session_status, sqlite_bytes=sqlite_bytes,
            archive_bytes=archive_path.stat().st_size,
            estimated_logical_bytes=estimated_logical_bytes,
            row_counts=row_counts)
    except BaseException:
        archive_path.unlink(missing_ok=True)
        raise
    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.rollback()
            source.close()
        sqlite_path.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm", "-journal"):
            Path(str(sqlite_path) + suffix).unlink(missing_ok=True)


__all__ = ["SessionExport", "SessionExportError", "export_session_archive"]
