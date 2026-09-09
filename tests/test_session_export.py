"""Single-session SQLite export regressions; no model or community calls."""
from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest import mock
import zipfile

from aurex.session_export import SessionExportError, export_session_archive
from aurex.sessiondb import SessionDB, encode


SESSION_TABLES = (
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


class SessionExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database_path = str(Path(self.temp.name) / "sessions.sqlite3")
        self.export_dir = Path(self.temp.name) / "exports"
        self.db = SessionDB(self.database_path)

    def seed_every_table(self, sid: str, marker: str, base: int) -> tuple[str, str]:
        """Create one internally consistent row in every session-owned table."""
        self.db.session(sid, title=marker + " title")
        rid = sid + "-run"
        self.db.begin(sid, marker + " prompt", rid)
        self.db.run_status(rid, "running")
        child_id = f"{base:032x}"
        created = time.time()
        with self.db.connect() as db:
            db.execute(
                "INSERT INTO messages(id,session_id,run_id,role,data,created) "
                "VALUES(?,?,?,?,?,?)",
                (base + 1, sid, rid, "tool", encode({"role": "tool", "content": marker}), created),
            )
            db.execute(
                "INSERT INTO messages(id,session_id,run_id,role,data,created) "
                "VALUES(?,?,?,?,?,?)",
                (base + 2, sid, rid, "assistant", encode({"role": "assistant", "content": marker}), created),
            )
            db.execute(
                "INSERT INTO events(id,session_id,run_id,kind,data,created) VALUES(?,?,?,?,?,?)",
                (base + 3, sid, rid, "seed", encode({"marker": marker}), created),
            )
            db.execute(
                "INSERT INTO artifacts(id,session_id,path,mime_type,label,created) VALUES(?,?,?,?,?,?)",
                (sid + "-artifact", sid, "/not/bundled/" + marker, "text/plain", marker, created),
            )
            db.execute(
                "INSERT INTO documents(id,session_id,title,content,created) VALUES(?,?,?,?,?)",
                (sid + "-tool-document", sid, marker + " tool", marker + " tool content", created),
            )
            db.execute(
                "INSERT INTO documents(id,session_id,title,content,created) VALUES(?,?,?,?,?)",
                (sid + "-child-document", sid, marker + " child", marker + " child content", created),
            )
            db.execute(
                "INSERT INTO tool_outcomes(run_id,call_id,session_id,name,ok,document_id,message_id,created) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (rid, sid + "-call", sid, "seed_tool", 1, sid + "-tool-document", base + 1, created),
            )
            db.execute(
                "INSERT INTO final_answers(run_id,session_id,review_id,message_id,created) VALUES(?,?,?,?,?)",
                (rid, sid, sid + "-review", base + 2, created),
            )
            db.execute(
                "INSERT INTO run_checkpoints(run_id,session_id,summary,compacted_until,updated) "
                "VALUES(?,?,?,?,?)",
                (rid, sid, marker + " checkpoint", base + 2, created),
            )
            db.execute(
                "INSERT INTO task_plan_items(run_id,session_id,item_id,ordinal,title,status,note,evidence,created,updated) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (rid, sid, "inspect", 0, marker + " plan", "in_progress", marker, "[]", created, created),
            )
            db.execute(
                "INSERT INTO subagent_runs(id,session_id,parent_run_id,depth,objective,context,status,report,deadline_at,created,updated) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (child_id, sid, rid, 1, marker + " objective", encode({"marker": marker}),
                 "running", "{}", None, created, created),
            )
            db.execute(
                "INSERT INTO subagent_messages(id,subagent_id,session_id,parent_run_id,role,data,created) "
                "VALUES(?,?,?,?,?,?,?)",
                (base + 4, child_id, sid, rid, "tool",
                 encode({"role": "tool", "content": marker + " child"}), created),
            )
            db.execute(
                "INSERT INTO subagent_events(id,subagent_id,session_id,parent_run_id,kind,data,created) "
                "VALUES(?,?,?,?,?,?,?)",
                (base + 5, child_id, sid, rid, "seed", encode({"marker": marker}), created),
            )
            db.execute(
                "INSERT INTO subagent_tool_outcomes(subagent_id,call_id,session_id,parent_run_id,name,ok,document_id,message_id,created) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (child_id, child_id + ":call", sid, rid, "seed_child_tool", 1,
                 sid + "-child-document", base + 4, created),
            )
        return rid, child_id

    def unpack(self, archive_path: Path) -> tuple[zipfile.ZipInfo, bytes]:
        with zipfile.ZipFile(archive_path, "r") as archive:
            members = archive.infolist()
            self.assertEqual(len(members), 1)
            self.assertIsNone(archive.testzip())
            return members[0], archive.read(members[0])

    def connect_export(self, raw: bytes):
        extracted = Path(self.temp.name) / ("unpacked-" + str(time.time_ns()) + ".sqlite3")
        extracted.write_bytes(raw)
        self.addCleanup(extracted.unlink, missing_ok=True)
        connection = sqlite3.connect(extracted)
        connection.row_factory = sqlite3.Row
        self.addCleanup(connection.close)
        return connection

    def test_archive_is_deflated_fresh_sqlite_with_only_one_session_across_all_tables(self):
        wanted_marker = "WANTED-SESSION-5f9c"
        foreign_marker = "FOREIGN-SESSION-7a2e"
        self.seed_every_table("wanted", wanted_marker, 100)
        self.seed_every_table("foreign", foreign_marker, 100_000)

        result = export_session_archive(self.database_path, "wanted", str(self.export_dir))

        self.assertTrue(result.path.is_file())
        self.assertEqual(result.path.suffixes[-2:], [".sqlite3", ".zip"])
        self.assertEqual(os.stat(result.path).st_mode & 0o777, 0o600)
        self.assertTrue(result.filename.endswith(".sqlite3.zip"))
        self.assertFalse((self.export_dir / result.inner_filename).exists(),
                         "an uncompressed SQLite file escaped the export directory")
        member, raw = self.unpack(result.path)
        self.assertEqual(member.filename, result.inner_filename)
        self.assertEqual(member.compress_type, zipfile.ZIP_DEFLATED)
        self.assertEqual(raw[:16], b"SQLite format 3\x00")
        self.assertIn(wanted_marker.encode(), raw)
        self.assertNotIn(foreign_marker.encode(), raw,
                         "another session survived in a live or freelist page")

        exported = self.connect_export(raw)
        self.assertEqual(exported.execute("PRAGMA quick_check").fetchone()[0], "ok")
        self.assertEqual(exported.execute("PRAGMA freelist_count").fetchone()[0], 0)
        for table in SESSION_TABLES:
            column = "id" if table == "sessions" else "session_id"
            values = {row[0] for row in exported.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}"')}
            self.assertEqual(values, {"wanted"}, table)
            self.assertEqual(result.row_counts[table], exported.execute(
                f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        metadata = dict(exported.execute("SELECT key,value FROM export_metadata"))
        self.assertEqual(metadata["schema"], "aurex.session-export.v1")
        self.assertEqual(metadata["session_id"], "wanted")
        self.assertEqual(metadata["session_status"], "running")
        self.assertEqual(metadata["point_in_time"], "true")
        self.assertEqual(metadata["compression_required"], "true")
        self.assertEqual(metadata["external_artifacts_included"], "false")
        self.assertEqual(metadata["private_subagent_history_included"], "true")
        self.assertGreater(int(metadata["estimated_logical_bytes"]), 0)
        self.assertEqual(result.estimated_logical_bytes,
                         int(metadata["estimated_logical_bytes"]))
        self.assertEqual(json.loads(metadata["row_counts"]), result.row_counts)
        sequences = dict(exported.execute("SELECT name,seq FROM sqlite_sequence"))
        self.assertLess(sequences["messages"], 100_000,
                        "the source database's cross-session sequence leaked")

    def test_running_writer_gets_one_cross_table_snapshot_without_pausing_or_changing_status(self):
        rid, _ = self.seed_every_table("active", "ACTIVE-SESSION", 300)
        writer_started = threading.Event()
        stop_writer = threading.Event()
        errors: list[BaseException] = []

        def write_pairs():
            try:
                connection = sqlite3.connect(self.database_path, timeout=30)
                connection.execute("PRAGMA wal_autocheckpoint=0")
                index = 0
                while not stop_writer.is_set():
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "INSERT INTO messages(session_id,run_id,role,data,created) VALUES(?,?,?,?,?)",
                        ("active", rid, "user", encode({"writer_pair": index}), time.time()),
                    )
                    connection.execute(
                        "INSERT INTO events(session_id,run_id,kind,data,created) VALUES(?,?,?,?,?)",
                        ("active", rid, "writer_pair", encode({"writer_pair": index}), time.time()),
                    )
                    connection.commit()
                    index += 1
                    writer_started.set()
                    time.sleep(0.001)
                connection.close()
            except BaseException as exc:  # pragma: no cover - reported by assertion
                errors.append(exc)
                writer_started.set()

        writer = threading.Thread(target=write_pairs, name="session-export-writer")
        writer.start()
        self.addCleanup(lambda: (stop_writer.set(), writer.join(5)))
        self.assertTrue(writer_started.wait(5))
        result = export_session_archive(self.database_path, "active", str(self.export_dir))
        stop_writer.set()
        writer.join(5)
        self.assertFalse(writer.is_alive())
        self.assertFalse(errors)

        _, raw = self.unpack(result.path)
        exported = self.connect_export(raw)
        messages = exported.execute(
            "SELECT COUNT(*) FROM messages WHERE data LIKE '%writer_pair%'").fetchone()[0]
        events = exported.execute(
            "SELECT COUNT(*) FROM events WHERE kind='writer_pair'").fetchone()[0]
        self.assertGreater(messages, 0)
        self.assertEqual(messages, events,
                         "tables came from different moments rather than one read transaction")
        self.assertEqual(result.session_status, "running")
        self.assertEqual(self.db.get_task(rid)["status"], "running")
        self.assertEqual(self.db.get("active")["status"], "running")

    def test_wal_commit_is_captured_but_open_transaction_is_not_leaked(self):
        rid, _ = self.seed_every_table("wal", "WAL-SESSION", 500)
        writer = sqlite3.connect(self.database_path, timeout=30)
        self.addCleanup(writer.close)
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO messages(session_id,run_id,role,data,created) VALUES(?,?,?,?,?)",
            ("wal", rid, "user", encode({"text": "COMMITTED-WAL-MARKER"}), time.time()),
        )
        writer.commit()
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO messages(session_id,run_id,role,data,created) VALUES(?,?,?,?,?)",
            ("wal", rid, "user", encode({"text": "UNCOMMITTED-SECRET-MARKER"}), time.time()),
        )

        result = export_session_archive(self.database_path, "wal", str(self.export_dir))
        writer.rollback()

        _, raw = self.unpack(result.path)
        self.assertIn(b"COMMITTED-WAL-MARKER", raw)
        self.assertNotIn(b"UNCOMMITTED-SECRET-MARKER", raw)

    def test_compression_failure_removes_archive_sqlite_and_sidecars(self):
        self.seed_every_table("failure", "FAILURE-SESSION", 700)
        with mock.patch.object(zipfile.ZipFile, "write", side_effect=OSError("zip failed")):
            with self.assertRaisesRegex(OSError, "zip failed"):
                export_session_archive(self.database_path, "failure", str(self.export_dir))
        self.assertEqual(list(self.export_dir.iterdir()), [])

    def test_missing_session_creates_no_partial_payload(self):
        with self.assertRaisesRegex(SessionExportError, "session not found"):
            export_session_archive(self.database_path, "missing", str(self.export_dir))
        self.assertEqual(list(self.export_dir.iterdir()), [])

    def test_unknown_schema_and_broken_relationship_fail_without_payload(self):
        self.seed_every_table("unknown", "UNKNOWN-SCHEMA", 900)
        with self.db.connect() as db:
            db.execute("CREATE TABLE unexpected_private_data(secret TEXT)")
        with self.assertRaisesRegex(SessionExportError, "unsupported session database schema"):
            export_session_archive(self.database_path, "unknown", str(self.export_dir))
        self.assertEqual(list(self.export_dir.iterdir()), [])

        with self.db.connect() as db:
            db.execute("DROP TABLE unexpected_private_data")
            child = db.execute(
                "SELECT id FROM subagent_runs WHERE session_id='unknown'").fetchone()[0]
            db.execute("""INSERT INTO subagent_tool_outcomes
                (subagent_id,call_id,session_id,parent_run_id,name,ok,document_id,message_id,created)
                VALUES(?,?,?,?,?,?,?,?,?)""", (child, child + ":orphan", "unknown",
                "unknown-run", "broken", 0, "missing-document", 999999, time.time()))
        with self.assertRaisesRegex(SessionExportError, "incomplete session relationships"):
            export_session_archive(self.database_path, "unknown", str(self.export_dir))
        self.assertEqual(list(self.export_dir.iterdir()), [])

    def test_capacity_preflight_reserves_space_for_archive_and_active_wal(self):
        self.seed_every_table("capacity", "CAPACITY", 1100)
        actual = shutil.disk_usage(self.export_dir.parent)
        constrained = actual._replace(free=96 * 1024 * 1024)
        with mock.patch("aurex.session_export.shutil.disk_usage", return_value=constrained):
            with self.assertRaisesRegex(SessionExportError, "active WAL growth"):
                export_session_archive(self.database_path, "capacity", str(self.export_dir))
        self.assertEqual(list(self.export_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
