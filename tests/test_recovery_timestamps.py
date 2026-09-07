"""Isolated regression tests; import a specified snapshot, never the running service."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SOURCE = Path(os.environ.get('AUREX_SESSIONDB_TEST_MODULE', Path(__file__).resolve().parents[1] / 'src/aurex/sessiondb.py'))
spec = importlib.util.spec_from_file_location('timestamp_snapshot', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SessionDB = module.SessionDB


class RecoveryTimestampTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = str(Path(temporary.name) / 'isolated.sqlite3')
        self.db = SessionDB(self.path)

    def seed(self, sid, when, status='completed'):
        with mock.patch.object(module.time, 'time', return_value=when):
            self.db.session(sid, title=sid)
            if status != 'idle':
                self.db.enqueue_task(sid, 'original request', task_id='run-' + sid,
                                     images=['unchanged-image-path'])
                if status != 'queued':
                    self.db.run_status('run-' + sid, status)

    def timestamps(self):
        with self.db.connect() as db:
            return {r['id']: (r['status'], r['updated']) for r in db.execute('SELECT id,status,updated FROM sessions')}

    def recover_at(self, when):
        with mock.patch.object(module.time, 'time', return_value=when):
            return self.db.recover()

    def test_initialization_alone_does_not_touch_existing_activity_time(self):
        self.seed('old', 100)
        self.seed('idle', 200, 'idle')
        before = self.timestamps()
        with mock.patch.object(module.time, 'time', return_value=9000):
            SessionDB(self.path)
        self.assertEqual(self.timestamps(), before)

    def test_unchanged_sessions_keep_times_and_historical_sort_order(self):
        for sid, when, status in [('completed', 100, 'completed'), ('error', 200, 'error'),
                                  ('idle', 300, 'idle'), ('queued', 400, 'queued')]:
            self.seed(sid, when, status)
        before = self.timestamps()
        order = [row['id'] for row in self.db.list()]
        queued = self.recover_at(9000)
        self.assertEqual(self.timestamps(), before)
        self.assertEqual([row['id'] for row in self.db.list()], order)
        self.assertEqual(queued[0]['images'], ['unchanged-image-path'])
        self.assertEqual(queued[0]['updated'], 400)

    def test_status_only_aggregate_correction_preserves_last_activity(self):
        self.seed('old', 100)
        with self.db.connect() as db:
            db.execute("UPDATE sessions SET status='running' WHERE id='old'")
        self.recover_at(9000)
        self.assertEqual(self.timestamps()['old'], ('completed', 100))

    def test_real_interruption_gets_recovery_timestamp_but_other_history_does_not(self):
        self.seed('completed', 100)
        self.seed('active', 200, 'running')
        self.recover_at(9000)
        self.assertEqual(self.timestamps()['completed'], ('completed', 100))
        self.assertEqual(self.timestamps()['active'], ('interrupted', 9000))
        events = self.db.events('active')
        self.assertEqual([(e['kind'], e['created']) for e in events], [('interrupted', 9000)])
        self.assertEqual(self.db.get_task('run-active')['updated'], 9000)

    def test_real_cancel_recovery_remains_timestamped_and_does_not_resume(self):
        self.seed('old', 100)
        self.seed('pending-cancel', 200, 'queued')
        with self.db.connect() as db:
            db.execute("UPDATE runs SET cancel_requested=1 WHERE id='run-pending-cancel'")
        self.assertEqual(self.recover_at(9000), [])
        self.assertEqual(self.timestamps()['pending-cancel'], ('cancelled', 9000))
        self.assertEqual(self.timestamps()['old'], ('completed', 100))

    def test_active_sibling_interrupted_but_queued_sibling_keeps_aggregate_queued(self):
        self.seed('shared', 100, 'running')
        with mock.patch.object(module.time, 'time', return_value=200):
            self.db.enqueue_task('shared', 'next request', task_id='next')
        queued = self.recover_at(9000)
        self.assertEqual([row['id'] for row in queued], ['next'])
        self.assertEqual(self.db.get_task('run-shared')['status'], 'interrupted')
        self.assertEqual(self.timestamps()['shared'], ('queued', 9000))

    def test_repeated_recover_is_timestamp_and_event_idempotent(self):
        self.seed('active', 100, 'running')
        self.seed('idle', 200, 'idle')
        self.recover_at(9000)
        before, events = self.timestamps(), self.db.events('active')
        self.recover_at(10000)
        self.assertEqual(self.timestamps(), before)
        self.assertEqual(self.db.events('active'), events)

    def test_real_new_activity_still_updates_time_after_recovery(self):
        self.seed('old', 100)
        self.recover_at(9000)
        with mock.patch.object(module.time, 'time', return_value=10000):
            self.db.event('old', 'run-old', 'new_activity', {})
        self.assertEqual(self.timestamps()['old'][1], 10000)
        with mock.patch.object(module.time, 'time', return_value=11000):
            self.db.run_status('run-old', 'needs_attention')
        self.assertEqual(self.timestamps()['old'], ('needs_attention', 11000))


if __name__ == '__main__':
    unittest.main()
