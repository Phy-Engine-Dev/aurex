import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from aurex.config import ConfigError, load_config
from aurex.history_retention import HistoryRetention, HistoryRetentionWorker, _verify_archive


class HistoryRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.database = self.root / 'runtime' / 'aurex.sqlite3'
        self.cache = self.root / 'runtime' / 'cache'
        self.history = self.root / 'runtime' / 'backups'
        self.database.parent.mkdir(parents=True)
        self.database.write_bytes(b'live database must remain')
        self.cache.mkdir()
        self.history.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def retention(self, *, compress=1, delete=10**9):
        return HistoryRetention(
            str(self.history), database_path=str(self.database), cache_dir=str(self.cache),
            compress_at_bytes=compress, delete_at_bytes=delete, min_age_sec=0)

    @staticmethod
    def snapshot(path, content, timestamp):
        path.mkdir()
        (path / 'aurex-before.sqlite3').write_bytes(content)
        os.utime(path / 'aurex-before.sqlite3', (timestamp, timestamp))
        os.utime(path, (timestamp, timestamp))

    def test_compresses_each_snapshot_atomically_without_touching_live_storage(self):
        self.snapshot(self.history / 'old', b'A' * 32768, 100)
        self.snapshot(self.history / 'new', b'B' * 32768, 200)
        report = self.retention().enforce_once()
        self.assertEqual([item['source'] for item in report['compressed']], ['old', 'new'])
        self.assertFalse((self.history / 'old').exists())
        self.assertFalse((self.history / 'new').exists())
        self.assertTrue((self.history / 'old.tar.gz').is_file())
        self.assertTrue((self.history / 'new.tar.gz').is_file())
        self.assertEqual(self.database.read_bytes(), b'live database must remain')
        self.assertTrue(self.cache.is_dir())

    def test_managed_directory_and_archives_are_private_even_with_permissive_umask(self):
        self.snapshot(self.history / 'private', b'private history', 100)
        previous = os.umask(0o002)
        try:
            self.retention().enforce_once()
        finally:
            os.umask(previous)
        modes = {
            'root': stat.S_IMODE(self.history.stat().st_mode),
            'marker': stat.S_IMODE((self.history / '.aurex-history-v1').stat().st_mode),
            'lock': stat.S_IMODE((self.history / '.retention.lock').stat().st_mode),
            'archive': stat.S_IMODE((self.history / 'private.tar.gz').stat().st_mode),
        }
        self.assertEqual(modes, {key: 0o600 if key != 'root' else 0o700 for key in modes})

    def test_delete_watermark_expires_oldest_verified_archive_and_keeps_latest(self):
        self.snapshot(self.history / 'old', os.urandom(32768), 100)
        self.snapshot(self.history / 'new', os.urandom(32768), 200)
        report = self.retention(compress=1, delete=2).enforce_once()
        self.assertEqual(report['deleted'], ['old.tar.gz'])
        self.assertFalse((self.history / 'old.tar.gz').exists())
        self.assertTrue((self.history / 'new.tar.gz').exists())
        self.assertEqual(self.database.read_bytes(), b'live database must remain')

    def test_delete_stops_below_delete_watermark_instead_of_compress_watermark(self):
        for index, name in enumerate(('old', 'middle', 'new')):
            self.snapshot(self.history / name, os.urandom(65536), 100 + index)
        builder = self.retention()
        for name in ('old', 'middle', 'new'):
            builder._compress(self.history / name)
        archives = [self.history / (name + '.tar.gz') for name in ('old', 'middle', 'new')]
        sizes = [path.stat().st_blocks * 512 for path in archives]
        delete_at = sum(sizes) - min(sizes) // 2
        report = self.retention(compress=1, delete=delete_at).enforce_once()
        self.assertEqual(report['deleted'], ['old.tar.gz'])
        self.assertTrue((self.history / 'middle.tar.gz').exists())
        self.assertTrue((self.history / 'new.tar.gz').exists())
        self.assertLess(report['after_bytes'], delete_at)
        self.assertGreater(report['after_bytes'], 1)

    def test_delete_pressure_uses_worker_unit_before_compressing_more_raw_history(self):
        archived = self.history / 'archived-old'
        raw_one = self.history / 'raw-one'
        raw_two = self.history / 'raw-two'
        self.snapshot(archived, os.urandom(65536), 100)
        self.snapshot(raw_one, os.urandom(65536), 200)
        self.snapshot(raw_two, os.urandom(65536), 300)
        builder = self.retention()
        archive = builder._compress(archived)
        total = builder.usage_bytes()
        delete_at = total - archive.stat().st_blocks * 256

        report = self.retention(compress=1, delete=delete_at).enforce_once(max_operations=1)

        self.assertEqual(report['deleted'], ['archived-old.tar.gz'])
        self.assertEqual(report['compressed'], [])
        self.assertTrue(raw_one.is_dir())
        self.assertTrue(raw_two.is_dir())
        self.assertLess(report['after_bytes'], delete_at)

    def test_symlink_is_never_followed_or_removed(self):
        outside = self.root / 'outside'
        outside.write_bytes(b'outside')
        (self.history / 'linked').symlink_to(outside)
        self.snapshot(self.history / 'ordinary', b'C' * 8192, 100)
        self.retention().enforce_once()
        self.assertTrue(outside.exists())
        self.assertTrue((self.history / 'linked').is_symlink())

    def test_escaping_symlink_makes_snapshot_ineligible_without_touching_target(self):
        outside = self.root / 'outside-secret'
        outside.write_bytes(b'outside must survive')
        source = self.history / 'unsafe'
        self.snapshot(source, b'inside', 100)
        (source / 'escape').symlink_to('../../outside-secret')
        report = self.retention().enforce_once()
        self.assertTrue(source.exists())
        self.assertFalse((self.history / 'unsafe.tar.gz').exists())
        self.assertEqual(outside.read_bytes(), b'outside must survive')
        self.assertIn('link escapes', report['errors'][0]['error'])

    def test_failed_verification_preserves_source_and_removes_partial_archive(self):
        self.snapshot(self.history / 'only', b'D' * 8192, 100)
        with patch('aurex.history_retention._verify_archive', side_effect=OSError('bad archive')):
            report = self.retention().enforce_once()
        self.assertTrue((self.history / 'only').is_dir())
        self.assertFalse((self.history / 'only.tar.gz').exists())
        self.assertEqual(report['errors'][0]['operation'], 'compress')
        self.assertFalse(any(path.name.startswith('.retention-') for path in self.history.iterdir()))

    def test_crash_partial_is_removed_only_after_exclusive_lock(self):
        partial = self.history / '.retention-deadbeef.tar.gz.partial'
        partial.write_bytes(b'incomplete')
        report = self.retention(compress=10**9).enforce_once()
        self.assertFalse(partial.exists())
        self.assertEqual(report['partials_removed'], [partial.name])

    def test_matching_archive_and_source_crash_state_preserves_both(self):
        source = self.history / 'snapshot'
        self.snapshot(source, b'E' * 8192, 100)
        saved = self.root / 'saved-source'
        shutil.copytree(source, saved)
        self.retention()._compress(source)
        shutil.copytree(saved, source)
        os.utime(source, (100, 100))
        report = self.retention().enforce_once()
        self.assertTrue(source.exists())
        self.assertTrue((self.history / 'snapshot.tar.gz').exists())
        self.assertEqual(report['errors'][0]['operation'], 'compress')

    def test_mismatched_archive_and_source_crash_state_preserves_both(self):
        source = self.history / 'snapshot'
        self.snapshot(source, b'F' * 8192, 100)
        self.retention()._compress(source)
        self.snapshot(source, b'different', 100)
        report = self.retention().enforce_once()
        self.assertTrue(source.exists())
        self.assertTrue((self.history / 'snapshot.tar.gz').exists())
        self.assertEqual(report['errors'][0]['operation'], 'compress')

    def test_existing_archive_symlink_never_replaces_the_managed_copy(self):
        source = self.history / 'snapshot'
        self.snapshot(source, b'owned history', 100)
        saved = self.root / 'saved-source'
        shutil.copytree(source, saved)
        builder = self.retention()
        archive = builder._compress(source)
        outside = self.root / 'outside.tar.gz'
        archive.replace(outside)
        shutil.copytree(saved, source)
        (self.history / 'snapshot.tar.gz').symlink_to(outside)

        report = builder.enforce_once()

        self.assertTrue(source.exists())
        self.assertTrue((self.history / 'snapshot.tar.gz').is_symlink())
        self.assertTrue(outside.is_file())
        self.assertEqual(report['errors'][0]['operation'], 'compress')
        self.assertIn('already exists', report['errors'][0]['error'])

    def test_lock_symlink_fails_without_touching_the_external_file(self):
        builder = self.retention(compress=10**9)
        outside = self.root / 'external-lock-target'
        outside.write_bytes(b'external contents')
        outside.chmod(0o644)
        (self.history / '.retention.lock').symlink_to(outside)

        with self.assertRaises(OSError):
            builder.enforce_once()

        self.assertEqual(outside.read_bytes(), b'external contents')
        self.assertEqual(stat.S_IMODE(outside.stat().st_mode), 0o644)

    def test_change_after_archive_verification_preserves_source(self):
        source = self.history / 'moving'
        payload = source / 'aurex-before.sqlite3'
        self.snapshot(source, b'G' * 8192, 100)

        def verify_then_change(path, expected_root):
            result = _verify_archive(path, expected_root)
            payload.write_bytes(b'changed during compression')
            return result

        with patch('aurex.history_retention._verify_archive', side_effect=verify_then_change):
            report = self.retention().enforce_once()
        self.assertTrue(source.exists())
        self.assertFalse((self.history / 'moving.tar.gz').exists())
        self.assertEqual(report['errors'][0]['operation'], 'compress')

    def test_archive_payload_signature_must_equal_source_before_source_is_removed(self):
        source = self.history / 'payload'
        self.snapshot(source, b'H' * 8192, 100)

        def wrong_signature(path, expected_root):
            rows = list(_verify_archive(path, expected_root))
            rows[-1] = (*rows[-1][:3], '0' * 64)
            return tuple(rows)

        with patch('aurex.history_retention._verify_archive', side_effect=wrong_signature):
            report = self.retention().enforce_once()
        self.assertTrue(source.exists())
        self.assertFalse((self.history / 'payload.tar.gz').exists())
        self.assertIn('does not match', report['errors'][0]['error'])

    def test_archive_age_uses_newest_descendant_and_deletes_actual_oldest(self):
        actual_new = self.history / 'actual-new'
        actual_old = self.history / 'actual-old'
        self.snapshot(actual_new, os.urandom(32768), 100)
        os.utime(actual_new / 'aurex-before.sqlite3', (200, 200))
        self.snapshot(actual_old, os.urandom(32768), 150)
        builder = self.retention()
        builder._compress(actual_new)
        builder._compress(actual_old)
        self.assertAlmostEqual((self.history / 'actual-new.tar.gz').stat().st_mtime, 200, delta=1)
        self.assertAlmostEqual((self.history / 'actual-old.tar.gz').stat().st_mtime, 150, delta=1)
        archives = list(self.history.glob('*.tar.gz'))
        total = sum(path.stat().st_blocks * 512 for path in archives)
        delete_at = total - min(path.stat().st_blocks * 512 for path in archives) // 2
        report = self.retention(compress=1, delete=delete_at).enforce_once()
        self.assertEqual(report['deleted'], ['actual-old.tar.gz'])
        self.assertTrue((self.history / 'actual-new.tar.gz').exists())

    def test_one_operation_limit_leaves_remaining_snapshot_for_next_safe_boundary(self):
        self.snapshot(self.history / 'one', os.urandom(8192), 100)
        self.snapshot(self.history / 'two', os.urandom(8192), 200)
        report = self.retention().enforce_once(max_operations=1)
        self.assertEqual([item['source'] for item in report['compressed']], ['one'])
        self.assertTrue(report['limited'])
        self.assertTrue((self.history / 'two').exists())
        self.assertFalse((self.history / 'two.tar.gz').exists())

    def test_refuses_history_root_that_contains_live_database_or_cache(self):
        with self.assertRaises(ValueError):
            HistoryRetention(str(self.database.parent), database_path=str(self.database),
                             cache_dir=str(self.cache), compress_at_bytes=1, delete_at_bytes=2)

    def test_nonempty_custom_directory_requires_dedicated_history_marker(self):
        custom = self.root / 'not-a-history-directory'
        custom.mkdir()
        (custom / 'personal-file').write_text('must not be managed')
        with self.assertRaisesRegex(ValueError, 'lacks the Aurex history marker'):
            HistoryRetention(str(custom), database_path=str(self.database),
                             cache_dir=str(self.cache), compress_at_bytes=1, delete_at_bytes=2)
        self.assertEqual((custom / 'personal-file').read_text(), 'must not be managed')

    def test_worker_checks_immediately(self):
        called = threading.Event()

        class Stub:
            logger = None

            def enforce_once(self, **kwargs):
                called.set()
                return {'compressed': [], 'deleted': [], 'errors': [], 'limited': False}

        worker = HistoryRetentionWorker(Stub(), 3600)
        worker.start()
        self.assertTrue(called.wait(2))
        worker.close()

    def test_worker_close_stops_before_starting_the_next_snapshot(self):
        self.snapshot(self.history / 'one', os.urandom(8192), 100)
        self.snapshot(self.history / 'two', os.urandom(8192), 200)
        retention = self.retention()
        original = retention._compress
        first_done = threading.Event()

        def observable_compress(source):
            result = original(source)
            first_done.set()
            time.sleep(0.15)
            return result

        retention._compress = observable_compress
        worker = HistoryRetentionWorker(retention, 3600)
        worker.start()
        self.assertTrue(first_done.wait(2))
        worker.close()
        self.assertFalse(worker.thread.is_alive())
        self.assertTrue((self.history / 'one.tar.gz').exists())
        self.assertTrue((self.history / 'two').exists())
        self.assertFalse((self.history / 'two.tar.gz').exists())


class HistoryRetentionConfigTests(unittest.TestCase):
    def load(self, storage):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'storage': storage}), encoding='utf-8')
            return load_config(str(path)).storage

    def test_defaults_are_ten_and_twenty_gib(self):
        storage = self.load({})
        self.assertEqual(storage.history_dir, '')
        self.assertEqual(storage.history_compress_at_gib, 10.0)
        self.assertEqual(storage.history_delete_at_gib, 20.0)

    def test_explicit_retention_configuration(self):
        storage = self.load({'history_dir': '.aurex/backups',
                             'history_retention_enabled': False,
                             'history_compress_at_gib': 12.5,
                             'history_delete_at_gib': 25,
                             'history_check_interval_sec': 60,
                             'history_min_age_sec': 0})
        self.assertEqual(storage.history_dir, '.aurex/backups')
        self.assertFalse(storage.history_retention_enabled)
        self.assertEqual(storage.history_compress_at_gib, 12.5)

    def test_rejects_zero_or_reversed_thresholds(self):
        for value in (
            {'history_compress_at_gib': 0},
            {'history_compress_at_gib': 10, 'history_delete_at_gib': 9},
            {'history_compress_at_gib': True},
        ):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.load(value)


if __name__ == '__main__':
    unittest.main()
