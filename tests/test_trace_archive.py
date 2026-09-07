import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aurex.trace_archive import compact_snapshot, read_series, hydrate_snapshot
from aurex.tools.registry import ToolError


class TraceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'snapshot.pe-state.json'
        self.rows = [{'time_s':i*0.1, 'completed_steps':i,
            'components':[{'id':'unchanged-ID','digital':[0,1,2,3], 'voltage':[], 'details':'原始数据'*200}]} for i in range(1,6)]
        self.state = {'schema':'aurex.pe-state.v1', 'spec':{'components':[]}, 'scene':{},
            'measurements':{'components':[], 'transient':{'samples':self.rows, 'sample_count':5}}}

    def archived(self):
        with patch('aurex.trace_archive.THRESHOLD', 100):
            return compact_snapshot(self.path, self.state)

    def test_lossless_roundtrip_without_mutating_original(self):
        original = copy.deepcopy(self.state)
        saved = self.archived()
        self.assertEqual(self.state, original)
        self.assertNotIn('samples', saved['measurements']['transient'])
        self.assertEqual(read_series(self.path, saved['measurements']['transient'], 'samples'), self.rows)
        self.assertEqual(hydrate_snapshot(self.path, saved), original)

    def test_small_snapshots_remain_inline_and_legacy_readers_work(self):
        saved = compact_snapshot(self.path, self.state)
        self.assertEqual(saved, self.state)
        self.assertEqual(read_series(self.path, saved['measurements']['transient'], 'samples'), self.rows)

    def test_changed_archive_or_size_and_count_are_rejected(self):
        saved = self.archived()
        trace = saved['measurements']['transient']
        for key,value in (('sha256','0'*64), ('raw_sha256','0'*64), ('raw_bytes',1), ('count',4)):
            bad = copy.deepcopy(trace)
            bad['samples_archive'][key] = value
            with self.subTest(key=key), self.assertRaises(ToolError):
                read_series(self.path, bad, 'samples')

    def test_path_traversal_symlink_and_expansion_limit(self):
        trace = self.archived()['measurements']['transient']
        for name in ('../secret.json.gz','/etc/passwd','https://example.com/sample'):
            bad = copy.deepcopy(trace)
            bad['samples_archive']['file'] = name
            with self.assertRaises(ToolError):
                read_series(self.path,bad,'samples')
        bad = copy.deepcopy(trace)
        bad['samples_archive']['raw_bytes'] = 256*1024**2+1
        with self.assertRaises(ToolError):
            read_series(self.path,bad,'samples')
        target = self.path.parent/trace['samples_archive']['file']
        moved = target.with_suffix('.saved')
        target.rename(moved)
        target.symlink_to(moved)
        with self.assertRaises(ToolError):
            read_series(self.path,trace,'samples')

    def test_inline_and_archive_cannot_disagree(self):
        trace = self.archived()['measurements']['transient']
        trace['samples'] = []
        with self.assertRaises(ToolError):
            read_series(self.path,trace,'samples')
