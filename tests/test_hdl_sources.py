import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aurex.hdl_sources import archive_hdl_sources
from aurex.sessiondb import SessionDB
from aurex.tools.registry import ToolRegistry, ToolSpec
import test_aurex_v3 as support


def report(files):
    hashes = {item['name']: hashlib.sha256(item['content'].encode()).hexdigest() for item in files}
    return {'verified': False, 'source_files_sha256': hashes,
            'source_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}


class HDLSourceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.db = SessionDB(str(Path(temp.name) / 'db.sqlite3'))
        self.sid = self.db.session('HDL', source='admin')
        self.files = [{'name': 'dut.v', 'content': '// 原始代码\nmodule dut; endmodule\n'},
                      {'name': 'tb.sv', 'content': 'module tb; dut d(); endmodule'}]

    def test_exact_sources_retrievable_and_same_session_reuses_archive(self):
        refs = archive_hdl_sources(self.db, self.sid, {'files': self.files}, report(self.files))
        self.assertEqual(refs, archive_hdl_sources(self.db, self.sid, {'files': self.files}, report(self.files)))
        for item, ref in zip(self.files, refs):
            self.assertEqual(self.db.read_document(self.sid, ref['document_id'])['text'], item['content'])
            self.assertEqual(ref['sha256'], hashlib.sha256(item['content'].encode()).hexdigest())
        other = self.db.session('another', source='admin')
        other_refs = archive_hdl_sources(self.db, other, {'files': self.files}, report(self.files))
        self.assertNotEqual(refs[0]['document_id'], other_refs[0]['document_id'])
        with self.assertRaises(ValueError):
            self.db.read_document(other, refs[0]['document_id'])

    def test_hash_mismatch_and_path_names_fail_before_any_document_write(self):
        for files, outcome in ((self.files, {**report(self.files), 'source_sha256': 'bad'}),
                               ([{'name': '../dut.v', 'content': 'module dut; endmodule'}], report(self.files)),
                               (self.files + [self.files[0]], report(self.files))):
            with self.subTest(files=files), mock.patch.object(self.db, 'document') as write:
                with self.assertRaises(ValueError):
                    archive_hdl_sources(self.db, self.sid, {'files': files}, outcome)
                write.assert_not_called()


class HDLSourceIntegrationTests(unittest.TestCase):
    setUp = support.SessionAgentTests.setUp
    agent = support.SessionAgentTests.agent

    def test_failed_verification_still_preserves_exact_source_in_raw_outcome(self):
        files = [{'name': 'dut.v', 'content': 'module dut; endmodule'}]
        registry = ToolRegistry()
        registry.register(ToolSpec('hdl_simulate', 'HDL', {'type': 'object'}, lambda rt, args: report(files)))
        replies = [support.reply('', calls=[{'id': 'hdl-source', 'type': 'function', 'function': {
            'name': 'hdl_simulate', 'arguments': json.dumps({'files': files})}}], finish='tool_calls'),
            support.reply('Verification failed; this source is not a passing CPU.')]
        agent, _ = self.agent(replies, registry)
        handled = agent.handle(user_text='Verify the CPU')
        with agent.db.connect() as db:
            stored = db.execute('SELECT d.content FROM documents d JOIN tool_outcomes t ON t.document_id=d.id WHERE t.call_id=?',
                                ('hdl-source',)).fetchone()[0]
        data = json.loads(stored)['data']
        self.assertFalse(data['verified'])
        self.assertEqual(agent.db.read_document(handled['session_id'], data['source_documents'][0]['document_id'])['text'],
                         files[0]['content'])
