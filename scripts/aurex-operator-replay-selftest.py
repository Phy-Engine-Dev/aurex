#!/usr/bin/env python3
"""Temporary-DB-only operator replay tests; real network is explicitly denied."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from aurex.sessiondb import SessionDB

spec = importlib.util.spec_from_file_location('replay_under_test', Path(__file__).with_name('aurex-operator-replay.py'))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
SECRET = 'PRIVATE_SOURCE_TOKEN_NEVER_PRINT'


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')))
        self.stack.enter_context(mock.patch('requests.sessions.Session.request', side_effect=AssertionError('HTTP forbidden')))
        self.output = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.path = self.temp / 'tasks.sqlite3'
        self.ledger = self.temp / 'ledger.sqlite3'
        self.records = self.temp / 'records'
        self.db = SessionDB(str(self.path))
        self.sid = self.db.session('old-session', title='Existing original', source='community')
        self.prompt = 'CONTEXT_JSON:\n' + json.dumps({'comment': {'author_id': 'a' * 24},
                  'target': {'type': 'Experiment', 'id': 'b' * 24}, 'untrusted_text': SECRET}) + '\n\n原始问题：查看这个实验'
        self.rid = self.db.enqueue_task(self.sid, '原始问题：查看这个实验', task_id='source-task', prompt=self.prompt,
                    source='community', requester_user_id='a' * 24, requester_nickname='Original', reply_id='old-reply',
                    target={'type': 'Experiment', 'id': 'b' * 24}, explicit_publish_requested=True,
                    metadata={'dry_run': False, 'purpose': 'riscv_teaching_subset', 'old_scope': SECRET})
        self.db.run_status(self.rid, 'error')
        self.db.message(self.sid, self.rid, {'role': 'assistant', 'content': 'OLD ANSWER ' + SECRET})
        self.db.document(self.sid, 'old document', SECRET)
        self.db.event(self.sid, self.rid, 'reasoning_delta', {'text': 'PRIVATE REASONING ' + SECRET})

    def originals(self):
        with replay.readonly(self.path) as db:
            result = {}
            for table in ('sessions', 'runs', 'messages', 'documents', 'events', 'artifacts'):
                result[table] = [dict(row) for row in db.execute('SELECT * FROM ' + table)]
            return result

    def add_replay(self, request_append=None):
        return replay.enqueue(self.path, self.rid, self.records, execute=True, request_append=request_append)

    def make_ledger(self):
        with sqlite3.connect(self.ledger) as db:
            db.executescript('''
                CREATE TABLE task_scopes(task_id TEXT,binding TEXT);
                CREATE TABLE task_publications(task_id TEXT,state TEXT);
                CREATE TABLE approvals(run_id TEXT,state TEXT);
                CREATE TABLE task_final_answers(task_id TEXT,state TEXT,reply_receipt TEXT);
            ''')

    def test_prepare_is_readonly_and_redacts_original_material(self):
        before = self.originals()
        with mock.patch.object(SessionDB, '__init__', side_effect=AssertionError('No migration allowed')):
            result = replay.main(['prepare', '--database', str(self.path), '--task-id', self.rid])
        self.assertEqual(result, 0)
        self.assertEqual(before, self.originals())
        self.assertNotIn(SECRET, self.output.getvalue())
        self.assertNotIn('OLD ANSWER', self.output.getvalue())
        self.assertFalse(self.records.exists())

    def test_prepare_execute_flag_still_cannot_enqueue(self):
        before = self.originals()
        replay.main(['prepare', '--execute', '--database', str(self.path), '--task-id', self.rid])
        self.assertEqual(before, self.originals())

    def test_enqueue_without_execute_changes_nothing(self):
        before = self.originals()
        with self.assertRaises(ValueError):
            replay.enqueue(self.path, self.rid, self.records)
        self.assertEqual(before, self.originals())
        self.assertFalse(self.records.exists())

    def test_exact_new_admin_task_and_old_rows_unchanged(self):
        before = self.originals()
        with mock.patch.object(SessionDB, '__init__', side_effect=AssertionError('No migration allowed')):
            receipt = self.add_replay()
        new = replay.read_task(self.path, receipt['new_task_id'])
        self.assertNotEqual(new['session_id'], self.sid)
        self.assertEqual(new['prompt'], self.prompt)
        self.assertEqual(new['original_user_request'], '原始问题：查看这个实验')
        self.assertEqual(new['source'], 'admin')
        self.assertEqual(new['status'], 'queued')
        self.assertIs(new['metadata']['dry_run'], True)
        self.assertIs(new['metadata']['operator_replay'], True)
        self.assertEqual(new['metadata']['replay_of'], self.rid)
        self.assertFalse(new['explicit_publish_requested'])
        self.assertIsNone(new['requester_user_id'])
        self.assertIsNone(new['requester_nickname'])
        self.assertIsNone(new['reply_id'])
        self.assertNotIn('purpose', new['metadata'])
        self.assertNotIn('old_scope', new['metadata'])
        self.assertEqual(new['target'], {'type': 'Experiment', 'id': 'b' * 24})
        after = self.originals()
        self.assertEqual(len(after['runs']), len(before['runs']) + 1)
        self.assertEqual(len(after['sessions']), len(before['sessions']) + 1)
        for table, rows in before.items():
            for row in rows:
                self.assertIn(row, after[table])
        self.assertEqual(before['messages'], after['messages'])
        self.assertEqual(before['documents'], after['documents'])
        self.assertEqual(before['events'], after['events'])
        self.assertFalse(receipt['worker_started_by_script'])
        self.assertNotIn(SECRET, Path(receipt['record']).read_text())

    def test_default_and_explicit_none_preserve_utf8_bytes(self):
        original = replay.read_task(self.path, self.rid)
        receipts = [replay.enqueue(self.path, self.rid, self.records, execute=True),
                    replay.enqueue(self.path, self.rid, self.records, execute=True, request_append=None)]
        for receipt in receipts:
            new = replay.read_task(self.path, receipt['new_task_id'])
            for field in ('prompt', 'original_user_request'):
                self.assertEqual(new[field].encode('utf-8'), original[field].encode('utf-8'))
            self.assertIs(new['metadata']['request_append_present'], False)
            self.assertIsNone(new['metadata']['request_append_sha256'])
            self.assertEqual(new['metadata']['request_append_characters'], 0)
            self.assertEqual(receipt['source_prompt_sha256'], receipt['replay_prompt_sha256'])
            self.assertEqual(receipt['source_original_request_sha256'], receipt['replay_original_request_sha256'])

    def test_append_both_fields_verbatim_and_preserve_all_original_rows(self):
        before = self.originals()
        original = replay.read_task(self.path, self.rid)
        supplement = '  用户补充：只检查少量代表样例，明确这是部分验证。\n' + SECRET + '\n'
        with mock.patch.object(SessionDB, '__init__', side_effect=AssertionError('No migration allowed')):
            receipt = self.add_replay(supplement)
        new = replay.read_task(self.path, receipt['new_task_id'])
        suffix = replay.REQUEST_APPEND_SEPARATOR + supplement
        for field in ('prompt', 'original_user_request'):
            self.assertEqual(new[field], original[field] + suffix)
            self.assertEqual(new[field].encode('utf-8')[:len(original[field].encode('utf-8'))],
                             original[field].encode('utf-8'))
        # No parsing/re-encoding, replacing CONTEXT_JSON, or copying old grants.
        self.assertTrue(new['prompt'].startswith(self.prompt))
        self.assertEqual(new['source'], 'admin')
        self.assertIs(new['metadata']['dry_run'], True)
        self.assertFalse(new['explicit_publish_requested'])
        self.assertIsNone(new['requester_user_id'])
        self.assertIsNone(new['requester_nickname'])
        self.assertIsNone(new['reply_id'])
        self.assertEqual(new['target'], original['target'])
        self.assertNotIn('purpose', new['metadata'])
        self.assertNotIn('old_scope', new['metadata'])
        record = json.loads(Path(receipt['record']).read_text())
        expected = {'request_append_present': True,
                    'request_append_sha256': replay.digest(supplement),
                    'request_append_characters': len(supplement),
                    'source_prompt_sha256': replay.digest(original['prompt']),
                    'source_original_request_sha256': replay.digest(original['original_user_request']),
                    'replay_prompt_sha256': replay.digest(new['prompt']),
                    'replay_original_request_sha256': replay.digest(new['original_user_request'])}
        for key, value in expected.items():
            self.assertEqual(new['metadata'][key], value)
            self.assertEqual(record['metadata'][key], value)
            self.assertEqual(receipt[key], value)
        self.assertNotIn(SECRET, json.dumps(record))
        self.assertNotIn(SECRET, json.dumps(receipt))
        after = self.originals()
        for table, rows in before.items():
            for row in rows:
                self.assertIn(row, after[table])
        for table in ('messages', 'documents', 'events', 'artifacts'):
            self.assertEqual(before[table], after[table])
        self.assertEqual(len(after['runs']), len(before['runs']) + 1)
        self.assertEqual(len(after['sessions']), len(before['sessions']) + 1)

    def test_invalid_append_rejected_before_any_write(self):
        before = self.originals()
        for value in ('', ' \n\t', False, 0, [], {}, b'bytes'):
            with self.subTest(value=repr(value)), self.assertRaisesRegex(ValueError, 'nonempty string'):
                self.add_replay(value)
            self.assertEqual(before, self.originals())
            self.assertFalse(self.records.exists())

    def test_append_size_limit_counts_separator_and_checks_both_fields(self):
        addition = '补充'
        suffix_length = len(replay.REQUEST_APPEND_SEPARATOR + addition)
        for field in ('prompt', 'original_user_request'):
            task = {'prompt': '原始上下文', 'original_user_request': '原始要求'}
            task[field] = '字' * (500000 - suffix_length)
            prompt, request, _ = replay.replay_request(task, addition)
            self.assertEqual(len({'prompt': prompt, 'original_user_request': request}[field]), 500000)
            task[field] += '字'
            with self.assertRaisesRegex(ValueError, '500000'):
                replay.replay_request(task, addition)
        before = self.originals()
        with self.assertRaisesRegex(ValueError, '500000'):
            self.add_replay('字' * 500000)
        self.assertEqual(before, self.originals())
        self.assertFalse(self.records.exists())

    def test_prepare_append_is_readonly_and_only_reports_hashes(self):
        before = self.originals()
        supplement = '补充原话 ' + SECRET
        with mock.patch.object(SessionDB, '__init__', side_effect=AssertionError('No migration allowed')):
            result = replay.main(['prepare', '--execute', '--database', str(self.path), '--task-id', self.rid,
                                  '--records', str(self.records), '--request-append', supplement])
        self.assertEqual(result, 0)
        self.assertEqual(before, self.originals())
        self.assertFalse(self.records.exists())
        output = self.output.getvalue()
        self.assertNotIn(SECRET, output)
        provenance = json.loads(output)['sources'][0]['request_provenance']
        self.assertIs(provenance['request_append_present'], True)
        self.assertEqual(provenance['request_append_sha256'], replay.digest(supplement))

    def test_append_without_execute_cannot_write_in_function_or_cli(self):
        before = self.originals()
        with self.assertRaisesRegex(ValueError, '--execute'):
            replay.enqueue(self.path, self.rid, self.records, request_append='补充')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            replay.main(['enqueue', '--database', str(self.path), '--task-id', self.rid,
                         '--records', str(self.records), '--request-append', '补充'])
        self.assertEqual(before, self.originals())
        self.assertFalse(self.records.exists())

    def test_cli_explicit_append_is_forwarded_to_private_fifo(self):
        supplement = '补充目标，不给出解法。'
        result = replay.main(['enqueue', '--execute', '--database', str(self.path), '--task-id', self.rid,
                              '--records', str(self.records), '--request-append', supplement])
        self.assertEqual(result, 0)
        receipt = json.loads(self.output.getvalue())
        new = replay.read_task(self.path, receipt['new_task_id'])
        self.assertEqual(new['prompt'], self.prompt + replay.REQUEST_APPEND_SEPARATOR + supplement)
        self.assertTrue(new['original_user_request'].endswith(replay.REQUEST_APPEND_SEPARATOR + supplement))
        self.assertFalse(receipt['worker_started_by_script'])

    def test_cli_rejects_empty_or_status_append_without_writes(self):
        before = self.originals()
        for action, value in [('prepare', ''), ('enqueue', ' \n'), ('status', '补充')]:
            with self.subTest(action=action), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                replay.main([action, '--execute', '--database', str(self.path), '--task-id', self.rid,
                             '--records', str(self.records), '--request-append', value])
        self.assertEqual(before, self.originals())
        self.assertFalse(self.records.exists())

    def test_failed_enqueue_receipt_retains_append_provenance_not_text(self):
        before = self.originals()
        supplement = '真实补充 ' + SECRET
        with mock.patch.object(SessionDB, 'enqueue_task', side_effect=sqlite3.OperationalError('fixture failure')):
            with self.assertRaises(sqlite3.OperationalError):
                self.add_replay(supplement)
        self.assertEqual(before, self.originals())
        record = json.loads(next(self.records.glob('*.json')).read_text())
        self.assertEqual(record['submission'], 'uncertain_do_not_repeat_check_recorded_task_id')
        self.assertEqual(record['metadata']['request_append_sha256'], replay.digest(supplement))
        self.assertIs(record['metadata']['request_append_present'], True)
        self.assertNotIn(SECRET, json.dumps(record))

    def test_images_preserved_and_missing_images_refused(self):
        image = self.temp / 'original.png'
        image.write_bytes(b'fixture-image')
        with self.db.connect() as db:
            db.execute('UPDATE runs SET input_data=? WHERE id=?', (json.dumps({'images': [str(image)]}), self.rid))
        receipt = self.add_replay()
        self.assertEqual(replay.read_task(self.path, receipt['new_task_id'])['input_data']['images'], [str(image)])
        image.unlink()
        count = len(self.originals()['runs'])
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            self.add_replay()
        self.assertEqual(len(self.originals()['runs']), count)

    def test_unknown_task_does_not_create_database_or_records(self):
        with self.assertRaises(ValueError):
            replay.enqueue(self.path, 'missing', self.records, execute=True)
        self.assertFalse(self.records.exists())
        missing = self.temp / 'does-not-exist.sqlite3'
        with self.assertRaises(FileNotFoundError):
            replay.read_task(missing, self.rid)
        self.assertFalse(missing.exists())

    def test_readonly_connection_refuses_writes(self):
        with replay.readonly(self.path) as db, self.assertRaises(sqlite3.OperationalError):
            db.execute("DELETE FROM runs")

    def test_status_counts_tools_tokens_without_reasoning_or_arguments(self):
        for event, data in [('tool_start', {'name': 'read_context', 'call_id': 'a', 'arguments': {'text': SECRET}}),
                            ('tool_start', {'name': 'read_context', 'call_id': 'b', 'arguments': {'text': SECRET}}),
                            ('tool_end', {'name': 'read_context', 'call_id': 'a', 'ok': True, 'duration': 0.1, 'preview': SECRET}),
                            ('model_start', {'step': 0, 'thinking': True}),
                            ('model_end', {'step': 0, 'usage': {'prompt_tokens': 901, 'completion_tokens': 31}, 'reasoning_characters': 55, 'finish_reason': 'tool_calls'}),
                            ('context_budget', {'input_tokens': 932, 'context_limit': 90112, 'image_count': 2}),
                            ('error', {'type': 'ModelError', 'error': SECRET})]:
            self.db.event(self.sid, self.rid, event, data)
        report = replay.status(self.path, self.ledger, self.rid)
        raw = json.dumps(report)
        self.assertNotIn(SECRET, raw)
        self.assertNotIn('PRIVATE REASONING', raw)
        self.assertEqual(report['metrics']['repeated_call_signatures'][0]['count'], 2)
        self.assertEqual(report['metrics']['tool_results'][0]['duration'], 0.1)
        self.assertEqual(report['metrics']['model_requests'][1]['usage']['prompt_tokens'], 901)

    def test_missing_ledger_is_not_claimed_as_verified_zero_external_writes(self):
        new = self.add_replay()
        audit = replay.status(self.path, self.ledger, new['new_task_id'])['external_write_audit']
        self.assertFalse(audit['available'])
        self.assertIsNone(audit['no_recorded_external_writes'])
        self.assertFalse(self.ledger.exists())

    def test_local_review_is_allowed_but_external_reply_or_unknown_is_flagged(self):
        new = self.add_replay()
        self.make_ledger()
        with sqlite3.connect(self.ledger) as db:
            db.execute('INSERT INTO task_scopes VALUES(?,?)', (new['new_task_id'], json.dumps({
                'source': 'admin', 'dry_run': True, 'explicit_publish_requested': False})))
            db.execute('INSERT INTO task_final_answers VALUES(?,?,NULL)', (new['new_task_id'], 'reviewed'))
        audit = replay.status(self.path, self.ledger, new['new_task_id'])['external_write_audit']
        self.assertIs(audit['no_recorded_external_writes'], True)
        for state in ('replying', 'replied', 'unknown'):
            with sqlite3.connect(self.ledger) as db:
                db.execute('UPDATE task_final_answers SET state=?', (state,))
            audit = replay.status(self.path, self.ledger, new['new_task_id'])['external_write_audit']
            self.assertIs(audit['no_recorded_external_writes'], False)
            self.assertTrue(audit['violations'])

    def test_any_publication_record_is_a_replay_policy_violation(self):
        new = self.add_replay()
        self.make_ledger()
        with sqlite3.connect(self.ledger) as db:
            db.execute('INSERT INTO task_publications VALUES(?,?)', (new['new_task_id'], 'approved'))
        audit = replay.status(self.path, self.ledger, new['new_task_id'])['external_write_audit']
        self.assertEqual(audit['publication_rows'], 1)
        self.assertTrue(audit['violations'])

    def test_bad_scope_is_flagged_without_exposing_raw_binding(self):
        new = self.add_replay()
        self.make_ledger()
        with sqlite3.connect(self.ledger) as db:
            db.execute('INSERT INTO task_scopes VALUES(?,?)', (new['new_task_id'], json.dumps({
                'source': 'community', 'dry_run': False, 'original_user_request': SECRET})))
        audit = replay.status(self.path, self.ledger, new['new_task_id'])['external_write_audit']
        self.assertTrue(audit['violations'])
        self.assertNotIn(SECRET, json.dumps(audit))

    def test_observation_expiry_does_not_claim_cancel_or_requeue(self):
        new = self.add_replay()
        before = self.originals()
        self.assertEqual(replay.main(['status', '--database', str(self.path), '--ledger', str(self.ledger),
              '--task-id', new['new_task_id'], '--observe', '--timeout', '0.01']), 2)
        self.assertEqual(before, self.originals())
        self.assertIn('expired_task_unchanged', self.output.getvalue())

    def test_enqueue_failure_keeps_original_and_new_receipt_for_reconciliation(self):
        before = self.originals()
        with mock.patch.object(SessionDB, 'enqueue_task', side_effect=sqlite3.OperationalError('fixture unavailable')):
            with self.assertRaises(sqlite3.OperationalError):
                self.add_replay()
        self.assertEqual(before, self.originals())
        record = json.loads(next(self.records.glob('*.json')).read_text())
        self.assertEqual(record['submission'], 'uncertain_do_not_repeat_check_recorded_task_id')
        self.assertEqual(len(record['new_task_id']), 32)


if __name__ == '__main__':
    unittest.main()
