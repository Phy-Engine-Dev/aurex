import concurrent.futures
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aurex.tools.hdl_workspace import (hdl_workspace_create, hdl_workspace_read, hdl_workspace_edit,
    hdl_workspace_write, workspace_snapshot, register_hdl_workspace_tools)
from aurex.tools.hdl import hdl_simulate
from aurex.tools.phy_engine import _verified_hdl
from aurex.tools.registry import ToolRuntime, ToolRegistry, ToolError

DUT = 'module add8(input [7:0] a,b,output [7:0] y); assign y=a-b; endmodule\n'
TB = '''`timescale 1ns/1ps
module tb; reg[7:0] a,b; wire[7:0] y; add8 dut(a,b,y);
initial begin a=17; b=26; #1; if(y!==43) $fatal(1,"17+26 mismatch");
a=255; b=1; #1; if(y!==0) $fatal(1,"wrap mismatch");
a=0; b=0; #1; if(y!==0) $fatal(1,"zero mismatch");
$display("THREE_ADDER_SAMPLES_PASS"); $finish; end endmodule
'''


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.rt = ToolRuntime('task1', 'zh', '', None, self.temp.name, session_id='session1')
        self.workspace = hdl_workspace_create(self.rt, {'label': '多文件加法器', 'files': [
            {'name': 'add8.v', 'content': DUT}, {'name': 'tb.sv', 'content': TB, 'role': 'testbench'}]})
        self.wid = self.workspace['workspace_id']

    def edit(self, **kwargs):
        return hdl_workspace_edit(self.rt, {'workspace_id': self.wid, 'expected_revision': 1,
            'edits': [{'name': 'add8.v', 'old_text': 'a-b', 'new_text': 'a+b'}], **kwargs})

    def test_persists_restart_and_separates_sessions(self):
        restarted = replace(self.rt, task_id='task2')
        data = hdl_workspace_read(restarted, {'workspace_id': self.wid, 'name': 'add8.v'})
        self.assertEqual(data['text'], DUT)
        self.assertEqual(hdl_workspace_read(restarted, {})['workspaces'][0]['id'], self.wid)
        other = replace(self.rt, session_id='foreign')
        self.assertEqual(hdl_workspace_read(other, {})['workspaces'], [])
        with self.assertRaises(ToolError):
            hdl_workspace_read(other, {'workspace_id': self.wid})
        with self.assertRaises(ToolError):
            hdl_workspace_edit(other, {'workspace_id': self.wid, 'expected_revision': 1, 'edits': []})

    def test_exact_edit_preserves_history_and_other_file(self):
        changed = self.edit()
        self.assertEqual(changed['workspace_revision'], 2)
        self.assertEqual(changed['changes'][0]['replacements'], 1)
        self.assertTrue(changed['atomic'])
        self.assertIsNone(changed['last_verification'])
        self.assertEqual(hdl_workspace_read(self.rt, {'workspace_id': self.wid, 'name': 'add8.v'})['text'], DUT.replace('a-b', 'a+b'))
        self.assertEqual(hdl_workspace_read(self.rt, {'workspace_id': self.wid, 'revision': 1, 'name': 'add8.v'})['text'], DUT)
        self.assertEqual(hdl_workspace_read(self.rt, {'workspace_id': self.wid, 'name': 'tb.sv'})['text'], TB)
        with self.assertRaisesRegex(ToolError, 'Stale'):
            self.edit()

    def test_multi_edit_rolls_back_if_later_match_fails(self):
        with self.assertRaises(ToolError):
            self.edit(edits=[{'name': 'add8.v', 'old_text': 'a-b', 'new_text': 'a+b'},
                             {'name': 'tb.sv', 'old_text': 'missing', 'new_text': 'wrong'}])
        data = hdl_workspace_read(self.rt, {'workspace_id': self.wid, 'name': 'add8.v'})
        self.assertEqual(data['workspace_revision'], 1)
        self.assertEqual(data['text'], DUT)

    def test_ambiguous_empty_noop_and_path_edits_rejected(self):
        for old, new, name in (('', 'x', 'add8.v'), ('module', 'module', 'add8.v'),
                               ('module', 'x', 'add8.v'), ('a-b', 'a+b', '../add8.v')):
            with self.subTest(old=old, name=name), self.assertRaises(ToolError):
                self.edit(edits=[{'name': name, 'old_text': old, 'new_text': new}])
        replaced = self.edit(edits=[{'name': 'add8.v', 'old_text': 'module', 'new_text': 'MODULE', 'replace_all': True}])
        self.assertEqual(replaced['changes'][0]['replacements'], 2)

    def test_write_requires_hash_and_new_file_is_explicit(self):
        with self.assertRaises(ToolError):
            hdl_workspace_write(self.rt, {'workspace_id': self.wid, 'expected_revision': 1,
                'name': 'add8.v', 'content': DUT + '//x', 'expected_sha256': None})
        out = hdl_workspace_write(self.rt, {'workspace_id': self.wid, 'expected_revision': 1,
            'name': 'helper.v', 'content': 'module helper; endmodule', 'expected_sha256': None})
        self.assertEqual(out['workspace_revision'], 2)
        with self.assertRaises(ToolError):
            hdl_workspace_write(self.rt, {'workspace_id': self.wid, 'expected_revision': 2,
                'name': 'bad/path.v', 'content': DUT, 'expected_sha256': None})

    def test_character_paging_and_snapshot_pin(self):
        pieces = [hdl_workspace_read(self.rt, {'workspace_id': self.wid, 'name': 'add8.v', 'offset': i, 'length': 7})['text']
                  for i in range(0, len(DUT), 7)]
        self.assertEqual(''.join(pieces), DUT)
        with self.assertRaises(ToolError):
            workspace_snapshot(self.rt, self.wid, None)
        self.edit()
        with self.assertRaisesRegex(ToolError, 'Stale'):
            workspace_snapshot(self.rt, self.wid, 1)

    def test_concurrent_edits_only_one_can_commit(self):
        def attempt(_):
            try:
                return self.edit()['workspace_revision']
            except ToolError:
                return 'stale'
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(attempt, range(2)))
        self.assertCountEqual(results, [2, 'stale'])

    def test_missing_null_boolean_revision_cannot_bypass_edit_pin(self):
        for revision in (None, True, False, 0, '1'):
            with self.subTest(revision=revision), self.assertRaises(ToolError):
                self.edit(expected_revision=revision)
            with self.assertRaises(ToolError):
                hdl_workspace_write(self.rt, {'workspace_id': self.wid, 'expected_revision': revision,
                    'name': 'add8.v', 'content': DUT + '//changed',
                    'expected_sha256': hashlib.sha256(DUT.encode()).hexdigest()})
        self.assertEqual(hdl_workspace_read(self.rt, {'workspace_id': self.wid})['head_revision'], 1)

    def test_find_returns_exact_text_with_offsets(self):
        out = hdl_workspace_read(self.rt, {'workspace_id': self.wid, 'name': 'tb.sv', 'find': '255', 'length': 80})
        self.assertEqual(out['match_offset'], TB.index('255'))
        self.assertEqual(out['text'], TB[out['offset']:out['offset']+80])
        self.assertIn('255', out['text'])
        self.assertFalse(hdl_workspace_read(self.rt, {'workspace_id': self.wid, 'name': 'tb.sv', 'find': 'absent'})['found'])
        with self.assertRaises(ToolError):
            hdl_workspace_read(self.rt, {'workspace_id': self.wid, 'find': '255'})

    def test_workspace_does_not_bypass_simulation_sandbox_validation(self):
        self.edit(edits=[{'name': 'add8.v', 'old_text': 'assign y=a-b;',
                         'new_text': 'initial $system("touch /tmp/must-not-run"); assign y=a+b;'}])
        with patch('aurex.tools.hdl._run', side_effect=AssertionError('must reject before execution')):
            with self.assertRaises(ToolError):
                hdl_simulate(self.rt, {'workspace_id': self.wid, 'workspace_revision': 2,
                    'profile': 'custom', 'top': 'tb'})

    def test_check_requires_complete_report_and_cannot_promote_concurrent_edit(self):
        from aurex.tools.hdl_workspace import workspace_check
        self.edit()
        def edit_before_record(runtime, report):
            self.assertEqual(json.loads(Path(report['report_path']).read_text())['verification_id'], report['verification_id'])
            hdl_workspace_edit(runtime, {'workspace_id': self.wid, 'expected_revision': 2,
                'edits': [{'name': 'add8.v', 'old_text': 'a+b', 'new_text': 'a-b'}]})
            return workspace_check(runtime, report)
        with patch('aurex.tools.hdl_workspace.workspace_check', side_effect=edit_before_record):
            report = hdl_simulate(self.rt, {'workspace_id': self.wid, 'workspace_revision': 2,
                'profile': 'custom', 'top': 'tb', 'design_top': 'add8'})
        self.assertTrue(report['verified'])  # Frozen revision 2, not revision 3.
        self.assertFalse(report['workspace_current'])
        current = hdl_workspace_read(self.rt, {'workspace_id': self.wid})
        self.assertEqual(current['workspace_revision'], 3)
        self.assertIsNone(current['last_verification'])
        self.assertTrue(hdl_workspace_read(self.rt, {'workspace_id': self.wid, 'revision': 2})['last_verification']['verified'])
        with self.assertRaisesRegex(ToolError, 'Stale'):
            _verified_hdl(self.rt, report['report_path'])

    def test_journal_failure_preserves_report_and_missing_design_top_refuses_export(self):
        import sqlite3
        self.edit()
        with patch('aurex.tools.hdl_workspace.workspace_check', side_effect=sqlite3.OperationalError('database locked')):
            report = hdl_simulate(self.rt, {'workspace_id': self.wid, 'workspace_revision': 2,
                'profile': 'custom', 'top': 'tb'})
        self.assertTrue(report['verified'])
        self.assertIsNone(report['workspace_current'])
        self.assertIn('database locked', report['workspace_journal_error'])
        self.assertEqual(json.loads(Path(report['report_path']).read_text()), report)
        with self.assertRaisesRegex(ToolError, 'design_top'):
            _verified_hdl(self.rt, report['report_path'])

    def test_registered_schemas_validate_create_and_edit(self):
        import jsonschema
        reg = ToolRegistry()
        register_hdl_workspace_tools(reg)
        self.assertEqual(len(reg.list()), 4)
        jsonschema.validate({'files': [{'name': 'x.v', 'content': DUT}]}, reg.get('hdl_workspace_create').parameters)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({'workspace_id': self.wid, 'edits': []}, reg.get('hdl_workspace_edit').parameters)

    def test_compaction_retains_workspace_version_without_promoting_edits_to_pass(self):
        from aurex.config import ContextPolicyConfig, LLMConfig
        from aurex.context_budget import ContextBudget
        from aurex.sessiondb import SessionDB
        class Client:
            config = LLMConfig(context_length=65536, max_output_tokens=512)
            def count(self, messages, tools=None):
                return len(json.dumps([messages, tools or []], ensure_ascii=False).encode()) // 3 + 1
        db = SessionDB(str(Path(self.temp.name) / 'context.sqlite3'))
        sid = db.session('workspace compaction test', source='admin')
        rid = db.enqueue_task(sid, '修改同一份HDL，抽样验证，不外发', source='admin')
        changed = self.edit()
        for index, (tool, data) in enumerate((('hdl_workspace_create', self.workspace), ('hdl_workspace_edit', changed))):
            cid = f'workspace_call_{index}'
            call = {'id':cid, 'type':'function', 'function':{'name':tool, 'arguments':'{}'}}
            db.message(sid, rid, {'role':'assistant', 'content':'', 'tool_calls':[call]})
            db.tool_outcome(sid, rid, cid, tool, json.dumps({'ok':True, 'data':data}), True)
        budget = ContextBudget(Client(), db, sid, rid, 65536, lambda *args: None,
            policy=ContextPolicyConfig(safety_tokens=128, summary_max_tokens=4096))
        text = budget._tool_index(budget._journal_boundary(), 2048)
        payload = json.loads(text.split('\n', 1)[1])
        refs = payload['machine_evidence']['hdl_outcome_refs']
        self.assertEqual([ref['workspace_revision'] for ref in refs], [1, 2])
        self.assertTrue(all(ref['workspace_id'] == self.wid for ref in refs))
        self.assertTrue(all(ref['classification'] == 'workspace_source_operation_not_compilation_or_simulation' for ref in refs))
        self.assertTrue(all('verified' not in ref for ref in refs))
        self.assertLessEqual(Client().count([{'role':'user','content':text}]), 2048)

    def test_real_failed_then_one_fragment_fix_passes_exact_revision(self):
        args = {'workspace_id': self.wid, 'workspace_revision': 1, 'profile': 'custom', 'top': 'tb', 'design_top': 'add8'}
        failed = hdl_simulate(self.rt, args)
        self.assertFalse(failed['verified'])
        self.assertIn('17+26 mismatch', failed['simulation']['log'])
        self.assertFalse(hdl_workspace_read(self.rt, {'workspace_id': self.wid})['last_verification']['verified'])
        self.edit()
        passed = hdl_simulate(self.rt, {**args, 'workspace_revision': 2})
        self.assertTrue(passed['verified'], passed)
        self.assertIn('THREE_ADDER_SAMPLES_PASS', passed['simulation']['log'])
        self.assertEqual(passed['source_files_sha256']['add8.v'], hashlib.sha256(DUT.replace('a-b', 'a+b').encode()).hexdigest())
        self.assertEqual(passed['design_source_files'], ['add8.v'])
        metadata, design = _verified_hdl(self.rt, passed['report_path'])
        self.assertEqual(metadata['design_top'], 'add8')
        self.assertEqual(design, DUT.replace('a-b', 'a+b'))
        self.assertNotIn('module tb', design)
        hdl_workspace_edit(self.rt, {'workspace_id': self.wid, 'expected_revision': 2,
            'edits': [{'name': 'add8.v', 'old_text': 'a+b', 'new_text': 'a-b'}]})
        with self.assertRaisesRegex(ToolError, 'Stale'):
            _verified_hdl(self.rt, passed['report_path'])
        # Original immutable compiled snapshot was not overwritten by editing.
        self.assertEqual(Path(passed['sources_paths'][0]).read_text(), DUT.replace('a-b', 'a+b'))

    def test_no_inline_workspace_mixing_and_cross_session_verification(self):
        args = {'workspace_id': self.wid, 'workspace_revision': 1, 'profile': 'custom', 'top': 'tb'}
        with self.assertRaises(ToolError):
            hdl_simulate(self.rt, {**args, 'files': []})
        with self.assertRaises(ToolError):
            hdl_simulate(replace(self.rt, session_id='other'), args)
        with self.assertRaises(ToolError):
            hdl_simulate(self.rt, {**args, 'workspace_revision': None})


if __name__ == '__main__':
    unittest.main()
