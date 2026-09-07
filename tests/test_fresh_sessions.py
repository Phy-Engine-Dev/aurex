"""Fresh-request isolation: no model, network, or production database."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aurex.config import AurexConfig, ContextPolicyConfig, LLMConfig, StorageConfig, TrackingConfig
from aurex.context_budget import ContextBudget
from aurex.session_agent import SessionAgent
from aurex.sessiondb import SessionDB
from aurex.tools.hdl_workspace import hdl_workspace_create, hdl_workspace_read
from aurex.tools.registry import ToolError, ToolRegistry, ToolRuntime
from aurex.web import PersistentTaskQueue
from test_aurex_v3 import FakeLLM, reply

OLD = 'OLD_CONTEXT_CANARY_DO_NOT_REPLAY_6ca1bb'
NEW = 'NEW_INDEPENDENT_REQUEST_8bc20d'

class FreshSessionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cfg = replace(AurexConfig(), storage=StorageConfig(cache_dir=str(self.root)),
            tracking=TrackingConfig(database_path=str(self.root/'db.sqlite3')),
            llm=LLMConfig(enabled=True, context_length=32768, max_output_tokens=512))
        self.db = SessionDB(self.cfg.tracking.database_path)
        self.sid = self.db.session('legacy-shared')
        self.old = self.db.enqueue_task(self.sid, OLD, task_id='old-task')
        self.db.message(self.sid, self.old, {'role':'user', 'content':OLD})
        until = self.db.message(self.sid, self.old, {'role':'assistant', 'content':OLD+' answer',
            'tool_calls':[{'id':'old-call','type':'function','function':{'name':'read','arguments':'{}'}}]})
        self.db.compact(self.sid, OLD+' SUMMARY', until)
        self.db.finish_run(self.sid, self.old, 'cancelled')
        self.old_doc = self.db.document(self.sid, OLD, OLD)
        self.rt = ToolRuntime(self.old, 'zh', '', self.cfg, str(self.root), session_id=self.sid)
        self.workspace = hdl_workspace_create(self.rt, {'files':[{'name':'dut.v','content':'module dut; endmodule'}]})

    def test_actual_agent_first_and_final_model_requests_exclude_all_legacy_context(self):
        fake = FakeLLM(self.cfg.llm, [reply('Fresh request handled.')])
        with patch('aurex.session_agent.VLLMClient', return_value=fake):
            agent = SessionAgent(cfg=self.cfg, config_path=str(self.root/'cfg.json'), tools=ToolRegistry())
        queue = PersistentTaskQueue(self.db, agent)
        before = self.db.messages(self.sid)
        rid = queue.enqueue(self.sid, NEW, source='admin')
        task = self.db.get_task(rid)
        self.assertNotEqual(task['session_id'], self.sid)
        self.assertEqual(self.db.checkpoint(task['session_id'], rid), {'summary':'','compacted_until':0})
        self.assertTrue(queue.run_next())
        self.assertEqual(self.db.get_task(rid)['status'], 'completed')
        self.assertEqual(len(fake.requests), 2)  # Initial answer + independent final review.
        for messages, options in fake.requests:
            self.assertNotIn(OLD, json.dumps(messages))
            self.assertIn(NEW, json.dumps(messages))
        self.assertTrue(fake.requests[0][1]['thinking'])
        self.assertEqual(self.db.messages(self.sid), before)
        self.assertEqual(self.db.get_task(self.old)['status'], 'cancelled')
        with self.assertRaises(ValueError):
            self.db.read_document(task['session_id'], self.old_doc)
        with self.assertRaises(ToolError):
            hdl_workspace_read(replace(self.rt, session_id=task['session_id'], task_id=rid),
                {'workspace_id':self.workspace['workspace_id']})

    def test_task_scoped_projection_defends_even_legacy_shared_session_rows(self):
        rid = self.db.enqueue_task(self.sid, NEW, task_id='current-task')
        self.db.message(self.sid, rid, {'role':'user','content':NEW})
        fake = FakeLLM(self.cfg.llm, [])
        budget = ContextBudget(fake, self.db, self.sid, rid, 32768, lambda *a:None,
            policy=ContextPolicyConfig(safety_tokens=128))
        messages = budget.messages('system', [])
        self.assertNotIn(OLD, json.dumps(messages))
        self.assertNotIn('old-call', json.dumps(messages))
        self.assertIn(NEW, json.dumps(messages))
        self.db.repair_tools(self.sid, rid)
        self.assertEqual(len(self.db.messages(self.sid, run_id=rid)), 1)

    def test_task_checkpoint_survives_restart_and_is_not_used_by_sibling(self):
        a = self.db.enqueue_task(self.sid, NEW, task_id='checkpoint-a')
        until = self.db.message(self.sid, a, {'role':'user','content':NEW})
        self.db.compact(self.sid, 'CURRENT_TASK_CHECKPOINT', until, run_id=a)
        b = self.db.enqueue_task(self.sid, 'different', task_id='checkpoint-b')
        reopened = SessionDB(self.db.path)
        self.assertEqual(reopened.checkpoint(self.sid, a)['summary'], 'CURRENT_TASK_CHECKPOINT')
        self.assertEqual(reopened.checkpoint(self.sid, b), {'summary':'','compacted_until':0})
        with self.assertRaises(ValueError):
            reopened.compact('wrong-session','bad',until,run_id=a)

    def test_duplicate_task_receipt_does_not_create_or_resume_a_session(self):
        queue = PersistentTaskQueue(self.db, None)
        first = queue.enqueue(self.sid, NEW, task_id='new-receipt')
        sid = self.db.get_task(first)['session_id']
        self.db.finish_run(sid, first, 'cancelled')
        count = len(self.db.list())
        second = queue.enqueue(self.sid, NEW, task_id='new-receipt')
        self.assertEqual(first, second)
        self.assertEqual(self.db.get_task(first)['status'], 'cancelled')
        self.assertEqual(len(self.db.list()), count)
        with self.assertRaises(ValueError):
            queue.enqueue(self.sid, 'changed request', task_id='new-receipt')

    def test_final_review_fallback_is_also_task_scoped(self):
        from aurex.publishing import bind_task_actions
        from aurex.task_reply import review_final_answer
        rid = self.db.enqueue_task(self.sid, NEW, task_id='review-task', source='admin')
        self.db.message(self.sid, rid, {'role':'user','content':NEW})
        bind_task_actions(str(self.root),task_id=rid,session_id=self.sid,source='admin',
            original_user_request=NEW,explicit_publish_requested=False,dry_run=True)
        fake = FakeLLM(self.cfg.llm, [])
        fake.last_output = reply('Fresh reviewed answer')
        rt = replace(self.rt, task_id=rid)
        with patch('aurex.context_budget.ContextBudget.messages', side_effect=RuntimeError('fixture fallback')):
            review_final_answer(rt, 'Fresh reviewed answer', fake, self.db, self.sid, rid, lambda *a:None)
        self.assertEqual(len(fake.requests), 1)
        self.assertNotIn(OLD, json.dumps(fake.requests[0][0]))
        self.assertIn(NEW, json.dumps(fake.requests[0][0]))

if __name__ == '__main__':
    unittest.main()
