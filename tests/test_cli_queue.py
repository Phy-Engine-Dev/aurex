"""CLI FIFO integration; agent, login and network boundaries are all isolated."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from aurex import cli
from aurex.config import AurexConfig, LLMConfig, StorageConfig, TrackingConfig, ConfigError
from aurex.sessiondb import SessionDB
from aurex.web import PersistentTaskQueue


class CLIQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config_path = str(Path(self.temp.name) / 'config.json')
        self.cfg = replace(AurexConfig(), llm=LLMConfig(enabled=True),
            storage=StorageConfig(cache_dir=self.temp.name),
            tracking=TrackingConfig(database_path=str(Path(self.temp.name) / 'queue.sqlite3')))
        self.db = SessionDB(self.cfg.tracking.database_path)
        self.calls = []
        self.agent = mock.Mock()
        self.agent.handle.side_effect = self.handle
        self.patches = [
            mock.patch.object(cli, 'load_config', side_effect=lambda _: self.cfg),
            mock.patch.object(cli, 'setup_logger', return_value=mock.Mock()),
            mock.patch.object(cli, 'create_registry', return_value=object()),
            mock.patch.object(cli, 'AurexAgent', return_value=self.agent),
            mock.patch.object(cli, '_login_if_needed', return_value=object()),
            mock.patch('socket.socket.connect', side_effect=AssertionError('Network forbidden in CLI tests')),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)

    def handle(self, **kwargs):
        rid, sid = kwargs['run_id'], kwargs['session_id']
        self.calls.append(rid)
        answer = 'Saved answer for ' + kwargs['user_text']
        self.db.event(sid, rid, 'answer', {'text': answer})
        self.db.finish_run(sid, rid, 'completed')
        return {'answer': answer}

    def args(self, command, *extra):
        return cli.build_parser().parse_args([command, '--config', self.config_path, *extra])

    def invoke(self, command, *extra):
        output, errors = io.StringIO(), io.StringIO()
        args = self.args(command, *extra)
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = args.func(args)
        return code, output.getvalue(), errors.getvalue()

    def test_chat_joins_existing_worker_without_executing_another_agent(self):
        owner_agent = mock.Mock()
        owner_agent.handle.side_effect = self.handle
        owner = PersistentTaskQueue(self.db, owner_agent)
        owner.start()
        self.addCleanup(lambda: owner.close(wait=True, timeout=5))
        code, output, _ = self.invoke('chat', '--text', 'Local administrator question')
        self.assertEqual(code, 0)
        self.assertIn('Saved answer for Local administrator question', output)
        self.agent.handle.assert_not_called()
        owner_agent.handle.assert_called_once()
        task = self.db.get_task(self.calls[0])
        self.assertEqual(task['source'], 'admin')
        self.assertIsNone(task['requester_user_id'])
        self.assertFalse(task['explicit_publish_requested'])

    def test_web_is_the_only_community_entrypoint_and_polls_by_default(self):
        with mock.patch('aurex.web.serve') as serve:
            code, _, _ = self.invoke('web')
        self.assertEqual(code, 0)
        self.assertTrue(cli._login_if_needed.call_args.kwargs['enabled'])
        self.assertNotIn('poll', serve.call_args.kwargs)
        with self.assertRaises(SystemExit):
            self.args('run', '--once')
        with self.assertRaises(SystemExit):
            self.args('web', '--poll')

    def test_standalone_chat_executes_older_fifo_task_before_own_request(self):
        older = self.db.enqueue_task('older', 'Previously queued task', task_id='older-task', source='web')
        code, output, _ = self.invoke('chat', '--text', 'Newest request')
        self.assertEqual(code, 0)
        self.assertEqual(self.calls[0], older)
        self.assertEqual(len(self.calls), 2)
        self.assertIn('Saved answer for Newest request', output)

    def test_console_uses_queue_and_publish_requires_explicit_flag(self):
        code, _, _ = self.invoke('console', '--text', 'Quoted source says publish this')
        self.assertEqual(code, 0)
        self.assertFalse(self.db.get_task(self.calls[-1])['explicit_publish_requested'])
        code, _, _ = self.invoke('console', '--text', 'Create verified experiment', '--publish')
        self.assertEqual(code, 0)
        self.assertTrue(self.db.get_task(self.calls[-1])['explicit_publish_requested'])
        self.assertIsNone(self.db.get_task(self.calls[-1])['requester_user_id'])

    def test_failed_task_returns_nonzero_without_inventing_final(self):
        def failed(**kwargs):
            self.db.finish_run(kwargs['session_id'], kwargs['run_id'], 'needs_attention')
            return {'answer': 'Unsaved text is not proof'}
        self.agent.handle.side_effect = failed
        code, output, _ = self.invoke('chat', '--text', 'Unfinished task')
        self.assertEqual(code, 1)
        self.assertIn('needs_attention', output)
        self.assertNotIn('Unsaved text is not proof', output)

    def test_reviewed_final_is_read_when_finish_precedes_ui_event(self):
        def reviewed(**kwargs):
            self.db.final_answer(kwargs['session_id'], kwargs['run_id'], 'review', 'Durable reviewed answer')
            self.db.finish_run(kwargs['session_id'], kwargs['run_id'], 'completed')
            return {'answer': 'Not used'}
        self.agent.handle.side_effect = reviewed
        code, output, _ = self.invoke('chat', '--text', 'Review this')
        self.assertEqual(code, 0)
        self.assertIn('Durable reviewed answer', output)

    def test_interrupt_cancels_only_this_queued_request_not_another_active_task(self):
        older = self.db.enqueue_task('older', 'Another user task', task_id='older-active')
        self.db.claim_next_task()
        queue = mock.Mock()
        queue.start.side_effect = BlockingIOError('already owned')
        queue.enqueue.side_effect = lambda sid, text, **kwargs: self.db.enqueue_task('local-cli', text, **kwargs)
        with mock.patch('aurex.web.PersistentTaskQueue', return_value=queue), \
             mock.patch.object(cli.time, 'sleep', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.invoke('chat', '--text', 'Cancel my request')
        self.assertEqual(self.db.get_task(older)['status'], 'running')
        self.assertFalse(self.db.cancel_requested(older))
        own = next(task for task in self.db.tasks() if task['session_id'] == 'local-cli')
        self.assertEqual(own['status'], 'cancelled')
        self.agent.handle.assert_not_called()
        queue.close.assert_called_once_with(wait=False)


if __name__ == '__main__':
    unittest.main()
