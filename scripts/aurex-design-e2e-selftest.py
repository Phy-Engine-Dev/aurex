#!/usr/bin/env python3
"""Isolated runner contract tests: all real HTTP/socket connections are denied."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import requests


SCRIPT = Path(__file__).with_name('aurex-design-e2e.py')
PLAN = Path(__file__).with_name('aurex-blind-tasks.json')
spec = importlib.util.spec_from_file_location('design_e2e_under_test', SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Response:
    def __init__(self, data, code=200):
        self.data, self.status_code = data, code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('fixture HTTP ' + str(self.status_code))

    def json(self):
        return self.data


class FakeHTTP:
    def __init__(self, *, status='completed', fail_post=None, events=None):
        self.posts, self.gets = [], []
        self.status, self.fail_post = status, fail_post
        self.events = events or []
        self.current = '1' * 32
        self.session = 'shared-session'

    def post(self, url, *, json, timeout):
        self.posts.append((url, json))
        if self.fail_post:
            if isinstance(self.fail_post, Exception):
                raise self.fail_post
            return self.fail_post
        self.current = format(len(self.posts), '032x')
        self.session = json.get('session_id') or 'new-session-' + self.current
        return Response({'task_id': self.current, 'run_id': self.current,
                         'session_id': self.session, 'status': 'queued'}, 202)

    def get(self, url, *, timeout, params=None):
        self.gets.append((url, params))
        if '/events' in url:
            rows = [{**row, 'run_id': row.get('run_id', self.current)} for row in self.events
                    if row['id'] > params['after']]
            return Response(rows[:500])
        if '/api/tasks/' in url:
            return Response({'id': self.current, 'session_id': self.session, 'status': self.status,
                             'source': 'admin', 'explicit_publish_requested': False})
        if url.endswith('/api/tasks'):
            return Response([])
        raise AssertionError('Unexpected fake endpoint: ' + url)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        # Even an incorrectly mocked transport fails locally, never reaches vLLM.
        self.stack.enter_context(mock.patch('requests.sessions.Session.request', side_effect=AssertionError('Real HTTP is forbidden')))
        self.stack.enter_context(mock.patch('socket.socket.connect', side_effect=AssertionError('Real socket is forbidden')))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.args = SimpleNamespace(case='555', blind_task=None, plan=PLAN, input_file=None,
                                    session=None, execute=True, base_url='http://fixture.invalid',
                                    timeout=0.03, task_id=None)

    def test_original_cases_are_exact_and_only_explicit_publish_cases(self):
        for case in ('555', 'riscv'):
            self.args.case = case
            _, body = runner.build_request(self.args)
            self.assertEqual(body['original_user_request'], runner.PROMPTS[case])
            self.assertIs(body['explicit_publish_requested'], True)
            self.assertEqual(set(body), {'original_user_request', 'title', 'explicit_publish_requested'})

    def test_all_blind_tasks_are_nonpublishing(self):
        rows = runner.matrix(PLAN)
        self.assertEqual(len(rows), 14)
        self.assertEqual({x['category'] for x in rows}, {'analog', 'digital', 'context'})
        for row in rows:
            self.assertIs(row['explicit_publication_requested'], False)
            # Mentioning an existing publication or its author is a read-only
            # question, not authority to publish. The explicit flag is binding.
            if row['category'] in {'analog', 'digital'}:
                self.assertNotIn('发布', row['prompt'])

    def test_run_creates_one_admin_api_task_and_does_not_set_server_fields(self):
        http = FakeHTTP()
        self.assertEqual(runner.run(http, self.args, self.tmp), 0)
        self.assertEqual(len(http.posts), 1)
        url, body = http.posts[0]
        self.assertEqual(url, 'http://fixture.invalid/api/tasks')
        self.assertFalse(set(body) & {'purpose', 'source', 'metadata', 'task_id', 'requester_user_id'})
        report = json.loads(next(self.tmp.glob('555-*.json')).read_text())
        self.assertEqual(report['task_id'], http.current)
        self.assertEqual(report['assessment'], 'not_scored')

    def test_same_session_two_explicit_submissions_get_two_task_ids(self):
        self.args.session = 'shared-session'
        http = FakeHTTP()
        runner.run(http, self.args, self.tmp)
        first = http.current
        runner.run(http, self.args, self.tmp)
        self.assertNotEqual(first, http.current)
        self.assertEqual(len(http.posts), 2)
        self.assertTrue(all(body['session_id'] == 'shared-session' for _, body in http.posts))

    def test_no_execute_never_submits(self):
        self.args.execute = False
        http = FakeHTTP()
        with self.assertRaises(ValueError):
            runner.run(http, self.args, self.tmp)
        self.assertEqual(http.posts, [])

    def test_prepare_never_constructs_an_http_client(self):
        with mock.patch.object(runner, 'REPO', self.tmp), mock.patch.object(runner, 'client', side_effect=AssertionError('Must remain offline')):
            self.assertEqual(runner.main(['prepare', '--plan', str(PLAN)]), 0)
        saved = json.loads((self.tmp / '.aurex/cache/design-e2e/blind-task-plan.json').read_text())
        self.assertIs(saved['enabled'], False)

    def test_submission_timeout_is_unknown_once_not_retried(self):
        http = FakeHTTP(fail_post=requests.ReadTimeout('fixture timeout'))
        self.assertEqual(runner.run(http, self.args, self.tmp), 3)
        self.assertEqual(len(http.posts), 1)
        self.assertEqual(http.gets, [])
        report = json.loads(next(self.tmp.glob('submission-*.json')).read_text())
        self.assertEqual(report['submission'], 'unknown_do_not_resubmit')

    def test_bad_or_error_receipt_does_not_fallback_to_messages_endpoint(self):
        for response in (Response({}), Response({}, 500), Response({}, 400)):
            http = FakeHTTP(fail_post=response)
            self.assertEqual(runner.run(http, self.args, self.tmp), 3)
            self.assertEqual(len(http.posts), 1)
            self.assertEqual(http.gets, [])

    def test_observer_timeout_keeps_original_task_without_post_or_cancel(self):
        http = FakeHTTP(status='running')
        self.assertEqual(runner.observe(http, self.args, self.tmp, http.current), 2)
        self.assertEqual(http.posts, [])
        report = json.loads(next(self.tmp.glob('observed-*.json')).read_text())
        self.assertEqual(report['status'], 'running')
        self.assertEqual(report['observation'], 'client_wait_expired_task_unchanged')
        self.assertFalse(any('cancel' in url for url, _ in http.gets))
        http.status = 'completed'
        self.assertEqual(runner.observe(http, self.args, self.tmp, http.current), 0)
        self.assertEqual(http.posts, [])

    def test_answer_event_is_not_completion_and_reasoning_is_only_counted(self):
        events = [{'id': 1, 'kind': 'reasoning_delta', 'data': {'text': 'private analysis'}},
                  {'id': 2, 'kind': 'answer', 'data': {'text': 'candidate'}},
                  {'id': 3, 'kind': 'publication_review', 'data': {'reasoning': 'private review', 'ok': True}}]
        http = FakeHTTP(status='running', events=events)
        self.assertEqual(runner.observe(http, self.args, self.tmp, http.current), 2)
        raw = next(self.tmp.glob('observed-*.json')).read_text()
        self.assertNotIn('private analysis', raw)
        self.assertNotIn('private review', raw)
        report = json.loads(raw)
        self.assertEqual(report['observed_reasoning_delta_characters'], 16)
        self.assertEqual(report['status'], 'running')

    def test_terminal_error_is_not_success_even_if_an_answer_exists(self):
        http = FakeHTTP(status='needs_attention', events=[{'id': 1, 'kind': 'answer', 'data': {'text': 'done'}}])
        self.assertEqual(runner.observe(http, self.args, self.tmp, http.current), 1)

    def test_paginated_events_and_resume_do_not_duplicate_archive(self):
        events = [{'id': i, 'kind': 'fixture', 'data': {}} for i in range(1, 503)]
        http = FakeHTTP(events=events)
        self.args.timeout = 1
        runner.observe(http, self.args, self.tmp, http.current)
        runner.observe(http, self.args, self.tmp, http.current)
        report = json.loads(next(self.tmp.glob('observed-*.json')).read_text())
        self.assertEqual(len(report['events']), 502)
        self.assertEqual(report['last_event_id'], 502)

    def test_context_requires_real_input_and_rejects_oracle_fields(self):
        self.args.case, self.args.blind_task = None, 'context-post-grounding'
        with self.assertRaises(ValueError):
            runner.build_request(self.args)
        path = self.tmp / 'input.json'
        path.write_text(json.dumps({'oracle': 'answer', 'reference_text': 'source'}))
        self.args.input_file = path
        with self.assertRaises(ValueError):
            runner.build_request(self.args)
        path.write_text(json.dumps({'reference_text': 'raw source only'}))
        _, body = runner.build_request(self.args)
        self.assertIn('raw source only', body['original_user_request'])
        self.assertIs(body['explicit_publish_requested'], False)

    def test_plan_verifier_notes_are_never_sent(self):
        path = self.tmp / 'plan.json'
        path.write_text(json.dumps({'tasks': [{'id': 'example', 'prompt': 'Natural goal',
                        'explicit_publication_requested': False, 'oracle': 'SECRET EXPECTATION'}]}))
        self.args.case, self.args.blind_task, self.args.plan = None, 'example', path
        _, body = runner.build_request(self.args)
        self.assertEqual(body['original_user_request'], 'Natural goal')
        self.assertNotIn('SECRET', json.dumps(body))

    def test_status_is_read_only(self):
        http = FakeHTTP()
        runner.status(http, self.args)
        self.assertEqual(http.posts, [])
        self.assertEqual(http.gets[0][0], 'http://fixture.invalid/api/tasks')


if __name__ == '__main__':
    unittest.main()
