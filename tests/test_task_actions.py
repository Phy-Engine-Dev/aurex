import json
import os
import subprocess
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from aurex import publishing
from aurex.config import LLMConfig, ContextPolicyConfig
from aurex.sessiondb import SessionDB
from aurex.task_reply import (FINAL_SYSTEM, finalize_timeout_reply,
                              review_final_answer, post_reviewed_reply,
                              saved_final_answer)
from aurex.tools.registry import ToolError
from aurex.vllm_client import ModelReply
from plar import api


UID = '1234567890abcdef12345678'
TARGET = 'abcdefabcdefabcdefabcdef'
COMMENT = '111111111111111111111111'


class Client:
    def __init__(self):
        self.config = LLMConfig()
        self.outcome = 'completed'
        self.answer = '已核对实际记录，任务要求的分析已完成；未进行额外社区发布。'
        self.calls = []
        self.finish = 'stop'
        self.cancel = None
        self.invalid = False

    def capacity(self):
        return 65536

    def count(self, *args, **kwargs):
        return 1000

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        kwargs['on_delta']('reasoning', 'PRIVATE_FINAL_THINKING')
        if self.cancel:
            self.cancel()
        return ModelReply('not json' if self.invalid else json.dumps({'outcome': self.outcome, 'answer': self.answer}), 'PRIVATE_FINAL_THINKING', [], {}, self.finish)


class TaskActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = self.temp.name
        self.db = SessionDB(str(Path(self.cache) / 'sessions.sqlite3'))
        self.sid = self.db.session('session')
        self.rid = self.db.begin(self.sid, '分析此实验', 'task-one')
        self.user = SimpleNamespace(user_id='actor', token='SECRET', auth_code='SECRET')
        self.runtime = SimpleNamespace(cache_dir=self.cache, session_id=self.sid, task_id=self.rid, user=self.user,
            config=SimpleNamespace(agent=SimpleNamespace(dry_run=False)), task_metadata={})
        self.client, self.events = Client(), []

    def bind(self, **override):
        fields = dict(task_id=self.rid, session_id=self.sid, source='admin',
                      original_user_request='分析此实验', explicit_publish_requested=False)
        fields.update(override)
        return publishing.bind_task_actions(self.cache, **fields)

    def community(self, **override):
        fields = dict(source='community', requester_user_id=UID, requester_nickname='提问者',
                      target={'type': 'Experiment', 'id': TARGET, 'comment_id': COMMENT})
        fields.update(override)
        return self.bind(**fields)

    def review(self):
        return review_final_answer(self.runtime, '候选答复', self.client, self.db, self.sid, self.rid,
            lambda kind, value: self.events.append((kind, value)), context_messages=[{'role': 'user', 'content': '真实任务和实测证据'}])

    def test_conservative_intent_admits_only_direct_original_requests(self):
        accepted = ['请发布这个实验', '帮我设计一个电路并发布', '@aurex 帮我验证然后发布',
                    '发布这个作品', 'Please publish this experiment.', 'Can you publish this experiment?',
                    'Please design a circuit and publish it.']
        rejected = ['不要发布这个实验', '请验证，但不要发布', '这个电路什么时候发布？', '请问发布这个实验有什么风险？',
                    '请介绍发布这个实验的方式', '他让我发布这个实验', '如果没问题就发布这个实验',
                    '“请发布这个实验”是什么意思', '> 请发布这个实验', 'How can I publish this?',
                    "Don't publish this", 'Should I publish it?', 'If it works, publish it', 'He said publish this',
                    'Please explain how to publish this experiment', 'Analyze the experiment']
        for text in accepted:
            with self.subTest(text=text):
                self.assertTrue(publishing.conservative_publish_request(text))
        for text in rejected:
            with self.subTest(text=text):
                self.assertFalse(publishing.conservative_publish_request(text))

    def test_timeout_template_posts_once_with_bound_community_reply_id(self):
        self.community()
        self.runtime.check_cancel = mock.Mock(side_effect=AssertionError(
            'Timeout delivery must not restart the cancelled agent'))
        with mock.patch('aurex.publishing.plar_api.post_task_comment_once',
                        return_value={'Status': 200}) as post:
            first = finalize_timeout_reply(self.runtime, 1800)
            second = finalize_timeout_reply(self.runtime, 1800)
        self.assertEqual(first['state'], 'replied')
        self.assertEqual(second['state'], 'replied')
        self.assertEqual(first['answer'],
            '<user=' + UID + '>@提问者</user> 当前任务到达时间上限1800s，已经停止，请简化问题。')
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs['requester_user_id'], UID)
        self.runtime.check_cancel.assert_not_called()

    def test_new_tasks_are_not_limited_to_two_global_test_sessions(self):
        for i in range(5):
            scope = self.bind(task_id=f'task-{i}', explicit_publish_requested=True,
                              original_user_request='请发布这个电学实验')
            self.assertEqual(scope['publication_limit'], 1)
            self.assertEqual(publishing.publication_authorization(self.cache, self.sid, task_id=f'task-{i}')['state'], 'authorized')

    def test_publish_intent_strips_only_exact_leading_verified_bot_mention(self):
        bot = '698726c4fc064466378176b8'
        prefix = f'<user={bot}>@aurex</user>'
        self.assertTrue(publishing.conservative_publish_request(prefix + ' 请设计一个电路并发布', bot_user_id=bot))
        self.assertTrue(publishing.conservative_publish_request(prefix + ' Please publish this experiment.', bot_user_id=bot))
        self.assertTrue(publishing.conservative_publish_request(f'<user={bot}>@机器人</user> 请发布这个实验',
                        bot_user_id=bot, mention_tag='@机器人'))
        denied = [prefix + ' 请发布这个实验', '> ' + prefix + ' 请发布这个实验',
                  '“' + prefix + ' 请发布这个实验”', '他说' + prefix + ' 请发布这个实验',
                  prefix + ' <user=68b49a54cb66416aa48183d7>@别人</user> 请发布这个实验',
                  prefix + prefix + ' 请发布这个实验', prefix + ' 不要发布这个实验',
                  prefix.replace('@aurex', '@其他机器人') + ' 请发布这个实验']
        self.assertFalse(publishing.conservative_publish_request(denied[0]))
        self.assertFalse(publishing.conservative_publish_request(denied[0], bot_user_id='68b49a54cb66416aa48183d7'))
        for value in denied[1:]:
            with self.subTest(value=value):
                self.assertFalse(publishing.conservative_publish_request(value, bot_user_id=bot))

    def test_explicit_publication_veto_is_distinct_from_unrelated_negative_constraints(self):
        for value in ('不要发布', '请勿对外发布', '只分析不发布', '暂时不自动发布',
                      "Please do not publish this", "Don't publish it", 'Analyze without publishing'):
            with self.subTest(value=value):
                self.assertTrue(publishing.explicitly_forbids_publication(value))
        for value in ('设计一个电路', '请发布，不要夸大验证结论',
                      'Do not change the circuit, publish the validated design'):
            with self.subTest(value=value):
                self.assertFalse(publishing.explicitly_forbids_publication(value))

    def test_binding_cannot_change_requester_intent_destination_or_source(self):
        self.community()
        for overrides in ({'requester_user_id': TARGET}, {'source': 'admin'}, {'explicit_publish_requested': True},
                          {'target': {'type': 'User', 'id': TARGET}}, {'original_user_request': '发布'}):
            with self.subTest(overrides=overrides), self.assertRaises(publishing.PublicationError):
                self.community(**overrides)

    def test_missing_scope_or_nonexplicit_intent_never_grants_publication(self):
        publishing.authorize_session(self.cache, self.sid, '555_state_table')
        with self.assertRaises(publishing.PublicationError):
            publishing.publication_authorization(self.cache, self.sid, task_id=self.rid)
        self.bind()
        with self.assertRaisesRegex(publishing.PublicationError, 'explicitly'):
            publishing.publication_authorization(self.cache, self.sid, task_id=self.rid)

    def test_web_final_thinks_once_has_no_mention_and_reuses_unique_answer(self):
        self.bind(source='web')
        result = self.review()
        self.assertEqual(result['outcome'], 'completed')
        self.assertFalse(result['answer'].startswith('<user='))
        self.assertIs(self.client.calls[0][1]['thinking'], False)
        self.assertEqual(self.client.calls[0][1]['tools'], [])
        again = self.review()
        self.assertEqual(again['review_id'], result['review_id'])
        self.assertEqual(len(self.client.calls), 1)
        with mock.patch.object(api, 'post_task_comment_once') as post:
            delivered = post_reviewed_reply(self.runtime, result['review_id'])
            self.assertEqual(delivered['state'], 'local_delivered')
            post.assert_not_called()
        self.assertNotIn('PRIVATE_FINAL_THINKING', result['answer'])
        self.assertFalse(any(k == 'text_delta' for k, _ in self.events))
        self.assertFalse(any(k == 'reasoning_delta' for k, _ in self.events))
        self.assertEqual(saved_final_answer(self.runtime)['review_id'], result['review_id'])
        self.assertEqual(len(self.client.calls), 1)

    def test_continue_does_not_consume_final_slot_and_completed_review_can_follow(self):
        self.bind()
        self.client.outcome = 'continue'
        first = self.review()
        self.assertIsNone(first['review_id'])
        with publishing._db(self.cache) as store:
            self.assertEqual(store.execute('SELECT COUNT(*) FROM task_final_answers').fetchone()[0], 0)
        self.client.outcome = 'completed'
        second = self.review()
        self.assertTrue(second['review_id'])
        self.assertEqual(len(self.client.calls), 2)

    def test_required_publication_without_success_cannot_be_marked_completed(self):
        self.bind(explicit_publish_requested=True, original_user_request='验证并发布这个实验')
        result = self.review()
        self.assertEqual(result['outcome'], 'continue')
        self.assertIsNone(result['review_id'])

    def test_community_reply_targets_original_user_once_not_comment_id(self):
        self.community()
        result = self.review()
        self.assertTrue(result['answer'].startswith(f'<user={UID}>@提问者</user> '))
        with mock.patch.object(api, 'post_task_comment_once', return_value={'posted': True, 'comment_id': COMMENT}) as post:
            first = post_reviewed_reply(self.runtime, result['review_id'])
            second = post_reviewed_reply(self.runtime, result['review_id'])
            self.assertEqual(first['state'], 'replied')
            self.assertEqual(second['state'], 'replied')
            post.assert_called_once()
            self.assertEqual(post.call_args.kwargs['requester_user_id'], UID)
            self.assertEqual(post.call_args.kwargs['target_id'], TARGET)
            self.assertNotIn('reply_id', post.call_args.kwargs)

    def test_ambiguous_reply_never_reposts_after_restart(self):
        self.community()
        result = self.review()
        with mock.patch.object(api, 'post_task_comment_once', side_effect=TimeoutError()) as post:
            self.assertEqual(post_reviewed_reply(self.runtime, result['review_id'])['state'], 'unknown')
            self.assertEqual(post_reviewed_reply(self.runtime, result['review_id'])['state'], 'unknown')
            post.assert_called_once()
        self.assertTrue(saved_final_answer(self.runtime)['needs_attention'])
        self.assertEqual(saved_final_answer(self.runtime)['outcome'], 'blocked')

    def test_invalid_final_json_returns_continue_without_consuming_unique_answer(self):
        self.bind()
        self.client.invalid = True
        result = self.review()
        self.assertEqual(result['outcome'], 'continue')
        self.assertIsNone(result['review_id'])
        self.assertIsNone(saved_final_answer(self.runtime))
        self.assertEqual(len(self.client.calls), 2)

    def test_short_invalid_final_json_is_repaired_without_reentering_execution_agent(self):
        self.bind()
        original = self.client.chat
        calls = 0
        def chat(messages, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                self.client.calls.append((messages, kwargs))
                return ModelReply('{}', '', [], {}, 'stop')
            return original(messages, **kwargs)
        self.client.chat = chat
        result = self.review()
        self.assertEqual(result['outcome'], 'completed')
        self.assertIsNotNone(result['review_id'])
        self.assertEqual(len(self.client.calls), 2)
        self.assertTrue(any(kind == 'final_review_contract_retry' for kind, _ in self.events))

    def test_single_markdown_json_fence_does_not_restart_completed_task(self):
        self.bind()
        answer = '已核对真实工具证据，直接返回结论。'
        self.client.chat = lambda messages, **kwargs: ModelReply(
            '```json\n' + json.dumps({'outcome': 'completed', 'answer': answer},
                                      ensure_ascii=False) + '\n```',
            '', [], {}, 'stop')
        result = self.review()
        self.assertEqual(result['outcome'], 'completed')
        self.assertEqual(result['answer'], answer)
        self.assertIsNotNone(result['review_id'])

    def test_fenced_json_with_trailing_prose_is_still_rejected(self):
        self.bind()
        self.client.chat = lambda messages, **kwargs: ModelReply(
            '```json\n{"outcome":"completed","answer":"x"}\n```\nextra',
            '', [], {}, 'stop')
        result = self.review()
        self.assertEqual(result['outcome'], 'continue')
        self.assertIsNone(result['review_id'])
        self.assertIsNone(saved_final_answer(self.runtime))

    def test_oversized_final_input_is_archived_compacted_and_then_reviewed(self):
        self.bind()
        self.runtime.config.context = ContextPolicyConfig(safety_tokens=512, summary_max_tokens=512)
        self.client.config = LLMConfig(context_length=12000, max_output_tokens=1024)
        self.client.capacity = lambda: 12000
        self.client.count = lambda messages, tools=None: len(json.dumps(messages, ensure_ascii=False))
        ordinary = self.client.chat
        def chat(messages, **kwargs):
            if kwargs.get('thinking') is False and messages[0].get('content') != FINAL_SYSTEM:
                self.client.calls.append((messages, kwargs))
                return ModelReply('Original evidence retained: actual facts and unfinished objective.', '', [], {}, 'stop')
            return ordinary(messages, **kwargs)
        self.client.chat = chat
        result = review_final_answer(self.runtime, '候选答复', self.client, self.db, self.sid, self.rid,
            lambda k, v: self.events.append((k, v)), context_messages=[{'role': 'user', 'content': 'ORIGINAL_HEAD' + 'x' * 24000 + 'ORIGINAL_TAIL'}])
        self.assertEqual(result['outcome'], 'completed')
        self.assertTrue(any(not opts['thinking'] for _, opts in self.client.calls))
        self.assertFalse(self.client.calls[-1][1]['thinking'])
        with self.db.connect() as store:
            originals = [row['content'] for row in store.execute('SELECT content FROM documents')]
        self.assertTrue(any('ORIGINAL_HEAD' in text and 'ORIGINAL_TAIL' in text and len(text) > 24000 for text in originals))

    def test_disabled_compaction_overflow_is_explicit_attention_not_final_answer(self):
        self.bind()
        self.runtime.config.context = ContextPolicyConfig(auto_compact=False)
        self.client.count = lambda *args, **kwargs: 100000
        result = self.review()
        self.assertEqual(result['outcome'], 'blocked')
        self.assertTrue(result['needs_attention'])
        self.assertIsNone(result['review_id'])
        self.assertFalse(self.client.calls)

    def test_dry_run_and_missing_requester_never_send(self):
        self.community(dry_run=True)
        result = self.review()
        with mock.patch.object(api, 'post_task_comment_once') as post:
            self.assertEqual(post_reviewed_reply(self.runtime, result['review_id'])['state'], 'dry_run')
            post.assert_not_called()
        self.rid = 'missing-requester'
        self.runtime.task_id = self.rid
        self.community(requester_user_id=None)
        with self.assertRaises(publishing.PublicationError):
            self.review()

    def test_cancel_after_final_thinking_does_not_store_or_post_answer(self):
        self.bind()
        cancelled = False
        def check():
            if cancelled:
                raise RuntimeError('cancelled')
        def cancel():
            nonlocal cancelled
            cancelled = True
        self.runtime.check_cancel = check
        self.client.cancel = cancel
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.review()
        with publishing._db(self.cache) as store:
            self.assertEqual(store.execute('SELECT COUNT(*) FROM task_final_answers').fetchone()[0], 0)

    def test_comment_api_uses_bound_sdk_reply_id_and_canonical_prefix_once(self):
        response = {'Status': 200, 'Data': {'Comment': {'ID': COMMENT}}}
        with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, json.dumps(response), '')) as run:
            result = api.post_task_comment_once(self.user, target_id=TARGET, target_type='Experiment',
                requester_user_id=UID, content=f'<user={UID}>@提问者</user> 实测分析结果')
            self.assertTrue(result['posted'])
            self.assertEqual(result['comment_id'], COMMENT)
            self.assertEqual(result['notification_status'], 'unverified')
            self.assertEqual(result['request_format'], 'sdk_plain_reply_prefix')
            payload = json.loads(run.call_args.kwargs['input'])
            self.assertEqual(payload['requester_user_id'], UID)
            self.assertEqual(payload['content'], '回复@提问者: 实测分析结果')
            self.assertEqual(run.call_args.args[0][1:], ['-m', 'plar.comment_worker'])
            self.assertNotIn('SECRET', json.dumps(run.call_args.args))
            self.assertEqual(run.call_args.kwargs['timeout'], 60.0)
            run.assert_called_once()

    def test_live_flat_comment_receipt_does_not_claim_notification_delivery(self):
        response = {'Status': 200, 'Data': {
            'ID': COMMENT, 'TargetID': TARGET, 'Hidden': False, 'Replies': [], 'Flags': None}}
        with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, json.dumps(response), '')) as run:
            result = api.post_task_comment_once(self.user, target_id=TARGET, target_type='User',
                requester_user_id=UID, content=f'<user={UID}>@提问者</user> 通知测试')
            self.assertEqual(result['comment_id'], COMMENT)
            self.assertIs(result['comment_hidden'], False)
            self.assertEqual(result['notification_status'], 'unverified')
            self.assertEqual(len(result['request_content_sha256']), 64)
            self.assertNotEqual(result['request_content_sha256'], result['reviewed_content_sha256'])
            run.assert_called_once()
        # Experiment publication still requires its own versioned contract.
        self.assertEqual(api._publication_headers(self.user)['x-API-Version'], '2503')

    def test_unknown_comment_receipt_id_does_not_trigger_a_second_post(self):
        for data in (None, {}, {'ID': 'not-an-id'}, {'Comment': None}):
            with self.subTest(data=data), mock.patch('subprocess.run', return_value=subprocess.CompletedProcess(
                    [], 0, json.dumps({'Status': 200, 'Data': data}), '')) as run:
                result = api.post_task_comment_once(self.user, target_id=TARGET, target_type='User',
                    requester_user_id=UID, content=f'<user={UID}>@提问者</user> 通知测试')
                self.assertIsNone(result['comment_id'])
                self.assertEqual(result['notification_status'], 'unverified')
                run.assert_called_once()

    def test_review_to_sdk_reply_routing_for_all_community_targets(self):
        # Exercise the real reply adapter up to the isolated SDK process.
        # Original comment, requester and wall/experiment IDs are distinct.
        for target_type in ('User', 'Experiment', 'Discussion'):
            with self.subTest(target_type=target_type):
                self.rid = 'route-' + target_type
                self.runtime.task_id = self.rid
                self.community(target={'type': target_type, 'id': TARGET, 'comment_id': COMMENT})
                reviewed = self.review()
                response = {'Status': 200, 'Data': {'ID': '2' * 24}}
                with mock.patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, json.dumps(response), '')) as run:
                    result = post_reviewed_reply(self.runtime, reviewed['review_id'])
                    post_reviewed_reply(self.runtime, reviewed['review_id'])
                    run.assert_called_once()
                    body = json.loads(run.call_args.kwargs['input'])
                    self.assertEqual(body['requester_user_id'], UID)
                    self.assertNotEqual(body['requester_user_id'], COMMENT)
                    self.assertNotEqual(body['requester_user_id'], TARGET)
                    self.assertEqual(body['target_id'], TARGET)
                    self.assertEqual(body['target_type'], target_type)
                    self.assertEqual(body['content'], '回复@提问者: ' + self.client.answer)
                    self.assertTrue(result['answer'].startswith(f'<user={UID}>@提问者</user> '))
                    self.assertEqual(result['state'], 'replied')
                with publishing._db(self.cache) as store:
                    receipt = json.loads(store.execute('SELECT reply_receipt FROM task_final_answers WHERE task_id=?', (self.rid,)).fetchone()[0])
                self.assertEqual(receipt['request_reply_user_id'], UID)
                self.assertEqual(receipt['request_target_id'], TARGET)

    def test_malformed_or_mismatched_mentions_are_rejected_before_sending(self):
        for content in (f'<user={TARGET}>@他人</user> 答案', f'<user={UID}>@</user> 答案',
                        f'<user={UID}>@提问 者</user> 答案', f'<user={UID}>@提问者:其他人</user> 答案',
                        f'<user={UID}>@提问者</user> ', '回复@提问者: 未经绑定的答案'):
            with self.subTest(content=content), mock.patch('subprocess.run') as run:
                with self.assertRaises(api.PLARError):
                    api.post_task_comment_once(self.user, target_id=TARGET, target_type='User',
                        requester_user_id=UID, content=content)
                run.assert_not_called()

    def test_sdk_timeout_is_durable_unknown_and_never_falls_back_or_retries(self):
        self.community()
        reviewed = self.review()
        with mock.patch('subprocess.run', side_effect=subprocess.TimeoutExpired(['python'], 60, output='SECRET')) as run, \
                mock.patch('requests.Session') as requests_session:
            result = post_reviewed_reply(self.runtime, reviewed['review_id'])
            repeated = post_reviewed_reply(self.runtime, reviewed['review_id'])
            self.assertEqual(result['state'], 'unknown')
            self.assertEqual(repeated['state'], 'unknown')
            self.assertNotIn('SECRET', json.dumps(result))
            run.assert_called_once()
            requests_session.assert_not_called()

    def test_wire_conversion_preserves_body_and_only_replaces_the_leading_mention(self):
        body = f'  第一行\n<user={TARGET}>@被引用者</user>\n  第三行'
        self.assertEqual(api._sdk_reply_content(f'<user={UID}>@提问者</user> ' + body, UID),
                         '回复@提问者: ' + body)

    def test_real_child_process_uses_installed_sdk_and_only_sends_once_offline(self):
        # A child-only transport stub closes all real sockets before importing
        # the SDK. Exercise review -> ledger -> subprocess -> official method.
        marker = Path(self.cache) / 'sdk-calls.jsonl'
        (Path(self.cache) / 'sitecustomize.py').write_text(
            "import json, socket\n"
            "def blocked(*args, **kwargs):\n    raise AssertionError('Network forbidden')\n"
            "socket.socket.connect = blocked\n"
            "from physicsLab.web import _request\n"
            "def post(**kwargs):\n"
            "    assert kwargs['path'] == 'Messages/PostComment'\n"
            "    safe = {'body': kwargs['body'], 'header_names': sorted(kwargs['header'])}\n"
            f"    with open({str(marker)!r}, 'a') as f:\n        f.write(json.dumps(safe) + '\\n')\n"
            f"    return {{'Status': 200, 'Data': {{'ID': {COMMENT!r}, 'Hidden': False}}}}\n"
            "_request.post_https = post\n", encoding='utf-8')
        self.community()
        reviewed = self.review()
        with mock.patch.dict(os.environ, {'PYTHONPATH': self.cache + os.pathsep + os.environ.get('PYTHONPATH', '')}):
            first = post_reviewed_reply(self.runtime, reviewed['review_id'])
            second = post_reviewed_reply(self.runtime, reviewed['review_id'])
        self.assertEqual(first['state'], 'replied')
        self.assertEqual(second['state'], 'replied')
        calls = [json.loads(line) for line in marker.read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['body'], {'TargetID': TARGET, 'TargetType': 'Experiment',
            'Language': 'Chinese', 'ReplyID': UID, 'Content': '回复@提问者: ' + self.client.answer, 'Special': None})
        self.assertEqual(set(calls[0]['header_names']), {'Content-Type', 'x-API-Token', 'x-API-AuthCode'})

    def test_sdk_reply_wrapper_preserves_explicit_recipient(self):
        user = SimpleNamespace(post_comment=mock.Mock())
        api.post_comment(user, target_id=TARGET, target_type='User',
                         content=f'<user={UID}>@提问者</user> 回答', reply_id=UID)
        self.assertEqual(user.post_comment.call_args.kwargs['reply_id'], UID)
        self.assertEqual(user.post_comment.call_args.kwargs['content'], '回复@提问者: 回答')


if __name__ == '__main__':
    unittest.main()
