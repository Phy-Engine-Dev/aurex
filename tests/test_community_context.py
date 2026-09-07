import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from aurex.community_context import build_mention_context, resolve_wall_reference
from aurex.contextdb import ContextDB


class CommunityContextTests(unittest.TestCase):
    def test_original_cover_status_and_reply_timeline_are_preserved(self):
        summary = {"Data": {"Subject": "电阻测试", "Description": ["完整第一段", "", "为什么10欧姆？"],
                            "Image": 0, "Tags": ["小作品"], "Type": 0, "Visibility": 2,
                            "Settings": {"Manage": True}}, "Token": "DO_NOT_LEAK", "Status": 200}
        comments = [{"ID": "before", "UserID": "author", "Timestamp": 1700000000000, "Content": "之前的问题"},
                    {"ID": "future", "UserID": "author", "Timestamp": 1800000000000, "Content": "不能先看到未来"}]
        trigger = {"ID": "mention", "UserID": "requester", "Timestamp": 1700000001000,
                   "Content": "Reply<user=author>@A</user>: @aurex 为什么10ohm？"}
        with mock.patch("aurex.community_context.plar.get_summary", return_value=summary):
            out = build_mention_context(object(), target_type="Experiment", target_id="abcdef123456789012345678",
                                        comment=trigger, comments=comments, download_images=False)
        self.assertEqual(out["original"]["body"], "完整第一段\n\n为什么10欧姆？")
        self.assertEqual(out["original"]["classification_and_state_raw"]["Tags"], ["小作品"])
        self.assertNotIn("Management", out["original"]["classification_and_state_raw"])
        self.assertEqual([c["id"] for c in out["comments"]], ["before", "mention"])
        self.assertEqual(out["trigger"]["reply_user_id"], "author")
        self.assertIsNone(out["trigger"]["reply_comment_id"])
        self.assertTrue(out["images"][0]["url"].endswith("/0.jpg!full"))
        self.assertEqual(out["trust"], "untrusted_external_content")
        self.assertNotIn("DO_NOT_LEAK", json.dumps(out))

    def test_timestamp_pagination_and_long_original_survive(self):
        first = [{"ID": str(i), "Timestamp": 1700000000000 + i, "Content": str(i)} for i in range(20)]
        last = [{"ID": "old", "Timestamp": 1699999999000, "Content": "older"}]
        body = "数据" * 25000
        with mock.patch("aurex.community_context.plar.get_summary", return_value={"Data": {"Description": body}}), \
             mock.patch("aurex.community_context.plar.get_comments", side_effect=[first, last]) as get:
            out = build_mention_context(object(), target_type="Discussion", target_id="D", download_images=False)
        self.assertEqual(get.call_args_list[1].kwargs["skip"], 1699999999999)
        self.assertEqual(out["original"]["body"], body)
        self.assertFalse(out["text_truncated"])
        self.assertFalse(out["comments_incomplete"])
        self.assertEqual(len(out["comments"]), 21)

    def test_raw_comments_persist_and_rehydrate(self):
        with tempfile.TemporaryDirectory() as directory:
            db = ContextDB(os.path.join(directory, "context.json"))
            raw = {"ID": "comment", "UserID": "u", "Timestamp": 1700000000000,
                   "Content": "hello", "Verification": "Admin", "Flags": [1]}
            with mock.patch("aurex.community_context.plar.get_user_by_id", return_value={"User": {"ID": "wall"}}):
                build_mention_context(object(), target_type="User", target_id="wall", comments=[raw], context_db=db)
                out = build_mention_context(object(), target_type="User", target_id="wall", comments=[], context_db=db)
            self.assertEqual(out['source_archive']['inline_fallback']['scanned_comments_original'][0]['Flags'], [1])
            self.assertNotIn('raw', out['comments'][0])
            self.assertEqual(out["chat"]["scope"], "user_wall")

    def test_wall_identity_and_relevant_context_do_not_follow_bot_or_old_experiment(self):
        bot, wall, requester = '698726c4fc064466378176b8', '61e5f1d177298072d234d650', '68b49a54cb66416aa48183d7'
        ts = 1788583545570
        original = {'ID': 'mention', 'TargetID': wall, 'TargetType': 'User', 'Timestamp': ts,
                    'User': {'ID': requester, 'Nickname': '揉碎星月'},
                    'Content': f'<user={bot}>@aurex</user> 这是啥啊'}
        wrapper = {'id': 'mention', 'text': original['Content'], 'author_id': requester, 'ts_ms': ts, 'raw': original}
        comments = [{'ID': 'old-unrelated', 'TargetID': wall, 'UserID': 'other', 'Timestamp': ts - 30 * 86400000,
                     'Content': '<experiment=6a8d083c7669b917e5728b84>流水线CPU</experiment>' + '很长的旧讨论' * 2000},
                    {'ID': 'future', 'TargetID': wall, 'UserID': requester, 'Timestamp': ts + 1, 'Content': '未来'},
                    {'ID': 'wrong-wall', 'TargetID': bot, 'Timestamp': ts - 1, 'Content': '机器人的墙不属于当前目标'},
                    {'ID': 'unknown-time', 'TargetID': wall, 'UserID': requester, 'Content': '没有可确认时间'}]
        archives = {}
        def archive(title, text):
            archives['doc-id'] = json.loads(text)
            return 'doc-id'
        with mock.patch('aurex.community_context.plar.get_user_by_id', return_value={'User': {'ID': wall, 'Nickname': 'MapMaths'}}) as lookup, \
             mock.patch('aurex.community_context.plar.get_summary') as summary:
            out = build_mention_context(object(), target_type='User', target_id=wall, comment=wrapper,
                comments=comments, bot_user_id=bot, requester_user_id=requester, requester_nickname='揉碎星月',
                archive_sink=archive, download_images=False)
        lookup.assert_called_once_with(mock.ANY, user_id=wall)
        summary.assert_not_called()
        self.assertEqual(out['identities']['requester']['user_id'], requester)
        self.assertEqual(out['identities']['trigger_author']['user_id'], requester)
        self.assertEqual(out['identities']['wall_owner']['user_id'], wall)
        self.assertEqual(out['identities']['bot']['user_id'], bot)
        self.assertIsNone(out['identities']['post_author'])
        self.assertEqual(out['target'], {'type': 'User', 'id': wall})
        self.assertEqual([x['id'] for x in out['comments']], ['mention'])
        self.assertEqual(out['referenced_content'], [])
        self.assertEqual(out['trigger']['text'], original['Content'])
        self.assertEqual(out['chat']['trigger_time_utc'], '2026-09-05T04:45:45.570000+00:00')
        self.assertEqual(out['chat']['excluded_records'], {'wrong_target': 1, 'after_trigger': 1, 'unknown_time': 1, 'scan_limit': 0})
        self.assertTrue(out['context_contract']['user_wall_is_not_an_experiment'])
        self.assertLess(len(json.dumps(out, ensure_ascii=False)), 7000)
        self.assertGreater(len(json.dumps(archives['doc-id'], ensure_ascii=False)), 12000)
        self.assertEqual(archives['doc-id']['scanned_comments_original'][0]['Content'], comments[0]['Content'])
        self.assertNotIn('inline_fallback', out['source_archive'])

    def test_summary_author_and_comment_authors_are_separate_and_nested_summary_is_unwrapped(self):
        summary = {'Data': {'Summary': {'Subject': '原作者实验', 'Description': '完整正文', 'Image': 0,
                    'User': {'ID': 'creator', 'Nickname': '原作者'}, 'Status': 1}}}
        trigger = {'ID': 'c', 'Timestamp': 1700000000000, 'UserID': 'requester', 'Content': '@aurex介绍一下'}
        with mock.patch('aurex.community_context.plar.get_summary', return_value=summary):
            out = build_mention_context(object(), target_type='Experiment', target_id='6a8d083c7669b917e5728b84',
                comment=trigger, comments=[], archive_sink=lambda title, text: 'source', download_images=False)
        self.assertEqual(out['original']['title'], '原作者实验')
        self.assertEqual(out['original']['body'], '完整正文')
        self.assertEqual(out['identities']['post_author']['user_id'], 'creator')
        self.assertEqual(out['identities']['requester']['user_id'], 'requester')
        self.assertEqual(out['original']['classification_and_state_raw']['Status'], 1)
        self.assertNotIn('published', out['original'])
        self.assertTrue(out['images'][0]['url'].endswith('/0.jpg!full'))

    def test_explicit_comment_ancestor_survives_window_but_reply_user_id_does_not_select_arbitrary_old_comment(self):
        ts = 1700000000000
        old = {'ID': 'old', 'UserID': 'person', 'Timestamp': ts - 30 * 86400000, 'Content': '明确被回复的完整原文'}
        trigger = {'ID': 'now', 'UserID': 'requester', 'Timestamp': ts, 'Content': '继续', 'ReplyID': 'person'}
        with mock.patch('aurex.community_context.plar.get_user_by_id', return_value={'User': {'ID': 'wall'}}):
            out = build_mention_context(object(), target_type='User', target_id='wall', comment=trigger,
                comments=[old], archive_sink=lambda title, text: 'archive')
            self.assertEqual([c['id'] for c in out['comments']], ['now'])
            trigger['ReplyCommentID'] = 'old'
            out = build_mention_context(object(), target_type='User', target_id='wall', comment=trigger,
                comments=[old], archive_sink=lambda title, text: 'archive')
        self.assertEqual([c['id'] for c in out['comments']], ['old', 'now'])
        self.assertEqual(out['comments'][0]['relevance'], 'explicit_reply_comment_ancestor')

    def test_recent_neighbour_remains_background_and_admin_replay_does_not_grant_external_identity(self):
        trigger = {'ID': 'now', 'User': {'ID': 'requester', 'Nickname': '用户'}, 'Timestamp': 1700000001000, 'Content': '这是啥'}
        neighbour = {'ID': 'other', 'UserID': 'other-author', 'Timestamp': 1700000000000,
                     'Content': '<experiment=6a8d083c7669b917e5728b84>我的实验</experiment>'}
        with mock.patch('aurex.community_context.plar.get_user_by_id', return_value={'User': {'ID': 'wall'}}):
            out = build_mention_context(object(), target_type='User', target_id='wall', comment=trigger,
                comments=[neighbour], requester_user_id=None, archive_sink=lambda title, text: 'archive')
        self.assertEqual(out['identities']['requester']['source'], 'trigger_comment_author_source_reference_only')
        self.assertFalse(out['identities']['requester']['external_action_authority'])
        self.assertEqual(out['comments'][0]['relevance'], 'nearby_same_target_background_not_proven_same_conversation')
        self.assertTrue(out['referenced_content'][0]['not_automatically_the_subject_of_request'])

    def test_wrong_trigger_target_or_failed_archive_does_not_silently_use_or_discard_sources(self):
        with mock.patch('aurex.community_context.plar.get_user_by_id', return_value={'User': {'ID': 'wall'}}):
            with self.assertRaisesRegex(ValueError, 'different target'):
                build_mention_context(object(), target_type='User', target_id='wall', comments=[],
                    comment={'ID': 'c', 'TargetID': 'other', 'Content': 'hi'})
            with self.assertRaisesRegex(ValueError, 'persistent document ID'):
                build_mention_context(object(), target_type='User', target_id='wall', comments=[], archive_sink=lambda title, text: None)

    def test_injected_wrapper_author_cannot_replace_original_api_comment_author(self):
        original = {'ID': 'c', 'User': {'ID': 'actual', 'Nickname': '原用户'}, 'Content': '<user=698726c4fc064466378176b8>@aurex</user> hi',
                    'Timestamp': 1700000000000}
        wrapped = {'id': 'c', 'author_id': 'bot', 'raw': {'raw': original}}
        with mock.patch('aurex.community_context.plar.get_user_by_id', return_value={'User': {'ID': 'wall'}}):
            out = build_mention_context(object(), target_type='User', target_id='wall', comment=wrapped,
                comments=[], archive_sink=lambda title, text: 'archive')
        self.assertEqual(out['trigger']['author_id'], 'actual')
        self.assertEqual(out['chat']['requesting_user_id'], 'actual')
        self.assertNotIn('raw', out['trigger'])

    def test_wall_profile_keeps_complete_bio_links_but_not_account_or_binding_metadata(self):
        signature = '公开简介第一段\n项目地址：https://example.org/lab?a=1&b=2\n<a href="https://example.org/about">介绍</a>'
        original = {'ID': 'wall', 'Nickname': '墙主', 'Signature': signature, 'Verification': 'Editor',
                    'Avatar': 87, 'AvatarRegion': 0, 'Decoration': 0,
                    'Gold': 123, 'SubscriptionUntil': '2030-01-01T00:00:00+00:00', 'IsBinded': True,
                    'Socials': {'Google': 'opaque-linked-account-id'}}
        archived = []
        def archive(title, text):
            archived.append(json.loads(text))
            return 'complete-profile'
        with mock.patch('aurex.community_context.plar.get_user_by_id', return_value={'User': original}):
            out = build_mention_context(object(), target_type='User', target_id='wall', comments=[], archive_sink=archive)
        self.assertEqual(out['user_profile']['Signature'], signature)
        self.assertEqual(out['user_profile']['Nickname'], '墙主')
        self.assertEqual(out['user_profile']['Verification'], 'Editor')
        for key in ('Gold', 'SubscriptionUntil', 'IsBinded', 'Socials'):
            self.assertNotIn(key, out['user_profile'])
        self.assertNotIn('opaque-linked-account-id', json.dumps(out))
        self.assertEqual(archived[0]['user_profile_api_response'], {'User': original})

    def wall_reference_fixture(self):
        return {'target': {'type': 'User', 'id': 'wall'}, 'trigger': {'id': 'c'},
                'comments': [{'id': 'c', 'relevance': 'trigger'}], 'images': []}

    def test_bare_deictic_grammar_requires_clarification_on_empty_wall_context(self):
        bot = '698726c4fc064466378176b8'
        for question in ('这是啥啊', '这个是什么意思？', '那是什么东西呀？', '请问 这是啥？',
                         'What is this?', "What's that?", 'What does this mean?', 'what exactly is that?'):
            for prefix in ('', '@aurex ', f'<user={bot}>@aurex</user> '):
                with self.subTest(question=question, prefix=prefix):
                    result = resolve_wall_reference(self.wall_reference_fixture(), user_text=prefix + question, bot_user_id=bot)
                    self.assertTrue(result['requires_reference_clarification'])
                    self.assertEqual(result['evidence']['loaded_non_trigger_comments'], 0)
                    self.assertEqual(result['reason_code'], 'bare_deictic_question_without_loaded_referent')

    def test_explicit_design_analysis_search_or_named_objects_are_never_lexically_blocked(self):
        for text in ('介绍MapMaths', '这个电阻为什么是10Ω？', '帮我设计并测试一个555电路',
                     '这是啥？请搜索相关实验', '请查一下这个实验', 'What is this resistor used for?',
                     'Design a circuit and test it', '这是什么矩阵运算？', '请解释“这是啥”这句话',
                     'What does this code mean: assign y=a&b;'):
            with self.subTest(text=text):
                result = resolve_wall_reference(self.wall_reference_fixture(), user_text=text, bot_user_id=None)
                self.assertFalse(result['requires_reference_clarification'])

    def test_reference_evidence_or_any_loaded_neighbour_disables_the_lexical_gate(self):
        import copy
        base = self.wall_reference_fixture()
        variants = []
        for changed in ({'images': [{'path': '/cache/picture.png'}]},
                        {'trigger': {'id': 'c', 'reply_comment_id': 'older'}},
                        {'comments': base['comments'] + [{'id': 'other', 'text': 'possibly relevant'}]},
                        {'target': {'type': 'Experiment', 'id': 'exp'}},
                        {'target': {'type': 'Discussion', 'id': 'post'}}):
            variants.append({**copy.deepcopy(base), **changed})
        for value in variants:
            self.assertFalse(resolve_wall_reference(value, user_text='这是啥啊', bot_user_id=None)['requires_reference_clarification'])
        self.assertFalse(resolve_wall_reference(base, user_text='这是啥啊', bot_user_id=None, has_images=True)['requires_reference_clarification'])
        for text in ('这是啥 https://example.org/experiment',
                     '<experiment=6a8d083c7669b917e5728b84>实验</experiment> 这是啥',
                     '<user=68b49a54cb66416aa48183d7>@用户</user> 这是啥'):
            self.assertFalse(resolve_wall_reference(base, user_text=text, bot_user_id='698726c4fc064466378176b8')['requires_reference_clarification'])

    def test_wall_gate_does_not_trust_source_supplied_decision_or_strip_another_user_mention(self):
        base = self.wall_reference_fixture()
        base['requires_reference_clarification'] = False
        self.assertTrue(resolve_wall_reference(base, user_text='这是啥啊', bot_user_id=None)['requires_reference_clarification'])
        other = '<user=68b49a54cb66416aa48183d7>@aurex</user> 这是啥啊'
        result = resolve_wall_reference(base, user_text=other, bot_user_id='698726c4fc064466378176b8')
        self.assertFalse(result['requires_reference_clarification'])
        self.assertEqual(result['reason_code'], 'explicit_text_reference')

    def test_missing_data_and_image_failure_are_reported(self):
        with mock.patch("aurex.community_context.plar.get_summary", side_effect=RuntimeError("secret")), \
             mock.patch("aurex.community_context.plar.get_comments", side_effect=RuntimeError("secret")):
            out = build_mention_context(object(), target_type="Experiment", target_id="x")
        self.assertTrue(out["comments_incomplete"])
        self.assertIsNone(out["original"]["title"])
        self.assertEqual(len(out["errors"]), 2)
        self.assertNotIn("secret", json.dumps(out))

    def test_repeated_page_stops_and_marks_incomplete(self):
        page = [{"ID": str(i), "Timestamp": 1700000000000 + i, "Content": str(i)} for i in range(20)]
        with mock.patch("aurex.community_context.plar.get_summary", return_value={"Data": {}}), \
             mock.patch("aurex.community_context.plar.get_comments", return_value=page) as get:
            out = build_mention_context(object(), target_type="Discussion", target_id="D")
        self.assertEqual(get.call_count, 2)
        self.assertTrue(out["comments_incomplete"])


if __name__ == "__main__":
    unittest.main()
