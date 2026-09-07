"""Behavioral v3 regressions: no GPU/model service is contacted."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aurex.config import AurexConfig, LLMConfig, StorageConfig, TrackingConfig
from aurex.context_budget import ContextBudget
from aurex.session_agent import (SHORT_COMMUNITY_SYSTEM, SessionAgent,
                                 _is_archived_full_netlist,
                                 _is_short_community_lookup,
                                 _mutate_task_plan,
                                 _short_community_required_tools,
                                 _review_requests_answer_revision,
                                 _progress_fingerprint)
from aurex.sessiondb import SessionDB
from aurex.tools.registry import ToolRegistry, ToolSpec
from aurex.vllm_client import DegenerateGeneration, ModelError, ModelReply, VLLMClient


def reply(content="done", *, reasoning="", calls=None, finish="stop"):
    return ModelReply(content, reasoning, calls or [], {}, finish)


class FakeLLM:
    def __init__(self, config, replies):
        self.config = config
        self.replies = iter(replies)
        self.requests = []
        self.final_reviews = None
        self.short_final_answers = None
        self.last_output = None
        self.summary_response = None

    def capacity(self):
        return self.config.context_length

    def count(self, messages, tools=None):
        total = 100 + len(json.dumps(tools or [])) // 4
        for message in messages:
            content = message.get("content") or ""
            if isinstance(content, list):
                total += sum(500 if part.get("type") == "image_url" else len(part.get("text", "")) // 4 for part in content)
            else:
                total += len(content) // 4
        return total

    def chat(self, messages, **options):
        self.requests.append((messages, options))
        from aurex.task_reply import FINAL_SYSTEM, SHORT_COMMUNITY_FINAL_SYSTEM
        from aurex.context_budget import SUMMARY_PROMPT
        if messages[0].get('content') == SHORT_COMMUNITY_FINAL_SYSTEM:
            result = (next(self.short_final_answers) if self.short_final_answers is not None else
                      {'answer': self.last_output.content})
            output = reply(json.dumps(result, ensure_ascii=False))
        elif messages[0].get('content') == FINAL_SYSTEM:
            result = (next(self.final_reviews) if self.final_reviews is not None else
                      {'outcome': 'completed', 'answer': self.last_output.content})
            output = reply(json.dumps(result, ensure_ascii=False), reasoning='PRIVATE_FINAL_REVIEW')
        elif messages[0].get('content') == SUMMARY_PROMPT and self.summary_response is not None:
            output = reply(self.summary_response)
        else:
            output = next(self.replies)
            if isinstance(output, Exception):
                raise output
            self.last_output = output
        if options.get("on_delta"):
            if output.reasoning:
                options["on_delta"]("reasoning", output.reasoning)
            if output.content:
                options["on_delta"]("text", output.content)
        return output


class SessionAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = replace(AurexConfig(),
                           storage=StorageConfig(cache_dir=self.temp.name),
                           tracking=TrackingConfig(database_path=os.path.join(self.temp.name, "sessions.sqlite3")),
                           llm=LLMConfig(enabled=True, context_length=32768, max_output_tokens=512, max_images=2),
                           agent=replace(AurexConfig().agent, max_tool_loops=3))

    def agent(self, outputs, registry=None):
        fake = FakeLLM(self.cfg.llm, outputs)
        with mock.patch("aurex.session_agent.VLLMClient", return_value=fake):
            agent = SessionAgent(cfg=self.cfg, config_path=os.path.join(self.temp.name, "config.json"), tools=registry or ToolRegistry())
        return agent, fake

    def test_plan_mutations_do_not_repeat_all_durable_notes_in_tool_history(self):
        db = SessionDB(os.path.join(self.temp.name, "plan.sqlite3"))
        sid = db.session("plan-session")
        rid = db.enqueue_task(sid, "audit", source="admin")
        initial = _mutate_task_plan(db, sid, rid, {"action": "set", "items": [
            {"id": "fetch", "title": "Fetch"},
            {"id": "verify", "title": "Verify"},
        ]})
        self.assertEqual(initial["remaining"], 2)
        self.assertNotIn("note", initial["items"][0])
        updated = _mutate_task_plan(db, sid, rid, {
            "action": "update", "id": "fetch", "status": "completed",
            "note": "exact durable finding", "next_id": "verify",
        })
        self.assertEqual(updated["updated"]["note"], "exact durable finding")
        self.assertTrue(all("note" not in item for item in updated["items"]))
        recovered = _mutate_task_plan(db, sid, rid, {"action": "get"})
        self.assertEqual(recovered["items"][0]["note"], "exact durable finding")

    def test_full_netlist_documents_are_not_model_paging_targets(self):
        db = SessionDB(os.path.join(self.temp.name, "documents.sqlite3"))
        sid = db.session("document-session")
        netlist = db.document(sid, "circuit_query_many: full netlist_path", '{"components": []}')
        ordinary = db.document(sid, "Experiment summary", "text")
        self.assertTrue(_is_archived_full_netlist(db, sid, netlist))
        self.assertFalse(_is_archived_full_netlist(db, sid, ordinary))
        self.assertFalse(_is_archived_full_netlist(db, sid, "missing"))

    def test_completed_durable_plan_disables_execution_tools_while_answering(self):
        agent, fake = self.agent([reply("依据现有证据，验证已完成。")])
        sid = agent.db.session("completed-plan")
        rid = agent.db.enqueue_task(sid, "设计然后验证一个电路", source="admin")
        agent.db.set_task_plan(sid, rid, [{"id": "verify", "title": "验证"}])
        agent.db.update_task_plan_item(sid, rid, "verify", "completed", note="5 V measured")
        result = agent.handle(user_text="设计然后验证一个电路", session_id=sid, run_id=rid)
        self.assertEqual(result["status"], "completed")
        # The first request is the execution/final-draft turn. The subsequent
        # request is the independent final reviewer and has its own contract.
        self.assertEqual(fake.requests[0][1]["tools"], [])
        self.assertIn("全部持久化步骤已完成", fake.requests[0][0][0]["content"])

    def test_completed_plan_wording_correction_uses_answer_revision_mode(self):
        plan = [{'id': 'verify', 'status': 'completed'}]
        review = {'review_document_id': 'review-doc',
                  'answer': '候选结论与实际轨迹矛盾，请修正公开答案，删除错误表述。'}
        self.assertTrue(_review_requests_answer_revision(review, plan, []))
        self.assertFalse(_review_requests_answer_revision(
            {'review_document_id': 'review-doc', 'answer': '请重新运行瞬态验证。'}, plan, []))
        self.assertFalse(_review_requests_answer_revision(review, plan, [
            {'id': 'verify', 'status': 'in_progress'}]))
        self.assertFalse(_review_requests_answer_revision(
            {'review_document_id': None, 'answer': '请修正公开答案。'}, plan, []))

    def test_short_community_route_is_narrow_and_never_captures_cpu_work(self):
        self.assertTrue(_is_short_community_lookup(
            'community', '<user=' + 'a' * 24 + '>@aurex</user> 介绍一下这个用户',
            explicit_publish_requested=False))
        self.assertTrue(_is_short_community_lookup(
            'community', '总结一下这个用户发布的作品',
            explicit_publish_requested=False))
        self.assertTrue(_is_short_community_lookup(
            'community', '<user=' + 'a' * 24 + '>@aurex</user> 总结一下 <user=' +
            'b' * 24 + '>@Target</user> 这个用户发布的内容',
            explicit_publish_requested=False))
        self.assertTrue(_is_short_community_lookup(
            'community', '请总结用户 <user=' + 'b' * 24 + '>@MapMaths</user> 最近发布的内容；'
                         '只查询少量最新公开作品，不要分析电路。',
            explicit_publish_requested=False))
        self.assertTrue(_is_short_community_lookup(
            'admin', '只读审计：介绍用户 <user=' + 'b' * 24 + '>@MapMaths</user>；不要发布或评论。',
            explicit_publish_requested=False))
        for request in ('介绍并验证他的CPU设计', '介绍这个用户并仿真电路',
                        '介绍这个用户然后发布实验'):
            self.assertFalse(_is_short_community_lookup(
                'community', request, explicit_publish_requested=False))
        self.assertFalse(_is_short_community_lookup(
            'web', '介绍一下这个用户', explicit_publish_requested=False))
        self.assertFalse(_is_short_community_lookup(
            'community', '<user=' + 'a' * 24 + '>@aurex</user> 介绍一下这个实验',
            explicit_publish_requested=False))

    def test_short_community_evidence_checklist_is_intent_scoped(self):
        self.assertEqual(_short_community_required_tools('介绍用户 MapMaths 的创作概况，不要评论'),
                         {'plar_get_user', 'plar_query_experiments'})
        self.assertEqual(_short_community_required_tools('查询用户最新发布的实验，回答评论区最新一条评论是谁发布的'),
                         {'plar_query_experiments', 'plar_get_comments'})

    def test_task_hard_timeout_stops_and_returns_fixed_template_without_review(self):
        self.cfg = replace(self.cfg, agent=replace(self.cfg.agent, task_timeout_sec=1))
        agent, fake = self.agent([])
        sid = agent.db.session('timeout-task')
        rid = agent.db.enqueue_task(sid, 'Long task', source='admin')

        def stalled_chat(*args, **kwargs):
            time.sleep(1.05)
            fake.on_tick()
            self.fail('The deadline callback must stop generation')

        fake.chat = stalled_chat
        result = agent.handle(user_text='Long task', session_id=sid, run_id=rid)
        self.assertTrue(result['timed_out'])
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['answer'], '当前任务到达时间上限1s，已经停止，请简化问题。')
        self.assertEqual(agent.db.get_task(rid)['status'], 'cancelled')
        kinds = [event['kind'] for event in agent.db.events(sid)]
        self.assertIn('task_timeout', kinds)
        self.assertEqual(kinds.count('answer'), 1)

    def test_extreme_single_response_repetition_recovers_without_disabling_tools(self):
        tools = ToolRegistry()
        execute = mock.Mock(return_value={'value': 17})
        tools.register(ToolSpec('probe', 'Read value', {'type': 'object'}, execute))
        partial = reply('', calls=[{'id': 'partial', 'type': 'function',
            'function': {'name': 'probe', 'arguments': '{'}}],
            finish='repetition_guard')
        stopped = DegenerateGeneration(
            'extreme repetition', partial,
            {'reason': 'same_token_run', 'generated_tokens': 256,
             'max_same_token_run': 256,
             'max_exact_repeat': {'period_tokens': 1, 'copies': 256, 'span_tokens': 256}})
        call = {'id': 'complete', 'type': 'function',
                'function': {'name': 'probe', 'arguments': '{}'}}
        agent, fake = self.agent(
            [stopped, reply('', calls=[call], finish='tool_calls'), reply('value is 17')],
            tools)
        result = agent.handle(user_text='Read the value once')
        self.assertEqual(result['status'], 'completed')
        execute.assert_called_once()
        self.assertTrue(fake.requests[1][1]['tools'])
        events = agent.db.events(result['session_id'])
        guard = next(event for event in events if event['kind'] == 'generation_repetition_guard')
        self.assertFalse(guard['data']['executed'])
        self.assertEqual(guard['data']['reason'], 'same_token_run')
        self.assertEqual(len([event for event in events if event['kind'] == 'answer']), 1)

    def test_short_community_lookup_uses_read_only_tools_and_one_no_think_finalizer(self):
        from types import SimpleNamespace
        tools = ToolRegistry()
        tools.register(ToolSpec('plar_get_user', 'Get user', {'type': 'object'},
            lambda *_: {'id': 'e' * 24, 'nickname': 'H₂CO₃', 'level': 18,
                        'stats': {'experiment_count': 62, 'star_count': 607}}))
        tools.register(ToolSpec('plar_query_experiments', 'Query works', {'type': 'object'},
            lambda *_: [{'subject': 'PID调节器电路', 'popularity': 850},
                        {'subject': '混沌电路', 'popularity': 404}]))
        # This unrelated capability must not be visible on the short route.
        tools.register(ToolSpec('circuit_analyze', 'Analyze circuit', {'type': 'object'},
                                lambda *_: self.fail('Circuit tool must not run')))
        get_user = {'id': 'profile', 'type': 'function', 'function': {
            'name': 'plar_get_user', 'arguments': '{"name":"H₂CO₃"}'}}
        works = {'id': 'works', 'type': 'function', 'function': {
            'name': 'plar_query_experiments', 'arguments': '{"user_id":"' + 'e' * 24 + '"}'}}
        agent, fake = self.agent([
            reply('', calls=[get_user], finish='tool_calls'),
            reply('', calls=[works], finish='tool_calls'),
            reply('@Alice @H₂CO₃ 是活跃创作者，还问过一个并不存在的问题。'),
        ], tools)
        fake.short_final_answers = iter([{'answer': 'H₂CO₃ 是社区用户，公开资料显示等级 18，发布过 62 个实验，获得 607 星。作品示例包括《PID调节器电路》和《混沌电路》。'}])
        sid = agent.db.session('short-community')
        uid, target = 'a' * 24, 'b' * 24
        rid = agent.db.enqueue_task(sid,
            '<user=' + 'f' * 24 + '>@aurex</user> 介绍一下 <user=' + 'e' * 24 + '>@H₂CO₃</user> 这个用户',
            source='community', requester_user_id=uid, requester_nickname='Alice',
            target={'type': 'Discussion', 'id': target}, reply_id='c' * 24)
        user = SimpleNamespace(user_id='d' * 24)
        with mock.patch('aurex.community_context.build_mention_context', return_value={}), \
             mock.patch('aurex.community_context.resolve_wall_reference', return_value={
                 'requires_reference_clarification': False, 'reason_code': 'explicit_user'}), \
             mock.patch('plar.api.post_task_comment_once', return_value={'Status': 200}) as post:
            result = agent.handle(user_text=agent.db.get_task(rid)['prompt'],
                                  session_id=sid, run_id=rid, user=user)
        self.assertEqual(result['status'], 'completed')
        self.assertIn('H₂CO₃ 是社区用户', result['answer'])
        self.assertNotIn('@Alice @H₂CO₃', result['answer'])
        self.assertEqual([options['thinking'] for _, options in fake.requests],
                         [True, False, False])
        self.assertEqual(fake.requests[-1][0][0]['content'],
                         __import__('aurex.task_reply', fromlist=['SHORT_COMMUNITY_FINAL_SYSTEM']).SHORT_COMMUNITY_FINAL_SYSTEM)
        self.assertEqual(fake.requests[-1][1]['tools'], [])
        for messages, options in fake.requests[:-1]:
            if options['tools']:
                names = {tool['function']['name'] for tool in options['tools']}
                self.assertLessEqual(names, {'plar_get_user', 'plar_query_experiments',
                                             'plar_get_comments', 'plar_get_oldest_comment',
                                             'plar_oldest_by_user', 'plar_get_relations',
                                             'plar_check_following', 'plar_list_builtin_tags',
                                             'read_context'})
        self.assertEqual(
            [{tool['function']['name'] for tool in options['tools']}
             for _, options in fake.requests[:-1]],
            [{'plar_get_user', 'plar_query_experiments'},
             {'plar_query_experiments'}])
        post.assert_called_once()
        self.assertTrue(any(event['kind'] == 'short_community_finalized'
                            for event in agent.db.events(sid)))
        self.assertFalse(any(event['kind'] == 'task_continues'
                             for event in agent.db.events(sid)))

    def test_admin_dry_run_exercises_short_lookup_and_finishes_locally(self):
        tools = ToolRegistry()
        tools.register(ToolSpec('plar_get_user', 'Get user', {'type': 'object'},
            lambda *_: {'id': 'e' * 24, 'nickname': 'MapMaths',
                        'signature': 'Trismegistus',
                        'stats': {'experiment_count': 197}}))
        tools.register(ToolSpec('plar_query_experiments', 'Query works', {'type': 'object'},
            lambda *_: [{'subject': '串行ADC'}]))
        get_user = {'id': 'profile', 'type': 'function', 'function': {
            'name': 'plar_get_user', 'arguments': '{"user_id":"' + 'e' * 24 + '"}'}}
        agent, fake = self.agent([
            reply('', calls=[get_user], finish='tool_calls'),
            reply('', calls=[{'id': 'works', 'type': 'function', 'function': {
                'name': 'plar_query_experiments', 'arguments': '{"user_id":"' + 'e' * 24 + '"}'}}],
                finish='tool_calls'),
        ], tools)
        fake.short_final_answers = iter([{'answer': 'MapMaths 的公开签名是 Trismegistus。'}])
        sid = agent.db.session('admin-short-dry-run', source='admin')
        request = '只读审计：介绍用户 <user=' + 'e' * 24 + '>@MapMaths</user>。'
        rid = agent.db.enqueue_task(sid, request, source='admin',
                                    metadata={'admin_scenario': True, 'dry_run': True})
        result = agent.handle(user_text=request, session_id=sid, run_id=rid)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['answer'], 'MapMaths 的公开签名是 Trismegistus。')
        self.assertEqual([options['thinking'] for _, options in fake.requests],
                         [True, False, False])
        self.assertTrue(any(event['kind'] == 'short_community_finalized'
                            for event in agent.db.events(sid)))

    def test_first_request_thinks_then_tool_turn_is_no_think_and_reasoning_is_not_replayed(self):
        tools = ToolRegistry()
        tools.register(ToolSpec("read_voltage", "Read fixture", {"type": "object", "properties": {}}, lambda rt, args: {"voltage": 5, "units": "V"}))
        call = {"id": "call1", "type": "function", "function": {"name": "read_voltage", "arguments": "{}"}}
        agent, fake = self.agent([reply("", reasoning="private reasoning marker", calls=[call], finish="tool_calls"), reply("Measured 5 V")], tools)
        result = agent.handle(user_text="Measure voltage")
        self.assertEqual([options["thinking"] for _, options in fake.requests], [True, False, False])
        self.assertEqual(fake.requests[0][1]["max_tokens"], 4096)
        self.assertIsNone(fake.requests[1][1]["max_tokens"])
        self.assertEqual(fake.requests[-1][1]['tools'], [])
        self.assertEqual(result["answer"], "Measured 5 V")
        self.assertNotIn("private reasoning marker", json.dumps(agent.db.messages(result["session_id"])))
        self.assertNotIn("private reasoning marker", json.dumps(fake.requests[-1][0]))
        self.assertFalse(any(message.get('role') == 'tool' or 'tool_calls' in message
                             for message in fake.requests[-1][0]))
        self.assertTrue(any(e["kind"] == "reasoning_delta" for e in agent.db.events(result["session_id"])))

    def test_reasoning_only_first_turn_recovers_without_replaying_private_reasoning(self):
        agent, fake = self.agent([
            reply("", reasoning="private loop that never handed off"),
            reply("Recovered concise answer"),
        ])
        result = agent.handle(user_text="Check this bounded question")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["answer"], "Recovered concise answer")
        self.assertEqual([options["thinking"] for _, options in fake.requests], [True, False, False])
        self.assertEqual(fake.requests[0][1]["max_tokens"], 4096)
        self.assertNotIn("private loop", json.dumps(fake.requests[1][0]))
        self.assertNotIn("private loop", json.dumps(agent.db.messages(result["session_id"])))
        events = agent.db.events(result["session_id"])
        recovery = next(event for event in events if event["kind"] == "thinking_handoff_recovery")
        self.assertGreater(recovery["data"]["reasoning_characters"], 0)

    def test_explicit_tool_images_are_attached_to_next_model_request(self):
        from PIL import Image
        path = os.path.join(self.temp.name, "circuit.png")
        Image.new("RGB", (30, 30), "white").save(path)
        tools = ToolRegistry()
        tools.register(ToolSpec("circuit_inspect", "Read schematic", {"type": "object", "properties": {"with_image": {"type": "boolean"}}}, lambda rt, args: {"images": [{"path": path}]}))
        call = {"id": "draw1", "type": "function", "function": {"name": "circuit_inspect", "arguments": '{"with_image":true}'}}
        agent, fake = self.agent([reply("", calls=[call], finish="tool_calls"), reply("Read diagram")], tools)
        result = agent.handle(user_text="Look at circuit")
        attached = [p for m in fake.requests[1][0] if isinstance(m.get("content"), list) for p in m["content"] if p.get("type") == "image_url"]
        self.assertEqual(len(attached), 1)
        self.assertTrue(attached[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertTrue(result["tool_results"][0].ok)
        self.assertTrue(any(row['message'].get('_image_requested_by') == result['task_id'] for row in agent.db.messages(result['session_id'])))

    def test_legacy_tool_returned_images_are_archived_but_never_automatically_attached(self):
        from PIL import Image
        path = os.path.join(self.temp.name, "legacy.png")
        Image.new("RGB", (10, 10), "white").save(path)
        for arguments in ('{}', '{"with_image":false}', '{"with_image":"true"}'):
            with self.subTest(arguments=arguments):
                tools = ToolRegistry()
                tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object', 'properties': {'with_image': {'type': 'boolean'}}},
                                        lambda *_: {'images': [{'path': path}], 'measurements': {'voltage': 3}}))
                call = {'id': 'legacy', 'type': 'function', 'function': {'name': 'circuit_inspect', 'arguments': arguments}}
                agent, fake = self.agent([reply(calls=[call], finish='tool_calls'), reply('Use measured values')], tools)
                result = agent.handle(user_text='Measure the voltage')
                self.assertNotIn('"type": "image_url"', json.dumps([r[0] for r in fake.requests]))
                self.assertNotIn('data:image/', json.dumps([r[0] for r in fake.requests]))

    def test_view_image_explicitly_reopens_initial_unseen_upload(self):
        from PIL import Image
        path = os.path.join(self.temp.name, "requested.png")
        Image.new("RGB", (10, 10), "white").save(path)
        call = {'id': 'look', 'type': 'function', 'function': {'name': 'view_image', 'arguments': json.dumps({'path': path})}}
        agent, fake = self.agent([reply(calls=[call], finish='tool_calls'), reply('Visible diagram checked')])
        result = agent.handle(user_text='Look at this screenshot', images=[path])
        self.assertNotIn('"type": "image_url"', json.dumps(fake.requests[0][0]))
        self.assertIn('"type": "image_url"', json.dumps(fake.requests[1][0]))
        self.assertIn('"type": "image_url"', json.dumps(fake.requests[-1][0]))
        self.assertTrue(result['tool_results'][0].ok)

    def test_truncated_response_continues_task_without_executing_partial_tools(self):
        tools = ToolRegistry()
        executed = mock.Mock(return_value={"measured": 5})
        tools.register(ToolSpec("measure", "Measure", {"type": "object"}, executed))
        incomplete = {"id": "cut", "type": "function", "function": {"name": "measure", "arguments": "{\"x\":"}}
        valid = {"id": "actual", "type": "function", "function": {"name": "measure", "arguments": "{}"}}
        agent, fake = self.agent([reply("unfinished", calls=[incomplete], finish="length"),
                                 reply("", calls=[valid], finish="tool_calls"), reply("Measured result verified")], tools)
        fake.final_reviews = iter([
            {'outcome': 'continue', 'answer': 'No measurement was executed; submit a complete focused measurement.'},
            {'outcome': 'completed', 'answer': 'Measured result verified'},
        ])
        result = agent.handle(user_text="Run long task", session_id="length-test")
        self.assertEqual(result["answer"], "Measured result verified")
        executed.assert_called_once()
        self.assertEqual([x[1]["thinking"] for x in fake.requests], [True, False, False, False, False])
        events = agent.db.events("length-test")
        self.assertTrue(any(e["kind"] == "generation_continuation" for e in events))
        self.assertFalse(any(e["kind"] == "tool_start" and e["data"]["call_id"] == "cut" for e in events))

    def test_uploaded_image_has_reopenable_reference_in_archived_message(self):
        from PIL import Image
        path = os.path.join(self.temp.name, "upload.png")
        Image.new("RGB", (10, 10), "white").save(path)
        agent, fake = self.agent([reply("Read upload")])
        result = agent.handle(user_text="Remember this diagram", images=[path])
        saved = json.dumps(agent.db.messages(result["session_id"]))
        self.assertIn(path, saved, "Archived upload needs a path that remains available after image eviction/compaction")
        self.assertNotIn('"type": "image_url"', json.dumps([r[0] for r in fake.requests]))

    def test_plain_long_web_input_is_archived_and_compacted(self):
        agent, fake = self.agent([reply("answer")])
        fake.summary_response = 'Resistor R1 is 10 ohm; all repeated original source remains archived.'
        long_text = "Resistor R1 is 10 ohm. " * 11000
        result = agent.handle(user_text=long_text)
        self.assertTrue(result["answer"])
        events = agent.db.events(result["session_id"])
        self.assertTrue(any(e["kind"] == "compaction_start" for e in events))
        self.assertEqual(agent.db.get(result["session_id"])["status"], "completed")

    def test_modern_toolset_exposes_raw_file_download_but_not_sdk_loaders(self):
        from aurex.tools import create_registry
        agent, fake = self.agent([reply("safe")], create_registry())
        fake.count = lambda *args, **kwargs: 100
        agent.handle(user_text="List safe tools")
        names = {tool["function"]["name"] for tool in fake.requests[0][1]["tools"]}
        self.assertIn("plar_get_experiment_file", names)
        self.assertIn("plar_get_summary", names)
        self.assertNotIn("plar_get_status_save", names)
        self.assertNotIn("plar_get_experiment_context", names)

    def test_unadvertised_legacy_sdk_call_is_rejected_before_execution(self):
        called = mock.Mock(side_effect=AssertionError("unsafe old loader was executed"))
        tools = ToolRegistry()
        tools.register(ToolSpec("plar_get_status_save", "legacy SDK", {"type": "object"}, called))
        call = {"id": "unsafe", "type": "function", "function": {"name": "plar_get_status_save", "arguments": "{}"}}
        agent, fake = self.agent([reply("", calls=[call], finish="tool_calls"), reply("Use safe raw file download")], tools)
        result = agent.handle(user_text="Read community experiment")
        called.assert_not_called()
        self.assertFalse(result["tool_results"][0].ok)
        self.assertIn("not enabled", result["tool_results"][0].error)

    def test_legacy_round_budget_does_not_end_v3_tasks(self):
        self.cfg = replace(self.cfg, agent=replace(self.cfg.agent, max_tool_loops=1))
        tools = ToolRegistry()
        tools.register(ToolSpec("read", "read", {"type": "object", "properties": {}}, lambda *args: {"value": 1}))
        call = {"id": "once", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        agent, fake = self.agent([reply("", calls=[call], finish="tool_calls"), reply("More investigation is needed")], tools)
        result = agent.handle(user_text="Investigate")
        self.assertEqual(agent.db.get(result["session_id"])["status"], "completed")
        self.assertTrue(fake.requests[-2][1]["tools"])
        self.assertEqual(fake.requests[-1][1]['tools'], [])
        answers = [event for event in agent.db.events(result["session_id"]) if event["kind"] == "answer"]
        self.assertFalse(answers[-1]["data"]["tool_limit_reached"])

    def test_publish_is_intercepted_by_server_review_not_direct_tool_handler(self):
        tools = ToolRegistry()
        direct = mock.Mock(side_effect=AssertionError("Direct publication bypassed review"))
        tools.register(ToolSpec("plar_publish_experiment", "Publish with review", {"type": "object"}, direct))
        call = {"id": "publish1", "type": "function", "function": {"name": "plar_publish_experiment", "arguments": "{}"}}
        agent, fake = self.agent([reply("", calls=[call], finish="tool_calls"), reply("Published receipt")], tools)
        with mock.patch("aurex.publication_review.review_and_publish", return_value={"published": True}) as reviewed:
            result = agent.handle(user_text="Publish verified test", session_id="authorized-fixture")
        direct.assert_not_called()
        reviewed.assert_called_once()
        runtime, _, client, db, sid, rid, emit = reviewed.call_args.args
        self.assertEqual(runtime.session_id, sid)
        self.assertEqual(sid, "authorized-fixture")
        self.assertEqual(runtime.task_id, rid)
        self.assertIs(client, fake)
        self.assertIs(db, agent.db)
        self.assertTrue(result["tool_results"][0].ok)

    def test_successful_tool_is_journaled_before_bad_image_presentation(self):
        tools = ToolRegistry()
        tools.register(ToolSpec("measure", "Measure", {"type": "object"},
                                lambda *_: {"voltage": 5, "images": [{"path": "/missing/image.png"}]}))
        call = {"id": "m1", "type": "function", "function": {"name": "measure", "arguments": "{}"}}
        agent, _ = self.agent([reply("", calls=[call], finish="tool_calls"), reply("Measured 5V")], tools)
        result = agent.handle(user_text="Measure")
        self.assertTrue(result["tool_results"][0].ok)
        self.assertEqual(result["tool_results"][0].data["voltage"], 5)
        saved = agent.db.get_tool_outcome(result["session_id"], result["task_id"], "m1")
        self.assertTrue(json.loads(saved["full_json"])["ok"])
        self.assertTrue(any(e["kind"] == "artifact_error" for e in agent.db.events(result["session_id"])))

    def test_task_continues_pending_publication_until_review_reports_a_real_blocker(self):
        agent, fake = self.agent([reply("All done"), reply("Cannot access publishing account")])
        fake.final_reviews = iter([
            {'outcome': 'completed', 'answer': 'All done'},
            {'outcome': 'blocked', 'answer': 'Publishing account unavailable; experiment has not been published.'},
        ])
        sid = agent.db.session('pending-publication')
        rid = agent.db.enqueue_task(sid, 'Complete and publish my experiment', source='admin',
                                    explicit_publish_requested=True)
        result = agent.handle(user_text='Complete and publish my experiment', session_id=sid, run_id=rid)
        self.assertEqual(agent.db.get(result["session_id"])["status"], "needs_attention")
        self.assertIn('not been published', result['answer'])
        self.assertTrue(any(e['kind'] == 'task_continues' for e in agent.db.events(sid)))
        self.assertEqual(len([e for e in agent.db.events(sid) if e['kind'] == 'answer']), 1)
        self.assertEqual([x[1]['thinking'] for x in fake.requests], [True, False, False, False])

    def test_admin_reply_is_local_and_idempotent_without_requester_id(self):
        agent, fake = self.agent([reply('Measured 5 V')])
        sid = agent.db.session('admin-test')
        rid = agent.db.enqueue_task(sid, 'Measure', source='admin')
        with mock.patch('plar.api.post_task_comment_once', side_effect=AssertionError('Admin must not post')):
            result = agent.handle(user_text='Measure', session_id=sid, run_id=rid)
            again = agent.handle(user_text='Measure', session_id=sid, run_id=rid)
        self.assertEqual(result['answer'], again['answer'])
        self.assertNotIn('@', result['answer'])
        self.assertEqual(len(fake.requests), 2)
        finals = [m['message'] for m in agent.db.messages(sid) if m['message'].get('_final_review_id')]
        self.assertEqual(len(finals), 1)
        self.assertIsNone(fake.on_tick)

    def test_admin_publication_selection_is_pinned_as_server_metadata(self):
        agent, fake = self.agent([reply('Design unavailable')])
        fake.final_reviews = iter([{'outcome': 'blocked', 'answer': 'Design unavailable'}])
        sid = agent.db.session('selected-publish')
        rid = agent.db.enqueue_task(sid, 'Design a circuit', source='admin',
                                    explicit_publish_requested=True)
        agent.handle(user_text='Design a circuit', session_id=sid, run_id=rid)
        text = json.dumps(fake.requests[0][0], ensure_ascii=False)
        self.assertIn('trusted_task_binding', text)
        self.assertIn('explicit_publish_requested', text)
        self.assertIn('admin', text)

    def test_compact_circuit_result_has_lossless_readable_source_documents(self):
        raw = json.dumps({'components': [{'ref': 'C' + str(i), 'value': i} for i in range(1000)]})
        path = Path(self.temp.name) / 'full-netlist.json'
        path.write_text(raw)
        registry = ToolRegistry()
        registry.register(ToolSpec('circuit_fixture', 'Inspect', {'type': 'object'},
            lambda rt, args: {'count': 1000, 'artifact': {'netlist_path': str(path)}}))
        call = {'id': 'inspect', 'type': 'function', 'function': {'name': 'circuit_fixture', 'arguments': '{}'}}
        agent, fake = self.agent([reply(calls=[call], finish='tool_calls'), reply('1000 components')], registry)
        result = agent.handle(user_text='Inspect the whole scene')
        with agent.db.connect() as connection:
            source = connection.execute("SELECT id,content FROM documents WHERE title='circuit_fixture: full netlist_path'").fetchone()
        self.assertEqual(source['content'], raw)
        messages = [row['message'] for row in agent.db.messages(result['session_id'])]
        tool_text = next(m['content'] for m in messages if m['role'] == 'tool')
        self.assertIn(source['id'], tool_text)
        self.assertIn('Full source documents', tool_text)
        self.assertNotIn('C999', tool_text)

    def test_community_final_reply_uses_bound_user_id_once_not_comment_id(self):
        from types import SimpleNamespace
        agent, fake = self.agent([reply('Measured 5 V')])
        sid = agent.db.session('community-fixture')
        uid, target, comment = 'a' * 24, 'b' * 24, 'c' * 24
        rid = agent.db.enqueue_task(sid, 'Measure', source='community', requester_user_id=uid,
                                    requester_nickname='Alice', target={'type': 'Experiment', 'id': target},
                                    reply_id=comment)
        user = SimpleNamespace(user_id='d' * 24)
        with mock.patch('plar.api.post_task_comment_once', return_value={'Status': 200}) as post:
            result = agent.handle(user_text='Measure', session_id=sid, run_id=rid, user=user)
            agent.handle(user_text='Measure', session_id=sid, run_id=rid, user=user)
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs['requester_user_id'], uid)
        self.assertTrue(result['answer'].startswith('<user=' + uid + '>@Alice</user> '))

    def test_no_stream_callback_cancellation_stops_before_final_review(self):
        agent, fake = self.agent([])
        sid = agent.db.session('cancel-silent')
        rid = agent.db.enqueue_task(sid, 'Wait for tool', source='admin')
        def silent(*args, **kwargs):
            agent.db.request_cancel(sid, rid)
            fake.on_tick()
            self.fail('Cancellation callback must stop the silent generation')
        fake.chat = silent
        result = agent.handle(user_text='Wait for tool', session_id=sid, run_id=rid)
        self.assertTrue(result['cancelled'])
        self.assertEqual(agent.db.get_task(rid)['status'], 'cancelled')
        self.assertFalse(any(e['kind'] == 'answer' for e in agent.db.events(sid)))

    def test_repeated_tools_stay_enabled_and_can_continue_after_review(self):
        tools = ToolRegistry()
        executed = mock.Mock(return_value={'voltage': 5})
        tools.register(ToolSpec('measure', 'Measure', {'type': 'object'}, executed))
        def call(cid, args='{}'):
            return {'id': cid, 'type': 'function', 'function': {'name': 'measure', 'arguments': args}}
        outputs = [reply('', calls=[call('m' + str(i))], finish='tool_calls') for i in range(3)]
        outputs += [reply('Need a second point'), reply('', calls=[call('new', '{"point":2}')], finish='tool_calls'), reply('Measured both points')]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([
            {'outcome': 'continue', 'answer': '第二个测点尚未取得，继续测量。'},
            {'outcome': 'completed', 'answer': 'Measured both points'},
        ])
        result = agent.handle(user_text='Measure both points')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(executed.call_count, 4, 'Live tool calls are not blindly memoized or skipped')
        self.assertTrue(fake.requests[3][1]['tools'])
        self.assertFalse(fake.requests[3][1]['thinking'])
        self.assertTrue(fake.requests[5][1]['tools'])
        self.assertEqual(len([e for e in agent.db.events(result['session_id']) if e['kind'] == 'answer']), 1)
        self.assertTrue(any(e['kind'] == 'loop_recovery' for e in agent.db.events(result['session_id'])))
        self.assertIn('read_context(document_id=', json.dumps(fake.requests[5][0]))

    def test_identical_live_queries_with_new_results_do_not_trigger_recovery(self):
        tools = ToolRegistry()
        executed = mock.Mock(side_effect=[{'voltage': 1}, {'voltage': 2}, {'voltage': 3}])
        tools.register(ToolSpec('measure', 'Measure', {'type': 'object'}, executed))
        calls = [{'id': 'm' + str(i), 'type': 'function', 'function': {'name': 'measure', 'arguments': '{}'}} for i in range(3)]
        agent, fake = self.agent([*[reply(calls=[c], finish='tool_calls') for c in calls], reply('Measured rising voltage')], tools)
        result = agent.handle(user_text='Read voltage changes')
        self.assertEqual(executed.call_count, 3)
        self.assertTrue(fake.requests[3][1]['tools'])
        self.assertFalse(any(e['kind'] == 'loop_recovery' for e in agent.db.events(result['session_id'])))

    def test_missing_wall_reference_keeps_thinking_but_cannot_investigate_unrelated_sources(self):
        tools = ToolRegistry()
        investigate = mock.Mock(side_effect=AssertionError('No speculative investigation'))
        tools.register(ToolSpec('investigate_wall', 'Investigate', {'type': 'object'}, investigate))
        call = {'id': 'unrequested', 'type': 'function', 'function': {'name': 'investigate_wall', 'arguments': '{}'}}
        agent, fake = self.agent([reply(calls=[call], finish='tool_calls'), reply('这是用户留言板，你指哪条内容？')], tools)
        fake.final_reviews = iter([{'outcome': 'completed', 'answer': '这是用户留言板，你指哪条内容？'}])
        resolution = {'requires_reference_clarification': True, 'reason_code': 'missing_deictic_referent'}
        with mock.patch('aurex.community_context.resolve_wall_reference', return_value=resolution):
            result = agent.handle(user_text='这是啥啊')
        investigate.assert_not_called()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([opts['thinking'] for _, opts in fake.requests], [True, False])
        self.assertTrue(all(not opts.get('tools') for _, opts in fake.requests))
        self.assertTrue(any(e['kind'] == 'tool_calls_deferred' for e in agent.db.events(result['session_id'])))

    def test_circuit_repeated_facts_ignore_only_artifact_path_noise(self):
        original = {'ok': True, 'data': {'artifact': {'png_path': '/first.png'},
            'images': [{'path': '/first.png'}], 'measurements': {'v': 3, 'component_scope': {'complete_state_path': '/first.json'}}}}
        changed = json.loads(json.dumps(original).replace('/first', '/second'))
        self.assertEqual(_progress_fingerprint('circuit_analyze', original), _progress_fingerprint('circuit_analyze', changed))
        self.assertNotEqual(_progress_fingerprint('other_tool', original), _progress_fingerprint('other_tool', changed))
        changed['data']['measurements']['v'] = 4
        self.assertNotEqual(_progress_fingerprint('circuit_analyze', original), _progress_fingerprint('circuit_analyze', changed))
        self.assertEqual(original['data']['images'][0]['path'], '/first.png')

    def test_progress_fingerprint_ignores_key_order_but_preserves_semantic_paths(self):
        first = {'ok': True, 'data': {'critical_path': ['A', 'B'], 'voltage': {'a': 1, 'b': 2}}}
        reordered = {'data': {'voltage': {'b': 2, 'a': 1}, 'critical_path': ['A', 'B']}, 'ok': True}
        self.assertEqual(_progress_fingerprint('circuit_analyze', first), _progress_fingerprint('circuit_analyze', reordered))
        reordered['data']['critical_path'] = ['B', 'A']
        self.assertNotEqual(_progress_fingerprint('circuit_analyze', first), _progress_fingerprint('circuit_analyze', reordered))

    def test_fresh_circuit_paths_do_not_hide_repeated_simulation_from_review(self):
        tools = ToolRegistry()
        execute = mock.Mock(side_effect=[{'state_path': '/unused/' + str(i), 'measurements': {'v': 3}} for i in range(3)])
        tools.register(ToolSpec('circuit_fixture', 'Simulate', {'type': 'object'}, execute))
        calls = [{'id': 'r' + str(i), 'type': 'function', 'function': {
            'name': 'circuit_fixture', 'arguments': json.dumps({'a': 1, 'b': 2} if i % 2 else {'b': 2, 'a': 1})}} for i in range(3)]
        agent, fake = self.agent([*[reply(calls=[c], finish='tool_calls') for c in calls], reply('Measured 3 V')], tools)
        result = agent.handle(user_text='Measure voltage')
        self.assertEqual(execute.call_count, 3)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(fake.requests[3][1]['tools'])
        self.assertTrue(any(e['kind'] == 'loop_recovery' for e in agent.db.events(result['session_id'])))

    def test_circuit_parameter_cycle_requests_review_without_skipping_or_finishing(self):
        tools = ToolRegistry()
        execute = mock.Mock(side_effect=[{'state_path': '/result/' + str(i), 'measurements': {'v': 3 if i % 2 == 0 else 4}} for i in range(5)])
        tools.register(ToolSpec('circuit_analyze', 'Simulate', {'type': 'object'}, execute))
        calls = [{'id': 'cycle' + str(i), 'type': 'function', 'function': {
            'name': 'circuit_analyze', 'arguments': json.dumps({'path': '/revision/' + str(i), 'analysis': 'dc'})}} for i in range(5)]
        agent, fake = self.agent([*[reply(calls=[c], finish='tool_calls') for c in calls], reply('Compare recorded candidates')], tools)
        result = agent.handle(user_text='Verify the design')
        self.assertEqual(execute.call_count, 5)
        self.assertTrue(fake.requests[4][1]['tools'])
        self.assertTrue(fake.requests[5][1]['tools'])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len([e for e in agent.db.events(result['session_id']) if e['kind'] == 'answer']), 1)

    def test_archived_json_page_is_not_double_escaped_for_the_model(self):
        agent, fake = self.agent([])
        sid = agent.db.session('readable-context')
        source = '{"R1":{"resistance_ohm":10},"label":"测点"}'
        did = agent.db.document(sid, 'Original source', source)
        call = {'id': 'page1', 'type': 'function', 'function': {
            'name': 'read_context', 'arguments': json.dumps({'document_id': did})}}
        fake.replies = iter([reply('', calls=[call], finish='tool_calls'), reply('R1 is 10 ohm')])
        result = agent.handle(user_text='Read archived resistance', session_id=sid)
        page = next(m['content'] for m in fake.requests[1][0] if m['role'] == 'tool')
        self.assertIn(source, page)
        self.assertIn('next_offset=' + str(len(source)), page)
        saved = agent.db.get_tool_outcome(sid, result['task_id'], 'page1')
        self.assertEqual(json.loads(saved['full_json'])['data']['text'], source)

    def test_same_solver_failure_with_different_parameters_requests_review_then_can_continue(self):
        tools = ToolRegistry()
        execute = mock.Mock(side_effect=[RuntimeError('Transient trace failed (rc=3, completed_steps=0, time_s=0.0)')
                                        for _ in range(3)] + [{'measurements': {'v': 3}}])
        tools.register(ToolSpec('circuit_analyze', 'Simulate', {'type': 'object'}, execute))
        def call(i):
            return {'id': 'failure' + str(i), 'type': 'function', 'function': {
                'name': 'circuit_analyze', 'arguments': json.dumps({'path': '/revision/' + str(i), 'dt': 0.01 / (i + 1)})}}
        agent, fake = self.agent([*[reply(calls=[call(i)], finish='tool_calls') for i in range(3)],
                                 reply('Solver failed; inspect the circuit constraints'),
                                 reply(calls=[call(3)], finish='tool_calls'), reply('Measured 3 V')], tools)
        fake.final_reviews = iter([{'outcome': 'continue', 'answer': '存在可验证的修正，继续同一任务。'},
                                   {'outcome': 'completed', 'answer': 'Measured 3 V'}])
        result = agent.handle(user_text='Verify the circuit')
        self.assertEqual(execute.call_count, 4)
        self.assertTrue(fake.requests[3][1]['tools'])
        self.assertFalse(fake.requests[3][1]['thinking'])
        self.assertTrue(fake.requests[5][1]['tools'])
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len([e for e in agent.db.events(result['session_id']) if e['kind'] == 'answer']), 1)

    def test_successful_analysis_breaks_same_failure_streak(self):
        tools = ToolRegistry()
        execute = mock.Mock(side_effect=[RuntimeError('No convergence'), RuntimeError('No convergence'),
                                        {'measurements': {'v': 3}}, RuntimeError('No convergence')])
        tools.register(ToolSpec('circuit_analyze', 'Simulate', {'type': 'object'}, execute))
        calls = [{'id': 'streak' + str(i), 'type': 'function', 'function': {
            'name': 'circuit_analyze', 'arguments': json.dumps({'revision': i})}} for i in range(4)]
        agent, fake = self.agent([*[reply(calls=[c], finish='tool_calls') for c in calls], reply('One verified case')], tools)
        result = agent.handle(user_text='Try these cases')
        self.assertTrue(fake.requests[4][1]['tools'])
        self.assertFalse(any(e['kind'] == 'loop_recovery' for e in agent.db.events(result['session_id'])))


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = SessionDB(os.path.join(self.temp.name, "sessions.sqlite3"))

    def test_queued_uploaded_image_paths_survive_database_reopen(self):
        sid = self.db.session()
        path = os.path.join(self.temp.name, "input.png")
        rid = self.db.begin(sid, "Inspect upload", images=[path])
        reopened = SessionDB(self.db.path)
        queued = reopened.recover()
        self.assertEqual(queued[0]["id"], rid)
        self.assertEqual(queued[0]["images"], [path])

    def test_restart_records_failure_for_unfinished_tool_calls(self):
        sid = self.db.session()
        rid = self.db.begin(sid, "Simulate")
        self.db.run_status(rid, "running")
        self.db.message(sid, rid, {"role": "assistant", "content": None, "tool_calls": [
            {"id": "tool-finished", "type": "function", "function": {"name": "read", "arguments": "{}"}},
            {"id": "tool-interrupted", "type": "function", "function": {"name": "simulate", "arguments": "{}"}},
        ]})
        self.db.message(sid, rid, {"role": "tool", "tool_call_id": "tool-finished", "content": "measured 5V"})
        self.db.recover()
        messages = [x["message"] for x in self.db.messages(sid)]
        results = [x for x in messages if x["role"] == "tool"]
        self.assertEqual({x["tool_call_id"] for x in results}, {"tool-finished", "tool-interrupted"})
        failure = next(x for x in results if x["tool_call_id"] == "tool-interrupted")
        self.assertIn("interrupt", failure["content"].lower())
        self.assertEqual(self.db.get(sid)["status"], "interrupted")
        self.db.recover()
        self.assertEqual(len(self.db.messages(sid)), len(messages), "Recovery must not duplicate tool results")

    def test_document_access_is_session_scoped_and_paginated(self):
        sid = self.db.session()
        other = self.db.session()
        doc = self.db.document(sid, "evidence", "ABCDEFGHIJK")
        part = self.db.read_document(sid, doc, offset=2, length=3)
        self.assertEqual(part["text"], "CDE")
        self.assertTrue(part["has_more"])
        with self.assertRaises(ValueError):
            self.db.read_document(other, doc)

    def test_large_tool_source_remains_complete_after_model_summary(self):
        sid = self.db.session()
        config = LLMConfig(context_length=32768, max_output_tokens=512)
        fake = FakeLLM(config, [reply("A concise evidence summary") for _ in range(20)])
        events = []
        budget = ContextBudget(fake, self.db, sid, "run", 32768, lambda kind, data: events.append((kind, data)))
        source = '{"raw_original_state":"' + "R1:10ohm; " * 13000 + 'TAIL_MUST_SURVIVE"}'
        compacted = budget.document("Large original state", source)
        original_id = next(data["document_id"] for kind, data in events if kind == "compaction_start")
        tail = self.db.read_document(sid, original_id, offset=len(source) - 30, length=30)
        self.assertEqual(tail["text"], source[-30:])
        self.assertEqual(tail["total_chars"], len(source))
        self.assertIn(original_id, compacted)
        self.assertIn("read_context", compacted)


class TransportTests(unittest.TestCase):
    def response(self, packets):
        return iter([b"data: " + json.dumps(x).encode() for x in packets] + [b"data: [DONE]"])

    def test_stream_separates_reasoning_and_assembles_tool_arguments(self):
        config = LLMConfig()
        client = VLLMClient(config)
        response = self.response([
            {"choices": [{"delta": {"reasoning_content": "analysis", "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": "{\"x\":"}}]}}]},
            {"choices": [{"delta": {"content": "answer", "tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"}]},
            {"usage": {"completion_tokens": 3}, "choices": []},
        ])
        with mock.patch.object(client, "_stream_lines", return_value=response) as post:
            result = client.chat([{"role": "user", "content": "question"}], thinking=True)
        self.assertEqual(result.content, "answer")
        self.assertEqual(result.reasoning, "analysis")
        self.assertEqual(result.tool_calls[0]["function"]["arguments"], '{"x":1}')
        self.assertIs(post.call_args.args[0]["chat_template_kwargs"]["enable_thinking"], True)

    def test_missing_finish_reason_is_not_success(self):
        client = VLLMClient(LLMConfig())
        with mock.patch.object(client, "_stream_lines", return_value=self.response([{"choices": [{"delta": {"content": "unfinished"}}]}])), self.assertRaises(ModelError):
            client.chat([{"role": "user", "content": "question"}], thinking=False)

    def test_default_does_not_send_a_token_budget_or_null_reasoning_effort(self):
        client = VLLMClient(LLMConfig())
        response = self.response([{"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]}])
        with mock.patch.object(client, "_stream_lines", return_value=response) as post:
            client.chat([{"role": "user", "content": "question"}], thinking=True)
        body = post.call_args.args[0]
        self.assertNotIn("max_tokens", body)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})

    def test_filtered_or_aborted_stream_cannot_be_executed_as_a_tool(self):
        for finish in ("content_filter", "abort", "unknown"):
            client = VLLMClient(LLMConfig())
            response = self.response([{"choices": [{"delta": {"content": "done", "tool_calls": [
                {"index": 0, "id": "x", "function": {"name": "publish", "arguments": "{}"}}]}, "finish_reason": finish}]}])
            with self.subTest(finish=finish), mock.patch.object(client, "_stream_lines", return_value=response), self.assertRaises(ModelError):
                client.chat([{"role": "user", "content": "question"}])

    def test_complete_stream_with_duplicate_call_ids_is_rejected(self):
        client = VLLMClient(LLMConfig())
        response = self.response([{"choices": [{"delta": {"tool_calls": [
            {"index": i, "id": "duplicate", "function": {"name": "read", "arguments": "{}"}} for i in range(2)]},
            "finish_reason": "tool_calls"}]}])
        with mock.patch.object(client, "_stream_lines", return_value=response), self.assertRaises(ModelError):
            client.chat([{"role": "user", "content": "question"}])


@unittest.skipUnless(shutil.which("node"), "Node.js is needed for tracking UI concurrency tests")
class TrackingUITests(unittest.TestCase):
    def run_ui(self, scenario):
        html = (Path(__file__).resolve().parents[1] / "src/aurex/tracking.html").read_text()
        source = re.search(r"<script>(.*?)</script>", html, flags=re.S).group(1)
        harness = r'''
const fs = require('node:fs'), vm = require('node:vm');
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const context = vm.createContext({URLSearchParams, console, setTimeout});
vm.runInContext(`
class Element {
 constructor(){this.children=[];this.style={};this.textContent='';this.value='';this.files=[];this.scrollHeight=10;this.scrollTop=0;this.clientHeight=10;}
 append(...children){this.children.push(...children)}
 replaceChildren(...children){this.children=children}
}
globalThis.elements=new Map();
globalThis.document={getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id)},createElement(){return new Element()}};
globalThis.location={search:'',hash:'',pathname:'/'};
globalThis.history={replaceState(){}};
globalThis.setInterval=()=>0;
globalThis.response=value=>({ok:true,status:200,json:async()=>value});
globalThis.fetchCalls=[];globalThis.fetchHook=null;
globalThis.fetch=async(path,options={})=>{fetchCalls.push({path,options});return fetchHook?fetchHook(path,options):response([])};
globalThis.readers=[];
globalThis.FileReader=class {readAsDataURL(file){readers.push(this)}};
globalThis.assert=(condition,message)=>{if(!condition)throw Error(message)};
`, context);
vm.runInContext(payload.source, context);
(async()=>{
 await new Promise(resolve=>setImmediate(resolve));
 await vm.runInContext('(async()=>{'+payload.scenario+'})()',context);
})().catch(error=>{console.error(error.stack);process.exitCode=1});
'''
        result = subprocess.run([shutil.which("node"), "-e", harness],
                                input=json.dumps({"source": source, "scenario": scenario}),
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_late_events_do_not_cross_sessions_and_new_session_refresh_is_not_blocked(self):
        self.run_ui(r'''
sid='A';let resolveOld;
fetchHook=(path)=>path.includes('/A/events')?new Promise(resolve=>resolveOld=resolve):response([]);
const old=events();await select('B');
assert(fetchCalls.some(x=>x.path.includes('/B/events')),'Switch must refresh B while A is in flight');
resolveOld(response([{id:500,run_id:'old',kind:'user',data:{text:'WRONG_SESSION'},created:1}]));await old;
assert(!JSON.stringify($('timeline')).includes('WRONG_SESSION'),'A events contaminated B');
assert(cursor===0,'A cursor contaminated B');
''')

    def test_reselect_same_session_rejects_old_generation(self):
        self.run_ui(r'''
sid='A';let resolveOld,requests=0;
fetchHook=path=>{if(!path.includes('/events'))return response([]);requests++;return requests===1?new Promise(resolve=>resolveOld=resolve):response([{id:10,run_id:'fresh',kind:'user',data:{text:'FRESH'},created:1}])};
const old=events();await select('A');
resolveOld(response([{id:500,run_id:'old',kind:'user',data:{text:'OBSOLETE'},created:1}]));await old;
assert(cursor===10,'Reselect accepted an obsolete cursor');
assert(!JSON.stringify($('timeline')).includes('OBSOLETE'),'Reselect rendered obsolete events');
''')

    def test_image_upload_creates_fresh_request_without_overwriting_selected_session(self):
        self.run_ui(r'''
sid='A';$('prompt').value='Question for A';$('file').files=[{name:'a.png'}];
fetchHook=()=>response([]);
const sending=$('send').onclick();
assert(readers.length===1,'Expected pending image read');
await select('B');$('prompt').value='Draft for B';
readers[0].result='data:image/png;base64,dGVzdA==';readers[0].onload();await sending;
const submitted=fetchCalls.filter(x=>x.path==='/api/requests');
assert(submitted.length===1,'Upload must go to the new-request endpoint');
const payload=JSON.parse(submitted[0].options.body);
assert(payload.text==='Question for A'&&!('session_id' in payload),'Upload inherited another conversation');
assert(sid==='B','Late submission moved the user away from selected B');
assert($('prompt').value==='Draft for B','A completion erased the new B draft');
''')

    def test_old_session_list_does_not_overwrite_newer_response(self):
        self.run_ui(r'''
let resolveOld,count=0;
fetchHook=()=>++count===1?new Promise(resolve=>resolveOld=resolve):response([{id:'B',title:'LATEST_TITLE',status:'idle',updated:1}]);
const old=list();await list();
resolveOld(response([{id:'A',title:'STALE_TITLE',status:'idle',updated:1}]));await old;
assert(JSON.stringify($('sessions')).includes('LATEST_TITLE'),'Latest sidebar response was lost');
assert(!JSON.stringify($('sessions')).includes('STALE_TITLE'),'Older sidebar response overwrote latest');
''')

    def test_late_new_session_does_not_undo_a_subsequent_user_selection(self):
        self.run_ui(r'''
sid='A';let finishCreate;
fetchHook=(path,options)=>options.method==='POST'?new Promise(resolve=>finishCreate=resolve):response([]);
const creating=$('new').onclick();await select('B');finishCreate(response({id:'NEW'}));await creating;
assert(sid==='B','Delayed new session stole navigation from the subsequent B selection');
''')

    def test_successful_login_clears_previous_auth_errors(self):
        self.run_ui(r'''
$('error').textContent='需要访问码';$('login-error').textContent='Incorrect access token';
$('login').style.display='flex';$('token').value='stale input';fetchHook=()=>response([]);
await login('test-token');
assert($('error').textContent===''&&$('login-error').textContent==='','Successful login left stale authentication errors visible');
assert($('login').style.display==='none'&&$('token').value==='','Login overlay did not close and clear input');
''')

    def test_needs_attention_has_chinese_status_and_explicit_progress_warning(self):
        self.run_ui(r'''
sid='A';fetchHook=()=>response([{id:'A',title:'Circuit',status:'needs_attention',updated:1}]);await list();
assert($('status').textContent==='待继续处理','Needs-attention status is not clearly translated');
assert(JSON.stringify($('sessions')).includes('待继续处理'),'Sidebar omits the pending-work status');
add({kind:'answer',run_id:'r',created:1,data:{text:'Partial result',tool_limit_reached:true}});
assert(JSON.stringify($('timeline')).includes('仍需继续处理'),'Tool-limit answer looks like completed work');
''')


if __name__ == "__main__":
    unittest.main()
