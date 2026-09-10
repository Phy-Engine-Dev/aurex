"""Whole-agent recovery tests: no network, no arbitrary truncation of content."""
import json
import unittest
from dataclasses import replace
from unittest import mock

import test_aurex_v3 as support
from aurex.tools.registry import ToolRegistry, ToolSpec
from aurex.vllm_client import DegenerateGeneration, VLLMClient
from aurex.config import LLMConfig


def call(cid, name='probe', **args):
    return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}


class GenerationRecoveryTests(unittest.TestCase):
    setUp = support.SessionAgentTests.setUp
    agent = support.SessionAgentTests.agent

    def test_length_empty_and_repetition_share_finite_recovery(self):
        repetitive = DegenerateGeneration('repetitive', support.reply('unfinished'), {'reason': 'same_token_run'})
        agent, fake = self.agent([support.reply('partial', finish='length'), support.reply(''), repetitive])
        result = agent.handle(user_text='Evaluate the evidence')
        self.assertEqual(result['status'], 'needs_attention')
        self.assertIn('INCONCLUSIVE', result['answer'])
        self.assertEqual(len(fake.requests), 3)
        self.assertEqual(sum(e['kind'] == 'answer' for e in agent.db.events(result['session_id'])), 1)
        self.assertNotIn('partial', result['answer'])

    def test_new_tool_evidence_resets_recovery_but_duplicate_receipt_does_not(self):
        tools = ToolRegistry()
        probe = mock.Mock(return_value={'value': 17})
        tools.register(ToolSpec('probe', 'Read evidence', {'type': 'object'}, probe))
        outputs = [support.reply('', finish='length'), support.reply('', finish='length'),
            support.reply('', calls=[call('a')], finish='tool_calls'),
            support.reply('', finish='length'), support.reply('', finish='length'),
            support.reply('', calls=[call('b')], finish='tool_calls'), support.reply('', finish='length')]
        agent, fake = self.agent(outputs, tools)
        result = agent.handle(user_text='Measure a value')
        self.assertEqual(len(fake.requests), 7)
        self.assertEqual(probe.call_count, 2, 'Repeated tools remain allowed')
        self.assertIn('INCONCLUSIVE', result['answer'])
        self.assertIn('17', result['answer'])

    def test_truncated_tool_calls_are_never_executed(self):
        tools = ToolRegistry()
        probe = mock.Mock(return_value={})
        tools.register(ToolSpec('probe', 'Read', {'type': 'object'}, probe))
        partial = call('broken')
        partial['function']['arguments'] = '{"incomplete":'
        agent, fake = self.agent([support.reply('', calls=[partial], finish='length')] * 3, tools)
        result = agent.handle(user_text='Read')
        probe.assert_not_called()
        self.assertIn('INCONCLUSIVE', result['answer'])

    def test_navigation_timeout_never_executes_partial_call_and_uses_finite_recovery(self):
        tools = ToolRegistry()
        seed = mock.Mock(return_value={'value': 17})
        partial_target = mock.Mock(return_value={'must_not': 'run'})
        tools.register(ToolSpec('seed', 'Read evidence', {'type': 'object'}, seed))
        tools.register(ToolSpec('probe', 'Read more', {'type': 'object'}, partial_target))
        partial = support.reply('', calls=[call('PRIVATE_CALL', secret='PRIVATE_ARGUMENT')],
                                finish='generation_timeout')
        timeout = DegenerateGeneration(
            'bounded response timeout', partial,
            {'reason': 'generation_wall_timeout', 'tool_name': 'probe',
             'tool_argument_characters': 29, 'elapsed_seconds': 60.25})
        agent, fake = self.agent([
            support.reply('', calls=[call('seed-call', 'seed')], finish='tool_calls'),
            timeout, timeout, timeout,
        ], tools)
        result = agent.handle(user_text='Read and evaluate')
        self.assertEqual(result['status'], 'needs_attention')
        self.assertEqual(seed.call_count, 1)
        partial_target.assert_not_called()
        guards = [e['data'] for e in agent.db.events(result['session_id'])
                  if e['kind'] == 'generation_navigation_guard']
        self.assertEqual(len(guards), 3)
        for guard in guards:
            self.assertFalse(guard['executed'])
            self.assertEqual(guard['tool_name'], 'probe')
            self.assertEqual(guard['tool_argument_characters'], 29)
            self.assertNotIn('PRIVATE_ARGUMENT', json.dumps(guard))
            self.assertNotIn('PRIVATE_CALL', json.dumps(guard))
        self.assertEqual([kw['generation_timeout_sec'] for _, kw in fake.requests],
                         [None, 60, 60, 60])

    def test_first_context_thinking_omits_unconfigured_output_limit(self):
        agent, fake = self.agent([support.reply('done')])
        config = replace(agent.cfg.llm, max_output_tokens=None, context_length=90112)
        agent.cfg = replace(agent.cfg, llm=config)
        fake.config = config
        agent.handle(user_text='Answer')
        self.assertIsNone(fake.requests[0][1]['max_tokens'])
        self.assertIsNone(fake.requests[0][1]['generation_timeout_sec'])
        system_text = '\n'.join(str(message.get('content', ''))
                                for message in fake.requests[0][0]
                                if message.get('role') == 'system')
        self.assertIn('SERVER_CURRENT_TURN_HANDOFF', system_text)
        self.assertIn('不得比较多个答案草稿', system_text)
        self.assertIn('一个有限反例已经足以回答', system_text)
        self.assertIn('必须区分通用原理与当前平台实证', system_text)
        self.assertIn('不得写成所有数字网络必然一次传播', system_text)
        self.assertIn('没有当前平台基准时，不给“快几个数量级”等性能数字', system_text)
        self.assertIn('不把正文描述升级为“实际电路已忠实实现”', system_text)
        self.assertIn('正式答案不得保留“等等/让我重算/现在给结论”等草稿式自我纠错', system_text)
        self.assertIn('数字传播也可能迭代或不稳定', system_text)
        self.assertIn('一次成功DC/TR和目标测量足以回答就直接结束', system_text)
        self.assertIn('unambiguous=false必须原样保留并列带', system_text)
        self.assertIn('禁止给出严格全序', system_text)
        self.assertIn('必须逐行重排刺激位和对应输出', system_text)
        self.assertIn('连续批次拼成全元件目录', system_text)
        self.assertIn('已有成功求解后不再补扫结构', system_text)
        self.assertIn('诊断阻止执行本身不能区分', system_text)
        self.assertIn('验证是否符合介绍”不等于验证所有子电路', system_text)
        self.assertIn('clk、d、in等输入脚是负载', system_text)
        self.assertIn('connections_truncated或external_connections非零', system_text)
        self.assertIn('first_stable_sampled_window/settled_for_remainder', system_text)
        self.assertIn('不能概括成“全部上升”', system_text)
        self.assertIn('不写已证明周期振荡', system_text)
        self.assertIn('不为逐项关闭计划而连续调用task_plan', system_text)
        self.assertIn('不把“整理结论”设成必须调用工具关闭的步骤', system_text)
        self.assertIn('必填path必须逐字取自最近成功回执', system_text)
        self.assertIn('漏参失败后只修正一次', system_text)
        self.assertIn('末尾免责声明不能抵消前文的绝对断言', system_text)
        self.assertIn('source_ref是原PLSAV编号', system_text)
        self.assertIn('ref是当前导入视图编号', system_text)
        self.assertIn('trace只有id时只能按同一id回接旧映射', system_text)
        self.assertIn('按interaction_events切分操作前、按下和松开后', system_text)
        self.assertIn('节点电压不能冒充元件状态', system_text)
        self.assertIn('继电器动作只据对应id的model_state.engaged', system_text)
        self.assertIn('first_stable_sampled_window只证明所报窗口', system_text)
        self.assertIn('每个被问信号各取操作前、操作中', system_text)
        self.assertIn('禁止写“全程不变”', system_text)
        self.assertIn('task_plan的title/note只是导航草稿而不是测量证据', system_text)
        self.assertIn('与后续trace冲突时必须丢弃旧note', system_text)
        self.assertIn('记录间隔不得等于或大于已知周期', system_text)
        self.assertIn('只能写“记录点未观察到”或未确认', system_text)
        self.assertIn('pin_voltage_v的label=out那一行', system_text)
        self.assertIn('NC/COM/NO是接点而不是线圈', system_text)
        self.assertIn('VCVS=V/V、VCCS=A/V、CCVS=V/A、CCCS=A/A', system_text)
        self.assertIn('不能用端电压冒充电流', system_text)
        self.assertIn('query_coverage.complete=true后，空间查询即已完成', system_text)
        self.assertIn('projection_truncated只是展示裁剪', system_text)
        self.assertIn('成功DC/TR后不返回重查空间或正文', system_text)

    def test_first_context_thinking_respects_explicit_output_limit(self):
        agent, fake = self.agent([support.reply('done')])
        config = replace(agent.cfg.llm, max_output_tokens=12000,
                         context_length=90112)
        agent.cfg = replace(agent.cfg, llm=config)
        fake.config = config
        agent.handle(user_text='Answer')
        self.assertEqual(fake.requests[0][1]['max_tokens'], 12000)

    def test_long_valid_text_is_not_semantically_truncated(self):
        text = '\n'.join('Detailed measured result ' + str(i) for i in range(2500))
        agent, fake = self.agent([support.reply(text)])
        result = agent.handle(user_text='Explain all measured results')
        self.assertEqual(result['answer'], text)

    def test_recovery_window_is_restored_after_fresh_evidence(self):
        tools = ToolRegistry()
        tools.register(ToolSpec('probe', 'Read evidence', {'type': 'object'}, lambda *_: {'value': 17}))
        agent, fake = self.agent([support.reply('', finish='length'),
            support.reply('', calls=[call('fresh')], finish='tool_calls'), support.reply('value=17')], tools)
        config = replace(agent.cfg.llm, max_output_tokens=None, context_length=90112)
        agent.cfg = replace(agent.cfg, llm=config)
        fake.config = config
        result = agent.handle(user_text='Read the value')
        self.assertEqual(result['status'], 'completed')
        # The first full-context turn remains generous.  Every later
        # no-thinking tool-capable handoff is a bounded navigation response,
        # including recovery and the final concise answer.
        self.assertEqual([kw['max_tokens'] for _, kw in fake.requests], [None, 2048, 2048])
        self.assertEqual([kw['generation_timeout_sec'] for _, kw in fake.requests],
                         [None, 60, 60])

    def test_post_first_navigation_mode_is_bounded_and_model_start_is_auditable(self):
        tools = ToolRegistry()
        tools.register(ToolSpec('probe', 'Read evidence', {'type': 'object'},
                                lambda *_: {'value': 17}))
        agent, fake = self.agent([
            support.reply('', calls=[call('read')], finish='tool_calls'),
            support.reply('value=17'),
        ], tools)
        config = replace(agent.cfg.llm, max_output_tokens=None, context_length=90112)
        agent.cfg = replace(agent.cfg, llm=config)
        fake.config = config
        result = agent.handle(user_text='Read one value')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([kw['max_tokens'] for _, kw in fake.requests], [None, 2048])
        self.assertEqual([kw['generation_timeout_sec'] for _, kw in fake.requests],
                         [None, 60])
        starts = [e['data'] for e in agent.db.events(result['session_id'])
                  if e['kind'] == 'model_start']
        self.assertEqual(starts, [
            {'step': 0, 'thinking': True, 'requested_max_tokens': None,
             'generation_wall_timeout_sec': None,
             'response_mode': 'first_context_thinking'},
            {'step': 1, 'thinking': False, 'requested_max_tokens': 2048,
             'generation_wall_timeout_sec': 60,
             'response_mode': 'bounded_tool_navigation'},
        ])

    def test_text_only_navigation_truncation_gets_one_unbounded_answer_only_turn(self):
        tools = ToolRegistry()
        tools.register(ToolSpec('probe', 'Read evidence', {'type': 'object'},
                                lambda *_: {'value': 17}))
        agent, fake = self.agent([
            support.reply('', calls=[call('read')], finish='tool_calls'),
            support.reply('partial answer must not be delivered', finish='length'),
            support.reply('complete final answer with value 17'),
        ], tools)
        config = replace(agent.cfg.llm, max_output_tokens=None,
                         context_length=90112)
        agent.cfg = replace(agent.cfg, llm=config)
        fake.config = config
        result = agent.handle(user_text='Read and explain')
        self.assertEqual(result['answer'], 'complete final answer with value 17')
        self.assertEqual([kw['max_tokens'] for _, kw in fake.requests],
                         [None, 2048, None])
        self.assertEqual([kw['generation_timeout_sec'] for _, kw in fake.requests],
                         [None, 60, None])
        self.assertTrue(fake.requests[0][1]['tools'])
        self.assertTrue(fake.requests[1][1]['tools'])
        self.assertEqual(fake.requests[2][1]['tools'], [])
        self.assertEqual(sum(e['kind'] == 'answer' for e in
                             agent.db.events(result['session_id'])), 1)
        self.assertNotIn('partial answer', json.dumps(
            agent.db.messages(result['session_id'], run_id=result['task_id'])))

    def test_text_only_navigation_timeout_gets_answer_only_turn(self):
        tools = ToolRegistry()
        tools.register(ToolSpec('probe', 'Read evidence', {'type': 'object'},
                                lambda *_: {'value': 17}))
        timeout = DegenerateGeneration(
            'bounded response timeout',
            support.reply('partial final text', finish='generation_timeout'),
            {'reason': 'generation_wall_timeout', 'tool_name': '',
             'tool_argument_characters': 0, 'elapsed_seconds': 60.1,
             'request_elapsed_seconds': 75.2})
        agent, fake = self.agent([
            support.reply('', calls=[call('read')], finish='tool_calls'),
            timeout,
            support.reply('complete answer'),
        ], tools)
        result = agent.handle(user_text='Read and explain')
        self.assertEqual(result['answer'], 'complete answer')
        self.assertEqual(fake.requests[2][1]['tools'], [])
        self.assertIsNone(fake.requests[2][1]['generation_timeout_sec'])
        self.assertEqual(len([e for e in agent.db.events(result['session_id'])
                              if e['kind'] == 'generation_final_answer_handoff']), 1)

    def test_ui_notice_is_deduplicated_and_not_injected(self):
        agent, fake = self.agent([support.reply('done')])
        original = fake.chat
        def chat(messages, **options):
            with mock.patch('aurex.session_agent.time.monotonic', return_value=10**12):
                options['on_tick']()
                options['on_tick']()
            return original(messages, **options)
        fake.chat = chat
        # Disable only the fixture deadline; this test advances the clock.
        agent.cfg = replace(agent.cfg, agent=replace(agent.cfg.agent, task_timeout_sec=0))
        result = agent.handle(user_text='Answer')
        notices = [e for e in agent.db.events(result['session_id'])
                   if e['kind'] == 'generation_notice' and e['data']['reason'] == 'generating']
        self.assertEqual(len(notices), 1)
        self.assertNotIn('模型仍在生成', json.dumps(agent.db.messages(result['session_id']), ensure_ascii=False))

    def test_large_repetitive_hdl_tool_payload_is_not_a_text_loop(self):
        client = VLLMClient(LLMConfig(stream_token_progress=True))
        arguments = json.dumps({'source': 'assign q = 0;\n' * 5000})
        packets = [{'choices': [{'index': 0, 'token_ids': [1] * 2000,
            'delta': {'tool_calls': [{'index': 0, **call('hdl', 'workspace_edit', source='placeholder')}]},
            'finish_reason': 'tool_calls'}]}]
        packets[0]['choices'][0]['delta']['tool_calls'][0]['function']['arguments'] = arguments
        lines = [b'data: ' + json.dumps(packet).encode() for packet in packets] + [b'data: [DONE]']
        with mock.patch.object(client, '_stream_lines', return_value=iter(lines)):
            result = client.chat([], tools=[{'type': 'function', 'function': {'name': 'workspace_edit'}}], thinking=False)
        self.assertEqual(result.tool_calls[0]['function']['arguments'], arguments)
