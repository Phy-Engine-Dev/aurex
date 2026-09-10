"""Recovery must make progress even if a provider ignores tools=[]."""
import json
import unittest
from unittest import mock

import test_aurex_v3 as support
from aurex.session_agent import (_auto_bind_task_plan_evidence,
                                 _is_controlled_source_verification,
                                 _task_plan_prompt)
from aurex.tools.registry import ToolRegistry, ToolSpec


class ProgressReviewTests(unittest.TestCase):
    def test_task_plan_current_is_a_discardable_candidate_not_a_gate(self):
        prompt = _task_plan_prompt([{
            'id': 'locate_pipeline', 'title': '枚举全部 C/N 定位四级流水线',
            'status': 'in_progress', 'note': '', 'evidence_document_ids': [],
        }], required=False)
        payload = json.loads(prompt.split('\n', 1)[1])
        self.assertNotIn('current', payload)
        self.assertEqual(payload['current_candidate']['id'], 'locate_pipeline')
        self.assertTrue(payload['current_candidate_is_not_a_completion_gate'])
        self.assertIn('枚举大量C/N编号', payload['completion_rule'])
        self.assertIn('立即舍弃', payload['completion_rule'])

    def test_cpu_planning_first_turn_respects_configured_generation_allocation(self):
        tools = ToolRegistry()
        agent, fake = self.agent([support.reply('')], tools)
        sid = agent.db.session('cpu-planning-bound', source='admin')
        rid = agent.db.enqueue_task(
            sid, '设计并验证 RV32I CPU', source='admin')
        plan = {'id': 'plan', 'type': 'function', 'function': {
            'name': 'task_plan', 'arguments': json.dumps({'action': 'set', 'items': [
                {'id': 'rtl', 'title': '实现 RTL'}, {'id': 'verify', 'title': '代表性验证'}]})}}
        fake.replies = iter([support.reply('', calls=[plan], finish='tool_calls')])
        # Stop before the post-plan turn; the assertion concerns only the first
        # generation handoff and does not make a task budget.
        original_chat = fake.chat
        calls = 0
        def chat(messages, **options):
            nonlocal calls
            calls += 1
            if calls > 1:
                agent.db.request_cancel(sid, rid)
                fake.on_tick()
            return original_chat(messages, **options)
        fake.chat = chat
        result = agent.handle(user_text='设计并验证 RV32I CPU', session_id=sid, run_id=rid)
        self.assertTrue(result['cancelled'])
        self.assertEqual(fake.requests[0][1]['max_tokens'], 512)
        first_prompt = json.dumps(fake.requests[0][0], ensure_ascii=False)
        self.assertIn('SERVER_CPU_ACCEPTANCE_PROTOCOL', first_prompt)
        self.assertIn('rv32i_teaching_v1', first_prompt)
        self.assertIn('aurex_rv32i_teaching', first_prompt)
        self.assertIn('同一设计源hash已在固定profile中verified=true', first_prompt)
        self.assertIn('32位word数组须用addr>>2索引', first_prompt)
        self.assertIn('绝不能在同一次hdl_simulate里同时传workspace_id和files', first_prompt)
        self.assertIn("ADDI x1,x0,5 = 32'h00500093", first_prompt)
        self.assertIn('每次改变选择器后先#1', first_prompt)
        self.assertIn('不能单独证明它们分别构成取指/译码/执行/写回四级', first_prompt)
        self.assertIn('不得把作者介绍改写成“结构上已有对应网络”', first_prompt)
        first_tools = {schema['function']['name']
                       for schema in fake.requests[0][1]['tools']}
        self.assertIn('task_plan', first_tools)
        self.assertIn('spawn_subagent', first_tools)
        self.assertNotIn('read_context', first_tools)
        self.assertNotIn('read_content', first_tools)
        self.assertNotIn('web_search', first_tools)
        self.assertFalse(any(event['kind'] == 'tool_limit_reached'
                             for event in agent.db.events(sid)))

    def test_existing_cpu_verification_gets_preflight_before_navigation_protocol(self):
        agent, fake = self.agent([support.reply('当前没有独立预期，结论为 INCONCLUSIVE。')])
        result = agent.handle(user_text='测试这个现有 CPU 作品的流水线部分是否符合介绍。')
        self.assertEqual(result['status'], 'completed')
        first_prompt = json.dumps(fake.requests[0][0], ensure_ascii=False)
        self.assertIn('SERVER_EXISTING_CPU_VERIFICATION_PROTOCOL', first_prompt)
        self.assertNotIn('SERVER_CPU_ACCEPTANCE_PROTOCOL', first_prompt)
        self.assertIn('circuit_diagnose(mode=\\"preflight\\"', first_prompt)
        self.assertIn('不得先按C/N编号', first_prompt)
        self.assertIn('立即给出INCONCLUSIVE', first_prompt)

    def test_controlled_source_protocol_can_be_selected_from_existing_post_body(self):
        target = {'type': 'Experiment', 'id': 'a' * 24}
        enriched = {'original': {
            'title': '四类受控源',
            'body': '包含 VCVS、VCCS、CCVS 与 CCCS 的对照实验。',
        }}
        self.assertTrue(_is_controlled_source_verification(
            '请验证这个实验是否符合介绍。', target, enriched))
        self.assertFalse(_is_controlled_source_verification(
            '请简单介绍这个实验。', target, enriched))
        self.assertFalse(_is_controlled_source_verification(
            '请验证这个实验。', target, {'original': {'title': '普通RC电路'}}))

    def test_existing_controlled_source_verification_gets_black_box_protocol(self):
        agent, fake = self.agent([support.reply('缺少独立预期的象限为 INCONCLUSIVE。')])
        result = agent.handle(
            user_text='测试这个现有受控源实验里的 VCVS、VCCS、CCVS、CCCS 是否正确。')
        self.assertEqual(result['status'], 'completed')
        first_prompt = json.dumps(fake.requests[0][0], ensure_ascii=False)
        self.assertIn('SERVER_CONTROLLED_SOURCE_VERIFICATION_PROTOCOL', first_prompt)
        self.assertIn('本任务不调用task_plan', first_prompt)
        self.assertIn('interface_only最多一次', first_prompt)
        self.assertIn('controls_only最多一次', first_prompt)
        self.assertIn('不再调用plar_get_summary', first_prompt)
        self.assertIn('成功DC后，对state_path只调用一次circuit_query_many', first_prompt)
        self.assertIn('limit=8', first_prompt)
        self.assertIn('measurements.voltage_across_0_to_1.real', first_prompt)
        self.assertIn('measurements.derived_current_0_to_1.real', first_prompt)
        self.assertIn('edit.i', first_prompt)
        self.assertIn('edit.r', first_prompt)
        self.assertIn('禁止为了匹配k而同时倍增', first_prompt)
        self.assertIn('不得把它说成由假定1Ω负载换算', first_prompt)
        self.assertIn('不得请求spatial字段或with_image', first_prompt)
        self.assertIn('queries固定为', first_prompt)
        self.assertIn('不得调用任何更多电路工具', first_prompt)
        self.assertIn('measurements.digital只是数字引脚', first_prompt)
        self.assertIn('measurements.voltage_across_0_to_1.real或measurements.derived_current_0_to_1.real', first_prompt)
        self.assertIn('整个验证合计1到2次DC/TR', first_prompt)
        self.assertIn('不得换字段、换序、拆批', first_prompt)
        self.assertIn('第一次成功DC/TR后不再回到内部结构检索', first_prompt)
        self.assertIn('Vout/Vcontrol（V/V，无量纲）', first_prompt)
        self.assertIn('Iout/Vcontrol（A/V，即S）', first_prompt)
        self.assertIn('Vout/Icontrol（V/A，即Ω）', first_prompt)
        self.assertIn('Iout/Icontrol（A/A，无量纲）', first_prompt)
        self.assertIn('禁止按C编号枚举内部运放、电阻', first_prompt)
        self.assertIn('立即记为INCONCLUSIVE', first_prompt)

    setUp = support.SessionAgentTests.setUp
    agent = support.SessionAgentTests.agent

    def test_cpu_task_uses_durable_evidence_bound_plan_before_completion(self):
        tools = ToolRegistry()
        analyze = mock.Mock(return_value={
            'measurement_source': 'fresh solve',
            'samples': [{'input': 1, 'actual': 1}],
        })
        tools.register(ToolSpec('circuit_analyze', 'Analyze', {'type': 'object'}, analyze))

        def call(cid, name, arguments):
            return {'id': cid, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments)}}

        outputs = [
            support.reply('', calls=[call('plan-set', 'task_plan', {'action': 'set', 'items': json.dumps([
                {'id': 'sample', 'title': '对原CPU执行代表性输入并读取实际输出'}], ensure_ascii=False)})],
                finish='tool_calls'),
            support.reply('', calls=[call('measure', 'circuit_analyze', {})], finish='tool_calls'),
            support.reply('', calls=[call('plan-done', 'task_plan', {
                'action': 'update', 'id': 'sample', 'status': 'completed',
                'note': '代表性样例已取得实际输出。', 'evidence_call_ids': ['measure']})],
                finish='tool_calls'),
            support.reply('代表性样例实际输出为1；未穷举全部指令。'),
        ]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '代表性样例实际输出为1；未穷举全部指令。'}])
        result = agent.handle(user_text='测试这个CPU是否正确，做非穷举验证。')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(analyze.call_count, 1)
        first_tools = {schema['function']['name'] for schema in fake.requests[0][1]['tools']}
        self.assertTrue({'task_plan', 'circuit_analyze', 'spawn_subagent'} <= first_tools)
        self.assertNotIn('read_context', first_tools)
        self.assertIn('SERVER_TASK_PLAN_JSON', json.dumps(fake.requests[1][0], ensure_ascii=False))
        plan = agent.db.task_plan(result['session_id'], result['task_id'])
        self.assertEqual(plan[0]['status'], 'completed')
        self.assertEqual(len(plan[0]['evidence_document_ids']), 1)
        from aurex.sessiondb import SessionDB
        self.assertEqual(SessionDB(agent.db.path).task_plan(result['session_id'], result['task_id']), plan)
        from aurex.task_reply import FINAL_SYSTEM
        self.assertFalse(any(messages[0].get('content') == FINAL_SYSTEM
                             for messages, _ in fake.requests))
        self.assertEqual(sum(event['kind'] == 'answer'
                             for event in agent.db.events(result['session_id'])), 1)

    def test_completed_plan_auto_binds_fresh_successful_tool_evidence(self):
        tools = ToolRegistry()
        inspect = mock.Mock(return_value={'measurement_source': 'fresh solve', 'value': 5.0})
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, inspect))

        def call(cid, name, arguments):
            return {'id': cid, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments)}}

        outputs = [
            support.reply('', calls=[call('plan-set', 'task_plan', {'action': 'set', 'items': [
                {'id': 'inspect', 'title': '读取实际测量'}]})], finish='tool_calls'),
            support.reply('', calls=[call('fresh', 'circuit_inspect', {})], finish='tool_calls'),
            support.reply('', calls=[call('plan-done', 'task_plan', {
                'action': 'update', 'id': 'inspect', 'status': 'completed',
                'note': '测量完成。'})], finish='tool_calls'),
            support.reply('实测值为 5。'),
        ]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed', 'answer': '实测值为 5。'}])
        result = agent.handle(user_text='读取并报告实际测量。')
        outcome = agent.db.get_tool_outcome(result['session_id'], result['task_id'], 'fresh')
        plan = agent.db.task_plan(result['session_id'], result['task_id'])
        self.assertEqual(plan[0]['evidence_document_ids'], [outcome['document_id']])
        events = [event for event in agent.db.events(result['session_id'], run_id=result['task_id'])
                  if event['kind'] == 'task_plan_evidence_auto_bound']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['data']['evidence_call_ids'], ['fresh'])
        self.assertEqual(events[0]['data']['source'],
                         'successful_same_run_tools_since_last_plan_change')

    def test_blocked_plan_auto_binds_the_fresh_diagnostic_evidence(self):
        tools = ToolRegistry()
        diagnose = mock.Mock(return_value={
            'verdict': 'INCONCLUSIVE',
            'failure_class': 'invalid_netlist_or_drive_contract',
            'execution': {'started': False},
        })
        tools.register(ToolSpec('circuit_diagnose', 'Diagnose', {'type': 'object'}, diagnose))

        def call(cid, name, arguments):
            return {'id': cid, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments)}}

        outputs = [
            support.reply('', calls=[call('plan-set', 'task_plan', {'action': 'set', 'items': [
                {'id': 'simulate', 'title': '验证目标输出'}]})], finish='tool_calls'),
            support.reply('', calls=[call('target-slice', 'circuit_diagnose', {})],
                          finish='tool_calls'),
            support.reply('', calls=[call('plan-blocked', 'task_plan', {
                'action': 'update', 'id': 'simulate', 'status': 'blocked',
                'note': '目标锥存在阻断，当前诊断契约未启动求解。'})], finish='tool_calls'),
            support.reply('本轮未启动，因此无法验证。'),
        ]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed',
                                    'answer': '本轮未启动，因此无法验证。'}])
        result = agent.handle(user_text='验证目标输出。')
        outcome = agent.db.get_tool_outcome(
            result['session_id'], result['task_id'], 'target-slice')
        plan = agent.db.task_plan(result['session_id'], result['task_id'])
        self.assertEqual(plan[0]['status'], 'blocked')
        self.assertEqual(plan[0]['evidence_document_ids'], [outcome['document_id']])
        events = [event for event in agent.db.events(result['session_id'], run_id=result['task_id'])
                  if event['kind'] == 'task_plan_evidence_auto_bound']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['data']['item_id'], 'simulate')
        self.assertEqual(events[0]['data']['evidence_call_ids'], ['target-slice'])

    def test_stale_explicit_plan_evidence_is_augmented_with_fresh_diagnosis(self):
        agent, _ = self.agent([])
        sid = agent.db.session('stale-plan-evidence', source='admin')
        rid = agent.db.enqueue_task(sid, 'diagnose', source='admin')
        agent.db.set_task_plan(sid, rid, [
            {'id': 'interface', 'title': '读取接口'},
            {'id': 'preflight', 'title': '诊断目标'},
        ])
        old_doc, _ = agent.db.tool_outcome(
            sid, rid, 'interface-call', 'circuit_inspect',
            json.dumps({'ok': True, 'data': {'ports': ['C11']}}), True)
        agent.db.update_task_plan_item(
            sid, rid, 'interface', 'completed', evidence_document_ids=[old_doc])
        fresh_doc, _ = agent.db.tool_outcome(
            sid, rid, 'diagnose-call', 'circuit_diagnose',
            json.dumps({'ok': True, 'data': {'verdict': 'INCONCLUSIVE'}}), True)
        args, candidates = _auto_bind_task_plan_evidence(agent.db, sid, rid, {
            'action': 'update', 'id': 'preflight', 'status': 'completed',
            'evidence_document_ids': [old_doc],
        })
        self.assertEqual(args['evidence_document_ids'], [old_doc, fresh_doc])
        self.assertEqual([item['call_id'] for item in candidates], ['diagnose-call'])

    def test_explicit_plan_evidence_wins_and_failed_results_are_not_auto_bound(self):
        tools = ToolRegistry()
        inspect = mock.Mock(side_effect=[{'value': 1}, {'value': 2}, RuntimeError('failed solve')])
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, inspect))

        def call(cid, name, arguments):
            return {'id': cid, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments)}}

        outputs = [
            support.reply('', calls=[call('plan-set', 'task_plan', {'action': 'set', 'items': [
                {'id': 'one', 'title': '第一阶段'}, {'id': 'two', 'title': '第二阶段'}]})],
                          finish='tool_calls'),
            support.reply('', calls=[call('one-a', 'circuit_inspect', {}),
                                     call('one-b', 'circuit_inspect', {})], finish='tool_calls'),
            support.reply('', calls=[call('complete-one', 'task_plan', {
                'action': 'update', 'id': 'one', 'status': 'completed',
                'evidence_call_ids': ['one-a']})], finish='tool_calls'),
            support.reply('', calls=[call('failed-two', 'circuit_inspect', {})], finish='tool_calls'),
            support.reply('', calls=[call('complete-two', 'task_plan', {
                'action': 'update', 'id': 'two', 'status': 'completed'})], finish='tool_calls'),
            support.reply('已完成。'),
        ]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed', 'answer': '已完成。'}])
        result = agent.handle(user_text='执行两个阶段。')
        first = agent.db.get_tool_outcome(result['session_id'], result['task_id'], 'one-a')
        plan = agent.db.task_plan(result['session_id'], result['task_id'])
        self.assertEqual(plan[0]['evidence_document_ids'], [first['document_id']])
        self.assertEqual(plan[1]['evidence_document_ids'], [])
        self.assertFalse(any(event['kind'] == 'task_plan_evidence_auto_bound'
                             for event in agent.db.events(result['session_id'], run_id=result['task_id'])))

    def test_auto_bound_evidence_is_cleared_before_the_next_plan_stage(self):
        tools = ToolRegistry()
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'},
                                mock.Mock(return_value={'value': 3})))

        def call(cid, name, arguments):
            return {'id': cid, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments)}}

        outputs = [
            support.reply('', calls=[call('plan-set', 'task_plan', {'action': 'set', 'items': [
                {'id': 'one', 'title': '第一阶段'}, {'id': 'two', 'title': '第二阶段'}]})],
                          finish='tool_calls'),
            support.reply('', calls=[call('stage-one-result', 'circuit_inspect', {})],
                          finish='tool_calls'),
            support.reply('', calls=[call('complete-one', 'task_plan', {
                'action': 'update', 'id': 'one', 'status': 'completed'})], finish='tool_calls'),
            support.reply('', calls=[call('complete-two', 'task_plan', {
                'action': 'update', 'id': 'two', 'status': 'completed'})], finish='tool_calls'),
            support.reply('已完成。'),
        ]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed', 'answer': '已完成。'}])
        result = agent.handle(user_text='执行两个阶段。')
        outcome = agent.db.get_tool_outcome(
            result['session_id'], result['task_id'], 'stage-one-result')
        plan = agent.db.task_plan(result['session_id'], result['task_id'])
        self.assertEqual(plan[0]['evidence_document_ids'], [outcome['document_id']])
        self.assertEqual(plan[1]['evidence_document_ids'], [])
        events = [event for event in agent.db.events(result['session_id'], run_id=result['task_id'])
                  if event['kind'] == 'task_plan_evidence_auto_bound']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['data']['item_id'], 'one')

    def test_plan_evidence_candidates_are_bounded_to_the_exact_task(self):
        agent, _ = self.agent([])
        sid = agent.db.session('plan-evidence-run-scope', source='admin')
        first_rid = agent.db.enqueue_task(sid, 'first', source='admin')
        second_rid = agent.db.enqueue_task(sid, 'second', source='admin')
        agent.db.set_task_plan(sid, first_rid, [{'id': 'work', 'title': 'first work'}])
        agent.db.set_task_plan(sid, second_rid, [{'id': 'work', 'title': 'second work'}])
        first_doc, _ = agent.db.tool_outcome(
            sid, first_rid, 'same-readable-call', 'circuit_inspect',
            json.dumps({'ok': True, 'data': {'value': 1}}), True)
        self.assertEqual([item['document_id'] for item in
                          agent.db.task_plan_evidence_candidates(sid, first_rid)], [first_doc])
        self.assertEqual(agent.db.task_plan_evidence_candidates(sid, second_rid), [])

    def test_explicit_plan_evidence_cannot_cross_runs_in_one_session(self):
        agent, _ = self.agent([])
        sid = agent.db.session('plan-explicit-run-scope', source='admin')
        first_rid = agent.db.enqueue_task(sid, 'first', source='admin')
        second_rid = agent.db.enqueue_task(sid, 'second', source='admin')
        agent.db.set_task_plan(sid, second_rid, [{'id': 'work', 'title': 'second work'}])
        first_doc, _ = agent.db.tool_outcome(
            sid, first_rid, 'first-call', 'circuit_inspect',
            json.dumps({'ok': True, 'data': {'value': 1}}), True)
        for field in ('evidence_document_ids', 'evidence_call_ids'):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'exact task|this task'):
                agent.db.update_task_plan_item(
                    sid, second_rid, 'work', 'completed', **{field: [first_doc]})

    def test_completed_plan_history_cannot_be_replaced_or_reopened(self):
        agent, _ = self.agent([])
        sid = agent.db.session('plan-history', source='admin')
        rid = agent.db.enqueue_task(sid, '复杂验证', source='admin')
        agent.db.set_task_plan(sid, rid, [{'id': 'measure', 'title': '取得测量证据'}])
        did, _ = agent.db.tool_outcome(
            sid, rid, 'measurement-call', 'circuit_inspect', '{"voltage":5}', True)
        agent.db.update_task_plan_item(sid, rid, 'measure', 'completed',
                                       evidence_call_ids=[did])
        with self.assertRaisesRegex(ValueError, 'already exists'):
            agent.db.set_task_plan(sid, rid, [{'id': 'again', 'title': '重复'}])
        with self.assertRaisesRegex(ValueError, 'immutable'):
            agent.db.update_task_plan_item(sid, rid, 'measure', 'in_progress')

    def test_plan_completion_does_not_force_retrieval_only_for_bookkeeping(self):
        agent, _ = self.agent([])
        sid = agent.db.session('plan-navigation-only', source='admin')
        rid = agent.db.enqueue_task(sid, '复杂验证', source='admin')
        agent.db.set_task_plan(sid, rid, [{'id': 'inspect', 'title': '检查当前源码'}])
        plan = agent.db.update_task_plan_item(
            sid, rid, 'inspect', 'completed', note='真实工作已在当前轮完成。')
        self.assertEqual(plan[0]['status'], 'completed')
        self.assertEqual(plan[0]['evidence_document_ids'], [])

    def test_completed_plan_remains_navigation_and_does_not_block_recheck(self):
        tools = ToolRegistry()
        inspect = mock.Mock(side_effect=lambda _rt, args: {
            'node_query': {'node': args['query'], 'exact': True, 'match_count': 1,
                           'offset': 0, 'limit': 8, 'next_offset': None},
            'netlist': {'components': [{'id': args['query']}]}})
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, inspect))

        def call(cid, name, arguments):
            return {'id': cid, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments)}}

        outputs = [support.reply('', calls=[call('plan', 'task_plan', {'action': 'set', 'items': [
            {'id': 'map', 'title': '取得有界节点映射证据'}]})], finish='tool_calls')]
        outputs += [support.reply('', calls=[call('node-' + str(i), 'circuit_inspect', {
            'path': '/original.sav', 'query': 'N' + str(i % 4)})], finish='tool_calls') for i in range(8)]
        outputs += [support.reply('', calls=[call('get-only', 'task_plan', {'action': 'get'})],
                                  finish='tool_calls'),
            support.reply('', calls=[call('advance', 'task_plan', {
            'action': 'update', 'id': 'map', 'status': 'completed',
            'note': '八个目标节点已取得有界映射证据。',
            'evidence_call_ids': ['node-' + str(i) for i in range(8)]})], finish='tool_calls'),
            support.reply('', calls=[call('repeat-after-checkpoint', 'circuit_inspect', {
                'path': '/original.sav', 'query': 'N7'})], finish='tool_calls'),
            support.reply('已完成有界映射核对。')]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed', 'answer': '已完成有界映射核对。'}])
        result = agent.handle(user_text='测试并验证这个CPU的有界节点映射。')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(inspect.call_count, 9)
        recovery_messages, recovery_options = fake.requests[9]
        names = {schema['function']['name'] for schema in recovery_options['tools']}
        self.assertTrue({'task_plan', 'circuit_inspect', 'spawn_subagent'} <= names)
        self.assertNotIn('read_context', names)
        self.assertTrue(any(message.get('role') == 'tool' or 'tool_calls' in message
                            for message in recovery_messages))
        second_recovery_messages, second_recovery_options = fake.requests[10]
        self.assertIn('circuit_inspect', {schema['function']['name']
                                          for schema in second_recovery_options['tools']})
        self.assertTrue(agent.db.get_tool_outcome(result['session_id'], result['task_id'],
                                                  'get-only')['ok'])
        resumed_names = {schema['function']['name'] for schema in fake.requests[11][1]['tools']}
        self.assertIn('circuit_inspect', resumed_names)
        repeated = agent.db.get_tool_outcome(result['session_id'], result['task_id'],
                                             'repeat-after-checkpoint')
        self.assertTrue(repeated['ok'])
        from aurex.task_reply import FINAL_SYSTEM
        self.assertFalse(any(messages[0].get('content') == FINAL_SYSTEM
                             for messages, _ in fake.requests))

    def test_recovery_keeps_registered_tools_available_and_records_failures(self):
        tools = ToolRegistry()
        execute = mock.Mock(side_effect=[RuntimeError('Unknown input ID: short-id')] * 3
                           + [{'measurement_source': 'fresh solve', 'samples': [{'input': 1, 'actual': 1}]}])
        attempted = mock.Mock(side_effect=RuntimeError('External write rejected by tool policy'))
        tools.register(ToolSpec('circuit_analyze', 'Analyze', {'type': 'object'}, execute))
        tools.register(ToolSpec('external_write', 'External write', {'type': 'object'}, attempted))
        def call(cid, name='circuit_analyze', arguments='{}'):
            return {'id': cid, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}
        outputs = [support.reply('', calls=[call('bad-' + str(i))], finish='tool_calls') for i in range(3)]
        outputs += [support.reply('', calls=[call('must-not-run', 'external_write')], finish='tool_calls'),
                    support.reply('', calls=[call('corrected', arguments='{"id":"exact-input-id"}')], finish='tool_calls'),
                    support.reply('One representative vector matched; other instructions were not tested.')]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': 'One representative vector matched; other instructions were not tested.'}])
        result = agent.handle(user_text='Introduce the circuit and slightly test it')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(execute.call_count, 4)
        attempted.assert_called_once()
        self.assertTrue(fake.requests[3][1]['tools'])
        events = agent.db.events(result['session_id'])
        self.assertEqual(sum(e['kind'] == 'tool_calls_deferred' for e in events), 0)
        self.assertEqual(sum(e['kind'] == 'progress_review_fallback' for e in events), 0)
        self.assertEqual(sum(e['kind'] == 'answer' for e in events), 1)
        self.assertFalse(agent.db.get_tool_outcome(result['session_id'], result['task_id'],
                                                   'must-not-run')['ok'])

    def test_repeated_calls_execute_in_same_task_without_round_or_token_deadline(self):
        tools = ToolRegistry()
        execute = mock.Mock(return_value={'voltage': 5})
        tools.register(ToolSpec('measure', 'Measure', {'type': 'object'}, execute))
        outputs = []
        for cycle in range(2):
            for i in range(3):
                call = {'id': f'c{cycle}-{i}', 'type': 'function', 'function': {'name': 'measure', 'arguments': '{}'}}
                outputs.append(support.reply('', calls=[call], finish='tool_calls'))
            call = {'id': f'unexecuted-{cycle}', 'type': 'function', 'function': {'name': 'measure', 'arguments': '{}'}}
            outputs.append(support.reply('', calls=[call], finish='tool_calls'))
        outputs.append(support.reply('Final measured result'))
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([
            {'outcome': 'completed', 'answer': 'Final measured result'},
        ])
        result = agent.handle(user_text='Measure all requested points')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(execute.call_count, 8)
        self.assertEqual(sum(e['kind'] == 'progress_review_fallback'
                             for e in agent.db.events(result['session_id'])), 0)
        self.assertEqual(len(agent.db.tasks(result['session_id'])), 1)

    def test_repeated_small_target_connectivity_walk_keeps_tools_available(self):
        tools = ToolRegistry()
        analyze = mock.Mock(return_value={'measurement_source': 'fresh solve', 'state_path': '/state',
                                          'measurements': {'stimulus_scope': {'total_steps': 4}}})
        inspect = mock.Mock(side_effect=lambda _rt, args: {
            'node_query': {'node': args['query'], 'match_count': 2},
            'netlist': {'components': [{'id': args['query']}]}})
        tools.register(ToolSpec('circuit_analyze', 'Analyze', {'type': 'object'}, analyze))
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, inspect))
        def call(cid, name, arguments):
            return {'id': cid, 'type': 'function',
                    'function': {'name': name, 'arguments': json.dumps(arguments)}}
        outputs = [support.reply('', calls=[call('measure', 'circuit_analyze', {})], finish='tool_calls')]
        outputs += [support.reply('', calls=[call('node-' + str(i), 'circuit_inspect',
                                                   {'path': '/original.sav', 'query': 'N' + str(i % 4)})],
                                  finish='tool_calls') for i in range(8)]
        outputs.append(support.reply('当前引擎中代表性刺激到达时钟后成为X；不能据此判CPU错误，其余ISA未测。'))
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '当前引擎中代表性刺激到达时钟后成为X；不能据此判CPU错误，其余ISA未测。'}])
        result = agent.handle(user_text='对原CPU做一个代表性动态测试，不要遍历完整网表。')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(inspect.call_count, 8)
        self.assertTrue(fake.requests[-1][1]['tools'])
        events = agent.db.events(result['session_id'])
        self.assertFalse(any(event['kind'] == 'connectivity_notice' for event in events))
        self.assertFalse(any(event['kind'] == 'loop_recovery' and
                             event['data'].get('connectivity_walk') for event in events))
        final_system = __import__('aurex.task_reply', fromlist=['FINAL_SYSTEM']).FINAL_SYSTEM
        self.assertFalse(any(messages[0].get('content') == final_system
                             for messages, _ in fake.requests))
        self.assertEqual(sum(event['kind'] == 'answer' for event in events), 1)

    def test_query_many_a_b_a_unchanged_outcome_adds_hint_without_disabling_tools(self):
        tools = ToolRegistry()
        query = mock.Mock(return_value={
            'query_manifest': {'query_coverage': {
                'requested': 1, 'represented': 1, 'complete': True}},
            'results': [{'query': 'N676', 'ok': True, 'component_ids': ['dff-1']}]})
        inspect = mock.Mock(return_value={
            'node_query': {'node': 'N1', 'exact': True, 'match_count': 1},
            'netlist': {'components': [{'id': 'gate-1'}]}})
        tools.register(ToolSpec('circuit_query_many', 'Query', {'type': 'object'}, query))
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, inspect))

        def call(cid, name, arguments):
            return {'id': cid, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments)}}

        same = {'path': '/cpu.sav', 'queries': ['N676'], 'fields': ['pins']}
        outputs = [
            support.reply('', calls=[call('query-a1', 'circuit_query_many', same)],
                          finish='tool_calls'),
            support.reply('', calls=[call('inspect-b', 'circuit_inspect', {
                'path': '/cpu.sav', 'query': 'N1'})], finish='tool_calls'),
            support.reply('', calls=[call('query-a2', 'circuit_query_many', same)],
                          finish='tool_calls'),
            support.reply('重复快照不是新证据，当前结论为 INCONCLUSIVE。'),
        ]
        agent, fake = self.agent(outputs, tools)
        result = agent.handle(user_text='测试这个现有 CPU，但不要重复扫网表。')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(query.call_count, 2)
        self.assertEqual(inspect.call_count, 1)
        events = agent.db.events(result['session_id'], run_id=result['task_id'])
        recovery = next(event for event in events if event['kind'] == 'loop_recovery')
        self.assertEqual(recovery['data']['circuit_repeated_result']['pattern'], 'A-B-A')
        self.assertTrue(recovery['data']['circuit_repeated_result']['tools_remain_enabled'])
        final_prompt = json.dumps(fake.requests[-1][0], ensure_ascii=False)
        self.assertIn('完整重复结果不能当作新证据', final_prompt)
        self.assertIn('所有正常工具仍可使用', final_prompt)
        self.assertTrue(fake.requests[-1][1]['tools'])

    def test_explicit_exact_pagination_exposes_only_the_required_next_node_page(self):
        tools = ToolRegistry()
        def page(_rt, args):
            offset = args.get('offset', 0)
            return {'node_query': {'node': 'N24', 'exact': True, 'match_count': 12,
                                   'offset': offset, 'limit': 8, 'requested_limit': 24,
                                   'next_offset': 8 if offset == 0 else None},
                    'netlist': {'components': [{'id': f'component-{offset}',
                                                'ref': f'C{offset + 1}',
                                                'type': 'D Flipflop',
                                                'selection_role': 'primary',
                                                'pins': [{'pin': 3, 'label': 'clk', 'node': 'N24'}]}]}}
        inspect = mock.Mock(side_effect=page)
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, inspect))
        def call(cid, offset=None):
            args = {'path': '/original.sav', 'query': 'N24', 'limit': 24}
            if offset is not None:
                args['offset'] = offset
            return {'id': cid, 'type': 'function', 'function': {
                'name': 'circuit_inspect', 'arguments': json.dumps(args)}}
        outputs = [support.reply('', calls=[call('page-0')], finish='tool_calls'),
                   support.reply('', calls=[call('page-8', 8)], finish='tool_calls'),
                   support.reply('已完成精确分页；实际行为仍需另行动态测量。')]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '已完成精确分页；实际行为仍需另行动态测量。'}])
        result = agent.handle(user_text='对N24使用circuit_inspect精确query和分页后回答。')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual([call.args[1].get('offset', 0) for call in inspect.call_args_list], [0, 8])
        final_system = __import__('aurex.task_reply', fromlist=['FINAL_SYSTEM']).FINAL_SYSTEM
        execution = [options for messages, options in fake.requests if messages[0].get('content') != final_system]
        self.assertIn('circuit_inspect',
                      [schema['function']['name'] for schema in execution[1]['tools']])
        self.assertTrue(any(event['kind'] == 'tool_end' for event in agent.db.events(result['session_id'])))

    def test_explicit_single_interface_scan_wording_does_not_disable_retrieval(self):
        tools = ToolRegistry()
        inspect = mock.Mock(return_value={'interface_only': True, 'ports': [{
            'id': 'input-id', 'ref': 'C1', 'direction': 'input', 'node': 'N1',
            'node_connection_count': 2, 'connected_to_other_components': True,
            'logic': 0, 'logic_text': 'L'}], 'total_inputs': 1,
            'total_outputs': 0, 'total_ports': 1})
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, inspect))
        def call(cid):
            return {'id': cid, 'type': 'function', 'function': {
                'name': 'circuit_inspect', 'arguments': json.dumps({
                    'path': '/original.sav', 'interface_only': True})}}
        outputs = [support.reply('', calls=[call('interface-first')], finish='tool_calls'),
                   support.reply('', calls=[call('interface-duplicate')], finish='tool_calls'),
                   support.reply('已使用首次接口证据；未把重复调用当成新测量。')]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '已使用首次接口证据；未把重复调用当成新测量。'}])
        result = agent.handle(user_text='只调用一次 interface_only，然后使用持久化证据回答。')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(inspect.call_count, 2)
        duplicate = agent.db.get_tool_outcome(result['session_id'], result['task_id'],
                                              'interface-duplicate')
        self.assertTrue(duplicate['ok'])
        repeated = [event for event in agent.db.events(result['session_id'])
                    if event['kind'] == 'repeated_evidence_call']
        self.assertEqual(len(repeated), 1)
        self.assertTrue(repeated[0]['data']['executed'])
        self.assertIn('相同参数允许重新取证', repeated[0]['data']['message'])

    def test_exact_inspection_page_may_be_rechecked_after_resume(self):
        tools = ToolRegistry()
        inspect = mock.Mock(return_value={'node_query': {'node': 'N642', 'exact': True,
            'match_count': 2, 'offset': 0, 'limit': 8, 'next_offset': None},
            'netlist': {'components': []}})
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, inspect))
        agent, fake = self.agent([], tools)
        sid = agent.db.session('durable-inspection', source='admin')
        request = '使用已保存的精确拓扑证据继续CPU验证。'
        rid = agent.db.enqueue_task(sid, request, source='admin')
        agent.db.message(sid, rid, {'role': 'user', 'content': request})
        arguments = json.dumps({'path': '/original.sav', 'query': 'N642', 'limit': 8})
        original_call = {'id': 'before-compaction', 'type': 'function', 'function': {
            'name': 'circuit_inspect', 'arguments': arguments}}
        agent.db.message(sid, rid, {'role': 'assistant', 'content': None,
                                    'tool_calls': [original_call]})
        agent.db.tool_outcome(sid, rid, 'before-compaction', 'circuit_inspect', json.dumps({
            'ok': True, 'data': {'node_query': {'node': 'N642', 'exact': True,
                'match_count': 2, 'offset': 0, 'limit': 8, 'next_offset': None},
                'netlist': {'components': []}}}), True)
        duplicate_call = {'id': 'after-compaction', 'type': 'function', 'function': {
            'name': 'circuit_inspect', 'arguments': arguments}}
        fake.replies = iter([
            support.reply('', calls=[duplicate_call], finish='tool_calls'),
            support.reply('已重新查询 N642，并保留前后两份证据。')])
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '已重新查询 N642，并保留前后两份证据。'}])
        result = agent.handle(user_text=request, session_id=sid, run_id=rid)
        self.assertEqual(result['status'], 'completed')
        inspect.assert_called_once()
        repeated = agent.db.get_tool_outcome(sid, rid, 'after-compaction')
        self.assertTrue(repeated['ok'])
        event = next(event for event in agent.db.events(sid, run_id=rid)
                     if event['kind'] == 'repeated_evidence_call')
        self.assertTrue(event['data']['executed'])

    def test_identical_successful_analysis_may_be_rechecked_after_pruning_or_resume(self):
        tools = ToolRegistry()
        analyze = mock.Mock(return_value={'state_path': '/new-state',
            'measurements': {'analysis': 'tr'}})
        tools.register(ToolSpec('circuit_analyze', 'Analyze', {'type': 'object'}, analyze))
        agent, fake = self.agent([], tools)
        sid = agent.db.session('durable-analysis', source='admin')
        request = '复用相同仿真，只有激励改变时才重跑。'
        rid = agent.db.enqueue_task(sid, request, source='admin')
        agent.db.message(sid, rid, {'role': 'user', 'content': request})
        arguments = json.dumps({'path': '/immutable.sav', 'analysis': 'tr',
            'stimulus': [{'set': {'clock-id': 0}}, {'set': {'clock-id': 1}}]},
            sort_keys=True)
        old = {'id': 'analysis-before-prune', 'type': 'function', 'function': {
            'name': 'circuit_analyze', 'arguments': arguments}}
        agent.db.message(sid, rid, {'role': 'assistant', 'content': None, 'tool_calls': [old]})
        agent.db.tool_outcome(sid, rid, old['id'], 'circuit_analyze', json.dumps({
            'ok': True, 'data': {'state_path': '/recorded-state',
                'measurements': {'analysis': 'tr'}}}), True)
        duplicate = {'id': 'analysis-after-prune', 'type': 'function', 'function': {
            'name': 'circuit_analyze', 'arguments': arguments}}
        fake.replies = iter([
            support.reply('', calls=[duplicate], finish='tool_calls'),
            support.reply('相同激励已重新仿真，并保留新的状态证据。')])
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '相同激励已重新仿真，并保留新的状态证据。'}])
        result = agent.handle(user_text=request, session_id=sid, run_id=rid)
        self.assertEqual(result['status'], 'completed')
        analyze.assert_called_once()
        repeated = agent.db.get_tool_outcome(sid, rid, duplicate['id'])
        self.assertTrue(repeated['ok'])
        self.assertTrue(any(event['kind'] == 'repeated_evidence_call' and
                            event['data']['tool'] == 'circuit_analyze'
                            for event in agent.db.events(sid, run_id=rid)))

    def test_unlabelled_cpu_input_does_not_create_a_server_side_tool_gate(self):
        tools = ToolRegistry()
        input_id = 'unlabelled-input-id'
        def inspect_result(_rt, args):
            if args.get('interface_only'):
                return {'ports': [{'id': input_id, 'ref': 'C1', 'label': '',
                    'direction': 'input', 'node': 'N705', 'logic': 0}],
                    'total_inputs': 1, 'total_outputs': 0, 'total_ports': 1}
            return {'node_query': {'node': 'N705', 'exact': True, 'match_count': 2,
                'offset': 0, 'limit': 8, 'next_offset': None},
                'netlist': {'components': []}}
        inspect = mock.Mock(side_effect=inspect_result)
        analyze = mock.Mock(return_value={'state_path': '/measured',
            'measurements': {'analysis': 'tr'}})
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, inspect))
        tools.register(ToolSpec('circuit_analyze', 'Analyze', {'type': 'object'}, analyze))
        def call(cid, name, args):
            return {'id': cid, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(args)}}
        stimulus = {'path': '/cpu.sav', 'analysis': 'tr',
                    'stimulus': [{'set': {input_id: 0}}, {'set': {input_id: 1}}]}
        outputs = [
            support.reply('', calls=[call('interface', 'circuit_inspect', {
                'path': '/cpu.sav', 'interface_only': True})], finish='tool_calls'),
            support.reply('', calls=[call('blind', 'circuit_analyze', stimulus)], finish='tool_calls'),
            support.reply('', calls=[call('trace-node', 'circuit_inspect', {
                'path': '/cpu.sav', 'query': 'N705', 'limit': 8})], finish='tool_calls'),
            support.reply('', calls=[call('measured', 'circuit_analyze', stimulus)], finish='tool_calls'),
            support.reply('N705 was traced before applying the bounded CPU stimulus.')]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': 'N705 was traced before applying the bounded CPU stimulus.'}])
        result = agent.handle(user_text='对这个 CPU 做一个有界功能测试。')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(inspect.call_count, 2)
        self.assertEqual(analyze.call_count, 2)
        first = agent.db.get_tool_outcome(result['session_id'], result['task_id'], 'blind')
        self.assertTrue(first['ok'])

    def test_repeated_typed_body_reads_execute_without_archive_reader_or_tool_lock(self):
        tools = ToolRegistry()
        read_body = mock.Mock(return_value={
            'summary_id': 'a' * 24, 'offset': 0,
            'text': '作者正文中的有界片段', 'has_more': False})
        tools.register(ToolSpec('plar_read_body', 'Read bounded prose',
            {'type': 'object'}, read_body))
        arguments = json.dumps({'summary_id': 'a' * 24, 'offset': 0, 'length': 256})
        calls = [support.reply('', calls=[{'id': 'body-' + str(index),
            'type': 'function', 'function': {
                'name': 'plar_read_body', 'arguments': arguments}}], finish='tool_calls')
            for index in range(2)]
        agent, fake = self.agent(calls + [support.reply('正文片段已经核对。')], tools)
        result = agent.handle(user_text='核对正文中的指定片段。')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(read_body.call_count, 2)
        for index in range(2):
            self.assertTrue(agent.db.get_tool_outcome(
                result['session_id'], result['task_id'], 'body-' + str(index))['ok'])
        self.assertEqual(sum(event['kind'] == 'tool_end'
                             and event['data']['name'] == 'plar_read_body'
                             for event in agent.db.events(result['session_id'])), 2)
        names = {schema['function']['name'] for schema in fake.requests[-1][1]['tools']}
        self.assertIn('plar_read_body', names)
        self.assertNotIn('read_context', names)

    def test_repeated_custom_cpu_failures_review_testbench_without_disabling_tools(self):
        tools = ToolRegistry()
        custom_calls = 0

        def simulate(_runtime, args):
            nonlocal custom_calls
            if args['profile'] == 'rv32i_teaching_v1':
                return {'profile': 'rv32i_teaching_v1', 'verified': True,
                    'workspace_id': 'ws', 'workspace_revision': 3,
                    'source_files_sha256': {'aurex_rv32i_teaching.v': 'cpu-hash'},
                    'compile': {'exit_code': 0}, 'simulation': {'exit_code': 0,
                        'log': 'AUREX_CASE 0 PASS\nAUREX_VERIFIED_nonce\n'}}
            custom_calls += 1
            return {'profile': 'custom', 'verified': False,
                'workspace_id': 'ws', 'workspace_revision': 3 + custom_calls,
                'source_files_sha256': {'aurex_rv32i_teaching.v': 'cpu-hash',
                                        'sample_tb.v': 'tb-' + str(custom_calls)},
                'compile': {'exit_code': 0}, 'simulation': {'exit_code': 1,
                    'log': 'FAIL x1_addi_5: expected 00000005 got 00000000\n'
                           'SOME TESTS FAILED: 1 errors\n'}}

        tools.register(ToolSpec('hdl_simulate', 'Simulate',
            {'type': 'object', 'properties': {'profile': {'type': 'string'}},
             'required': ['profile']}, simulate))

        def call(cid, name, arguments):
            return {'id': cid, 'type': 'function', 'function': {
                'name': name, 'arguments': json.dumps(arguments)}}

        outputs = [
            support.reply('', calls=[call('plan-set', 'task_plan', {'action': 'set', 'items': [
                {'id': 'verify', 'title': '用固定验证器和小抽样验证CPU'}]})], finish='tool_calls'),
            support.reply('', calls=[call('fixed', 'hdl_simulate', {'profile': 'rv32i_teaching_v1'})], finish='tool_calls'),
            support.reply('', calls=[call('custom-1', 'hdl_simulate', {'profile': 'custom'})], finish='tool_calls'),
            support.reply('', calls=[call('custom-2', 'hdl_simulate', {'profile': 'custom'})], finish='tool_calls'),
            support.reply('', calls=[call('custom-3', 'hdl_simulate', {'profile': 'custom'})], finish='tool_calls'),
            support.reply('', calls=[call('plan-done', 'task_plan', {'action': 'update', 'id': 'verify',
                'status': 'completed', 'note': '固定验证通过；补充测试台失败未当作CPU失败。',
                'evidence_call_ids': ['fixed']})], finish='tool_calls'),
            support.reply('固定验证通过；自写抽样测试台编码有误，未冒充通过。'),
        ]
        agent, fake = self.agent(outputs, tools)
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '固定验证通过；自写抽样测试台编码有误，未冒充通过。'}])
        result = agent.handle(user_text='从头设计并验证RV32I CPU。')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(custom_calls, 3)
        events = agent.db.events(result['session_id'])
        notice = next(event for event in events if event['kind'] == 'hdl_testbench_notice')
        self.assertEqual(notice['data']['matching_custom_failures'], 3)
        recovery = next(event for event in events if event['kind'] == 'loop_recovery' and
                        event['data'].get('hdl_testbench_failures'))
        self.assertTrue(recovery['data']['hdl_testbench_failures']['tools_remain_enabled'])
        prompt = json.dumps(fake.requests[5][0], ensure_ascii=False)
        self.assertIn('ADDI x1,x0,5=00500093', prompt)
        self.assertIn('每次debug_reg_addr赋值后加#1', prompt)
        self.assertTrue(fake.requests[5][1]['tools'])

    def test_fixed_cpu_pass_is_invalidated_only_when_a_source_hash_changes(self):
        from aurex.session_agent import _latest_fixed_cpu_pass
        agent, _ = self.agent([])
        sid = agent.db.session('fixed-source-binding', source='admin')
        rid = agent.db.enqueue_task(sid, '设计RV32I CPU', source='admin')
        fixed = {'ok': True, 'data': {'profile': 'rv32i_teaching_v1', 'verified': True,
            'workspace_id': 'ws', 'workspace_revision': 3,
            'source_files_sha256': {'cpu.v': 'good'},
            'compile': {'exit_code': 0}, 'simulation': {'exit_code': 0}}}
        agent.db.tool_outcome(sid, rid, 'fixed', 'hdl_simulate', json.dumps(fixed), True)
        testbench_only = {'ok': True, 'data': {'workspace_id': 'ws', 'workspace_revision': 4,
            'files': [{'name': 'cpu.v', 'role': 'source', 'sha256': 'good'},
                      {'name': 'tb.v', 'role': 'testbench', 'sha256': 'tb'}]}}
        agent.db.tool_outcome(sid, rid, 'tb-edit', 'hdl_workspace_write',
                              json.dumps(testbench_only), True)
        self.assertIsNotNone(_latest_fixed_cpu_pass(agent.db, sid, rid))
        source_edit = {'ok': True, 'data': {'workspace_id': 'ws', 'workspace_revision': 5,
            'files': [{'name': 'cpu.v', 'role': 'source', 'sha256': 'changed'},
                      {'name': 'tb.v', 'role': 'testbench', 'sha256': 'tb'}]}}
        agent.db.tool_outcome(sid, rid, 'cpu-edit', 'hdl_workspace_edit',
                              json.dumps(source_edit), True)
        self.assertIsNone(_latest_fixed_cpu_pass(agent.db, sid, rid))

    def test_internal_resistance_barrier_stops_broad_guessing_without_tool_lock(self):
        tools = ToolRegistry()
        analyze = mock.Mock(side_effect=ValueError(
            '442b0e9a30e740558cd6170a92ac2c8d has nonzero internal resistance; '
            'model it explicitly as a series resistor before simulation'))
        tools.register(ToolSpec('circuit_analyze', 'Analyze', {'type': 'object'}, analyze))
        call = {'id': 'blocked-sim', 'type': 'function', 'function': {
            'name': 'circuit_analyze', 'arguments': '{}'}}
        agent, fake = self.agent([
            support.reply('', calls=[call], finish='tool_calls'),
            support.reply('原存档因非零内阻建模边界未能仿真；未删除元件或冒充通过。')], tools)
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '原存档因非零内阻建模边界未能仿真；未删除元件或冒充通过。'}])
        result = agent.handle(user_text='说明这个电路为什么不能运行。')
        self.assertEqual(result['status'], 'completed')
        events = agent.db.events(result['session_id'])
        barrier = next(event for event in events if event['kind'] == 'circuit_modeling_barrier')
        self.assertEqual(barrier['data']['component_id'], '442b0e9a30e740558cd6170a92ac2c8d')
        recovery = next(event for event in events if event['kind'] == 'loop_recovery')
        self.assertTrue(recovery['data']['circuit_modeling_barrier']['tools_remain_enabled'])
        self.assertTrue(fake.requests[1][1]['tools'])
        prompt = json.dumps(fake.requests[1][0], ensure_ascii=False)
        self.assertIn('只对错误中的精确component_id定位一次', prompt)
        self.assertIn('不删除器件', prompt)

    def test_image_tool_is_not_rejected_based_on_prior_queries_or_request_classification(self):
        tools = ToolRegistry()
        render = mock.Mock(return_value={'rendered': True})
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, render))
        image_call = {'id': 'agent-selected-image', 'type': 'function', 'function': {
            'name': 'circuit_inspect', 'arguments': json.dumps({'path': '/cpu.sav', 'with_image': True})}}
        agent, fake = self.agent([support.reply('', calls=[image_call], finish='tool_calls'),
                                  support.reply('已根据本轮判断读取图片。')], tools)
        fake.final_reviews = iter([{'outcome': 'completed', 'answer': '已根据本轮判断读取图片。'}])
        result = agent.handle(user_text='测试这个CPU是否正确')
        self.assertEqual(result['status'], 'completed')
        render.assert_called_once()
        outcome = agent.db.get_tool_outcome(result['session_id'], result['task_id'], 'agent-selected-image')
        self.assertTrue(outcome['ok'])

    def test_original_request_anchor_is_not_replaced_with_prepared_binding_markup(self):
        agent, fake = self.agent([support.reply('Introduction')])
        request = '介绍一下；我下次再选一个做仿真。'
        agent.handle(user_text=request)
        heads = [m['content'] for m in fake.requests[0][0] if isinstance(m.get('content'), str)
                 and m['content'].startswith('Current task original request')]
        self.assertEqual(heads, ['Current task original request (reference; do not replace with an older summary):\n' + request])
        bindings = [m['content'] for m in fake.requests[0][0] if isinstance(m.get('content'), str)
                    and m['content'].startswith('Current trusted server task binding')]
        self.assertEqual(len(bindings), 1)
        value = json.loads(bindings[0].split('\n', 1)[1])
        self.assertEqual(value['source'], 'web')
        self.assertIsNone(value['requester_user_id'])

    def test_clarification_call_echo_never_enables_tools_or_hidden_review(self):
        tools = ToolRegistry()
        execute = mock.Mock(side_effect=AssertionError('Clarification has no authorized investigation'))
        tools.register(ToolSpec('investigate', 'Investigate', {'type': 'object'}, execute))
        calls = [{'id': str(i), 'type': 'function', 'function': {'name': 'investigate', 'arguments': '{}'}} for i in range(2)]
        agent, fake = self.agent([support.reply('INVENTED EXPERIMENT IS THE SUBJECT', calls=[call], finish='tool_calls')
                                  for call in calls], tools)
        fake.final_reviews = iter([
            {'outcome': 'continue', 'answer': '当前只缺指代，基于已有留言板事实形成澄清。'},
            {'outcome': 'completed', 'answer': '这是用户留言板，你指的是哪条内容？'},
        ])
        resolution = {'requires_reference_clarification': True, 'reason_code': 'missing_deictic_referent'}
        with mock.patch('aurex.community_context.resolve_wall_reference', return_value=resolution):
            result = agent.handle(user_text='这是啥啊')
        self.assertEqual(result['status'], 'completed')
        self.assertIn('具体指代对象尚不明确', result['answer'])
        self.assertIn('请指出要查询的实验、讨论、评论或用户', result['answer'])
        execute.assert_not_called()
        self.assertTrue(all(not opts['tools'] for _, opts in fake.requests))
        self.assertTrue(fake.requests[0][1]['thinking'])
        self.assertTrue(all(not opts['thinking'] for _, opts in fake.requests[1:]))
        from aurex.task_reply import FINAL_SYSTEM
        reviews = [messages for messages, _ in fake.requests if messages[0]['content'] == FINAL_SYSTEM]
        self.assertEqual(reviews, [])
        self.assertNotIn('INVENTED EXPERIMENT IS THE SUBJECT', result['answer'])
        self.assertEqual(len(agent.db.tasks(result['session_id'])), 1)


class FinalBindingProjectionTests(unittest.TestCase):
    def test_large_reference_resolution_keeps_clarification_and_permissions_within_window(self):
        import test_task_actions as actions
        from aurex.config import ContextPolicyConfig, LLMConfig
        from aurex.task_reply import review_final_answer
        fixture = actions.TaskActionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.bind(source='admin', original_user_request='这是什么？', dry_run=True)
        fixture.client.config = LLMConfig(context_length=8192, max_output_tokens=512)
        fixture.client.capacity = lambda: 8192
        fixture.client.count = lambda messages, tools=None: len(json.dumps([messages, tools or []], ensure_ascii=False)) // 4 + 1
        fixture.runtime.config.context = ContextPolicyConfig(safety_tokens=128, summary_max_tokens=512)
        resolution = {'requires_reference_clarification': True, 'reason_code': 'missing_reference',
                      'candidate_references': [{'description': 'HUGE AUXILIARY REFERENCE' * 1000}] * 50}
        result = review_final_answer(fixture.runtime, '基于已有事实请求澄清。', fixture.client, fixture.db,
            fixture.sid, fixture.rid, lambda *a: None, context_messages=[], reference_resolution=resolution)
        self.assertEqual(result['outcome'], 'completed')
        messages = fixture.client.calls[-1][0]
        binding = next(json.loads(m['content'].split('\n', 1)[1]) for m in messages
                       if isinstance(m.get('content'), str) and m['content'].startswith('trusted_server_task_binding:'))
        self.assertTrue(binding['reference_resolution']['requires_reference_clarification'])
        self.assertEqual(binding['source'], 'admin')
        self.assertTrue(binding['dry_run'])
        self.assertFalse(binding['explicit_publish_requested'])
        self.assertIsNone(binding['requester_user_id'])
        self.assertLess(fixture.client.count(messages), 8192 - 512 - 128)
        self.assertNotIn('HUGE AUXILIARY REFERENCE', json.dumps(messages))
        with fixture.db.connect() as store:
            raw = json.loads(store.execute('SELECT content FROM documents WHERE id=?',
                                          (binding['complete_binding_document_id'],)).fetchone()[0])
        self.assertEqual(raw['reference_resolution'], resolution)


if __name__ == '__main__':
    unittest.main()
