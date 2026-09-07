"""Recovery must make progress even if a provider ignores tools=[]."""
import json
import unittest
from unittest import mock

import test_aurex_v3 as support
from aurex.tools.registry import ToolRegistry, ToolSpec


class ProgressReviewTests(unittest.TestCase):
    def test_cpu_planning_thinking_is_bounded_per_turn_not_per_task(self):
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
        self.assertEqual(fake.requests[0][1]['max_tokens'], 4096)
        first_prompt = json.dumps(fake.requests[0][0], ensure_ascii=False)
        self.assertIn('SERVER_CPU_ACCEPTANCE_PROTOCOL', first_prompt)
        self.assertIn('rv32i_teaching_v1', first_prompt)
        self.assertIn('aurex_rv32i_teaching', first_prompt)
        self.assertIn('同一设计源hash已在固定profile中verified=true', first_prompt)
        self.assertIn('32位word数组须用addr>>2索引', first_prompt)
        self.assertIn('绝不能在同一次hdl_simulate里同时传workspace_id和files', first_prompt)
        self.assertIn("ADDI x1,x0,5 = 32'h00500093", first_prompt)
        self.assertIn('每次改变选择器后先#1', first_prompt)
        self.assertTrue({'task_plan', 'read_context'} <= {schema['function']['name']
                                                          for schema in fake.requests[0][1]['tools']})
        self.assertFalse(any(event['kind'] == 'tool_limit_reached'
                             for event in agent.db.events(sid)))

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
        self.assertTrue({'task_plan', 'circuit_analyze', 'read_context'} <= first_tools)
        self.assertIn('SERVER_TASK_PLAN_JSON', json.dumps(fake.requests[1][0], ensure_ascii=False))
        plan = agent.db.task_plan(result['session_id'], result['task_id'])
        self.assertEqual(plan[0]['status'], 'completed')
        self.assertEqual(len(plan[0]['evidence_document_ids']), 1)
        from aurex.sessiondb import SessionDB
        self.assertEqual(SessionDB(agent.db.path).task_plan(result['session_id'], result['task_id']), plan)
        review_messages = fake.requests[-1][0]
        self.assertFalse(any(message.get('role') == 'tool' or 'tool_calls' in message
                             for message in review_messages))

    def test_completed_plan_history_cannot_be_replaced_or_reopened(self):
        agent, _ = self.agent([])
        sid = agent.db.session('plan-history', source='admin')
        rid = agent.db.enqueue_task(sid, '复杂验证', source='admin')
        agent.db.set_task_plan(sid, rid, [{'id': 'measure', 'title': '取得测量证据'}])
        did = agent.db.document(sid, 'measurement', '{"voltage":5}')
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

    def test_completed_plan_blocks_an_unrequested_recheck_after_recovery(self):
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
        self.assertEqual(inspect.call_count, 8)
        recovery_messages, recovery_options = fake.requests[9]
        names = {schema['function']['name'] for schema in recovery_options['tools']}
        self.assertTrue({'task_plan', 'circuit_inspect', 'read_context'} <= names)
        self.assertTrue(any(message.get('role') == 'tool' or 'tool_calls' in message
                            for message in recovery_messages))
        second_recovery_messages, second_recovery_options = fake.requests[10]
        self.assertIn('circuit_inspect', {schema['function']['name']
                                          for schema in second_recovery_options['tools']})
        self.assertTrue(agent.db.get_tool_outcome(result['session_id'], result['task_id'],
                                                  'get-only')['ok'])
        resumed_names = {schema['function']['name'] for schema in fake.requests[11][1]['tools']}
        self.assertNotIn('circuit_inspect', resumed_names)
        repeated = agent.db.get_tool_outcome(result['session_id'], result['task_id'],
                                             'repeat-after-checkpoint')
        self.assertIsNone(repeated)

    def test_wide_zero_padded_stimulus_identifies_only_the_changed_column(self):
        from aurex.session_agent import _changed_stimulus_inputs
        ids = ['input-' + str(i) for i in range(45)]
        ports = {cid: {'logic': 0} for cid in ids}
        first = [0] * 45
        second = [0] * 45
        second[17] = 1
        changed, width = _changed_stimulus_inputs({
            'stimulus_table': {'inputs': ids, 'vectors': [first, second]}}, ports)
        self.assertEqual(width, 45)
        self.assertEqual(changed, ['input-17'])

    def test_batch_connectivity_evidence_survives_compaction_and_cpu_walk_is_bounded(self):
        from aurex.session_agent import (_cpu_connectivity_scope_warning,
                                         _successful_batch_queries)
        selectors, nodes = _successful_batch_queries({'results': [
            {'query': 'N17', 'ok': True, 'nodes': [{'id': 'N17'}]},
            {'query': 'C9', 'ok': True, 'component_ids': ['button-id'], 'nodes': []},
            {'query': 'N18', 'ok': False, 'nodes': [{'id': 'N18'}]},
            {'query': 'clock', 'ok': True, 'nodes': [{'id': 'named-net'}]},
        ], 'component_catalog': [{'id': 'button-id', 'pins': [
            {'pin': 0, 'node': 'N9'}, {'pin': 1, 'node': 'N10'}]}]})
        self.assertEqual(selectors, {'N17', 'C9', 'clock'})
        self.assertEqual(nodes, {'N9', 'N10', 'N17'})
        self.assertIsNone(_cpu_connectivity_scope_warning(
            {'queries': ['N1', 'N2', 'N2']}, {'N0'}))
        self.assertIn('at most 8', _cpu_connectivity_scope_warning(
            {'queries': [f'N{i}' for i in range(9)]}, set()))
        self.assertIn('bounded maximum 12', _cpu_connectivity_scope_warning(
            {'queries': ['N12', 'N13']}, {f'N{i}' for i in range(12)}))

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

    def test_repeated_small_target_connectivity_walk_is_advisory_before_normal_final_review(self):
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
        execution_requests = [options for messages, options in fake.requests
                              if messages[0].get('content') != __import__('aurex.task_reply', fromlist=['FINAL_SYSTEM']).FINAL_SYSTEM]
        self.assertTrue(execution_requests[-1]['tools'])
        events = agent.db.events(result['session_id'])
        review = next(event for event in events if event['kind'] == 'connectivity_review')
        self.assertEqual(review['data']['targeted_inspections_since_measurement'], 8)
        self.assertEqual(review['data']['distinct_targets'], 4)
        self.assertTrue(any(event['kind'] == 'loop_recovery' and
                            event['data'].get('connectivity_walk') for event in events))
        final_system = __import__('aurex.task_reply', fromlist=['FINAL_SYSTEM']).FINAL_SYSTEM
        progress = [options for messages, options in fake.requests if messages[0].get('content') == final_system]
        self.assertEqual(len(progress), 1)
        self.assertFalse(progress[0]['thinking'])
        self.assertEqual(progress[0]['max_tokens'], 2048)

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
        self.assertEqual([schema['function']['name'] for schema in execution[1]['tools']], ['circuit_inspect'])
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

    def test_unlabelled_cpu_input_requires_exact_node_evidence_before_stimulus(self):
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
        self.assertEqual(analyze.call_count, 1)
        rejected = agent.db.get_tool_outcome(result['session_id'], result['task_id'], 'blind')
        self.assertFalse(rejected['ok'])
        self.assertIn('selected only from interface order', rejected['full_json'])

    def test_fully_overlapping_archived_page_is_reexecuted_without_tool_lock(self):
        agent, fake = self.agent([])
        sid = agent.db.session('coverage-review', source='admin')
        rid = agent.db.enqueue_task(sid, '读取必要证据一次并回答。', source='admin')
        document_id = agent.db.document(sid, 'fixture', '0123456789' * 30)
        arguments = json.dumps({'document_id': document_id, 'offset': 40, 'length': 80})
        first = support.reply('', calls=[{'id': 'page-1', 'type': 'function', 'function': {
            'name': 'read_context', 'arguments': arguments}}], finish='tool_calls')
        second = support.reply('', calls=[{'id': 'page-2', 'type': 'function', 'function': {
            'name': 'read_context', 'arguments': arguments}}], finish='tool_calls')
        fake.replies = iter([first, second, support.reply('已有区间足够，使用已保存证据继续结论。')])
        fake.final_reviews = iter([{'outcome': 'completed', 'answer': '已有区间足够，使用已保存证据继续结论。'}])
        result = agent.handle(user_text='读取必要证据一次并回答。', session_id=sid, run_id=rid)
        self.assertEqual(result['status'], 'completed')
        final_system = __import__('aurex.task_reply', fromlist=['FINAL_SYSTEM']).FINAL_SYSTEM
        execution = [options for messages, options in fake.requests
                     if messages[0].get('content') != final_system]
        self.assertTrue(execution[-1]['tools'])
        review = next(event for event in agent.db.events(sid) if event['kind'] == 'repeated_context_read')
        self.assertTrue(review['data']['no_new_source_coverage'])
        self.assertEqual(review['data']['offset'], 40)

    def test_varying_literal_misses_get_navigation_hint_without_disabling_reads(self):
        agent, fake = self.agent([])
        sid = agent.db.session('literal-miss-review', source='admin')
        rid = agent.db.enqueue_task(sid, '检查保存日志后继续验证。', source='admin')
        document_id = agent.db.document(sid, 'bounded log',
            'AUREX_PROFILE_SAMPLE cycle=25\nAUREX_PROFILE_SAMPLE cycle=26\n')
        calls = []
        for cycle in (24, 23, 22, 21):
            calls.append(support.reply('', calls=[{'id': 'miss-' + str(cycle), 'type': 'function',
                'function': {'name': 'read_context', 'arguments': json.dumps({
                    'document_id': document_id, 'find': 'cycle=' + str(cycle)})}}], finish='tool_calls'))
        fake.replies = iter(calls + [support.reply('日志只保存了 cycle 25–26；依据真实窗口继续。')])
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '日志只保存了 cycle 25–26；依据真实窗口继续。'}])
        result = agent.handle(user_text='检查保存日志后继续验证。', session_id=sid, run_id=rid)
        self.assertEqual(result['status'], 'completed')
        events = agent.db.events(sid)
        review = next(event for event in events if event['kind'] == 'literal_search_review')
        self.assertEqual(review['data']['consecutive_literal_misses'], 4)
        self.assertTrue(review['data']['executed'])
        recovery = next(event for event in events if event['kind'] == 'loop_recovery')
        self.assertTrue(recovery['data']['literal_search_misses']['tools_remain_enabled'])
        self.assertTrue(fake.requests[4][1]['tools'])
        self.assertIn('停止猜测式find', json.dumps(fake.requests[4][0], ensure_ascii=False))

    def test_four_identical_archived_reads_get_advisory_but_all_execute(self):
        agent, fake = self.agent([])
        sid = agent.db.session('identical-read-review', source='admin')
        rid = agent.db.enqueue_task(sid, '复核已保存日志后继续。', source='admin')
        document_id = agent.db.document(sid, 'bounded log',
            'AUREX_PROFILE_SAMPLE case=0 cycle=3 pc=0000000c\n')
        arguments = json.dumps({'document_id': document_id,
                                'find': 'AUREX_PROFILE_SAMPLE case=0 cycle=3'})
        calls = [support.reply('', calls=[{'id': 'same-' + str(index), 'type': 'function',
            'function': {'name': 'read_context', 'arguments': arguments}}], finish='tool_calls')
            for index in range(4)]
        fake.replies = iter(calls + [support.reply('已使用保存结果继续，不再做A/B回环。')])
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '已使用保存结果继续，不再做A/B回环。'}])
        result = agent.handle(user_text='复核已保存日志后继续。', session_id=sid, run_id=rid)
        self.assertEqual(result['status'], 'completed')
        outcomes = [agent.db.get_tool_outcome(sid, rid, 'same-' + str(index))
                    for index in range(4)]
        self.assertTrue(all(outcome and outcome['ok'] for outcome in outcomes))
        events = agent.db.events(sid)
        review = next(event for event in events if event['kind'] == 'source_read_review')
        self.assertEqual(review['data']['same_result_reads_in_window'], 4)
        self.assertTrue(review['data']['executed'])
        recovery = next(event for event in events if event['kind'] == 'loop_recovery')
        self.assertTrue(recovery['data']['repeated_source_result']['tools_remain_enabled'])
        self.assertTrue(fake.requests[4][1]['tools'])
        next_prompt = json.dumps(fake.requests[4][0], ensure_ascii=False)
        self.assertIn('重复读取仍然允许', next_prompt)
        self.assertIn('不要继续A/B offset回环', next_prompt)

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
        review = next(event for event in events if event['kind'] == 'hdl_testbench_review')
        self.assertEqual(review['data']['matching_custom_failures'], 3)
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

    def test_many_tiny_pages_get_bounded_read_guidance_without_blocking_reads(self):
        agent, fake = self.agent([])
        sid = agent.db.session('tiny-page-review', source='admin')
        rid = agent.db.enqueue_task(sid, '读取源码后继续验证。', source='admin')
        document_id = agent.db.document(sid, 'large source', 'x' * 6000)
        calls = [support.reply('', calls=[{'id': 'tiny-' + str(index), 'type': 'function',
            'function': {'name': 'read_context', 'arguments': json.dumps({
                'document_id': document_id, 'offset': index * 130, 'length': 130})}}],
            finish='tool_calls') for index in range(12)]
        fake.replies = iter(calls + [support.reply('已改用有界大窗口取得所需源码。')])
        fake.final_reviews = iter([{'outcome': 'completed',
            'answer': '已改用有界大窗口取得所需源码。'}])
        result = agent.handle(user_text='读取源码后继续验证。', session_id=sid, run_id=rid)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(all(agent.db.get_tool_outcome(sid, rid, 'tiny-' + str(index))['ok']
                            for index in range(12)))
        events = agent.db.events(sid)
        review = next(event for event in events if event['kind'] == 'source_paging_review')
        self.assertEqual(review['data']['small_pages_in_window'], 12)
        recovery = next(event for event in events if event['kind'] == 'loop_recovery')
        self.assertTrue(recovery['data']['small_source_pages']['tools_remain_enabled'])
        self.assertTrue(fake.requests[12][1]['tools'])
        prompt = json.dumps(fake.requests[12][0], ensure_ascii=False)
        self.assertIn('hdl_workspace_read', prompt)
        self.assertIn('工具没有被禁用', prompt)

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

    def test_read_coverage_counts_only_the_projected_visible_source_prefix(self):
        from aurex.session_agent import _record_read_coverage, _visible_read_coverage
        source = {'offset': 0, 'text': 'x' * 20000}
        projection = json.dumps({'kind': 'bounded_recorded_tool_result', 'fields': {},
            'verbatim_excerpt': {'text': 'x' * 7139, 'shown_characters': 7139,
                'source_offset_start': 0, 'source_offset_end': 7139}})
        self.assertEqual(_visible_read_coverage(projection, source), (0, 7139))
        coverage = {}
        key = ('source-id', '/data/ports')
        self.assertTrue(_record_read_coverage(coverage, key, *_visible_read_coverage(projection, source)))
        self.assertTrue(_record_read_coverage(coverage, key, 7139, 14185))
        self.assertTrue(_record_read_coverage(coverage, key, 14185, 20427))
        self.assertFalse(_record_read_coverage(coverage, key, 14185, 20427))

    def test_image_generation_requires_visual_or_spatial_user_need(self):
        from aurex.session_agent import (_targeted_complex_schematic_allowed,
                                         _visual_evidence_requested)
        for request in ('测试这个CPU是否正确', '分析输入输出连接', '对电路做功能验证',
                        '不要看图，只用结构化数据', 'verify this without image'):
            self.assertFalse(_visual_evidence_requested(request))
        for request in ('看看这张图片', '那个电阻旁边的电容是什么', 'adjust the camera view'):
            self.assertTrue(_visual_evidence_requested(request))
        complex_paths = {'/complex.sav'}
        self.assertTrue(_targeted_complex_schematic_allowed(
            {'path': '/complex.sav', 'view': 'schematic', 'query': 'N24'}, complex_paths))
        self.assertTrue(_targeted_complex_schematic_allowed(
            {'path': '/complex.sav', 'view': 'schematic', 'focus_ids': ['a', 'b']}, complex_paths))
        self.assertFalse(_targeted_complex_schematic_allowed(
            {'path': '/unknown.sav', 'view': 'schematic', 'query': 'N24'}, complex_paths))
        self.assertFalse(_targeted_complex_schematic_allowed(
            {'path': '/complex.sav', 'view': 'spatial', 'query': 'N24'}, complex_paths))
        self.assertFalse(_targeted_complex_schematic_allowed(
            {'path': '/complex.sav', 'view': 'schematic'}, complex_paths))

    def test_nonvisual_task_rejects_image_tool_before_renderer_execution(self):
        tools = ToolRegistry()
        render = mock.Mock(return_value={'images': [{'path': '/should/not/exist.png'}]})
        tools.register(ToolSpec('circuit_inspect', 'Inspect', {'type': 'object'}, render))
        image_call = {'id': 'unneeded-image', 'type': 'function', 'function': {
            'name': 'circuit_inspect', 'arguments': json.dumps({'path': '/cpu.sav', 'with_image': True})}}
        agent, fake = self.agent([support.reply('', calls=[image_call], finish='tool_calls'),
                                  support.reply('图片未生成；改用结构化连接和仿真证据。')], tools)
        fake.final_reviews = iter([{'outcome': 'completed', 'answer': '图片未生成；改用结构化连接和仿真证据。'}])
        result = agent.handle(user_text='测试这个CPU是否正确')
        self.assertEqual(result['status'], 'completed')
        render.assert_not_called()
        outcome = agent.db.get_tool_outcome(result['session_id'], result['task_id'], 'unneeded-image')
        self.assertIn('requires either a user-requested visual/spatial question', outcome['full_json'])

    def test_exact_select_repeated_twice_is_recorded_but_tools_stay_enabled(self):
        agent, fake = self.agent([])
        sid = agent.db.session('selector-review', source='admin')
        rid = agent.db.enqueue_task(sid, '选择所需记录后回答。', source='admin')
        document_id = agent.db.document(sid, 'fixture', json.dumps([{'id': i} for i in range(6)]))
        first_args = {'document_id': document_id, 'json_pointer': '',
                      'select': {'fields': ['id'], 'offset': 0, 'limit': 2}}
        next_args = {'document_id': document_id, 'json_pointer': '',
                     'select': {'fields': ['id'], 'offset': 2, 'limit': 2}}
        calls = [(first_args, 'first'), (next_args, 'next'), (first_args, 'duplicate')]
        fake.replies = iter([support.reply('', calls=[{'id': cid, 'type': 'function', 'function': {
            'name': 'read_context', 'arguments': json.dumps(args)}}], finish='tool_calls') for args, cid in calls]
            + [support.reply('已使用保存的选择结果。')])
        fake.final_reviews = iter([{'outcome': 'completed', 'answer': '已使用保存的选择结果。'}])
        result = agent.handle(user_text='选择所需记录后回答。', session_id=sid, run_id=rid)
        self.assertEqual(result['status'], 'completed')
        reviews = [event for event in agent.db.events(sid) if event['kind'] == 'repeated_context_read']
        self.assertEqual(len(reviews), 1)
        self.assertTrue(reviews[0]['data']['exact_selector_and_offset_repeated'])
        self.assertEqual(reviews[0]['data']['selector_repetitions'], 2)
        self.assertTrue(reviews[0]['data']['executed'])
        self.assertTrue(fake.requests[3][1]['tools'])

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

    def test_clarification_call_echo_enters_review_without_enabling_tools_or_replaying_claims(self):
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
        self.assertEqual(result['answer'], '这是用户留言板，你指的是哪条内容？')
        execute.assert_not_called()
        self.assertTrue(all(not opts['tools'] for _, opts in fake.requests))
        self.assertEqual([opts['thinking'] for _, opts in fake.requests], [True, False, False, False])
        from aurex.task_reply import FINAL_SYSTEM
        reviews = [messages for messages, _ in fake.requests if messages[0]['content'] == FINAL_SYSTEM]
        self.assertEqual(len(reviews), 2)
        self.assertTrue(all('INVENTED EXPERIMENT IS THE SUBJECT' not in json.dumps(messages) for messages in reviews))
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
