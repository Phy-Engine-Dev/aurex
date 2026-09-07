import json
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest import mock

import test_aurex_v3 as support
from aurex.session_agent import _token_progress_event
from aurex.tools.registry import ToolRegistry, ToolSpec


def progress(**updates):
    data = dict(generated_tokens=123, generated_sha256='a' * 64, max_same_token_run=2,
                observed_period_limit=64, elapsed_s=5.5,
                max_exact_repeat=dict(period_tokens=2, copies=3, span_tokens=6))
    return json.dumps(dict(data, **updates))


class ProgressTests(unittest.TestCase):
    def test_diagnostic_whitelist_never_persists_raw_model_fields(self):
        data = _token_progress_event(progress(token_ids=[100, 200], prompt_token_ids=[300],
                                             reasoning='PRIVATE', content='PRIVATE', arguments='PRIVATE'))
        self.assertEqual(data['generated_tokens'], 123)
        self.assertNotIn('PRIVATE', json.dumps(data))
        for key in ('token_ids', 'prompt_token_ids', 'reasoning', 'content', 'arguments'):
            self.assertNotIn(key, data)

    def test_malformed_optional_telemetry_is_ignored(self):
        for raw in ('[]', '{', 'x' * 4097, progress(generated_tokens=True),
                    progress(elapsed_s=float('nan')), progress(elapsed_s=-1),
                    progress(generated_sha256='not-a-digest'),
                    progress(max_exact_repeat={'period_tokens': 1, 'copies': True, 'span_tokens': 2})):
            with self.subTest(raw=raw[:80]):
                self.assertIsNone(_token_progress_event(raw))

    def test_web_progress_updates_one_folded_card_per_request(self):
        node = shutil.which('node')
        if not node:
            self.skipTest('Node is needed for the isolated DOM test')
        page = Path(__file__).resolve().parents[1] / 'src/aurex/tracking.html'
        code = page.read_text().split('<script>', 1)[1].split("$('new').onclick", 1)[0]
        test = r'''
const vm=require('vm'),assert=require('assert');
function element(){return {children:[],textContent:'',append(...x){this.children.push(...x)}}}
const timeline=element(),sandbox={document:{getElementById:()=>timeline,createElement:element},location:{search:''},URLSearchParams};
vm.createContext(sandbox); vm.runInContext(JSON.parse(process.argv[1]),sandbox);
function update(run,step,count){sandbox.add({kind:'model_progress',run_id:run,created:0,data:{step,generated_tokens:count}})}
update('one',1,100);update('one',1,200);update('one',1,300);
assert.equal(timeline.children.length,1);assert.equal(timeline.children[0].open,undefined);
assert(timeline.children[0].children[0].textContent.includes('300 tokens'));
update('one',2,2);update('two',1,5);assert.equal(timeline.children.length,3);
assert(!JSON.stringify(timeline).includes('<script>'));
'''
        subprocess.run([node, '-e', test, json.dumps(code)], check=True, capture_output=True, text=True, timeout=10)


class SessionProgressTests(unittest.TestCase):
    setUp = support.SessionAgentTests.setUp
    agent = support.SessionAgentTests.agent

    def test_progress_does_not_break_response_or_enter_model_history(self):
        registry = ToolRegistry()
        registry.register(ToolSpec('read_voltage', 'Read', {'type': 'object'}, lambda rt, args: {'voltage': 5}))
        call = {'id': 'read1', 'type': 'function', 'function': {'name': 'read_voltage', 'arguments': '{}'}}
        agent, fake = self.agent([support.reply('', calls=[call], finish='tool_calls'), support.reply('Measured 5 V')], registry)
        original = fake.chat
        def traced(messages, **options):
            if options.get('on_delta') and options.get('tools'):
                options['on_delta']('progress', progress(content='PRIVATE_PROGRESS'))
                options['on_delta']('progress', 'malformed optional telemetry')
            return original(messages, **options)
        with mock.patch.object(fake, 'chat', side_effect=traced):
            result = agent.handle(user_text='Read voltage')
        events = agent.db.events(result['session_id'])
        records = [e for e in events if e['kind'] == 'model_progress']
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['data']['step'], 1)
        self.assertEqual(result['answer'], 'Measured 5 V')
        self.assertEqual(len([e for e in events if e['kind'] == 'answer']), 1)
        self.assertNotIn('PRIVATE_PROGRESS', json.dumps(events))
        history = json.dumps(agent.db.messages(result['session_id']))
        self.assertNotIn('generated_sha256', history)
        self.assertNotIn('PRIVATE_PROGRESS', history)


if __name__ == '__main__':
    unittest.main()
