#!/usr/bin/env python3
"""Exercise the running Web agent without posting anything to the community."""
import argparse
import json
from pathlib import Path
import time
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--base-url', default='http://127.0.0.1:4097')
parser.add_argument('--text', default='请用工具创建并实际仿真一个直流电路：理想5V电源V1与10Ω电阻R1并联接地。生成电路图和实验存档，读取仿真得到的电流，解释电阻功耗。必须依据工具的实际结果，不要只口算。')
parser.add_argument('--session')
parser.add_argument('--target')
parser.add_argument('--timeout', type=int, default=600)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
http = requests.Session()
http.trust_env = False
token_file = root / '.config/web-token'
if token_file.exists():
    http.headers['Authorization'] = 'Bearer ' + token_file.read_text().strip()
sid = args.session
if not sid:
    r = http.post(args.base_url + '/api/sessions', json={'title': args.text[:80]}, timeout=10)
    r.raise_for_status()
    sid = r.json()['id']
body = {'text': args.text}
if args.target:
    kind, tid = args.target.split(':', 1)
    body['target'] = {'type': kind, 'id': tid}
r = http.post(args.base_url + '/api/sessions/' + sid + '/messages', json=body, timeout=10)
r.raise_for_status()
rid = r.json()['run_id']
print(json.dumps({'session_id': sid, 'run_id': rid}, ensure_ascii=False), flush=True)
cursor = 0
deadline = time.monotonic() + args.timeout
while time.monotonic() < deadline:
    r = http.get(args.base_url + '/api/sessions/' + sid + '/events', params={'after': cursor}, timeout=10)
    r.raise_for_status()
    for event in r.json():
        cursor = max(cursor, event['id'])
        if event['run_id'] != rid or event['kind'] not in {'started', 'model_start', 'model_end', 'tool_start', 'tool_end', 'answer', 'error', 'artifact', 'context_loaded', 'compaction_end'}:
            continue
        value = event['data']
        if event['kind'] == 'tool_end':
            value = {k: v for k, v in value.items() if k != 'preview'}
        print(json.dumps({'kind': event['kind'], 'data': value}, ensure_ascii=False), flush=True)
        if event['kind'] in {'answer', 'error'}:
            raise SystemExit(0 if event['kind'] == 'answer' else 1)
    time.sleep(1)
raise SystemExit('Timed out waiting for the session; it remains visible in the Web tracker')
