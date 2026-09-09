#!/usr/bin/env python3
"""Create a private deployment config from the example, without committing credentials."""
import argparse
import json
import os
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='.config/aurex3.json')
args = parser.parse_args()
output = root / args.output
if output.exists():
    raise SystemExit('Configuration already exists; refusing to overwrite it')
cfg = json.loads((root / 'aurex3.config.example.json').read_text())
for section, keys in [('storage', ['cache_dir', 'context_db_path', 'history_dir']), ('tracking', ['database_path']),
                      ('phy_engine', ['cmake_source_dir', 'cmake_build_dir'])]:
    for key in keys:
        cfg[section][key] = str(root / cfg[section][key])
output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, 'w') as file:
    json.dump(cfg, file, ensure_ascii=False, indent=2)
    file.write('\n')
token_file = root / '.config/web-token'
if not token_file.exists():
    fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as file:
        file.write(secrets.token_urlsafe(32) + '\n')
print('Created private config:', output)
print('Web access token is stored in:', token_file)
