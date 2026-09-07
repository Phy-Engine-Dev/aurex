"""Archive the exact submitted HDL that a tool report cryptographically binds."""
from __future__ import annotations

import hashlib
import json
import re


def archive_hdl_sources(db, sid: str, args: dict, result: dict) -> list[dict]:
    files = args.get('files')
    hashes = result.get('source_files_sha256')
    if not isinstance(files, list) or not 1 <= len(files) <= 16 or not isinstance(hashes, dict):
        raise ValueError('HDL source archive needs the actual submitted files and report hashes')
    sources = {}
    total = 0
    for item in files:
        if not isinstance(item, dict) or set(item) != {'name', 'content'}:
            raise ValueError('Malformed submitted HDL source')
        name, text = item['name'], item['content']
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,80}\.(?:v|sv)', name)
                or name in sources or not isinstance(text, str)):
            raise ValueError('Invalid or duplicate HDL source name')
        size = len(text.encode())
        total += size
        if not text.strip() or size > 256_000 or total > 512_000:
            raise ValueError('HDL source archive exceeds the submission limits')
        sources[name] = text
    actual = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sorted(sources.items())}
    bundle = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if actual != hashes or bundle != result.get('source_sha256'):
        raise ValueError('HDL source differs from the actual verification report; not archived as report-bound source')
    refs = []
    for name, text in sorted(sources.items()):
        title = f'HDL source {name} sha256={actual[name]}'
        with db.connect() as store:
            row = store.execute('SELECT id,content FROM documents WHERE session_id=? AND title=? ORDER BY created DESC LIMIT 1',
                                (sid, title)).fetchone()
        did = row['id'] if row is not None and row['content'] == text else db.document(sid, title, text)
        refs.append({'name': name, 'document_id': did, 'sha256': actual[name], 'characters': len(text),
                     'retrieval': 'read_context', 'source_kind': 'submitted_hdl_source',
                     'note': 'Exact tool-submitted source, not proof that verification passed.'})
    return refs
