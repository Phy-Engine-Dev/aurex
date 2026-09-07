"""Lossless, hash-bound large sample storage, separate from renderer input."""
import copy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat

from .tools.registry import ToolError

LIMIT = 256 * 1024**2
THRESHOLD = 4 * 1024**2


def compact_snapshot(path, snapshot):
    result = copy.deepcopy(snapshot)
    measured = result['measurements']
    for container, key in ((measured, 'stimulus_results'), (measured.get('transient', {}), 'samples')):
        rows = container.get(key)
        if not isinstance(rows, list):
            continue
        raw = json.dumps(rows, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        if len(raw) <= THRESHOLD:
            continue
        if len(raw) > LIMIT:
            raise ToolError('Recorded sample series exceeds 256 MiB; select fewer recorded times (solver steps are independent)')
        compressed = gzip.compress(raw, compresslevel=3, mtime=0)
        if len(compressed) > 32 * 1024**2:
            raise ToolError('Compressed sample series exceeds 32 MiB')
        name = f'trace-{key}-{hashlib.sha256(compressed).hexdigest()}.json.gz'
        target = Path(path).parent / name
        # Newly allocated solver revision only; never overwrite an existing file.
        with target.open('xb') as out:
            out.write(compressed)
        container.pop(key)
        container[key + '_archive'] = {'schema':'aurex.trace-archive.v1', 'file':name,
            'sha256':hashlib.sha256(compressed).hexdigest(), 'raw_sha256':hashlib.sha256(raw).hexdigest(),
            'bytes':len(compressed), 'raw_bytes':len(raw), 'count':len(rows), 'encoding':'gzip-json',
            'lossless':True, 'note':'All recorded values preserved; no downsampling or rerun'}
    return result


def read_series(source, container, key):
    rows = container.get(key)
    record = container.get(key + '_archive')
    if record is None:
        return rows
    if rows is not None:
        raise ToolError('Ambiguous inline and archived trace')
    if not isinstance(record, dict) or record.get('schema') != 'aurex.trace-archive.v1':
        raise ToolError('Invalid trace archive manifest')
    name = record.get('file')
    if not isinstance(name, str) or not re.fullmatch(r'trace-(samples|stimulus_results)-[0-9a-f]{64}\.json\.gz', name):
        raise ToolError('Trace archive must be a bound sibling filename')
    if type(record.get('raw_bytes')) is not int or not 0 < record['raw_bytes'] <= LIMIT:
        raise ToolError('Invalid trace expansion size')
    path = Path(source).parent / name
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 32 * 1024**2:
                raise ToolError('Invalid trace archive file')
            compressed = stream.read(32 * 1024**2 + 1)
        if len(compressed) != record.get('bytes') or hashlib.sha256(compressed).hexdigest() != record.get('sha256'):
            raise ToolError('Trace archive hash/size mismatch')
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            raw = stream.read(record['raw_bytes'] + 1)
        if len(raw) != record['raw_bytes'] or hashlib.sha256(raw).hexdigest() != record.get('raw_sha256'):
            raise ToolError('Trace expansion hash/size mismatch')
        rows = json.loads(raw)
    except (OSError, ValueError, EOFError) as error:
        raise ToolError('Recorded trace archive is unavailable or damaged') from error
    if not isinstance(rows, list) or len(rows) != record.get('count'):
        raise ToolError('Recorded trace count mismatch')
    return rows


def hydrate_snapshot(source, value):
    if value.get('schema') != 'aurex.pe-state.v1':
        return value
    measured = value.get('measurements', {})
    for container, key in ((measured,'stimulus_results'), (measured.get('transient',{}),'samples')):
        if key + '_archive' in container:
            container[key] = read_series(source, container, key)
            container.pop(key + '_archive')
    return value
