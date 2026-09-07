"""Server-only task-scoped publication authority and durable one-shot receipts.

No function which issues a permit or approval is registered as a model tool.
The model-facing tool is intercepted by SessionAgent; its fallback handler refuses.
Submit is at-most-once: an interrupted/ambiguous submit requires human reconciliation.
Cover/Confirm retries refer only to the already recorded SummaryID.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid
from typing import Any

from plar import api as plar_api
from plar.official_publish_api import hdl_source_carrier_status
from plar.physicslab import unwrap_user
from .replyfmt import prefix_user_mention


class PublicationError(ValueError):
    pass


PURPOSES = frozenset(("electrical_experiment", "riscv_teaching_subset", "555_state_table"))
MAX_SOURCE = 32 * 1024 * 1024
MAX_COVER = 1024 * 1024
MAX_PUBLISHED_ELEMENTS = 5000
_HAN = re.compile(r"[\u3400-\u9fff]")


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 200 or any(ord(c) < 32 for c in value):
        raise PublicationError(f"Invalid {name}")
    return value


def _private_dir(cache_dir: str) -> Path:
    root = Path(cache_dir).resolve()
    directory = root / ".publication"
    if directory.is_symlink():
        raise PublicationError("Publication ledger directory cannot be a symlink")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


@contextmanager
def _db(cache_dir: str):
    directory = _private_dir(cache_dir)
    path = directory / "ledger.sqlite3"
    if path.is_symlink():
        raise PublicationError("Publication ledger cannot be a symlink")
    connection = sqlite3.connect(path, timeout=15, isolation_level=None)
    os.chmod(path, 0o600)
    connection.row_factory = sqlite3.Row
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS scopes (
            session_id TEXT PRIMARY KEY, purpose TEXT UNIQUE NOT NULL, created REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS approvals (
            approval_id TEXT PRIMARY KEY, session_id TEXT UNIQUE NOT NULL, run_id TEXT NOT NULL,
            user_id TEXT NOT NULL, state TEXT NOT NULL, manifest TEXT NOT NULL,
            summary_id TEXT, credential TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_scopes (
            task_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, binding TEXT NOT NULL,
            binding_sha256 TEXT NOT NULL, created REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_publications (
            approval_id TEXT PRIMARY KEY, task_id TEXT UNIQUE NOT NULL, session_id TEXT NOT NULL,
            run_id TEXT NOT NULL, user_id TEXT NOT NULL, state TEXT NOT NULL, manifest TEXT NOT NULL,
            summary_id TEXT, credential TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_final_answers (
            task_id TEXT PRIMARY KEY, review_id TEXT UNIQUE NOT NULL, session_id TEXT NOT NULL,
            account_id TEXT, binding_sha256 TEXT NOT NULL, state TEXT NOT NULL,
            answer TEXT NOT NULL, review_document_id TEXT, reply_receipt TEXT, error TEXT,
            created REAL NOT NULL, updated REAL NOT NULL, outcome TEXT NOT NULL DEFAULT 'completed'
        );
    """)
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def _operation_lock(cache_dir: str, approval_id: str):
    import fcntl

    if not re.fullmatch(r"[0-9a-f]{32}", approval_id):
        raise PublicationError("Invalid approval ID")
    path = _private_dir(cache_dir) / (approval_id + ".lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PublicationError("Publication is already in progress; do not issue another request") from error
        yield
    finally:
        os.close(fd)


def authorize_session(cache_dir: str, session_id: str, purpose: str) -> dict[str, Any]:
    """Legacy test markers only: these no longer grant permission to publish."""
    _identity(session_id, "session ID")
    if purpose not in {"riscv_teaching_subset", "555_state_table"}:
        raise PublicationError("Only riscv_teaching_subset and 555_state_table are authorized")
    with _db(cache_dir) as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute("SELECT purpose FROM scopes WHERE session_id=?", (session_id,)).fetchone()
        if old:
            if old["purpose"] != purpose:
                raise PublicationError("Cannot change an authorized session's purpose")
        else:
            if db.execute("SELECT COUNT(*) FROM scopes").fetchone()[0] >= 2:
                raise PublicationError("The two authorized test-session slots are already allocated")
            if db.execute("SELECT 1 FROM scopes WHERE purpose=?", (purpose,)).fetchone():
                raise PublicationError("This test purpose already has its one authorized session")
            db.execute("INSERT INTO scopes VALUES (?,?,?)", (session_id, purpose, time.time()))
        db.execute("COMMIT")
    return {"session_id": session_id, "purpose": purpose, "legacy_only": True, "publication_allowed": False}


def bind_task_actions(cache_dir: str, *, task_id: str, session_id: str, source: str,
                      original_user_request: str, explicit_publish_requested: bool,
                      requester_user_id: str | None = None, requester_nickname: str | None = None,
                      target: dict | None = None, purpose: str = "electrical_experiment", dry_run: bool = False) -> dict[str, Any]:
    """SERVER ONLY: persist the original request and action scope, never model arguments."""
    _identity(task_id, "task ID")
    _identity(session_id, "session ID")
    if source not in {"web", "admin", "community"} or purpose not in PURPOSES:
        raise PublicationError("Task source/purpose must come from supported server metadata")
    if type(explicit_publish_requested) is not bool or type(dry_run) is not bool or not isinstance(original_user_request, str) or not original_user_request.strip():
        raise PublicationError("Task requires its original request and explicit publish-intent decision")
    if source == "community" and explicit_publish_requested and not conservative_publish_request(original_user_request):
        raise PublicationError("Community publication intent must be a direct, unambiguous request in the original comment")
    if len(original_user_request.encode('utf-8')) > 4 * 1024 * 1024:
        raise PublicationError("Original task request exceeds the durable metadata limit")
    if requester_user_id is not None and (not isinstance(requester_user_id, str) or not re.fullmatch(r'[0-9a-fA-F]{24}', requester_user_id)):
        raise PublicationError("Requester ID must be an original full 24-hex user ID, never a guessed name")
    if requester_nickname is not None and (not isinstance(requester_nickname, str) or re.search(r'[<>\x00-\x1f]', requester_nickname)):
        raise PublicationError("Requester nickname must be original safe user metadata")
    if target is not None:
        if (not isinstance(target, dict) or set(target) - {'type', 'id', 'comment_id'}
            or target.get('type') not in {'User', 'Experiment', 'Discussion'}
            or not isinstance(target.get('id'), str) or not re.fullmatch(r'[0-9a-fA-F]{24}', target['id'])
            or (target.get('comment_id') is not None and (not isinstance(target['comment_id'], str)
                or not re.fullmatch(r'[0-9a-fA-F]{24}', target['comment_id'])))):
            raise PublicationError("Reply target must be an exact original community type/ID/comment binding")
    binding = {'task_id': task_id, 'session_id': session_id, 'source': source,
        'original_user_request': original_user_request, 'explicit_publish_requested': explicit_publish_requested,
        'requester_user_id': requester_user_id, 'requester_nickname': requester_nickname,
        'target': target, 'purpose': purpose, 'dry_run': dry_run}
    serialized = json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    with _db(cache_dir) as db:
        db.execute('BEGIN IMMEDIATE')
        old = db.execute('SELECT binding_sha256 FROM task_scopes WHERE task_id=?', (task_id,)).fetchone()
        if old and old['binding_sha256'] != digest:
            raise PublicationError("A task's original requester, source, request, intent and destination are immutable")
        db.execute('INSERT OR IGNORE INTO task_scopes VALUES(?,?,?,?,?)', (task_id, session_id, serialized, digest, time.time()))
        db.execute('COMMIT')
    return {**binding, 'binding_sha256': digest, 'publication_limit': 1, 'final_reply_limit': 1}


def task_action_scope(cache_dir: str, *, task_id: str, session_id: str | None = None) -> dict[str, Any]:
    _identity(task_id, 'task ID')
    with _db(cache_dir) as db:
        row = db.execute('SELECT * FROM task_scopes WHERE task_id=?', (task_id,)).fetchone()
    if not row or (session_id is not None and row['session_id'] != session_id):
        raise PublicationError('This task has no matching server-issued external-action binding')
    return {**json.loads(row['binding']), 'binding_sha256': row['binding_sha256']}


def conservative_publish_request(text: str, *, bot_user_id: str | None = None,
                                 mention_tag: str = '@aurex') -> bool:
    """Allow-list direct requests in the ORIGINAL user comment, never quoted context.

    False negatives are intentional: ambiguous requests require clarification.
    True only admits deterministic server validation; it is not publication authority.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 12000:
        return False
    value = text.strip()
    # Only the caller's verified bot identity may be removed. A user reference in
    # quoted material, another account's tag, or a second tag remains untrusted.
    tag = mention_tag.strip() if isinstance(mention_tag, str) else ''
    if (isinstance(bot_user_id, str) and re.fullmatch(r'[0-9a-fA-F]{24}', bot_user_id)
            and tag and not re.search(r'[<>\r\n]', tag)):
        prefix = f'<user={bot_user_id}>{tag}</user>'
        if value.startswith(prefix):
            value = value[len(prefix):].lstrip(' \t,:，：')
    if re.search(r'[`"“”「」『』]|(?:^|\n)\s*>|<[^>]+>', value):
        return False
    if re.search(r'不要|别发布|不(?:需要|要|准|允许|能|想).*?(?:发布|上传)|勿发布|禁止|暂不|暂时不|如果|假如|别人|他说|她说|引用|如何|怎么发布|什么时候|是否应该|能否发布|请问|教程|介绍|解释|讨论|学习|预览|发布.*风险', value):
        return False
    if re.search(r"\b(?:not|never|don't|dont|should|whether|when|how|if|quote|quoted|said|says|someone)\b", value, re.I):
        return False
    if tag:
        value = re.sub(r'^' + re.escape(tag) + r'(?=$|[\s,:，：])[\s,:，：]*', '', value)
    chinese = re.search(r'^(?:请|帮我|麻烦(?:你)?|能不能帮我|可以帮我|完成后|做好后|验证后|将|把|发布|做|制作|设计|创建|构建)[^。！？!?\n]{0,160}(?:发布(?:这个|该|此|它|实验|到|一下|出去|作品|一份)|(?:并|然后|再)发布|发布[。！!\s]*$)', value)
    if re.match(r'^发布(?:这个|该|此|它|实验|到|一下|出去|作品|一份)', value):
        chinese = True
    english = re.search(r'^(?:(?:please|kindly)\s+|(?:can|could|will|would)\s+you\s+)?(?:publish\b|(?:create|make|build|design|verify|test)\b[^.!?\n]{0,180}\b(?:and|then)\s+publish\b)', value, re.I)
    return bool(chinese or english)


def explicitly_forbids_publication(text: str) -> bool:
    """Fail closed on a clear publication veto, including a checkbox conflict.

    This is an additional guard alongside the server-issued intent binding. It
    does not confer authority on any positive text.
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(
        r'(?:不要|请勿|勿|别|禁止|暂不|暂时不|无需|不必|不需要|不允许|不能|不准|不想|不愿|不)' 
        r'\s*(?:再|自动|直接|擅自|对外|公开|进行)?\s*(?:发布|上传到社区)|'
        r"\b(?:do\s+not|don't|dont|never|must\s+not|should\s+not|cannot|can't)\s+"
        r'(?:automatically\s+|publicly\s+)?(?:publish|post|upload)\b|'
        r'\b(?:no\s+publication|without\s+(?:publishing|posting|uploading))\b', text, re.I))


def runtime_dry_run(runtime, scope: dict | None = None) -> bool:
    metadata = getattr(runtime, 'task_metadata', None)
    return bool((scope or {}).get('dry_run') or (metadata.get('dry_run') if isinstance(metadata, dict) else False)
                or getattr(getattr(getattr(runtime, 'config', None), 'agent', None), 'dry_run', False))


def requester_mention(scope: dict, *, user=None) -> str:
    if scope['source'] in {'web', 'admin'}:
        return ''  # Administrator/local Web exception: no fabricated community identity.
    uid = scope.get('requester_user_id')
    if not isinstance(uid, str) or not re.fullmatch(r'[0-9a-fA-F]{24}', uid):
        raise PublicationError('Community external actions require the original requester user ID')
    nick = scope.get('requester_nickname')
    if not nick and user is not None:
        value = plar_api.get_user_by_id(user, user_id=uid)
        info = value.get('User', value) if isinstance(value, dict) else {}
        if info.get('ID') != uid:
            raise PublicationError('Requester lookup did not return the bound user ID')
        nick = info.get('Nickname')
    if not isinstance(nick, str) or not nick.strip() or re.search(r'[<>\x00-\x1f]', nick):
        raise PublicationError('The bound requester needs a verified nickname before a community mention can be emitted')
    tag = prefix_user_mention('', user_id=uid, nickname=nick)
    if not tag.startswith('<user=' + uid + '>'):
        raise PublicationError('Could not construct the existing ID-based user mention')
    return tag


def publication_authorization(cache_dir: str, session_id: str, *, task_id: str | None = None) -> dict[str, Any]:
    if task_id is None:
        raise PublicationError('Task-bound publication authority is required; legacy session permits cannot publish')
    scope = task_action_scope(cache_dir, task_id=task_id, session_id=session_id)
    if not scope['explicit_publish_requested']:
        raise PublicationError('The original user request did not explicitly authorize publication')
    if explicitly_forbids_publication(scope['original_user_request']):
        raise PublicationError('The original user request explicitly forbids publication; it conflicts with the publish authorization')
    if scope.get('dry_run'):
        raise PublicationError('Dry-run task: community publication is disabled')
    if scope['source'] == 'community' and not scope.get('requester_user_id'):
        raise PublicationError('Community publication requires the original requester user ID')
    with _db(cache_dir) as db:
        row = db.execute('SELECT * FROM task_publications WHERE task_id=?', (task_id,)).fetchone()
    return {**scope, 'category': 'Experiment', 'price': 0, 'limit': 1,
            'approval_id': row['approval_id'] if row else None, 'state': row['state'] if row else 'authorized',
            'summary_id': row['summary_id'] if row else None, 'run_id': row['run_id'] if row else None}


def _read_artifact(cache_dir: str, path: str, *, suffixes: tuple[str, ...], limit: int) -> tuple[Path, bytes]:
    root = Path(cache_dir).resolve()
    lexical = Path(os.path.abspath(path))
    resolved = lexical.resolve()
    if not resolved.is_relative_to(root) or not str(resolved).lower().endswith(suffixes):
        raise PublicationError("Publication inputs must be matching local artifact files inside cache_dir")
    if resolved.is_relative_to(root / ".publication"):
        raise PublicationError("Private publication ledger files are not agent evidence")
    if any(p.is_symlink() for p in (lexical, *lexical.parents) if p != root):
        raise PublicationError("Publication inputs must not use symlinks")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(lexical, flags)
    try:
        import stat

        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise PublicationError("Publication artifact is empty, oversized or not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise PublicationError("Publication artifact exceeds its size limit")
    finally:
        os.close(fd)
    return resolved, data


def _json(data: str | bytes) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise PublicationError("Duplicate JSON keys are not valid publication evidence")
            out[key] = value
        return out

    def constant(_value):
        raise PublicationError("Non-finite JSON numbers are not valid publication evidence")

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise PublicationError("Overflowed JSON numbers are not valid publication evidence")
        return parsed

    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    except (TypeError, ValueError, UnicodeError) as error:
        raise PublicationError("Publication artifact must contain complete valid JSON") from error


def _vector(value: Any, name: str) -> None:
    try:
        parts = value.split(",")
        if len(parts) != 3 or not all(math.isfinite(float(x)) for x in parts):
            raise ValueError
    except (AttributeError, ValueError, TypeError) as error:
        raise PublicationError(f"Every {name} must preserve three finite saved coordinates") from error


def _source(cache_dir: str, sav_path: str) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    path, raw = _read_artifact(cache_dir, sav_path, suffixes=(".sav", ".plsav"), limit=MAX_SOURCE)
    data = _json(raw)
    experiment = data.get("Experiment") if isinstance(data, dict) else None
    summary = data.get("Summary") if isinstance(data, dict) else None
    kind = experiment.get("Type") if isinstance(experiment, dict) else None
    if not isinstance(experiment, dict) or type(kind) is not int or kind not in (0, 3):
        raise PublicationError("A complete original electrical Type-0 PLSAV or fixed Type-3 HDL source template is required")
    if type(data.get("Type")) is not int or data["Type"] != kind:
        raise PublicationError("Outer and Experiment PLSAV Type must be equal explicit integers")
    if not isinstance(summary, dict) or summary.get("ID") or summary.get("ContentID") or experiment.get("ID"):
        raise PublicationError("Only a newly generated local experiment can be published; existing IDs are forbidden")
    if summary.get("Price", 0) not in (None, 0) or summary.get("ParentID"):
        raise PublicationError("Only free, original experiments can use the test publication permits")
    if not isinstance(experiment.get("StatusSave"), str) or not isinstance(experiment.get("CameraSave"), str):
        raise PublicationError("Complete original StatusSave and CameraSave JSON strings are required")
    status, camera = _json(experiment["StatusSave"]), _json(experiment["CameraSave"])
    if not isinstance(status, dict) or not isinstance(camera, dict):
        raise PublicationError("Invalid saved experiment state")
    if kind == 3:
        required_status = hdl_source_carrier_status()
        required_camera = {"Mode": 2, "Distance": 2.75, "VisionCenter": "0,1.08,0",
                           "TargetRotation": "90,0,0"}
        if (status != required_status or camera != required_camera or experiment.get("Components") != 3
            or experiment.get("Version") != 2503 or summary.get("Type") != 3
            or summary.get("Tags") != ["Type-3", "高中", "教学实验"]
            or data.get("InternalName") != "Aurex HDL 源码载体"):
            raise PublicationError("Type-3 publication must use the exact fixed official astronomy HDL source carrier")
        return data, raw, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                           "elements": 3, "wires": 0, "experiment_type": 3,
                           "publication_mode": "hdl_source_celestial_fallback",
                           "interactive_circuit": False, "allowed_followup": "comments_only"}
    elements, wires = status.get("Elements"), status.get("Wires")
    if (not isinstance(elements, list) or not 1 <= len(elements) <= MAX_PUBLISHED_ELEMENTS
        or not isinstance(wires, list) or len(wires) > 200000):
        raise PublicationError(f"Saved state requires 1..{MAX_PUBLISHED_ELEMENTS} elements and a complete wire array")
    ids = set()
    for item in elements:
        if not isinstance(item, dict):
            raise PublicationError("Every element must be an object")
        ident = item.get("Identifier")
        if not isinstance(ident, str) or not ident or ident in ids or not item.get("ModelID") or not isinstance(item.get("Properties"), dict):
            raise PublicationError("Elements require unique identifiers, model names and saved properties")
        ids.add(ident)
        _vector(item.get("Position"), "Position")
        _vector(item.get("Rotation"), "Rotation")
    for wire in wires:
        if not isinstance(wire, dict) or wire.get("Source") not in ids or wire.get("Target") not in ids:
            raise PublicationError("Every wire must connect existing element identifiers")
        if any(type(wire.get(k)) is not int or wire[k] < 0 for k in ("SourcePin", "TargetPin")):
            raise PublicationError("Wire pin indices must be nonnegative integers")
    return data, raw, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                       "elements": len(elements), "wires": len(wires), "experiment_type": 0}


def inspect_publication_source(cache_dir: str, sav_path: str) -> dict[str, Any]:
    """Local deterministic preflight; does not publish or allocate a permit."""
    return _source(cache_dir, sav_path)[2]


def _chinese_text(value: Any, name: str, maximum: int, minimum: int) -> str:
    if not isinstance(value, str):
        raise PublicationError(f"{name} must be Chinese text")
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not minimum <= len(text) <= maximum or not _HAN.search(text):
        raise PublicationError(f"{name} must contain Chinese and have {minimum}..{maximum} characters")
    # Judge the language of the explanatory prose, not source code embedded in
    # fenced Markdown blocks. Oversized HDL publications deliberately carry a
    # complete Verilog appendix; counting it as English prose rejects valid text.
    prose = re.sub(r"(?ms)^```[^\n]*\n.*?^```[ \t]*$", "", text)
    han = len(_HAN.findall(prose))
    # Technical names and Markdown/code are welcome, but one Chinese character
    # must not make a long English publication satisfy the Chinese-only review.
    letters = han + len(re.findall(r"[A-Za-z]", prose))
    if han / max(1, letters) < 0.15:
        raise PublicationError(f"{name} must use Chinese prose, not an English publication with a token Chinese character")
    if any(ord(c) < 32 and c not in "\n\t" for c in text) or re.search(r"<[^>]+>", text):
        raise PublicationError(f"{name} must be plain Chinese/Markdown text without control or community markup")
    if name == "title" and ("\n" in text or "\t" in text):
        raise PublicationError("Title must be a single line")
    return text


def _without_fenced_code(text: str) -> str:
    """Remove model-authored code blocks before attaching trusted HDL source."""
    return re.sub(r"(?ms)^```[^\n]*\n.*?^```[ \t]*(?:\n|$)", "", text).strip()


def approve_publication(
    cache_dir: str, *, session_id: str, run_id: str, user_id: str, sav_path: str,
    title: str, introduction: str, evidence_paths: list[str], review: dict[str, Any],
    cover_path: str | None, cover_manifest: dict[str, Any] | None,
    trusted_appendix: str | None = None, user: Any = None,
) -> dict[str, Any]:
    """SERVER ONLY: bind deterministic server validation to immutable source bytes.

    review is retained as an internal compatibility parameter, but it now records
    deterministic validation rather than a second model call.
    Electrical Type-0 needs the trusted fixed-view cover. Oversized HDL Type-3 is
    text-only and must not have a cover upload or model-selected circuit state.
    """
    _identity(run_id, "run ID")
    _identity(user_id, "account ID")
    scope = publication_authorization(cache_dir, session_id, task_id=run_id)
    if scope["state"] not in ("authorized", "approved"):
        raise PublicationError("This session's one-shot publication permit is already used; resume the recorded receipt instead")
    source, source_bytes, source_info = _source(cache_dir, sav_path)
    title = _chinese_text(title, "title", 80, 1)
    if source_info["experiment_type"] == 3:
        # The execution model may quote the HDL it was given. Never publish that
        # untrusted duplicate: the appendix below is rebuilt from exact source
        # files whose hashes the server revalidated.
        introduction = _without_fenced_code(introduction)
    introduction = _chinese_text(introduction, "introduction", 16000, 20)
    if source_info["experiment_type"] == 3:
        if (not isinstance(trusted_appendix, str) or not trusted_appendix.startswith("## 已验证 HDL 源码\n")
            or any(ord(c) < 32 and c not in "\n\t" for c in trusted_appendix)):
            raise PublicationError("Type-3 HDL source fallback requires the server-verified full source appendix")
        introduction = introduction.rstrip() + "\n\n" + trusted_appendix.strip()
        if len(introduction) > 16000:
            raise PublicationError("Reviewed prose plus complete HDL source exceeds the 16000-character publication limit")
    elif trusted_appendix is not None:
        raise PublicationError("Electrical Type-0 publication cannot carry a Type-3 source appendix")
    mention = requester_mention(scope, user=user)
    if mention:
        introduction = mention + '：\n\n' + introduction.lstrip()
        if len(introduction) > 16000:
            raise PublicationError('Publication text plus its required requester mention exceeds 16000 characters')
    if (not isinstance(review, dict) or review.get("approved") is not True or review.get("server_validated") is not True
        or review.get("thinking") is not False
        or review.get('publish_requested') is not True
        or review.get("source_sha256") != source_info["sha256"]):
        raise PublicationError("Trusted deterministic server validation of this exact source hash is required")
    review_summary = _chinese_text(review.get("summary"), "validation summary", 8000, 5)
    if not isinstance(evidence_paths, list) or not 1 <= len(evidence_paths) <= 16 or not all(isinstance(p, str) for p in evidence_paths):
        raise PublicationError("Supply 1..16 local verification evidence artifacts")
    evidence = []
    for path in dict.fromkeys(evidence_paths):
        file, raw = _read_artifact(cache_dir, path, suffixes=(".json", ".md", ".txt"), limit=8 * 1024 * 1024)
        evidence.append({"path": str(file), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    cover_bytes = None
    if source_info["experiment_type"] == 3:
        if cover_path is not None or cover_manifest is not None:
            raise PublicationError("Type-3 HDL source fallback is title/body only; PLSAV screenshots and covers are forbidden")
        cover_info = None
    else:
        if not isinstance(cover_path, str):
            raise PublicationError("Electrical publication requires a fixed-view cover")
        file, cover_bytes = _read_artifact(cache_dir, cover_path, suffixes=(".jpg", ".jpeg"), limit=MAX_COVER)
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(cover_bytes))
            if image.format != "JPEG" or not (256 <= image.width <= 8192 and 256 <= image.height <= 8192):
                raise ValueError("not a supported JPEG cover")
            image.verify()
        except (ValueError, OSError) as error:
            raise PublicationError("The fixed-view system cover must be a valid 256..8192 pixel JPEG") from error
        cover_info = {"path": str(file), "sha256": hashlib.sha256(cover_bytes).hexdigest(), "bytes": len(cover_bytes)}
        expected_view = {"yaw": 45, "pitch": 60, "projection": "orthographic", "fit": "all"}
        if (not isinstance(cover_manifest, dict) or cover_manifest.get("source_sha256") != source_info["sha256"]
            or cover_manifest.get("cover_sha256") != cover_info["sha256"]
            or cover_manifest.get("rendered_all") is not True
            or type(cover_manifest.get("total_elements")) is not int
            or cover_manifest.get("total_elements") != source_info["elements"]
            or type(cover_manifest.get("visible_elements")) is not int
            or cover_manifest.get("visible_elements") != source_info["elements"]
            or cover_manifest.get("clipped_ids") != [] or cover_manifest.get("view") != expected_view):
            raise PublicationError("Cover must prove the fixed system view contains every source element without clipping")
    manifest = {"source": source_info, "cover": cover_info, "cover_manifest": cover_manifest,
        "title": title, "introduction": introduction, "evidence": evidence,
        "validation": {"approved": True, "server_validated": True, "thinking": False,
            'publish_requested': True, "source_sha256": source_info["sha256"], "summary": review_summary},
        "purpose": scope["purpose"], "category": "Experiment", "price": 0,
        "task_id": run_id, "binding_sha256": scope['binding_sha256'], 'requester_mention': mention}
    serialized = json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False)
    approval_id = uuid.uuid4().hex
    directory = _private_dir(cache_dir)
    # Private immutable snapshots avoid TOCTOU between validation and HTTP transmission.
    snapshots = [(".sav", source_bytes)]
    if cover_bytes is not None:
        snapshots.append((".jpg", cover_bytes))
    for suffix, data in snapshots:
        fd = os.open(directory / (approval_id + suffix), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    with _db(cache_dir) as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute("SELECT * FROM task_publications WHERE task_id=?", (run_id,)).fetchone()
        if old:
            if old["state"] != "approved":
                raise PublicationError("This session's one-shot publication permit is already used; never submit again")
            # Revoking an unsubmitted approval is safe; old receipt IDs stop working.
            db.execute("DELETE FROM task_publications WHERE task_id=?", (run_id,))
        now = time.time()
        db.execute("INSERT INTO task_publications VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            approval_id, run_id, session_id, run_id, user_id, "approved", serialized, None, None, None, now, now))
        db.execute("COMMIT")
    return {"approval_id": approval_id, "state": "approved", "session_id": session_id,
            "task_id": run_id,
            "source_sha256": source_info["sha256"], "cover_sha256": cover_info["sha256"] if cover_info else None,
            "category": "Experiment", "price": 0}


def _receipt(row: sqlite3.Row) -> dict[str, Any]:
    manifest = json.loads(row["manifest"])
    result = {"approval_id": row["approval_id"], "state": row["state"], "published": row["state"] == "published",
        "category": "Experiment", "price": 0, "summary_id": row["summary_id"],
        "source_sha256": manifest["source"]["sha256"],
        "cover_sha256": manifest["cover"]["sha256"] if manifest.get("cover") else None,
        "title": manifest["title"], "session_id": row["session_id"]}
    result['task_id'] = row['task_id']
    if row["summary_id"]:
        result["experiment_tag"] = f"<experiment={row['summary_id']}>{manifest['title']}</experiment>"
    if row["error"]:
        result["error"] = row["error"]
    return result


def publish_approved(
    cache_dir: str, *, session_id: str, run_id: str, user: Any, approval_id: str,
    check_cancel=None,
) -> dict[str, Any]:
    """SERVER ONLY: consume a validated approval; never re-create an uncertain submission."""
    user_id = getattr(unwrap_user(user), "user_id", None)
    with _operation_lock(cache_dir, approval_id), _db(cache_dir) as db:
        row = db.execute("SELECT * FROM task_publications WHERE approval_id=?", (approval_id,)).fetchone()
        if not row or row["session_id"] != session_id or row["run_id"] != run_id or row["user_id"] != user_id:
            raise PublicationError("Approval does not belong to this authenticated session, run and account")
        scope = publication_authorization(cache_dir, session_id, task_id=run_id)
        if row["state"] == "published":
            return _receipt(row)
        def update(state, *, summary_id=None, credential=None, error=None):
            db.execute("UPDATE task_publications SET state=?,summary_id=COALESCE(?,summary_id),credential=?,error=?,updated=? WHERE approval_id=?",
                (state, summary_id, json.dumps(credential) if credential is not None else None, error, time.time(), approval_id))

        def current():
            return db.execute("SELECT * FROM task_publications WHERE approval_id=?", (approval_id,)).fetchone()

        if row["state"] in ("submitting", "unknown"):
            update("unknown", error="提交结果不确定：禁止再次创建，请人工核对社区记录。")
            return _receipt(current())
        manifest = json.loads(row["manifest"])
        if manifest.get('binding_sha256') != scope['binding_sha256'] or manifest.get('task_id') != run_id:
            raise PublicationError('Publication approval is not bound to this exact original task')
        if scope['source'] == 'community':
            expected = manifest.get('requester_mention')
            if (not isinstance(expected, str) or not expected.startswith('<user=' + scope['requester_user_id'] + '>')
                or not manifest['introduction'].startswith(expected + '：\n')):
                raise PublicationError('Community publication must begin with the bound requester ID mention and a newline')
        directory = _private_dir(cache_dir)
        source_bytes = (directory / (approval_id + ".sav")).read_bytes()
        text_only = manifest["source"].get("experiment_type") == 3
        cover_bytes = None if text_only else (directory / (approval_id + ".jpg")).read_bytes()
        if (hashlib.sha256(source_bytes).hexdigest() != manifest["source"]["sha256"]
            or (not text_only and (not manifest.get("cover") or
                hashlib.sha256(cover_bytes).hexdigest() != manifest["cover"]["sha256"]))
            or (text_only and manifest.get("cover") is not None)):
            raise PublicationError("Approved immutable source/cover snapshot failed its integrity check")
        if row["state"] == "approved":
            # Reject source/evidence changes since validation rather than publishing an old design.
            artifacts = [(manifest["source"], (".sav", ".plsav"), MAX_SOURCE)]
            if not text_only:
                artifacts.append((manifest["cover"], (".jpg", ".jpeg"), MAX_COVER))
            artifacts.extend((item, (".json", ".md", ".txt"), 8 * 1024 * 1024) for item in manifest["evidence"])
            for item, suffixes, maximum in artifacts:
                _, raw = _read_artifact(cache_dir, item["path"], suffixes=suffixes, limit=maximum)
                if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                    raise PublicationError("Source, cover or verification evidence changed after approval; new validation is required")
            if callable(check_cancel):
                check_cancel()
            # Atomic CAS prevents a concurrently revoked approval from initiating an external write.
            changed = db.execute("UPDATE task_publications SET state='submitting',updated=? WHERE approval_id=? AND state='approved'",
                                 (time.time(), approval_id)).rowcount
            if changed != 1:
                raise PublicationError("Approval was revoked or consumed")
            try:
                submitted = plar_api.submit_original_experiment(user, source=_json(source_bytes), title=manifest["title"],
                    introduction=manifest["introduction"], cover_bytes=0 if text_only else len(cover_bytes))
            except Exception as error:
                detail = getattr(error, "safe_diagnostic", "")
                if not isinstance(detail, str) or not re.fullmatch(r"[A-Za-z0-9:_-]{1,120}", detail):
                    detail = type(error).__name__
                update("unknown", error="提交结果不确定：禁止再次创建，请人工核对社区记录。"
                    + " 安全诊断：" + detail)
                return _receipt(current())
            expected_counter = 0 if text_only else 1
            if submitted.get("image_counter") != expected_counter:
                update("unknown", summary_id=submitted["summary_id"], error="提交回执的图片计数不符合发布模式，禁止自动继续。")
                return _receipt(current())
            update("confirm_pending" if text_only else "image_pending", summary_id=submitted["summary_id"],
                   credential=submitted.get("_cover_credential", {}) if not text_only else None)
            row = current()
        if row["state"] == "image_pending":
            if callable(check_cancel):
                check_cancel()  # Submit receipt is durable; do not start another request after cancellation.
            credential = json.loads(row["credential"] or "{}")
            try:
                plar_api.upload_experiment_cover(user=user, cover=cover_bytes, credential=credential)
            except Exception:
                update("image_pending", credential=credential, error="封面上传未确认，未进行公开确认；重试仅上传同一个封面位置。")
                return _receipt(current())
            update("confirm_pending")  # Clear short-lived upload credentials after success.
            row = current()
        if row["state"] == "confirm_pending":
            if callable(check_cancel):
                check_cancel()  # Uploaded cover is durable, same-ID confirmation may resume explicitly.
            try:
                plar_api.confirm_original_experiment(user, summary_id=row["summary_id"], image_counter=0 if text_only else 1)
            except Exception:
                update("confirm_pending", error="发布确认暂未成功，重试仅确认已记录的同一实验，不会重新创建。")
                return _receipt(current())
            update("published")
        return _receipt(current())
