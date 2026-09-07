"""Deterministic validation between an agent publication request and the one-shot ledger.

This module does not create publication authority. User/agent requests, circuit files,
reports and descriptions are untrusted data, never instructions for validation.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

from plar.physicslab import unwrap_user
from . import publishing
from . import analog_evidence
from .verilog_lowering import MemoryLoweringError, lower_unpacked_register_arrays
from .tools import circuits
from .tools.registry import ToolError


def _fail(message: str) -> None:
    raise ToolError("发布校验：" + message)


def _evidence(cache_dir: str, paths: Any) -> list[dict[str, Any]]:
    if not isinstance(paths, list) or not 1 <= len(paths) <= 16 or not all(isinstance(p, str) for p in paths):
        _fail("需要1至16个本地验证证据文件。")
    records = []
    for requested in dict.fromkeys(paths):
        path, raw = publishing._read_artifact(cache_dir, requested, suffixes=(".json", ".md", ".txt"), limit=8 * 1024 * 1024)
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeError:
            _fail("验证证据必须是完整UTF-8文本。")
        value = publishing._json(text) if path.suffix.lower() == ".json" else None
        records.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw), "text": text, "value": value})
    if not any(isinstance(r["value"], dict) and r["value"].get("verified") is True for r in records):
        _fail("至少需要一份完整JSON验证报告，其中verified必须为true。")
    return records


def _hdl(cache_dir: str, source: dict[str, Any], records: list[dict[str, Any]], *, required_profile: str | None = None) -> dict[str, Any]:
    root = Path(cache_dir).resolve()
    candidates = [r for r in records if isinstance(r["value"], dict)
                  and r["value"].get("profile") in {"custom", "rv32i_teaching_v1"} and r["value"].get("verified") is True
                  and (required_profile is None or r["value"].get("profile") == required_profile)]
    if len(candidates) != 1:
        _fail("数字电路需要且只能有一份匹配实际任务的完整HDL通过报告。")
    record, report = candidates[0], candidates[0]["value"]
    path = Path(record["path"])
    if (path.name != "verification.json" or path.parent.parent != root / "hdl"
        or not path.parent.name.startswith("verification-")):
        _fail("RISC-V报告必须来自服务端hdl_simulate独立教学验证器。")
    if (not isinstance(report.get('top'), str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', report['top'])
        or report.get("report_path") != str(path)
        or (required_profile == 'rv32i_teaching_v1' and report['top'] != 'aurex_rv32i_teaching')):
        _fail("HDL验证报告的顶层或文件出处不一致。")
    for key in ("compile", "simulation"):
        result = report.get(key)
        if not isinstance(result, dict) or type(result.get("exit_code")) is not int or result["exit_code"] != 0 or result.get("failure"):
            _fail("RISC-V编译及独立测试必须完整成功，不能使用超时或中断结果。")
    hashes = report.get("source_files_sha256")
    if not isinstance(hashes, dict) or not 1 <= len(hashes) <= 16:
        _fail("RISC-V报告缺少完整RTL源文件哈希。")
    sources = []
    actual_hashes = {}
    for name, expected in sorted(hashes.items()):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,80}\.(?:v|sv)", name):
            _fail("RTL证据文件名无效。")
        _, raw = publishing._read_artifact(cache_dir, str(path.parent / name), suffixes=(".v", ".sv"), limit=512000)
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            _fail("RTL源文件在测试后发生变化，必须重新运行验证。")
        actual_hashes[name] = digest
        try:
            sources.append({"name": name, "sha256": digest, "text": raw.decode("utf-8")})
        except UnicodeError:
            _fail("RTL源文件不是UTF-8文本。")
    bundle = hashlib.sha256(json.dumps(actual_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if report.get("source_sha256") != bundle:
        _fail("RISC-V验证报告的RTL集合哈希不一致。")
    manifests = [r for r in records if isinstance(r["value"], dict)
                 and r["value"].get("schema") == "aurex.hdl-export.v1"
                 and r["value"].get("sav_sha256") == source["sha256"]]
    if len(manifests) != 1:
        _fail("缺少将当前PLSAV绑定到已验证RTL的唯一导出清单。")
    exported = manifests[0]["value"]
    if (exported.get("source_sha256") != bundle or exported.get("source_files_sha256") != actual_hashes
        or not isinstance(exported.get('top'), str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_$]*', exported['top'])
        or (required_profile == 'rv32i_teaching_v1' and exported['top'] != 'aurex_rv32i_teaching')
        or exported.get("strict_export") is not True
        or exported.get("sav_path") != source["path"]
        or manifests[0]["path"] != source["path"] + ".export.json"
        or exported.get("verification_report_path") != str(path)
        or exported.get("verification_id") != report.get("verification_id")
        or not isinstance(report.get("verification_id"), str) or not report["verification_id"]):
        _fail("待发布PLSAV不是由同一份已验证RTL导出。")
    fallback = exported.get("publication_fallback")
    if source.get("experiment_type") == 3:
        if (not isinstance(fallback, dict)
            or fallback.get("schema") != "aurex.hdl-source-celestial-fallback.v1"
            or fallback.get("reason") != "physical_gate_element_limit_exceeded"
            or fallback.get("max_direct_elements") != publishing.MAX_PUBLISHED_ELEMENTS
            or type(fallback.get("gate_elements")) is not int
            or fallback["gate_elements"] <= publishing.MAX_PUBLISHED_ELEMENTS
            or type(fallback.get("gate_wires")) is not int or fallback["gate_wires"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(fallback.get("discarded_gate_plsav_sha256") or ""))
            or fallback.get("template_type") != 3
            or fallback.get("interactive_circuit") is not False
            or fallback.get("allowed_followup") != "comments_only"):
            _fail("超限HDL发布缺少可信的5000原件阈值与只读天文模板记录。")
    elif source.get("experiment_type") == 0:
        if fallback is not None:
            _fail("未超过5000原件的电学PLSAV不应带有天文源码降级记录。")
    else:
        _fail("HDL发布源类型无效。")
    design_names = set(report.get("design_source_files") or actual_hashes)
    expected_export_sources = []
    expected_lowered_files = []
    expected_arrays = []
    for item in sources:
        text = item["text"]
        if item["name"] in design_names:
            try:
                text, lowered = lower_unpacked_register_arrays(text)
            except MemoryLoweringError as error:
                _fail("无法重放RTL数组降级：" + str(error))
            expected_arrays.extend({"source": item["name"], **entry} for entry in lowered)
            expected_export_sources.append(text)
        expected_lowered_files.append((item["name"], text))
    expected_export = "\n".join(expected_export_sources)
    if exported.get("export_verilog_sha256") != hashlib.sha256(expected_export.encode("utf-8")).hexdigest():
        _fail("PLSAV导出输入不能由原始已验证RTL确定性重建。")
    lowering = exported.get("export_lowering")
    if expected_arrays:
        if (not isinstance(lowering, dict) or lowering.get("schema") != "aurex.hdl-export-lowering.v1"
            or lowering.get("kind") != "fixed_unpacked_register_array_to_explicit_registers"
            or lowering.get("verified") is not True or lowering.get("arrays") != expected_arrays
            or lowering.get("profile") != report.get("profile")):
            _fail("寄存器数组导出缺少可重放且已复验的降级记录。")
        lower_path = Path(str(lowering.get("verification_report_path") or "")).resolve()
        if (lower_path.name != "verification.json" or lower_path.parent.parent != root / "hdl"
            or not lower_path.parent.name.startswith("verification-")):
            _fail("数组降级复验报告不在服务端HDL证据目录。")
        try:
            lowered_report = json.loads(lower_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _fail("数组降级复验报告无法读取。")
        lowered_hashes = {}
        for name, text in expected_lowered_files:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            lowered_hashes[name] = digest
            try:
                actual = (lower_path.parent / name).read_bytes()
            except OSError:
                _fail("数组降级复验缺少完整RTL文件。")
            if hashlib.sha256(actual).hexdigest() != digest:
                _fail("数组降级复验RTL不是原始RTL的确定性展开结果。")
        lowered_bundle = hashlib.sha256(json.dumps(lowered_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (lowered_report.get("verified") is not True or lowered_report.get("source_files_sha256") != lowered_hashes
            or lowered_report.get("source_sha256") != lowered_bundle or lowering.get("source_sha256") != lowered_bundle
            or lowering.get("verification_id") != lowered_report.get("verification_id")
            or lowered_report.get("report_path") != str(lower_path)):
            _fail("数组降级复验报告的结果或哈希不一致。")
        for key in ("compile", "simulation"):
            result = lowered_report.get(key)
            if not isinstance(result, dict) or result.get("exit_code") != 0 or result.get("failure"):
                _fail("数组降级后的RTL没有完整通过同一验证流程。")
    elif lowering is not None:
        _fail("无数组的RTL不应带有数组降级记录。")
    # Custom verification.top is the testbench; export.top is the DUT. They may
    # differ, but the DUT declaration must exist in the exact compiled source set.
    source_text = '\n'.join(item['text'] for item in sources)
    without_comments = re.sub(r'/\*.*?\*/|//[^\n]*', '', source_text, flags=re.S)
    if not re.search(r'\bmodule\s+(?:automatic\s+)?' + re.escape(exported['top']) + r'(?=\s|#|\(|;)', without_comments):
        _fail('导出的DUT顶层未出现在已验证的完整RTL源文件集合中。')
    public_appendix = None
    if fallback is not None:
        blocks = []
        for item in (item for item in sources if item["name"] in design_names):
            longest = max((len(value) for value in re.findall(r"`+", item["text"])), default=0)
            fence = "`" * max(3, longest + 1)
            language = "systemverilog" if item["name"].endswith(".sv") else "verilog"
            blocks.append(f"### `{item['name']}`\n\n{fence}{language}\n{item['text'].rstrip()}\n{fence}")
        public_appendix = ("## 已验证 HDL 源码\n\n"
            f"> 门级展开为 {fallback['gate_elements']} 个原件，超过发布上限 {fallback['max_direct_elements']}。"
            "因此本帖使用固定空白天文模板，仅在正文公开已验证设计源码；不包含可操作电路存档或截图，后续只能评论。\n\n"
            + "\n\n".join(blocks))
        if len(public_appendix) > 14000:
            _fail("HDL源码超过单篇正文可安全发布的长度，不能截断后冒充完整源码。")
    return {"kind": "riscv_teaching_subset" if required_profile else 'digital_electrical_experiment',
            'profile': report['profile'], 'testbench_top': report['top'], 'export_top': exported['top'],
            "source_bundle_sha256": bundle, "rtl_sources": sources,
            "publication_fallback": fallback, "required_public_appendix": public_appendix,
            "limitation": ("Custom testbench was supplied with the design: successful compilation/simulation proves only those executed checks, "
                "not independent coverage, universal correctness, synthesis equivalence, or a full CPU compliance claim. "
                "Review the original user's goal, actual DUT, complete testbench and observed exit status independently." if report['profile'] == 'custom' else
                "Trusted teaching-profile simulation; not complete RV32I compliance or a proof of RTL-to-PLSAV equivalence.")}


def _riscv(cache_dir: str, source: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    return _hdl(cache_dir, source, records, required_profile='rv32i_teaching_v1')


def _analog_task(cache_dir: str, source: dict[str, Any], records: list[dict[str, Any]], *, required_stop_s: float | None = .5) -> dict[str, Any]:
    candidates = [r for r in records if isinstance(r['value'], dict) and r['value'].get('schema') == analog_evidence.SCHEMA]
    if len(candidates) != 1:
        _fail('555任务需要唯一原生模拟证据报告，不能仅凭通用verified字段发布。')
    try:
        checked = analog_evidence.validate_report(cache_dir, candidates[0]['path'])
    except (ValueError, OSError) as error:
        _fail('模拟原始证据复验失败：' + str(error))
    report, samples = checked['report'], checked['samples']
    transient = report.get('transient')
    if required_stop_s is not None and (report.get('analysis') not in ('tr', 'trop') or not isinstance(transient, dict)
        or not math.isclose(transient['actual_stop_s'], required_stop_s, rel_tol=1e-12, abs_tol=1e-15)
        or not isinstance(samples, list) or len(samples) < 2):
        _fail('需要完整运行至0.5秒的真实瞬态序列，不能只提供静态或末端点状态。')
    if report['sav']['path'] != source['path'] or report['sav']['sha256'] != source['sha256']:
        _fail('实际求解证据与当前待发布PLSAV不是同一源文件。')
    # Mode-neutral numerical features: no target topology, oscillator requirement,
    # timing formula or functional success is invented by this validator.
    nodes = {}
    points = samples or [{'time_s': None, 'rows': report['rows']}]
    for sample in points:
        seen = set()
        for row in sample['rows']:
            for node, volts in zip(row['nodes'], row['voltage_v']):
                if node not in seen:
                    nodes.setdefault(node, []).append(volts)
                    seen.add(node)
    statistics = []
    times = [sample['time_s'] for sample in points]
    for node, values in sorted(nodes.items()):
        low, high = min(values), max(values)
        midpoint = (low + high) / 2
        crossings = [{'from_s': times[i - 1], 'to_s': times[i],
                      'direction': 'up' if values[i] >= midpoint else 'down'}
                     for i in range(1, len(values)) if high > low and
                     ((values[i - 1] < midpoint <= values[i]) or (values[i - 1] >= midpoint > values[i]))]
        statistics.append({'node': node, 'first_v': values[0], 'final_v': values[-1], 'min_v': low,
            'max_v': high, 'range_midpoint_v': midpoint, 'midpoint_crossing_brackets': crossings})
    return {'kind': '555_state_table' if required_stop_s is not None else 'analog_electrical_experiment', 'generic_numerical_and_export_verified': True,
        'functional_verification': False, 'complete_native_spec': checked['spec'],
        'actual_stop_s': transient['actual_stop_s'] if transient else None, 'sample_count': len(samples or []),
        'sample_times_s': times if samples else [], 'sampled_node_statistics': statistics,
        'statistics_scope': 'Actual recorded transient samples' if samples else 'Final state only; no waveform sequence was recorded',
        'required_final_table': checked['final_table'],
        'omit_raw_evidence_paths': [report['state']['path'], report.get('trace_table', {}).get('path')],
        'limitation': 'Every raw sample and source hash was structurally/numerically revalidated by code. '
                     'The reviewer sees complete spec, final states and sample statistics, NOT every raw waveform point. '
                     'Midpoint crossings describe only adjacent recorded samples, not exact event times or functional proof. '
                     'Independently judge the claimed operating mode and user goal; insufficient evidence requires rejection. '
                     'The server appends the canonical final-state table to the public introduction; do not duplicate it.'}


def _electrical_task(cache_dir: str, source: dict, records: list[dict]) -> dict:
    analog = [r for r in records if isinstance(r['value'], dict) and r['value'].get('schema') == analog_evidence.SCHEMA]
    if len(analog) == 1:
        return _analog_task(cache_dir, source, records, required_stop_s=None)
    if any(isinstance(r['value'], dict) and r['value'].get('profile') in {'custom', 'rv32i_teaching_v1'} for r in records):
        return _hdl(cache_dir, source, records)
    _fail('缺少与当前源文件绑定的受支持机器验证证据，通用verified标记不能替代实际验证。')


# Server-side hooks only. The model cannot register a validator or choose its purpose.
PURPOSE_VALIDATORS: dict[str, Callable] = {"electrical_experiment": _electrical_task,
    "riscv_teaching_subset": _riscv, "555_state_table": _analog_task}


def _check_cancel(runtime) -> None:
    check = getattr(runtime, 'check_cancel', None)
    if callable(check):
        check()


def review_and_publish(runtime, args: dict[str, Any], client, db, session_id: str, run_id: str, emit) -> dict[str, Any]:
    """Validate and publish without a second model-based publication review.

    The legacy function name is kept so existing sessions and call sites remain
    compatible. Authorization, evidence hashes, purpose-specific verification,
    fixed covers and the one-shot ledger remain enforced by server code.
    """
    try:
        return _review_and_publish(runtime, args, client, db, session_id, run_id, emit)
    except (publishing.PublicationError, OSError, UnicodeError) as error:
        raise ToolError("发布校验未通过：" + str(error)) from error


def _review_and_publish(runtime, args, client, db, session_id, run_id, emit):
    _check_cancel(runtime)
    if not isinstance(args, dict):
        _fail("工具参数必须是对象。")
    with_image = args.get("with_image", False)
    if type(with_image) is not bool:
        _fail("with_image必须是布尔值。")
    if runtime.task_id != run_id or getattr(runtime, "session_id", session_id) not in (None, session_id):
        _fail("服务端会话或运行标识不匹配。")
    user = runtime.user
    user_id = getattr(unwrap_user(user), "user_id", None) if user else None
    if not isinstance(user_id, str) or not user_id:
        _fail("需要已登录的社区账号。")
    scope = publishing.publication_authorization(runtime.cache_dir, session_id, task_id=runtime.task_id)
    if publishing.runtime_dry_run(runtime, scope):
        _fail('当前任务为dry-run，不执行外部发布。')
    publishing.requester_mention(scope, user=user)  # Missing community identity fails before cover/model work.
    if scope["state"] not in ("authorized", "approved"):
        if not scope.get("approval_id") or not scope.get("run_id"):
            _fail("持久化发布回执不完整，需要人工核对。")
        emit("publication_resume", {"approval_id": scope["approval_id"], "state": scope["state"], "original_run_id": scope["run_id"]})
        _check_cancel(runtime)
        result = publishing.publish_approved(runtime.cache_dir, session_id=session_id, run_id=scope["run_id"],
                                            user=user, approval_id=scope["approval_id"],
                                            check_cancel=getattr(runtime, 'check_cancel', None))
        emit("publication_receipt", result)
        return result
    required = {"sav_path", "title", "introduction", "evidence_paths"}
    if not required.issubset(args) or set(args) - required - {"with_image"}:
        _fail("只接受sav_path、title、introduction、evidence_paths及可选with_image；封面和权限不可由模型指定。")
    title = publishing._chinese_text(args["title"], "title", 80, 1)
    introduction = publishing._chinese_text(args["introduction"], "introduction", 16000, 20)
    saved, raw_source, source = publishing._source(runtime.cache_dir, args["sav_path"])
    records = _evidence(runtime.cache_dir, args["evidence_paths"])
    validator = PURPOSE_VALIDATORS.get(scope["purpose"])
    if validator is None:
        _fail("没有适用于该授权任务的专用证据验证器。")
    checks = validator(runtime.cache_dir, source, records)
    text_only = source.get("experiment_type") == 3
    if text_only and with_image:
        _fail("超限HDL天文模板只允许标题和正文，不能生成、读取或上传截图。")
    # Archive full originals for audit. No publication-review model is called.
    source_doc = db.document(session_id, "Publication original PLSAV " + source["sha256"], raw_source.decode("utf-8-sig"))
    for record in records:
        record["document_id"] = db.document(session_id, "Publication evidence " + Path(record["path"]).name, record["text"])
    cover_path = cover_manifest = None
    cover = {"images": []}
    if not text_only:
        cover = circuits.publication_cover(runtime, source["path"])
        cover_path, cover_manifest = cover["cover_path"], cover["cover_manifest"]
        if cover_manifest.get("source_sha256") != source["sha256"]:
            _fail("系统封面与发布源文件哈希不一致。")
        _, image = publishing._read_artifact(runtime.cache_dir, cover_path, suffixes=(".jpg", ".jpeg"), limit=publishing.MAX_COVER)
        if hashlib.sha256(image).hexdigest() != cover_manifest.get("cover_sha256"):
            _fail("系统封面字节不匹配其生成清单。")
        aid = db.artifact(session_id, cover_path, "image/jpeg", "系统固定视角发布封面（全部元件）")
        emit("artifact", {"id": aid, "path": cover_path, "mime_type": "image/jpeg", "url": "/api/artifacts/" + aid,
                          "label": "系统固定视角发布封面（全部元件）"})
    table = checks.get('required_final_table')
    if table and table not in introduction:
        introduction = publishing._chinese_text(introduction + '\n\n' + table, 'introduction', 16000, 20)
    # Revalidate after cover generation so slow local rendering cannot introduce
    # a source/evidence race before the immutable publication snapshot is made.
    _, _, after_source = publishing._source(runtime.cache_dir, source["path"])
    after_records = _evidence(runtime.cache_dir, args["evidence_paths"])
    if after_source["sha256"] != source["sha256"] or [(r["path"], r["sha256"]) for r in after_records] != [(r["path"], r["sha256"]) for r in records]:
        _fail("源文件或验证证据在发布校验期间发生变化，需要重新校验。")
    validator(runtime.cache_dir, after_source, after_records)
    _check_cancel(runtime)
    approval = publishing.approve_publication(runtime.cache_dir, session_id=session_id, run_id=run_id, user_id=user_id,
        sav_path=source["path"], title=title, introduction=introduction,
        evidence_paths=args["evidence_paths"], review={"approved": True, "server_validated": True,
            "thinking": False, "publish_requested": True, "source_sha256": source["sha256"],
            "summary": "服务端已校验发布授权、源文件哈希、验证证据、原件上限与固定封面规则。"},
        cover_path=cover_path, cover_manifest=cover_manifest,
        trusted_appendix=checks.get("required_public_appendix"), user=user)
    emit("publication_validated", {"source_sha256": source["sha256"], "purpose": scope["purpose"],
        "source_document_id": source_doc, "model_review": False, "cover_pixels_in_context": False})
    _check_cancel(runtime)
    result = publishing.publish_approved(runtime.cache_dir, session_id=session_id, run_id=run_id,
                                         user=user, approval_id=approval["approval_id"],
                                         check_cancel=getattr(runtime, 'check_cancel', None))
    emit("publication_receipt", result)
    # Images remain available to the tracking UI/artifact archive. This is not
    # permission to attach them to the execution agent's next model request.
    return {**result, "images": cover["images"], "cover_pixels_in_context": False,
            "publication_review": False}
