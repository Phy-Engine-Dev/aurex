"""Bounded, isolated HDL simulation; the teaching profile is not full ISA certification."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any

from .registry import ToolError, ToolRegistry, ToolRuntime, ToolSpec

PROFILE = "rv32i_teaching_v1"
CONTRACT = """module aurex_rv32i_teaching(input clk,rst, output [31:0] imem_addr,
input [31:0] imem_rdata, output dmem_we, output [31:0] dmem_addr,dmem_wdata,
input [31:0] dmem_rdata, output halted,trap, input [4:0] debug_reg_addr,
output [31:0] debug_reg_data);
Harvard asynchronous reads, writes commit on rising clk. Synchronous active-high
reset: PC=0, all 32 registers=0; x0 is always zero. Supported: ADDI ADD SUB LW SW
BEQ JAL EBREAK, 32-bit wrapping arithmetic, byte-addressed PC/memory, aligned
32-bit LW/SW only. EBREAK latches halted until reset. Unsupported instructions
and misaligned LW/SW latch trap. Debug read is combinational. No complete RV32I,
pipeline, interrupt, privilege, CSR, ABI, synthesis, or hardware timing claim.
"""
_SYSTEM = {"display", "write", "fatal", "error", "warning", "info", "finish",
           "time", "realtime", "signed", "unsigned", "clog2", "bits", "size",
           "isunknown", "countones", "onehot", "onehot0"}
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_TIMESCALE = re.compile(r"`timescale[ \t]+(?:1|10|100)[ \t]*(?:s|ms|us|ns|ps|fs)[ \t]*/[ \t]*(?:1|10|100)[ \t]*(?:s|ms|us|ns|ps|fs)[ \t]*(?=\r?\n|$)")
_CUSTOM_FAILURE_LINES = re.compile(
    r'(?im)^\s*(?:fail(?:ed)?\s*(?::|\b)|some\s+tests\s+failed\b)')
_CUSTOM_NONZERO_ERRORS = re.compile(
    r'(?im)^\s*(?:total\s+)?errors?\s*[:=]\s*([1-9][0-9]*)\b')
_CUSTOM_ICARUS_ERROR_LINES = re.compile(
    r'(?m)^ERROR:\s+/work/[A-Za-z][A-Za-z0-9_-]{0,80}\.(?:v|sv):[0-9]+:')
_LIMIT_LAUNCH = """import os,resource,sys
resource.setrlimit(resource.RLIMIT_AS,(1073741824,1073741824))
resource.setrlimit(resource.RLIMIT_CPU,(20,21))
resource.setrlimit(resource.RLIMIT_FSIZE,(16777216,16777216))
resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.execv(sys.argv[1],sys.argv[1:])
"""


def validate_source(source: str) -> None:
    """Lex untrusted HDL: strings/comments are data, macros/escaped IDs are forbidden."""
    if not isinstance(source, str) or not source.strip() or len(source.encode()) > 256_000:
        raise ToolError("Each HDL source must be nonempty UTF-8 text, at most 256000 bytes")
    if "\x00" in source:
        raise ToolError("NUL is forbidden in HDL")
    i = 0
    while i < len(source):
        if source.startswith("//", i):
            j = source.find("\n", i)
            i = len(source) if j < 0 else j + 1
        elif source.startswith("/*", i):
            j = source.find("*/", i + 2)
            if j < 0:
                raise ToolError("Unterminated HDL comment")
            i = j + 2
        elif source[i] == '"':
            i += 1
            while i < len(source) and source[i] != '"':
                if source[i] == "\\":
                    i += 1
                i += 1
            if i >= len(source):
                raise ToolError("Unterminated HDL string")
            i += 1
        elif source[i] == "`":
            match = _TIMESCALE.match(source, i)
            if not match:
                raise ToolError("Only a literal `timescale directive is allowed; macros/includes are forbidden")
            i = match.end()
        elif source[i] == "\\":
            raise ToolError("Escaped HDL identifiers are forbidden")
        elif source[i] == "$":
            match = _IDENT.match(source, i + 1)
            if not match or match.group() not in _SYSTEM:
                raise ToolError("File/process/plugin or unknown HDL system calls are forbidden")
            i = match.end()
        elif (match := _IDENT.match(source, i)):
            if match.group() in {"import", "export", "bind"} or match.group().startswith("aurex_profile"):
                raise ToolError("DPI, bind and reserved verifier identifiers are forbidden")
            i = match.end()
        else:
            i += 1


def _i(op: int, rd: int, rs1: int, imm: int, funct: int = 0) -> int:
    return ((imm & 4095) << 20) | (rs1 << 15) | (funct << 12) | (rd << 7) | op


def _r(rd: int, rs1: int, rs2: int, sub: bool = False) -> int:
    return (int(sub) << 30) | (rs2 << 20) | (rs1 << 15) | (rd << 7) | 0x33


def _sw(rs2: int, rs1: int, imm: int) -> int:
    return ((imm & 0xfe0) << 20) | (rs2 << 20) | (rs1 << 15) | (2 << 12) | ((imm & 31) << 7) | 0x23


def _beq(rs1: int, rs2: int, imm: int) -> int:
    return ((imm & 4096) << 19) | ((imm & 0x7e0) << 20) | (rs2 << 20) | (rs1 << 15) | ((imm & 30) << 7) | ((imm & 2048) >> 4) | 0x63


def _jal(rd: int, imm: int) -> int:
    return ((imm & 0x100000) << 11) | ((imm & 0x7fe) << 20) | ((imm & 0x800) << 9) | (imm & 0xff000) | (rd << 7) | 0x6f


def profile_vectors() -> list[dict[str, Any]]:
    """Known instruction vectors/expected values, not a model implementation."""
    p = [_i(0x13, r, 0, r * 7 - 70) for r in range(1, 32)]
    regs = [0] + [(r * 7 - 70) & 0xffffffff for r in range(1, 32)]
    p += [_i(0x13, 0, 0, 99), _i(0x13, 1, 0, 256), _i(0x13, 2, 0, -17), _i(0x13, 3, 0, 9),
          _r(4, 2, 3), _r(5, 3, 2, True), _sw(4, 1, 0), _sw(5, 1, 4),
          _i(3, 6, 1, 0, 2), _i(3, 7, 1, 4, 2), _beq(6, 4, 8), _i(0x13, 8, 0, 777),
          _i(0x13, 8, 0, -100), _beq(6, 7, 8), _i(0x13, 9, 0, 123)]
    link = (len(p) + 1) * 4
    p += [_jal(10, 8), _i(0x13, 11, 0, 777), _i(0x13, 12, 10, 4),
          _i(0x13, 13, 0, 0), _i(0x13, 14, 0, 3), _i(0x13, 13, 13, 1),
          _beq(13, 14, 8), _jal(0, -8), 0x00100073]
    for r, value in {1: 256, 2: -17, 3: 9, 4: -8, 5: 26, 6: -8, 7: 26, 8: -100,
                     9: 123, 10: link, 12: link + 4, 13: 3, 14: 3}.items():
        regs[r] = value & 0xffffffff
    backwards = [_i(0x13, 1, 0, 1), _i(0x13, 2, 0, 1), _jal(3, 12),
                 _i(0x13, 1, 1, 1), _jal(0, 8), _beq(1, 2, -8), 0x00100073]
    backwards_regs = [0, 2, 1, 12] + [0] * 28
    return [
        {"name": "arithmetic_registers_memory_branches_jal", "program": p, "regs": regs,
         "memory": {64: 0xfffffff8, 65: 26}, "trap": False},
        {"name": "negative_branch_offset", "program": backwards, "regs": backwards_regs, "memory": {}, "trap": False},
        {"name": "unsupported_instruction", "program": [0xffffffff], "trap": True},
        {"name": "misaligned_load", "program": [_i(3, 1, 0, 2, 2)], "trap": True},
        {"name": "misaligned_store", "program": [_sw(0, 0, 2)], "trap": True},
    ]


def _profile_testbench(nonce: str) -> str:
    cases = []
    for index, case in enumerate(profile_vectors()):
        lines = [f"// {case['name']}", "@(negedge clk); rst=1;", "repeat(3) @(negedge clk);",
                 "for(i=0;i<256;i=i+1) begin rom[i]=32'h00100073; mem[i]=0; end"]
        lines += [f"rom[{i}]=32'h{v:08x};" for i, v in enumerate(case["program"])]
        loop_checks = "if($isunknown(imem_addr) || $isunknown(halted) || $isunknown(trap)) begin aurex_profile_diagnostics; $fatal(1,\"case %0d: control became X at cycle %0d; inspect the first bad instruction and preceding state update\",aurex_profile_case,cycles); end"
        if not case["trap"]:
            loop_checks += " if(trap===1'b1) begin aurex_profile_diagnostics; $fatal(1,\"case %0d: unexpected trap in non-trap program at cycle %0d after executing pc=%08h instruction=%08h; inspect preceding control flow and decoder, and do not weaken trap or EBREAK handling\",aurex_profile_case,cycles,aurex_profile_executed_pc,aurex_profile_executed_instruction); end"
        lines += [f"aurex_profile_case={index}; aurex_profile_trace_count=0; aurex_profile_trace_next=0;",
                  "rst=0; cycles=0;", "while(halted!==1'b1 && trap!==1'b1 && cycles<512) begin @(negedge clk); cycles=cycles+1; aurex_profile_record; " + loop_checks + " end"]
        if case["trap"]:
            lines += [f"if(trap!==1'b1) begin aurex_profile_diagnostics; $fatal(1,\"case {index}: expected trap\"); end"]
        else:
            lines += [f"if(halted!==1'b1 || trap!==1'b0) begin aurex_profile_diagnostics; $fatal(1,\"case {index}: expected EBREAK halt without trap\"); end"]
            for r, value in enumerate(case["regs"]):
                lines += [f"debug_reg_addr=5'd{r}; #1; if(debug_reg_data!==32'h{value:08x}) begin aurex_profile_diagnostics; $fatal(1,\"case {index}: x{r} got %h expected {value:08x}\",debug_reg_data); end"]
            lines += [f"if(mem[{addr}]!==32'h{val:08x}) begin aurex_profile_diagnostics; $fatal(1,\"case {index}: stored memory mismatch word_index={addr} got %h expected {val:08x}\",mem[{addr}]); end" for addr, val in case["memory"].items()]
        lines += [f'$display("AUREX_CASE {index} {case["name"]} PASS");']
        cases.append("\n".join(lines))
    return """`timescale 1ns/1ps
module aurex_profile_tb;
reg clk=0,rst=1; always #5 clk=~clk;
wire [31:0] imem_addr,dmem_addr,dmem_wdata,debug_reg_data;
wire dmem_we,halted,trap;
reg [4:0] debug_reg_addr=0;
reg [31:0] rom[0:255],mem[0:255];
wire [31:0] imem_rdata=(imem_addr[1:0]==0 && imem_addr<1024)?rom[imem_addr>>2]:32'hffffffff;
wire [31:0] dmem_rdata=(dmem_addr[1:0]==0 && dmem_addr<1024)?mem[dmem_addr>>2]:32'hxxxxxxxx;
aurex_rv32i_teaching dut(clk,rst,imem_addr,imem_rdata,dmem_we,dmem_addr,dmem_wdata,dmem_rdata,halted,trap,debug_reg_addr,debug_reg_data);
always @(posedge clk) if(!rst && dmem_we===1'b1 && dmem_addr[1:0]==0 && dmem_addr<1024) mem[dmem_addr>>2]<=dmem_wdata;
integer i,cycles;
// Diagnostics only: read interfaces at the existing loop's falling-edge
// checkpoint, after the preceding rising-edge update. Do not add DUT clocks,
// delays, writes or assertions. These are current fetch/bus values, not an
// instruction-retirement or committed-store trace. Keep only the last 16 rows.
integer aurex_profile_case=-1;
integer aurex_profile_trace_count=0,aurex_profile_trace_next=0;
reg [31:0] aurex_profile_executed_pc=0,aurex_profile_executed_instruction=0;
always @(posedge clk) if(!rst) begin
  aurex_profile_executed_pc<=imem_addr;
  aurex_profile_executed_instruction<=imem_rdata;
end
integer aurex_profile_trace_cycle[0:15];
time aurex_profile_trace_time[0:15];
reg [31:0] aurex_profile_trace_pc[0:15],aurex_profile_trace_instr[0:15];
reg [31:0] aurex_profile_trace_addr[0:15],aurex_profile_trace_wdata[0:15],aurex_profile_trace_rdata[0:15];
reg aurex_profile_trace_rst[0:15],aurex_profile_trace_halt[0:15],aurex_profile_trace_trap[0:15],aurex_profile_trace_we[0:15];
task aurex_profile_record;
begin
  aurex_profile_trace_cycle[aurex_profile_trace_next]=cycles;
  aurex_profile_trace_time[aurex_profile_trace_next]=$time;
  aurex_profile_trace_pc[aurex_profile_trace_next]=imem_addr;
  aurex_profile_trace_instr[aurex_profile_trace_next]=imem_rdata;
  aurex_profile_trace_addr[aurex_profile_trace_next]=dmem_addr;
  aurex_profile_trace_wdata[aurex_profile_trace_next]=dmem_wdata;
  aurex_profile_trace_rdata[aurex_profile_trace_next]=dmem_rdata;
  aurex_profile_trace_rst[aurex_profile_trace_next]=rst;
  aurex_profile_trace_halt[aurex_profile_trace_next]=halted;
  aurex_profile_trace_trap[aurex_profile_trace_next]=trap;
  aurex_profile_trace_we[aurex_profile_trace_next]=dmem_we;
  aurex_profile_trace_next=(aurex_profile_trace_next+1)%16;
  if(aurex_profile_trace_count<16) aurex_profile_trace_count=aurex_profile_trace_count+1;
end
endtask
task aurex_profile_diagnostics;
integer aurex_profile_n,aurex_profile_slot;
begin
  $display("AUREX_PROFILE_TRACE case=%0d rows=%0d max_rows=16 sample=existing_negedge_loop_checkpoint time_unit=ns clock_period_ns=10 pc_instruction=current_fetch dmem=current_interface_not_committed_store",aurex_profile_case,aurex_profile_trace_count);
  for(aurex_profile_n=0;aurex_profile_n<aurex_profile_trace_count;aurex_profile_n=aurex_profile_n+1) begin
    aurex_profile_slot=(aurex_profile_trace_next-aurex_profile_trace_count+aurex_profile_n+16)%16;
    $display("AUREX_PROFILE_SAMPLE case=%0d cycle=%0d time_ns=%0d rst=%b pc=%08h instruction=%08h trap=%b halted=%b dmem_we=%b dmem_addr=%08h dmem_wdata=%08h dmem_rdata=%08h",aurex_profile_case,aurex_profile_trace_cycle[aurex_profile_slot],aurex_profile_trace_time[aurex_profile_slot],aurex_profile_trace_rst[aurex_profile_slot],aurex_profile_trace_pc[aurex_profile_slot],aurex_profile_trace_instr[aurex_profile_slot],aurex_profile_trace_trap[aurex_profile_slot],aurex_profile_trace_halt[aurex_profile_slot],aurex_profile_trace_we[aurex_profile_slot],aurex_profile_trace_addr[aurex_profile_slot],aurex_profile_trace_wdata[aurex_profile_slot],aurex_profile_trace_rdata[aurex_profile_slot]);
  end
  $display("AUREX_PROFILE_CURRENT case=%0d cycle=%0d time_ns=%0d sample=immediately_before_failed_assertion rst=%b pc=%08h instruction=%08h trap=%b halted=%b dmem_we=%b dmem_addr=%08h dmem_wdata=%08h dmem_rdata=%08h debug_reg_addr=%0d debug_reg_data=%08h",aurex_profile_case,cycles,$time,rst,imem_addr,imem_rdata,trap,halted,dmem_we,dmem_addr,dmem_wdata,dmem_rdata,debug_reg_addr,debug_reg_data);
end
endtask
initial begin
""" + "\n".join(cases) + f'\n$display("AUREX_VERIFIED_{nonce}"); $finish;\nend\ninitial begin #100000; aurex_profile_diagnostics; $fatal(1,"profile simulation deadline"); end\nendmodule\n'


def _toolchain() -> tuple[Path, str]:
    override = os.environ.get("AUREX_IVERILOG_ROOT")
    candidates = [Path(override)] if override else []
    candidates += [Path(__file__).resolve().parents[3] / ".aurex/toolchains/iverilog-12.0-3/root", Path("/")]
    for root in candidates:
        if (root / "usr/bin/iverilog").is_file() and (root / "usr/bin/vvp").is_file():
            for lib in ("usr/lib/x86_64-linux-gnu/ivl", "usr/lib/ivl"):
                if (root / lib / "ivl").is_file():
                    return root.resolve(), lib
    raise ToolError("Icarus Verilog is unavailable; install iverilog or set operator AUREX_IVERILOG_ROOT to a locally extracted package root")


def _sandbox(root: Path, work: Path, argv: list[str]) -> list[str]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise ToolError("bubblewrap is required; refusing unsandboxed HDL execution")
    command = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
               "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib"]
    if Path("/lib64").exists():
        command += ["--ro-bind", "/lib64", "/lib64"]
    command += ["--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                "--dir", "/toolchain", "--ro-bind", str(root / "usr"), "/toolchain/usr",
                "--bind", str(work), "/work", "--chdir", "/work", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
                "--setenv", "HOME", "/tmp", "--setenv", "LANG", "C.UTF-8", "--", *argv]
    return [sys.executable, "-c", _LIMIT_LAUNCH, *command]


def _run(command: list[str], timeout: float = 30, max_output: int = 65536) -> dict[str, Any]:
    start = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    output = bytearray()
    failure = None
    try:
        while selector.get_map():
            if time.monotonic() - start > timeout:
                failure = "timeout"
                break
            for key, _ in selector.select(.1):
                block = os.read(key.fileobj.fileno(), 4096)
                if not block:
                    selector.unregister(key.fileobj)
                else:
                    output.extend(block)
                    if len(output) > max_output:
                        failure = "output_limit"
                        break
            if failure:
                break
    finally:
        if not failure and process.poll() is None:
            try:
                process.wait(timeout=max(.01, timeout - (time.monotonic() - start)))
            except subprocess.TimeoutExpired:
                failure = "timeout"
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=5)
        selector.close()
        process.stdout.close()
    return {"exit_code": process.returncode, "failure": failure, "elapsed_seconds": round(time.monotonic() - start, 3),
            "log": output[:max_output].decode("utf-8", "replace"), "log_truncated": len(output) > max_output}


def hdl_simulate(runtime: ToolRuntime, args: dict[str, Any]) -> dict[str, Any]:
    profile = args.get("profile", "custom")
    if profile not in ("custom", PROFILE):
        raise ToolError("Unknown HDL verification profile")
    workspace, snapshot = None, None
    if 'workspace_id' in args:
        if 'files' in args or 'workspace_revision' not in args:
            raise ToolError('Use a pinned workspace_id/workspace_revision OR inline files, not both')
        from .hdl_workspace import workspace_snapshot
        workspace, snapshot = workspace_snapshot(runtime, args['workspace_id'], args['workspace_revision'])
        files = [{'name': name, 'content': item['content']} for name, item in sorted(snapshot.items())]
    else:
        if 'workspace_revision' in args or 'design_top' in args:
            raise ToolError('workspace_revision/design_top require workspace_id')
        files = args.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= 16:
        raise ToolError("files must contain 1..16 HDL files")
    sources = {}
    for file in files:
        if not isinstance(file, dict) or set(file) != {"name", "content"}:
            raise ToolError("Each file needs exactly name and content")
        name = file["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,80}\.(?:v|sv)", name) or name in sources or name.startswith("aurex_profile"):
            raise ToolError("HDL filenames must be unique flat .v/.sv names, no paths or dot traversal")
        validate_source(file["content"])
        sources[name] = file["content"]
    if sum(len(s.encode()) for s in sources.values()) > 512_000:
        raise ToolError("Total HDL input exceeds 512000 bytes")
    top = args.get("top", "")
    if profile == PROFILE:
        if top and top != "aurex_rv32i_teaching":
            raise ToolError("Teaching profile requires top aurex_rv32i_teaching")
        top = "aurex_profile_tb"
    elif not isinstance(top, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,80}", top):
        raise ToolError("custom simulation requires a valid top module name")
    design_top = args.get('design_top')
    if design_top is not None and (profile != 'custom' or not isinstance(design_top, str)
            or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,80}', design_top)):
        raise ToolError('design_top is a valid design module name for a custom workspace testbench only')
    root, lib = _toolchain()
    base = Path(runtime.cache_dir).resolve() / "hdl"
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="verification-", dir=base))
    verification_id = uuid.uuid4().hex
    nonce = uuid.uuid4().hex
    hashes = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sorted(sources.items())}
    bundle_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for name, source in sources.items():
        (work / name).write_text(source, encoding="utf-8")
    compile_files = ["/work/" + name for name in sources]
    if profile == PROFILE:
        (work / "aurex_profile_tb.sv").write_text(_profile_testbench(nonce), encoding="utf-8")
        compile_files += ["/work/aurex_profile_tb.sv"]
    ivl = "/toolchain/" + lib
    compile_result = _run(_sandbox(root, work, ["/toolchain/usr/bin/iverilog", "-B", ivl, "-g2012", "-Wall", "-s", top,
                                                "-o", "/work/simulation.vvp", *compile_files]))
    simulation = None
    if compile_result["exit_code"] == 0 and not compile_result["failure"]:
        simulation = _run(_sandbox(root, work, ["/toolchain/usr/bin/vvp", "-M", "-", "-M", ivl, "-N", "/work/simulation.vvp"]))
    passed = bool(simulation and simulation["exit_code"] == 0 and not simulation["failure"])
    reported_failure_markers = []
    if profile == 'custom' and passed:
        log = simulation.get('log', '')
        if _CUSTOM_FAILURE_LINES.search(log):
            reported_failure_markers.append('explicit_FAIL_line')
        if _CUSTOM_ICARUS_ERROR_LINES.search(log):
            reported_failure_markers.append('icarus_runtime_ERROR_line')
        counts = sorted({int(match) for match in _CUSTOM_NONZERO_ERRORS.findall(log)})
        if counts:
            reported_failure_markers.append('nonzero_error_count:' + ','.join(map(str, counts)))
        if reported_failure_markers:
            passed = False
    if profile == PROFILE:
        passed = bool(passed and ("AUREX_VERIFIED_" + nonce) in simulation["log"])
    result = {
        "verification_id": verification_id, "profile": profile, "verified": passed,
        "scope": "Fixed five-program teaching subset checks; not complete RV32I compliance or synthesis proof" if profile == PROFILE else "Custom testbench process plus explicit failure-marker check only; assertions/coverage are supplied by the caller",
        "source_sha256": bundle_hash, "source_files_sha256": hashes,
        "sources_path": str(work), "sources_paths": [str(work / name) for name in sources],
        "report_path": str(work / "verification.json"), "top": "aurex_rv32i_teaching" if profile == PROFILE else top,
        "compile": compile_result, "simulation": simulation,
        "toolchain": "Icarus Verilog; isolated filesystem/PID/network, 1GiB/20CPU-s/30wall-s/64KiB output per stage",
    }
    if profile == 'custom':
        result['reported_failure_markers'] = reported_failure_markers
        if not passed:
            result['diagnostic_hints'] = [
                "Treat a custom-test failure as a testbench-or-design question, not proof that a previously hash-bound fixed-profile PASS regressed.",
                "After assigning a combinational selector such as debug_reg_addr, wait #1 (or an explicit delta/event boundary) before checking debug_reg_data; immediate back-to-back checks can repeatedly observe the prior selection.",
                "Audit exact instruction fields, reset release, sampling edge and byte-address-to-word-index mapping before changing a fixed-profile-passing design or searching for a simulator bug.",
            ]
    if profile == PROFILE:
        result["contract"] = CONTRACT
        result["cases"] = [v["name"] for v in profile_vectors()]
        result["diagnostic_hints"] = [
            "Update PC/registers/halted/trap only in posedge sequential logic; combinational logic computes next-state and buses.",
            "Accept ADDI opcode 0010011/funct3 000; ADD/SUB opcode 0110011/funct3 000 with funct7 0000000/0100000; LW 0000011/funct3 010; SW 0100011/funct3 010; BEQ 1100011/funct3 000; JAL 1101111.",
            "Sign-extend exact immediate layouts: I={{20{inst[31]}},inst[31:20]}, S={{20{inst[31]}},inst[31:25],inst[11:7]}, B={{19{inst[31]}},inst[31],inst[7],inst[30:25],inst[11:8],1'b0}, J={{11{inst[31]}},inst[31],inst[19:12],inst[20],inst[30:21],1'b0}.",
            "I/B/J immediates already contain their architectural low bit after construction; never shift them again before address or PC calculation.",
            "Use rs1+I-immediate for LW effective address and rs1+S-immediate for SW; do not share one S-immediate address wire. BEQ target is current PC+B-immediate. JAL target is current PC+J-immediate and writes current PC+4 to rd.",
            "Check LW/SW alignment on the computed effective byte address low bits, never on instruction[1:0]. Drive dmem_addr from the LW effective address for LW and from the SW effective address for SW.",
            "Only ADDI, ADD, SUB, LW and JAL write rd: ADDI writes rs1+I-immediate, ADD/SUB write their ALU result, LW writes dmem_rdata and JAL writes current PC+4. SW, BEQ and EBREAK do not write a register.",
            "Decode EBREAK exactly as 32'h00100073: it latches halted without asserting trap.",
            "Unsupported means none of the supported instruction decodes matched; do not use a negated sub-decode that accidentally traps ADDI. Misaligned LW/SW latch trap; x0 remains zero and debug register 0 reads x0, not PC.",
        ]
        compile_log = compile_result.get("log", "")
        if "Syntax error between internal '}' and closing '}' of repeat concatenation" in compile_log:
            result["compile_remediation"] = {
                "failure_kind": "missing_outer_concatenation_braces",
                "explanation": (
                    "This is invalid Verilog source, not an Icarus limitation. A replication such as "
                    "{20{bit}} is one item; combining it with another item requires a separate outer "
                    "concatenation, so the source must visibly begin with two opening braces '{{'."
                ),
                "exact_examples": [
                    "wire [31:0] imm_i = {{20{instr[31]}}, instr[31:20]};",
                    "wire [31:0] imm_s = {{20{instr[31]}}, instr[31:25], instr[11:7]};",
                    "wire [31:0] imm_b = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};",
                    "wire [31:0] imm_j = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};",
                ],
                "next_action": (
                    "Use one small hdl_workspace_edit on the current revision. Replace the malformed "
                    "declarations themselves; do not retain them, add unused imm_i2 alternatives, blame "
                    "the compiler, or resend an unchanged whole file."
                ),
            }
    if workspace is not None:
        result.update(workspace_id=workspace['workspace_id'], workspace_revision=workspace['workspace_revision'],
            design_source_files=[name for name, item in sorted(snapshot.items()) if item['role'] == 'source'],
            source_retrieval={'tool': 'hdl_workspace_read', 'workspace_id': workspace['workspace_id'],
                              'revision': workspace['workspace_revision']})
        if design_top is not None:
            result['design_top'] = design_top
        elif profile == PROFILE:
            result['design_top'] = 'aurex_rv32i_teaching'
    def save_report():
        temporary = work / 'verification.json.tmp'
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(work / 'verification.json')
    # A check may only point to a complete report. Keep the actual simulation
    # outcome available even if the optional workspace journal is unavailable.
    save_report()
    if workspace is not None:
        from .hdl_workspace import workspace_check
        import sqlite3
        try:
            result['workspace_current'] = workspace_check(runtime, result)
        except (sqlite3.Error, OSError, ToolError) as error:
            result['workspace_current'] = None
            result['workspace_journal_error'] = str(error)
        save_report()
    return result


def register_hdl_tools(registry: ToolRegistry) -> None:
    registry.register(ToolSpec(
        name="hdl_simulate", description="Compile/simulate local Verilog/SystemVerilog in a bounded offline sandbox. Prefer persistent workspace_id/workspace_revision after small exact edits; inline files remain supported. custom top is the testbench; design_top separately selects the design for report-based export (only source-role files, never testbench-role files). A custom testbench must use $fatal for every mismatch; Icarus runtime ERROR diagnostics, lines beginning FAIL/FAILED/SOME TESTS FAILED, or a nonzero 'errors:' count force verified=false even if vvp exits 0. No filesystem/system/DPI calls or macros other than timescale. For a 32-bit CPU teaching subset select rv32i_teaching_v1; a trusted independent testbench verifies the exact interface: " + CONTRACT,
        parameters={"type": "object", "additionalProperties": False, "required": ["profile"],
          "oneOf": [{"required": ["files"], "not": {"anyOf": [{"required": ["workspace_id"]}, {"required": ["workspace_revision"]}, {"required": ["design_top"]}]}},
                    {"required": ["workspace_id", "workspace_revision"], "not": {"required": ["files"]}}], "properties": {
            "files": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "object", "additionalProperties": False,
                      "required": ["name", "content"], "properties": {"name": {"type": "string"}, "content": {"type": "string"}}}},
            "top": {"type": "string", "description": "custom: testbench top; teaching profile: aurex_rv32i_teaching (or omit)"},
            "workspace_id": {"type": "string"},
            "workspace_revision": {"type": "integer", "minimum": 1, "description": "Exact current revision returned by workspace create/read/edit"},
            "design_top": {"type": "string", "description": "custom workspace only: design top for export, distinct from the testbench top. Required for workspace custom export."},
            "profile": {"enum": ["custom", PROFILE]},
        }}, handler=hdl_simulate))
