"""HDL sandbox/profile regressions. No CPU implementation or model call is provided."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aurex.tools.hdl import (CONTRACT, PROFILE, _beq, _i, _jal, _profile_testbench, _r, _run, _sandbox, _toolchain,
                             _sw, hdl_simulate, profile_vectors, register_hdl_tools, validate_source)
from aurex.tools.registry import ToolError, ToolRegistry, ToolRuntime


class HDLSecurityTests(unittest.TestCase):
    def test_only_literal_timescale(self):
        validate_source('`timescale 1ns/1ps\nmodule top; initial $display("safe"); endmodule')
        for source in ('`include "secret"', '`define X $system', '`default_nettype none', '`timescale 1ns/1ps `include "x"'):
            with self.subTest(source=source), self.assertRaises(ToolError):
                validate_source(source)

    def test_calls_and_dpi_blocked(self):
        for call in ("readmemh", "readmemb", "fopen", "fwrite", "system", "getenv", "dumpfile", "stop", "value$plusargs", "unknown"):
            with self.subTest(call=call), self.assertRaises(ToolError):
                validate_source(f'module top; initial ${call}("x"); endmodule')
        for code in ('import "DPI-C" function int f();', '\\$system foo;', 'bind foo top x();', 'aurex_profile_tb.foo=1;'):
            with self.subTest(code=code), self.assertRaises(ToolError):
                validate_source(code)

    def test_comments_strings_and_lexical_errors(self):
        validate_source('module top; // $system("ignored")\n/* `include "x" */ initial $display("$fopen \\\"data\\\""); endmodule')
        for code in ('/* unclosed', 'module x; "unclosed', 'module x;\x00endmodule'):
            with self.subTest(code=code), self.assertRaises(ToolError):
                validate_source(code)

    def test_rv_encodings_and_expected_profile(self):
        self.assertEqual(_i(0x13, 1, 0, -1), 0xfff00093)
        self.assertEqual(_r(3, 1, 2, True), 0x402081b3)
        self.assertEqual(_sw(2, 1, 4), 0x0020a223)
        self.assertEqual(_beq(0, 0, -4), 0xfe000ee3)
        self.assertEqual(_jal(1, -4), 0xffdff0ef)
        vectors = profile_vectors()
        self.assertEqual(len(vectors), 5)
        self.assertEqual(vectors[0]["regs"][0], 0)
        self.assertEqual(vectors[0]["regs"][31], 147)
        self.assertEqual(vectors[0]["memory"], {64: 0xfffffff8, 65: 26})
        self.assertIn("AUREX_VERIFIED_unique_nonce", _profile_testbench("unique_nonce"))
        self.assertIn("No complete RV32I", CONTRACT)

    def test_registry(self):
        registry = ToolRegistry()
        register_hdl_tools(registry)
        self.assertEqual(registry.get("hdl_simulate").handler, hdl_simulate)

    def test_runner_limits(self):
        timeout = _run([sys.executable, "-c", "while True: pass"], timeout=.2)
        self.assertEqual(timeout["failure"], "timeout")
        output = _run([sys.executable, "-c", "print('x'*20000)"], max_output=200)
        self.assertEqual(output["failure"], "output_limit")
        self.assertEqual(len(output["log"]), 200)


class HDLIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = ToolRuntime("hdl-test", "en", str(Path(self.temp.name)/"config.json"), None, self.temp.name)

    def run_hdl(self, code, profile="custom", top="tb"):
        return hdl_simulate(self.runtime, {"files": [{"name": "test.sv", "content": code}], "profile": profile, "top": top})

    def test_real_compile_simulate_and_provenance(self):
        source = '''`timescale 1ns/1ps
module tb; reg[7:0] a,b; wire[7:0] y=a+b;
initial begin a=17; b=26; #1; if(y!==43) $fatal(1,"sum"); $display("ADDER_OK"); $finish; end
endmodule'''
        result = self.run_hdl(source)
        self.assertTrue(result["verified"], result)
        self.assertIn("ADDER_OK", result["simulation"]["log"])
        self.assertEqual(result["compile"]["exit_code"], 0)
        self.assertEqual(result["source_files_sha256"]["test.sv"], hashlib.sha256(source.encode()).hexdigest())
        self.assertEqual(Path(result["sources_paths"][0]).read_text(), source)
        self.assertEqual(json.loads(Path(result["report_path"]).read_text())["verification_id"], result["verification_id"])

    def test_custom_log_failure_marker_cannot_be_reported_verified(self):
        result = self.run_hdl(
            'module tb; initial begin $display("FAIL: mismatch"); $finish; end endmodule')
        self.assertFalse(result['verified'])
        self.assertIn('explicit_FAIL_line', result['reported_failure_markers'])
        self.assertTrue(any('debug_reg_addr' in hint and 'wait #1' in hint
                            for hint in result['diagnostic_hints']))

    def test_custom_zero_error_summary_remains_process_success(self):
        result = self.run_hdl(
            'module tb; initial begin $display("Total errors: 0"); $finish; end endmodule')
        self.assertTrue(result['verified'], result)
        self.assertEqual(result['reported_failure_markers'], [])

    def test_real_assertion_fails(self):
        result = self.run_hdl('module tb; initial $fatal(1,"EXPECTED_ASSERTION_FAILURE"); endmodule')
        self.assertFalse(result["verified"])
        self.assertIn("EXPECTED_ASSERTION_FAILURE", result["simulation"]["log"])

    def test_real_sandbox_hides_host_data_and_network(self):
        root, _ = _toolchain()
        script = 'import os; assert not os.path.exists("/home/macromodel"); assert not os.path.exists("/Users"); assert len(open("/proc/net/route").read().splitlines())==1; print("ISOLATED")'
        result = _run(_sandbox(root, Path(self.temp.name), ["/usr/bin/python3", "-c", script]))
        self.assertEqual(result["exit_code"], 0, result)
        self.assertIn("ISOLATED", result["log"])

    def test_real_compile_error_fails(self):
        result = self.run_hdl('module tb; invalid syntax here endmodule')
        self.assertFalse(result["verified"])
        self.assertIsNone(result["simulation"])
        self.assertNotEqual(result["compile"]["exit_code"], 0)

    def test_real_compile_warnings_are_visible_but_not_failures(self):
        # Out-of-range selection is a warning and remains X, never coerced to 0.
        source = '''module tb; reg [3:0] value; wire selected=value[8];
initial begin value=0; #1; if(selected!==1'bx) $fatal(1,"unknown lost"); $finish; end
endmodule'''
        result = self.run_hdl(source)
        self.assertEqual(result["compile"]["exit_code"], 0, result)
        self.assertIn("warning", result["compile"]["log"].lower())
        self.assertIn("select", result["compile"]["log"].lower())
        self.assertTrue(result["verified"], result)

    def test_profile_failure_has_bounded_actual_interface_history(self):
        # A deliberately broken bus generator, not a CPU implementation or oracle.
        source = '''module aurex_rv32i_teaching(input clk,rst,output[31:0] imem_addr,input[31:0] imem_rdata,
output dmem_we,output[31:0] dmem_addr,dmem_wdata,input[31:0] dmem_rdata,output halted,trap,
input[4:0] debug_reg_addr,output[31:0] debug_reg_data);
reg [31:0] counter; always @(posedge clk) if(rst) counter<=0; else counter<=counter+4;
assign imem_addr=counter;assign dmem_we=1'bx;assign dmem_addr=32'hzzzzzzzz;assign dmem_wdata=32'hxxxxxxxx;
assign halted=0;assign trap=0;assign debug_reg_data=0;endmodule'''
        result = self.run_hdl(source, PROFILE, "aurex_rv32i_teaching")
        self.assertEqual(result["compile"]["exit_code"], 0, result)
        self.assertFalse(result["verified"])
        log = result["simulation"]["log"]
        self.assertIn("expected EBREAK halt without trap", log)
        self.assertIn("rows=16 max_rows=16 sample=existing_negedge_loop_checkpoint", log)
        self.assertIn("time_unit=ns clock_period_ns=10", log)
        self.assertIn("pc_instruction=current_fetch dmem=current_interface_not_committed_store", log)
        samples = [line for line in log.splitlines() if line.startswith("AUREX_PROFILE_SAMPLE ")]
        self.assertEqual(len(samples), 16)
        self.assertEqual([int(re.search(r"cycle=(\d+)", line)[1]) for line in samples], list(range(497, 513)))
        times = [int(re.search(r"time_ns=(\d+)", line)[1]) for line in samples]
        self.assertEqual([b-a for a,b in zip(times,times[1:])], [10]*15)
        pcs = [int(re.search(r"pc=([0-9a-f]+)", line)[1], 16) for line in samples]
        self.assertEqual([b-a for a,b in zip(pcs,pcs[1:])], [4]*15)
        self.assertTrue(all("trap=0 halted=0 dmem_we=x dmem_addr=zzzzzzzz dmem_wdata=xxxxxxxx" in line for line in samples))
        current = next(line for line in log.splitlines() if line.startswith("AUREX_PROFILE_CURRENT "))
        self.assertIn("sample=immediately_before_failed_assertion", current)
        self.assertIn("cycle=512", current)
        self.assertNotIn("AUREX_VERIFIED_", log)
        self.assertFalse(result["simulation"]["log_truncated"])
        self.assertTrue(any("Sign-extend exact immediate layouts" in hint
                            for hint in result["diagnostic_hints"]))
        self.assertTrue(any("rs1+I-immediate for LW" in hint
                            for hint in result["diagnostic_hints"]))
        self.assertTrue(any("never shift them again" in hint
                            for hint in result["diagnostic_hints"]))
        self.assertTrue(any("Only ADDI, ADD, SUB, LW and JAL write rd" in hint
                            for hint in result["diagnostic_hints"]))
        self.assertTrue(any("never on instruction[1:0]" in hint
                            for hint in result["diagnostic_hints"]))
        self.assertEqual(Path(result["sources_paths"][0]).read_text(), source)

    def test_profile_repeat_concatenation_compile_error_has_exact_remediation(self):
        source = '''module aurex_rv32i_teaching(input clk,rst,output[31:0] imem_addr,input[31:0] imem_rdata,
output dmem_we,output[31:0] dmem_addr,dmem_wdata,input[31:0] dmem_rdata,output halted,trap,
input[4:0] debug_reg_addr,output[31:0] debug_reg_data);
wire [31:0] instr=imem_rdata;
wire [31:0] imm_i={20{instr[31]},instr[31:20]};
assign imem_addr=0;assign dmem_we=0;assign dmem_addr=0;assign dmem_wdata=0;
assign halted=0;assign trap=0;assign debug_reg_data=imm_i;endmodule'''
        result = self.run_hdl(source, PROFILE, "aurex_rv32i_teaching")
        self.assertFalse(result["verified"])
        remediation = result["compile_remediation"]
        self.assertEqual(remediation["failure_kind"], "missing_outer_concatenation_braces")
        self.assertIn("{{20{instr[31]}}, instr[31:20]}", remediation["exact_examples"][0])
        self.assertIn("unused imm_i2", remediation["next_action"])

    def test_profile_stops_at_first_unexpected_trap_with_executed_instruction(self):
        source = '''module aurex_rv32i_teaching(input clk,rst,output reg[31:0] imem_addr,input[31:0] imem_rdata,
output dmem_we,output[31:0] dmem_addr,dmem_wdata,input[31:0] dmem_rdata,output halted,output reg trap,
input[4:0] debug_reg_addr,output[31:0] debug_reg_data);
always @(posedge clk) if(rst) begin imem_addr<=0; trap<=0; end else begin imem_addr<=imem_addr+4; trap<=1; end
assign dmem_we=0; assign dmem_addr=0; assign dmem_wdata=0; assign halted=0; assign debug_reg_data=0;
endmodule'''
        result = self.run_hdl(source, PROFILE, "aurex_rv32i_teaching")
        self.assertFalse(result["verified"])
        log = result["simulation"]["log"]
        self.assertIn("unexpected trap in non-trap program at cycle 1", log)
        self.assertIn("after executing pc=00000000 instruction=fc100093", log)
        self.assertIn("inspect preceding control flow and decoder", log)

    def test_profile_unknown_control_fails_on_first_observed_cycle(self):
        source = '''module aurex_rv32i_teaching(input clk,rst,output[31:0] imem_addr,input[31:0] imem_rdata,
output dmem_we,output[31:0] dmem_addr,dmem_wdata,input[31:0] dmem_rdata,output halted,trap,
input[4:0] debug_reg_addr,output[31:0] debug_reg_data);
assign imem_addr=32'hxxxxxxxx;assign dmem_we=0;assign dmem_addr=0;assign dmem_wdata=0;
assign halted=0;assign trap=0;assign debug_reg_data=0;endmodule'''
        result = self.run_hdl(source, PROFILE, "aurex_rv32i_teaching")
        self.assertFalse(result["verified"])
        self.assertIn("control became X at cycle 1", result["simulation"]["log"])
        self.assertIn("rows=1 max_rows=16", result["simulation"]["log"])

    def test_fixed_profile_rejects_broken_design(self):
        # Intentionally nonfunctional constant-output stub, not a CPU implementation.
        source = '''module aurex_rv32i_teaching(input clk,rst,output[31:0] imem_addr,input[31:0] imem_rdata,
output dmem_we,output[31:0] dmem_addr,dmem_wdata,input[31:0] dmem_rdata,output halted,trap,
input[4:0] debug_reg_addr,output[31:0] debug_reg_data);
assign imem_addr=0;assign dmem_we=0;assign dmem_addr=0;assign dmem_wdata=0;
assign halted=1;assign trap=0;assign debug_reg_data=0;endmodule'''
        result = self.run_hdl(source, PROFILE, "aurex_rv32i_teaching")
        self.assertEqual(result["compile"]["exit_code"], 0, result)
        self.assertFalse(result["verified"])
        self.assertIn("x1 got", result["simulation"]["log"])
        self.assertIn("rows=0 max_rows=16", result["simulation"]["log"])
        self.assertIn("debug_reg_addr=1 debug_reg_data=00000000", result["simulation"]["log"])

    def test_paths_and_reserved_names(self):
        for name in ("../x.sv", "/tmp/x.sv", "a/b.sv", "a.v;evil", "a..sv", "aurex_profile_tb.sv"):
            with self.subTest(name=name), self.assertRaises(ToolError):
                hdl_simulate(self.runtime, {"files": [{"name": name, "content": "module tb;endmodule"}], "top": "tb"})
        with self.assertRaises(ToolError):
            self.run_hdl('module tb;endmodule', PROFILE, "tb")


if __name__ == "__main__":
    unittest.main()
