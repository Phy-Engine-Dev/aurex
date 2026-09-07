import unittest

from aurex.verilog_lowering import MemoryLoweringError, lower_unpacked_register_arrays


class VerilogMemoryLoweringTests(unittest.TestCase):
    def test_lowers_dynamic_reads_and_writes(self):
        source = """module m(input clk, input [1:0] a, output [7:0] q);
reg [7:0] mem [0:3];
always @(posedge clk) mem[a] <= mem[a] + 1'b1;
assign q = mem[a];
endmodule
"""
        lowered, metadata = lower_unpacked_register_arrays(source)
        self.assertEqual(metadata, [{"name": "mem", "width": 8, "depth": 4}])
        self.assertNotIn("mem [0:3]", lowered)
        self.assertIn("reg [7:0] mem__aurex_mem_0;", lowered)
        self.assertIn("if ((a) == 0) mem__aurex_mem_0 <=", lowered)
        self.assertIn("((a) == 0 ? mem__aurex_mem_0", lowered)

    def test_comments_strings_and_same_name_in_modules_are_isolated(self):
        source = """module a; reg [1:0] mem [0:1]; // mem[bad]
initial $display("mem[also_bad]"); endmodule
module b; reg [3:0] mem [2:3]; wire [3:0] x = mem[2]; endmodule
"""
        lowered, metadata = lower_unpacked_register_arrays(source)
        self.assertEqual(len(metadata), 2)
        self.assertIn("// mem[bad]", lowered)
        self.assertIn('"mem[also_bad]"', lowered)
        self.assertIn("mem__aurex_mem_2", lowered)

    def test_rejects_element_part_select_instead_of_miscompiling(self):
        source = "module m(input [1:0] a, output q); reg [7:0] mem [0:3]; assign q=mem[a][0]; endmodule"
        with self.assertRaises(MemoryLoweringError):
            lower_unpacked_register_arrays(source)

    def test_no_array_is_byte_identical(self):
        source = "module m(input a, output q); assign q=a; endmodule\n"
        self.assertEqual(lower_unpacked_register_arrays(source), (source, []))


if __name__ == "__main__":
    unittest.main()
