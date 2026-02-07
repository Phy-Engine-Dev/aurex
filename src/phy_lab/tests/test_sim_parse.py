import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from tools import _parse_series_vdc_n_resistors  # type: ignore


class TestSimParse(unittest.TestCase):
    def test_parse_chinese_3x4ohm_no_v(self):
        v, rs = _parse_series_vdc_n_resistors("请你测试3个4ohm电阻和一个vdc串联的直流仿真结果")
        self.assertIsNone(v)
        self.assertEqual(rs, [4.0, 4.0, 4.0])

    def test_parse_english_3x4ohm_with_v(self):
        v, rs = _parse_series_vdc_n_resistors("simulate V=5V 3x4ohm resistors in series with a VDC")
        self.assertEqual(v, 5.0)
        self.assertEqual(rs, [4.0, 4.0, 4.0])

    def test_parse_two_resistors(self):
        v, rs = _parse_series_vdc_n_resistors("simulate V=12V R1=10ohm R2=20ohm")
        self.assertEqual(v, 12.0)
        self.assertEqual(rs, [10.0, 20.0])


if __name__ == "__main__":
    unittest.main()

