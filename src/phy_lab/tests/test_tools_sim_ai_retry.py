import os
import sys
import types
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

import tools as tools_mod


class _FakeOllama:
    def __init__(self):
        self.calls = 0

    def chat(self, *, messages):
        self.calls += 1
        return "{}"


class _PhyCfg:
    cmake_source_dir = "third-parties/Phy-Engine/src"
    cmake_build_dir = "cache/phy-engine-build"
    cmake_build_type = "Release"
    build_timeout_sec = 1
    phyengine_lib_path = "cache/phy-engine-build/libphyengine.so"
    auto_build = False


class TestSimAiRetry(unittest.TestCase):
    def test_sim_ai_json_retries_up_to_three_times_then_returns_error(self):
        ollama = _FakeOllama()

        def _raise(_text):
            raise tools_mod.PEBuilderError("bad spec")

        with mock.patch.object(tools_mod, "ensure_phyengine_lib", return_value="x"):
            with mock.patch.object(tools_mod, "parse_spec_json", side_effect=_raise):
                with mock.patch.object(tools_mod, "llm_build_pe_sim_spec_json", return_value="{}") as m_build:
                    with mock.patch.object(tools_mod, "llm_fix_pe_sim_spec_json", return_value="{}") as m_fix:
                        out = tools_mod.simulate_ai_circuit_with_phyengine(
                            ollama=ollama,
                            text="中文输入",
                            context_json=None,
                            phy_engine_cfg=_PhyCfg(),
                            config_base_dir=".",
                            max_attempts=2,
                        )

        self.assertIn("我没能从你的描述里构建出可仿真的电路规格", out)
        self.assertEqual(m_build.call_count, 1)
        self.assertEqual(m_fix.call_count, 1)


if __name__ == "__main__":
    unittest.main()
