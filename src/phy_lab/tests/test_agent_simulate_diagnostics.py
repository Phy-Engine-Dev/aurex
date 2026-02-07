import os
import sys
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

import agent as agent_mod


class _FakeOllama:
    def chat(self, *, messages):
        return "OK"


class _Cfg:
    class _Agent:
        system_prompt = "sys"
        max_reply_chars = 2000
        mention_tag = "@aurex"
        commands_enabled = False
        command_prefix = "!"
        auto_tool_routing = True
        require_mention = False
        user_targets_require_mention = False
        web_search_enabled = False
        simulation_enabled = True
        simulation_ai_enabled = True
        simulation_ai_max_components = 30
        simulation_ai_max_probes = 20
        simulation_max_elements = 300

    class _Ollama:
        base_url = "http://127.0.0.1:11434"
        model = "m"

    class _Phy:
        auto_build = False
        phyengine_lib_path = ""
        cmake_source_dir = "third-parties/Phy-Engine/src"
        cmake_build_dir = "cache/phy-engine-build"

    agent = _Agent()
    ollama = _Ollama()
    phy_engine = _Phy()


class TestSimDiagnostics(unittest.TestCase):
    def test_auto_routed_simulate_returns_diagnostics_on_failure(self):
        def _fake_route(**_kw):
            return {"action": "simulate", "arg": "仿真 RC transient", "publish": False}

        parse_failure = "我没能把命令脚本解析成可仿真的电路。错误：boom\n\n脚本：\nANALYSIS dc"

        def _fake_ai_script(**_kw):
            return parse_failure

        def _fake_ai_json(**_kw):
            raise RuntimeError("json path failed")

        def _fake_series_demo(**_kw):
            return "要进行串联直流仿真，请写清电阻数量与阻值。"

        with mock.patch.object(agent_mod, "_llm_route_tool", side_effect=_fake_route):
            with mock.patch.object(agent_mod, "simulate_ai_script_circuit_with_phyengine", side_effect=_fake_ai_script):
                with mock.patch.object(agent_mod, "simulate_ai_circuit_with_phyengine", side_effect=_fake_ai_json):
                    with mock.patch.object(agent_mod, "simulate_series_vdc_resistors", side_effect=_fake_series_demo):
                        out = agent_mod._handle_comment(
                            comment={"Content": "hi"},
                            user=object(),
                            ollama=_FakeOllama(),
                            cache_dir="cache",
                            config_base_dir=".",
                            cfg=_Cfg(),
                            dry_run=True,
                            logger=agent_mod.logging.getLogger("t"),
                            conversation_key=None,
                            experiment_context=None,
                            history=[],
                        )
        self.assertIn("仿真失败（诊断信息）", out)
        self.assertIn("ai_json_spec_failed", out)
        self.assertIn("AI 构建电路失败细节", out)


if __name__ == "__main__":
    unittest.main()
