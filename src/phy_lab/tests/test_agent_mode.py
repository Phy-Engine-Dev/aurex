import os
import sys
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

import agent as agent_mod


class _SeqOllama:
    def __init__(self, replies):
        self._replies = list(replies)

    def chat(self, *, messages):
        if not self._replies:
            return "{\"tool\":\"end\",\"final\":\"OK\"}"
        return self._replies.pop(0)


class _Cfg:
    class _Agent:
        system_prompt = "sys"
        max_reply_chars = 2000
        mention_tag = "@aurex"
        commands_enabled = True
        command_prefix = "!"
        auto_tool_routing = True
        require_mention = False
        user_targets_require_mention = False
        web_search_enabled = False
        web_search_provider = "duckduckgo"
        web_search_proxy = ""
        web_search_timeout_sec = 10
        web_search_cache_ttl_sec = 10
        web_search_max_results = 3
        web_search_fallback_to_ddg = True
        web_search_user_agent = ""
        web_search_searxng_base_url = ""
        simulation_ai_enabled = True
        simulation_ai_max_components = 30
        simulation_ai_max_probes = 20
        simulation_max_elements = 300
        enable_publish = False
        auto_publish = False
        circuit_max_attempts = 2
        publish_max_elements = 5000
        publish_category = "Discussion"
        publish_tags = []

    class _Storage:
        keep_temp = False

    class _Phy:
        auto_build = False
        phyengine_lib_path = ""
        cmake_source_dir = "third-parties/Phy-Engine/src"
        cmake_build_dir = "cache/phy-engine-build"

    agent = _Agent()
    storage = _Storage()
    phy_engine = _Phy()


class TestAgentMode(unittest.TestCase):
    def test_agent_mode_end_immediate(self):
        out = agent_mod.agent_mode_run(
            ollama=_SeqOllama(['{"tool":"end","final":"Hello world."}']),
            user=object(),
            cfg=_Cfg(),
            cache_dir="cache",
            config_base_dir=".",
            dry_run=True,
            logger=agent_mod.logging.getLogger("t"),
            task="do it",
            context_json=None,
            history=[],
            max_seconds=3,
            max_steps=3,
        )
        self.assertIn("Hello", out)

    def test_agent_mode_tool_then_end(self):
        replies = [
            '{"tool":"search_plar","args":{"query":"@someone","max_results":2}}',
            '{"tool":"end","final":"Done."}',
        ]
        with mock.patch.object(agent_mod, "_agent_execute_tool", return_value="tool_ok"):
            out = agent_mod.agent_mode_run(
                ollama=_SeqOllama(replies),
                user=object(),
                cfg=_Cfg(),
                cache_dir="cache",
                config_base_dir=".",
                dry_run=True,
                logger=agent_mod.logging.getLogger("t"),
                task="search and answer",
                context_json=None,
                history=[],
                max_seconds=3,
                max_steps=3,
            )
        self.assertEqual(out.strip(), "Done.")

    def test_agent_mode_last_minute_rejects_tools(self):
        # max_seconds=1 => tool cutoff at 0s => tools disabled immediately.
        replies = [
            '{"tool":"search_plar","args":{"query":"x"}}',
            '{"tool":"end","final":"Final answer."}',
        ]
        out = agent_mod.agent_mode_run(
            ollama=_SeqOllama(replies),
            user=object(),
            cfg=_Cfg(),
            cache_dir="cache",
            config_base_dir=".",
            dry_run=True,
            logger=agent_mod.logging.getLogger("t"),
            task="whatever",
            context_json=None,
            history=[],
            max_seconds=1,
            max_steps=3,
        )
        self.assertIn("Final answer", out)


if __name__ == "__main__":
    unittest.main()
