import os
import sys
import tempfile
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

    def chat(self, *, messages, response_format=None):
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
    def test_json_path_get(self):
        obj = {"experiment": {"items": [{"id": "x", "subject": "y"}]}}
        self.assertEqual(agent_mod._json_path_get(obj, "experiment.items[0].subject"), "y")
        self.assertEqual(agent_mod._json_path_get(obj, "experiment.items[0].id"), "x")

    def test_parse_context_ref(self):
        self.assertEqual(agent_mod._parse_context_ref("experiment:abc"), ("Experiment", "abc"))
        self.assertEqual(agent_mod._parse_context_ref("exp:abc"), ("Experiment", "abc"))
        self.assertEqual(agent_mod._parse_context_ref("discussion:abc"), ("Discussion", "abc"))
        self.assertEqual(agent_mod._parse_context_ref("disc:abc"), ("Discussion", "abc"))
        self.assertEqual(agent_mod._parse_context_ref("abc"), ("Experiment", "abc"))
        self.assertIsNone(agent_mod._parse_context_ref("x:abc"))
        self.assertIsNone(agent_mod._parse_context_ref("experiment:"))

    def test_shrink_context_omits_excerpts(self):
        ctx = {
            "summary_id": "abc",
            "category": "Experiment",
            "experiment_data_excerpt": "Z" * 12000,
            "summary_data_excerpt": "Y" * 9000,
            "body_text": "Hello",
        }
        shrunk = agent_mod._shrink_context_json_for_llm(ctx)
        self.assertEqual(shrunk.get("summary_id"), "abc")
        self.assertEqual(shrunk.get("category"), "Experiment")
        self.assertIn("<omitted:", str(shrunk.get("experiment_data_excerpt")))
        self.assertIn("<omitted:", str(shrunk.get("summary_data_excerpt")))
        dumped = agent_mod.json.dumps(shrunk, ensure_ascii=False)
        self.assertNotIn("Z" * 2000, dumped)
        self.assertNotIn("Y" * 2000, dumped)

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

    def test_agent_mode_accepts_tool_prefix_names(self):
        replies = [
            '{"tool":"tool_list_plar","args":{"kind":"hot","category":"Experiment","take":5}}',
            '{"tool":"end","final":"OK"}',
        ]
        seen = {"tool": None}

        def _exec(*, tool, args, **_kw):
            seen["tool"] = tool
            return "tool_ok"

        with mock.patch.object(agent_mod, "_agent_execute_tool", side_effect=_exec):
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
                max_seconds=120,
                max_steps=3,
            )
        self.assertEqual(seen["tool"], "list_plar")
        self.assertEqual(out.strip(), "OK")

    def test_agent_parse_tool_call_repairs_plar_get_user_by_name_to_list_plar(self):
        tool, args, final = agent_mod._agent_parse_tool_call(
            '{"tool":"plar_get_user_by_name","args":{"kind":"latest","category":"both","user_id":"u","take":1}}'
        )
        self.assertEqual(tool, "list_plar")
        self.assertEqual(final, "")
        self.assertEqual(args.get("kind"), "latest")

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

    def test_agent_mode_rejects_unknown_tools(self):
        replies = [
            '{"tool":"plagiarism_check","args":{"text":"x"}}',
            '{"tool":"end","final":"OK"}',
        ]
        with mock.patch.object(agent_mod, "_agent_execute_tool") as exec_tool:
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
                max_seconds=120,
                max_steps=3,
            )
        self.assertEqual(out.strip(), "OK")
        exec_tool.assert_not_called()

    def test_list_plar_tool_result_is_summarized(self):
        # Ensure large list_plar outputs are stored and only summarized in prompt.
        big = {
            "kind": "latest",
            "category": "both",
            "experiment": {"items": [{"id": str(i), "subject": f"s{i}"} for i in range(10)], "next_skip": 10},
            "discussion": {"items": [{"id": str(i), "subject": f"d{i}"} for i in range(8)], "next_skip": 8},
            "pad": "x" * 8000,
        }
        with tempfile.TemporaryDirectory() as d:
            msg = agent_mod._agent_tool_result_message(
                tool="list_plar", result=agent_mod.json.dumps(big, ensure_ascii=False), cache_dir=d
            )
            self.assertEqual(msg.get("role"), "system")
            content = str(msg.get("content") or "")
            self.assertIn("Tool result (list_plar)", content)
            blob = content[content.find("{") :]
            obj = agent_mod.json.loads(blob)
            self.assertTrue(obj.get("stored"))
            key = str(obj.get("key") or "")
            self.assertRegex(key, r"^[0-9a-f]{16,64}$")
            summary = obj.get("summary") or {}
            self.assertIn("experiment", summary)
            self.assertIn("items_total", summary["experiment"])
            self.assertEqual(summary["experiment"]["items_total"], 10)
            self.assertEqual(len(summary["experiment"]["items"]), 3)
            # The full payload should be on disk.
            stored_path = os.path.join(d, "llm_store", f"{key}.txt")
            self.assertTrue(os.path.isfile(stored_path))
            with open(stored_path, "r", encoding="utf-8") as f:
                stored_text = f.read()
            self.assertIn('"pad"', stored_text)


if __name__ == "__main__":
    unittest.main()
