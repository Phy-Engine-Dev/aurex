import json
import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

import agent as agent_mod
from unittest import mock


class _FakeOllama:
    def __init__(self):
        self.calls = []

    def chat(self, *, messages):
        self.calls.append(messages)
        # Return a simple deterministic response.
        return "OK"


class _Cfg:
    class _Agent:
        system_prompt = "sys"
        max_reply_chars = 2000
        mention_tag = "@aurex"
        commands_enabled = True
        command_prefix = "!"
        auto_tool_routing = False
        require_mention = False
        user_targets_require_mention = False
        web_search_enabled = False
        simulation_enabled = False

    agent = _Agent()


class TestAgentSearchFlow(unittest.TestCase):
    def test_search_command_calls_llm_to_summarize_hits(self):
        ollama = _FakeOllama()

        def _fake_hits(**_kw):
            return [{"ID": "abc", "Subject": "S"}]

        comment = {"Content": "!search opamp"}
        with mock.patch.object(agent_mod, "search_recent_experiments", side_effect=_fake_hits):
            out = agent_mod._handle_comment(
                comment=comment,
                user=object(),
                ollama=ollama,
                cache_dir="cache",
                config_base_dir=".",
                cfg=_Cfg(),
                dry_run=True,
                logger=agent_mod.logging.getLogger("t"),
                conversation_key=None,
                experiment_context=None,
                history=[],
            )
        self.assertEqual(out, "OK")
        # Ensure LLM got results JSON in system message.
        blob = "\n".join(m.get("content", "") for m in ollama.calls[-1] if m.get("role") == "system")
        self.assertIn("Internal Physics Lab search results", blob)
        self.assertIn("abc", blob)

    def test_auto_routed_user_search_returns_user_info(self):
        class _Cfg2:
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
                simulation_enabled = False

            agent = _Agent()

        ollama = _FakeOllama()
        user = object()
        comment = {"Content": "hi"}

        def _fake_route(**_kw):
            return {"action": "search_plar", "arg": "user: abc", "publish": False}

        def _fake_get_user(_user, *, name):
            return {"User": {"ID": "u1", "Nickname": name}}

        with mock.patch.object(agent_mod, "_llm_route_tool", side_effect=_fake_route):
            with mock.patch.object(agent_mod, "get_user_by_name", side_effect=_fake_get_user):
                out = agent_mod._handle_comment(
                    comment=comment,
                    user=user,
                    ollama=ollama,
                    cache_dir="cache",
                    config_base_dir=".",
                    cfg=_Cfg2(),
                    dry_run=True,
                    logger=agent_mod.logging.getLogger("t"),
                    conversation_key=None,
                    experiment_context=None,
                    history=[],
                )
        self.assertIn("Nickname: abc", out)
        self.assertIn("ID: u1", out)

    def test_search_command_at_user_queries_user_api(self):
        class _Cfg3:
            class _Agent:
                system_prompt = "sys"
                max_reply_chars = 2000
                mention_tag = "@aurex"
                commands_enabled = True
                command_prefix = "!"
                auto_tool_routing = False
                require_mention = False
                user_targets_require_mention = False
                web_search_enabled = False
                simulation_enabled = False

            agent = _Agent()

        def _fake_get_user(_user, *, name):
            return {"User": {"ID": "u9", "Nickname": name, "Signature": "sig"}}

        with mock.patch.object(agent_mod, "get_user_by_name", side_effect=_fake_get_user):
            out = agent_mod._handle_comment(
                comment={"Content": "!search @abc"},
                user=object(),
                ollama=_FakeOllama(),
                cache_dir="cache",
                config_base_dir=".",
                cfg=_Cfg3(),
                dry_run=True,
                logger=agent_mod.logging.getLogger("t"),
                conversation_key=None,
                experiment_context=None,
                history=[],
            )
        self.assertIn("Nickname: abc", out)
        self.assertIn("ID: u9", out)

    def test_circuit_command_defaults_to_discussion_category(self):
        class _Cfg4:
            class _Agent:
                system_prompt = "sys"
                max_reply_chars = 2000
                mention_tag = "@aurex"
                commands_enabled = True
                command_prefix = "!"
                auto_tool_routing = False
                require_mention = False
                user_targets_require_mention = False
                web_search_enabled = False
                simulation_enabled = False
                enable_publish = True
                publish_category = "Discussion"
                publish_tags = ["SmallProject"]
                circuit_max_attempts = 1
                publish_max_elements = 5000

            class _Storage:
                keep_temp = False

            class _Phy:
                pass

            agent = _Agent()
            storage = _Storage()
            phy_engine = _Phy()

        captured = {}

        def _fake_build(**kwargs):
            captured.update(kwargs)

            class _Res:
                published = True
                summary_id = "sid"
                publish_block_reason = None
                plsav_elements = 1

            return _Res()

        with mock.patch.object(agent_mod, "build_and_maybe_publish_circuit", side_effect=_fake_build):
            out = agent_mod._handle_comment(
                comment={"Content": "!circuit make a counter"},
                user=object(),
                ollama=_FakeOllama(),
                cache_dir="cache",
                config_base_dir=".",
                cfg=_Cfg4(),
                dry_run=True,
                logger=agent_mod.logging.getLogger("t"),
                conversation_key=None,
                experiment_context=None,
                history=[],
            )

        self.assertIn("SummaryID", out)
        self.assertEqual(captured.get("publish_category_value"), "Discussion")


if __name__ == "__main__":
    unittest.main()
