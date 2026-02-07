import json
import os
import sys
import tempfile
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from config import ConfigError, load_config, parse_config


class TestConfig(unittest.TestCase):
    def test_parse_valid_config(self):
        cfg = parse_config(
            {
                "schema_version": 1,
                "account": {"email": "user@example.com"},
                "ollama": {"model": "llama3.1"},
                "agent": {
                    "poll_interval_sec": 5,
                    "user_targets_require_mention": True,
                    "web_search_enabled": True,
                    "auto_web_search": True,
                    "web_search_proxy": "http://127.0.0.1:7897",
                    "web_search_fallback_to_ddg": True,
                    "auto_tool_routing": True,
                    "auto_publish": True,
                    "circuit_max_attempts": 3,
                    "publish_max_elements": 5000,
                    "overload_protection_enabled": True,
                    "overload_window_sec": 600,
                    "overload_max_requests": 40,
                    "overload_message_en": "Too many requests at the moment, please try again later.",
                    "targets": [],
                },
            },
            source="in-memory",
        )
        self.assertEqual(cfg.account.email, "user@example.com")
        self.assertEqual(cfg.ollama.model, "llama3.1")
        self.assertEqual(cfg.agent.poll_interval_sec, 5.0)
        self.assertEqual(cfg.agent.user_targets_require_mention, True)
        self.assertEqual(cfg.agent.web_search_enabled, True)
        self.assertEqual(cfg.agent.auto_web_search, True)
        self.assertEqual(cfg.agent.web_search_proxy, "http://127.0.0.1:7897")
        self.assertEqual(cfg.agent.web_search_fallback_to_ddg, True)
        self.assertEqual(cfg.agent.auto_tool_routing, True)
        self.assertEqual(cfg.agent.auto_publish, True)
        self.assertEqual(cfg.agent.circuit_max_attempts, 3)
        self.assertEqual(cfg.agent.publish_max_elements, 5000)
        self.assertEqual(cfg.agent.overload_protection_enabled, True)
        self.assertEqual(cfg.agent.overload_window_sec, 600)
        self.assertEqual(cfg.agent.overload_max_requests, 40)
        self.assertEqual(
            cfg.agent.overload_message_en,
            "Too many requests at the moment, please try again later.",
        )

    def test_parse_ollama_pool_config(self):
        cfg = parse_config(
            {
                "schema_version": 1,
                "account": {"email": "user@example.com"},
                "ollama": {
                    "base_urls": ["127.0.0.1:11434", "127.0.0.1:11435"],
                    "model": "gpt-oss:latest",
                    "max_parallel_requests": 2,
                },
                "agent": {"targets": []},
            },
            source="in-memory",
        )
        self.assertEqual(cfg.ollama.base_url, "http://127.0.0.1:11434")
        self.assertEqual(cfg.ollama.base_urls, ["http://127.0.0.1:11434", "http://127.0.0.1:11435"])
        self.assertEqual(cfg.ollama.max_parallel_requests, 2)

    def test_parse_invalid_schema_version(self):
        with self.assertRaises(ConfigError):
            parse_config(
                {"schema_version": 2, "account": {"email": "x@example.com"}},
                source="in-memory",
            )

    def test_parse_missing_email(self):
        with self.assertRaises(ConfigError):
            parse_config({"schema_version": 1, "account": {}}, source="in-memory")

    def test_load_config_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {"schema_version": 1, "account": {"email": "user@example.com"}},
                    f,
                )
            cfg = load_config(path)
            self.assertEqual(cfg.account.email, "user@example.com")


if __name__ == "__main__":
    unittest.main()
