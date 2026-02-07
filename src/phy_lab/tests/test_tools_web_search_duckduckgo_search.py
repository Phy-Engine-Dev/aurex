import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

import tools


class TestDuckDuckGoSearchBackend(unittest.TestCase):
    def test_duckduckgo_search_library_is_used_when_available(self):
        captured = {}

        class FakeDDGS:
            def __init__(self, proxy=None, proxies=None, headers=None, timeout=None):
                captured["proxy"] = proxy
                captured["proxies"] = proxies
                captured["headers"] = headers
                captured["timeout"] = timeout

            def text(self, q, max_results=None):
                captured["query"] = q
                captured["max_results"] = max_results
                return [
                    {"title": "One", "href": "https://example.com/1"},
                    {"title": "Two", "href": "https://example.com/2"},
                ]

        fake_mod = types.ModuleType("duckduckgo_search")
        fake_mod.DDGS = FakeDDGS  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(sys.modules, {"duckduckgo_search": fake_mod}):
                with mock.patch.object(tools, "_http_get_text", side_effect=AssertionError("should not fetch html")):
                    out = tools.web_search_duckduckgo(
                        query="Physics Lab AR",
                        cache_dir=td,
                        proxy="http://127.0.0.1:7897",
                        timeout_sec=12,
                        ttl_sec=3600,
                        max_results=2,
                        user_agent="UA-TEST",
                    )

            self.assertIn("DuckDuckGo results:", out)
            self.assertIn("https://example.com/1", out)
            self.assertEqual(captured["query"], "Physics Lab AR")
            self.assertEqual(captured["max_results"], 2)
            self.assertEqual(captured["proxy"], "http://127.0.0.1:7897")
            self.assertEqual(captured["headers"], {"User-Agent": "UA-TEST"})
            self.assertEqual(captured["timeout"], 12.0)

            cache_root = os.path.join(td, "web_cache", "duckduckgo_search")
            self.assertTrue(os.path.isdir(cache_root))
            cached_files = [p for p in os.listdir(cache_root) if p.endswith(".json")]
            self.assertTrue(cached_files)
            with open(os.path.join(cache_root, cached_files[0]), "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, list)

    def test_provider_alias_routes_to_duckduckgo(self):
        fake_mod = types.ModuleType("duckduckgo_search")

        class FakeDDGS:
            def __init__(self, **kwargs):
                pass

            def text(self, q, max_results=None):
                return [{"title": "X", "href": "https://example.com/x"}]

        fake_mod.DDGS = FakeDDGS  # type: ignore[attr-defined]

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(sys.modules, {"duckduckgo_search": fake_mod}):
                out = tools.web_search(
                    query="x",
                    cache_dir=td,
                    provider="duckduckgo-search",
                    ttl_sec=0,
                    max_results=1,
                )
        self.assertIn("https://example.com/x", out)

    def test_duckduckgo_search_recursionerror_falls_back_to_html(self):
        captured = {"http_called": 0}

        class FakeDDGS:
            def __init__(self, **kwargs):
                pass

            def text(self, q, max_results=None):
                raise RecursionError("boom")

        fake_mod = types.ModuleType("duckduckgo_search")
        fake_mod.DDGS = FakeDDGS  # type: ignore[attr-defined]

        def _fake_http_get_text(*, url, proxy="", timeout_sec=20.0, user_agent=""):
            captured["http_called"] += 1
            return (
                '<a class="result__a" href="https://example.com/a">Title A</a>'
                '<a class="result__a" href="https://example.com/b">Title B</a>'
            )

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(sys.modules, {"duckduckgo_search": fake_mod}):
                with mock.patch.object(tools, "_http_get_text", side_effect=_fake_http_get_text):
                    out = tools.web_search_duckduckgo(
                        query="x",
                        cache_dir=td,
                        ttl_sec=0,
                        max_results=2,
                    )
        self.assertIn("DuckDuckGo results:", out)
        self.assertIn("fell back to HTML endpoint", out)
        self.assertEqual(captured["http_called"], 1)


if __name__ == "__main__":
    unittest.main()
