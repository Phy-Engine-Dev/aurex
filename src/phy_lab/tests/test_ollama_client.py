import os
import sys
import types
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from ollama import OllamaClient, OllamaError


class TestOllamaClient(unittest.TestCase):
    def test_localhost_disables_trust_env(self):
        captured = {"trust_env": None, "url": None}

        class _Resp:
            ok = True

            def json(self):
                return {"message": {"content": "OK"}}

        class _Session:
            def __init__(self):
                self.trust_env = True

            def post(self, url, json=None, timeout=None):
                captured["trust_env"] = self.trust_env
                captured["url"] = url
                return _Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.Session = _Session  # type: ignore[attr-defined]
        fake_requests.RequestException = Exception  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            c = OllamaClient(base_url="http://127.0.0.1:11434", model="m")
            out = c.chat(messages=[{"role": "user", "content": "x"}])
        self.assertEqual(out, "OK")
        self.assertEqual(captured["trust_env"], False)
        self.assertEqual(captured["url"], "http://127.0.0.1:11434/api/chat")

    def test_empty_content_raises(self):
        class _Resp:
            ok = True

            def json(self):
                return {"message": {"content": "   "}}

        class _Session:
            def __init__(self):
                self.trust_env = True

            def post(self, *_a, **_kw):
                return _Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.Session = _Session  # type: ignore[attr-defined]
        fake_requests.RequestException = Exception  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            c = OllamaClient(base_url="http://127.0.0.1:11434", model="m")
            with self.assertRaises(OllamaError):
                c.chat(messages=[{"role": "user", "content": "x"}])

    def test_empty_content_retries_once(self):
        calls = {"n": 0}

        class _Resp:
            ok = True

            def __init__(self, content):
                self._content = content

            def json(self):
                return {"message": {"content": self._content}}

        class _Session:
            def __init__(self):
                self.trust_env = True

            def post(self, *_a, **_kw):
                calls["n"] += 1
                return _Resp("   " if calls["n"] == 1 else "OK")

        fake_requests = types.ModuleType("requests")
        fake_requests.Session = _Session  # type: ignore[attr-defined]
        fake_requests.RequestException = Exception  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            c = OllamaClient(base_url="http://127.0.0.1:11434", model="m")
            out = c.chat(messages=[{"role": "user", "content": "x"}])
        self.assertEqual(out, "OK")
        self.assertEqual(calls["n"], 2)

    def test_empty_content_with_tool_calls_returns_tool_json(self):
        class _Resp:
            ok = True

            def json(self):
                return {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_x",
                                "function": {
                                    "name": "plar_list_plar",
                                    "arguments": {"kind": "hot", "category": "Experiment", "take": 5},
                                },
                            }
                        ],
                    }
                }

        class _Session:
            def __init__(self):
                self.trust_env = True

            def post(self, *_a, **_kw):
                return _Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.Session = _Session  # type: ignore[attr-defined]
        fake_requests.RequestException = Exception  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            c = OllamaClient(base_url="http://127.0.0.1:11434", model="m")
            out = c.chat(messages=[{"role": "user", "content": "x"}])
        self.assertIn('"tool": "list_plar"', out)


if __name__ == "__main__":
    unittest.main()
