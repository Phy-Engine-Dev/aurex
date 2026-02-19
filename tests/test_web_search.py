import os
import sys
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from aurex.config import AurexConfig  # noqa: E402
from aurex.tools.registry import ToolRuntime  # noqa: E402
from aurex.tools.web_search import ddg_web_search  # noqa: E402


class _Resp:
    def __init__(self, *, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class TestWebSearch(unittest.TestCase):
    def test_ddg_web_search_parses_duckduckgo_html_and_unwraps_url(self):
        html = """
        <html><body>
          <div class="results">
            <div class="result__body">
              <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ffoo">  Example   Title </a>
              <a class="result__snippet">  Snippet   one </a>
            </div>
            <div class="result__body">
              <a class="result__a" href="https://example.org/bar">Second</a>
              <div class="result__snippet">Second snippet</div>
            </div>
          </div>
        </body></html>
        """

        cfg = AurexConfig()
        rt = ToolRuntime(task_id="T", user_lang="zh", config_path=os.path.join(ROOT, "dummy.json"), config=cfg, cache_dir=os.path.join(ROOT, ".aurex", "cache"))

        with mock.patch("aurex.tools.web_search.requests.post", return_value=_Resp(text=html)) as mp:
            out = ddg_web_search(rt, {"query": "test", "max_results": 10})

        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["title"], "Example Title")
        self.assertEqual(out[0]["url"], "https://example.com/foo")
        self.assertEqual(out[0]["snippet"], "Snippet one")
        self.assertEqual(out[1]["title"], "Second")
        self.assertEqual(out[1]["url"], "https://example.org/bar")

        self.assertEqual(mp.call_count, 1)
        _url = mp.call_args[0][0]
        self.assertIn("duckduckgo.com", _url)
        data = mp.call_args.kwargs.get("data") or {}
        self.assertEqual(data.get("q"), "test")

    def test_ddg_web_search_falls_back_to_bing_when_ddg_empty(self):
        ddg_empty = "<html><body>empty</body></html>"
        bing_html = """
        <html><body>
          <ol>
            <li class="b_algo">
              <h2><a href="https://bing.example/a">Bing Result</a></h2>
              <p>bing snippet</p>
            </li>
          </ol>
        </body></html>
        """

        cfg = AurexConfig()
        rt = ToolRuntime(task_id="T", user_lang="zh", config_path=os.path.join(ROOT, "dummy.json"), config=cfg, cache_dir=os.path.join(ROOT, ".aurex", "cache"))

        with mock.patch("aurex.tools.web_search.requests.post", return_value=_Resp(text=ddg_empty)):
            with mock.patch("aurex.tools.web_search.requests.get", return_value=_Resp(text=bing_html)):
                out = ddg_web_search(rt, {"query": "test", "max_results": 5})

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["title"], "Bing Result")
        self.assertEqual(out[0]["url"], "https://bing.example/a")
        self.assertIn("bing", out[0]["snippet"])


@unittest.skipUnless(os.environ.get("AUREX_NET_TESTS") == "1", "network tests disabled (set AUREX_NET_TESTS=1)")
class TestWebSearchNetwork(unittest.TestCase):
    def test_ddg_web_search_returns_results_for_beijing_weather(self):
        cfg = AurexConfig()
        rt = ToolRuntime(task_id="T", user_lang="zh", config_path=os.path.join(ROOT, "dummy.json"), config=cfg, cache_dir=os.path.join(ROOT, ".aurex", "cache"))
        out = ddg_web_search(rt, {"query": "今日 北京 天气", "max_results": 3, "time_range": "d"})
        self.assertIsInstance(out, list)
        self.assertGreater(len(out), 0)
        self.assertTrue(any((r.get("title") or "").strip() for r in out))


if __name__ == "__main__":
    unittest.main()

