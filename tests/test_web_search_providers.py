import os
import socket
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from aurex.tools.registry import ToolError
from aurex.tools.web_search import (
    web_search, web_fetch, fetch_public_bytes, _public_endpoint, _relevant_results,
)


def runtime(**options):
    defaults = dict(provider="auto", api_key_env="TEST_BRAVE_KEY", base_url="", timeout_sec=5)
    defaults.update(options)
    return SimpleNamespace(config=SimpleNamespace(web_search=SimpleNamespace(**defaults)))


class SearchProviderTests(unittest.TestCase):
    def test_brave_uses_real_auth_header_and_json_results(self):
        response = mock.Mock()
        response.json.return_value = {"web": {"results": [{"title": "TI resistor", "url": "https://ti.com/a", "description": "resistor datasheet"}]}}
        with mock.patch.dict(os.environ, {"TEST_BRAVE_KEY": "private-key"}), mock.patch("aurex.tools.web_search.requests.get", return_value=response) as get:
            out = web_search(runtime(), {"query": "resistor", "time_range": "m"})
        self.assertEqual(get.call_args.kwargs["headers"]["X-Subscription-Token"], "private-key")
        self.assertEqual(get.call_args.kwargs["params"]["freshness"], "pm")
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        self.assertEqual(out[0]["provider"], "brave")

    def test_searxng_json_configured_endpoint(self):
        response = mock.Mock()
        response.json.return_value = {"results": [{"title": "Ohm's law source", "url": "https://example.org", "content": "ohms law reference"}]}
        with mock.patch("aurex.tools.web_search.requests.get", return_value=response) as get:
            out = web_search(runtime(provider="searxng", base_url="http://127.0.0.1:8888"), {"query": "ohms law"})
        self.assertEqual(get.call_args.args[0], "http://127.0.0.1:8888/search")
        self.assertEqual(get.call_args.kwargs["params"]["format"], "json")
        self.assertEqual(out[0]["provider"], "searxng")

    def test_brave_unrelated_results_are_filtered_before_model_context(self):
        response = mock.Mock()
        response.json.return_value = {
            "web": {"results": [
                {"title": "Movie times", "url": "https://example.com/movie",
                 "description": "Cinema tickets and trailer"},
                {"title": "Resistor datasheet", "url": "https://example.com/r",
                 "description": "10 kOhm resistor specifications"},
            ]}
        }
        with mock.patch.dict(os.environ, {"TEST_BRAVE_KEY": "private-key"}), \
             mock.patch("aurex.tools.web_search.requests.get", return_value=response):
            out = web_search(runtime(provider="brave"), {"query": "resistor"})
        self.assertEqual([item["title"] for item in out], ["Resistor datasheet"])

    def test_searxng_all_unrelated_results_report_no_relevant_results(self):
        response = mock.Mock()
        response.json.return_value = {"results": [{
            "title": "Movie times", "url": "https://example.com/movie",
            "content": "Cinema tickets and trailer",
        }]}
        with mock.patch("aurex.tools.web_search.requests.get", return_value=response), \
             self.assertRaisesRegex(ToolError, "STOP_NO_USEFUL_RESULTS.*no provider returned relevant results"):
            web_search(runtime(provider="searxng", base_url="http://127.0.0.1:8888"),
                       {"query": "resistor"})

    def test_empty_provider_result_returns_terminal_stop_signal(self):
        response = mock.Mock()
        response.json.return_value = {"results": []}
        with mock.patch("aurex.tools.web_search.requests.get", return_value=response), \
             self.assertRaisesRegex(ToolError, "STOP_NO_USEFUL_RESULTS") as stopped:
            web_search(runtime(provider="searxng", base_url="http://127.0.0.1:8888"),
                       {"query": "very specific circuit fact"})
        self.assertIn("do not retry unchanged or with synonymous wording", str(stopped.exception))
        self.assertIn("NO_EVIDENCE", str(stopped.exception))

    def test_provider_transport_failure_is_retryable_not_semantic_no_evidence(self):
        with mock.patch("aurex.tools.web_search.requests.get", side_effect=TimeoutError("offline")), \
             self.assertRaisesRegex(ToolError, "SEARCH_UNAVAILABLE_RETRYABLE") as failed:
            web_search(runtime(provider="searxng", base_url="http://127.0.0.1:8888"),
                       {"query": "运放输出饱和"})
        self.assertNotIn("STOP_NO_USEFUL_RESULTS", str(failed.exception))

    def test_chinese_aliases_and_natural_questions_keep_relevant_results(self):
        cases = [
            ("物实热度怎么算", "物理实验室社区热度计算方法", "作品热度、点赞与收藏"),
            ("运算放大器为什么会饱和", "运放输出饱和原理", "输出电压受到电源轨限制"),
        ]
        for query, title, snippet in cases:
            item = {"title": title, "snippet": snippet, "url": "https://example.org", "provider": "test"}
            with self.subTest(query=query):
                self.assertEqual(_relevant_results([item], query), [item])

    def test_auto_falls_back_after_provider_failure(self):
        rss = mock.Mock(content=b"<rss><channel><item><title>TI resistor</title><link>https://ti.com</link><description>Source</description></item></channel></rss>")
        with mock.patch.dict(os.environ, {"TEST_BRAVE_KEY": "private-key"}), \
             mock.patch("aurex.tools.web_search.requests.get", side_effect=[RuntimeError("authentication private-key"), rss]):
            out = web_search(runtime(), {"query": "resistor"})
        self.assertEqual(out[0]["provider"], "bing_rss")
        self.assertEqual(out[0]["trust"], "untrusted_external_content")

    def test_no_key_does_not_pretend_brave_is_available(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ToolError, "not set"):
            web_search(runtime(provider="brave"), {"query": "resistor"})

    def test_unrelated_bing_results_fall_back_to_labeled_scholarly_metadata(self):
        unrelated = mock.Mock(content=b"<rss><channel><item><title>Movie times</title><link>https://example.com</link><description>Cinema tickets</description></item></channel></rss>")
        crossref = mock.Mock()
        crossref.json.return_value = {"message": {"items": [{"title": ["Voltage divider and resistor calibrations"], "URL": "https://doi.org/10.123/test", "publisher": "NIST"}]}}
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("aurex.tools.web_search.requests.get", side_effect=[unrelated, crossref]):
            out = web_search(runtime(), {"query": "voltage divider resistor"})
        self.assertEqual(out[0]["provider"], "crossref")
        self.assertEqual(out[0]["scope"], "scholarly_metadata_not_full_text")

    def test_clash_fake_dns_resolves_real_public_ip_without_allowing_literal_fake_ip(self):
        dns = mock.Mock()
        dns.json.return_value = {"Answer": [{"type": 1, "data": "8.8.8.8"}]}
        with mock.patch("aurex.tools.web_search.socket.getaddrinfo", return_value=[(socket.AF_INET, 1, 6, "", ("198.18.0.2", 80))]), \
             mock.patch("aurex.tools.web_search.requests.get", return_value=dns) as get:
            self.assertEqual(_public_endpoint("https://public.example")[1], "8.8.8.8")
            self.assertEqual(get.call_count, 1)
            with self.assertRaises(ToolError):
                _public_endpoint("http://198.18.0.2")
            self.assertEqual(get.call_count, 1)

    def test_public_fetch_blocks_private_and_mixed_dns(self):
        for ip in ("127.0.0.1", "10.1.2.3", "169.254.169.254", "100.123.133.75", "::1"):
            with self.subTest(ip=ip), mock.patch("aurex.tools.web_search.socket.getaddrinfo", return_value=[(socket.AF_INET, 1, 6, "", (ip, 80))]), self.assertRaises(ToolError):
                _public_endpoint("http://example.com")
        with mock.patch("aurex.tools.web_search.socket.getaddrinfo", return_value=[(socket.AF_INET, 1, 6, "", ("8.8.8.8", 80)), (socket.AF_INET, 1, 6, "", ("10.0.0.1", 80))]), self.assertRaises(ToolError):
            _public_endpoint("http://example.com")

    def test_fetch_rechecks_redirect_target(self):
        response = mock.Mock(status=302)
        response.getheader.side_effect = lambda key, default=None: "http://127.0.0.1/secret" if key == "Location" else default
        conn = mock.Mock()
        conn.getresponse.return_value = response
        with mock.patch("aurex.tools.web_search._public_endpoint", side_effect=[(__import__("urllib.parse", fromlist=["urlparse"]).urlparse("http://example.org"), "8.8.8.8", 80), ToolError("private blocked")]) as endpoints, \
             mock.patch("aurex.tools.web_search.socket.create_connection"), \
             mock.patch("aurex.tools.web_search.http.client.HTTPConnection", return_value=conn), \
             self.assertRaisesRegex(ToolError, "private blocked"):
            fetch_public_bytes("http://example.org")
        self.assertEqual(endpoints.call_args_list[-1].args[0], "http://127.0.0.1/secret")
        conn.close.assert_called_once()

    def test_fetch_bounds_text_and_labels_untrusted(self):
        html = b"<html><title>title</title><script>do bad things</script><p>" + b"x" * 3000 + b"</p></html>"
        with mock.patch("aurex.tools.web_search.fetch_public_bytes", return_value=(html, "text/html", "https://example.com")):
            result = web_fetch(runtime(), {"url": "https://example.com", "max_chars": 1000})
        self.assertEqual(len(result["text"]), 1000)
        self.assertTrue(result["truncated"])
        self.assertNotIn("bad things", result["text"])


if __name__ == "__main__":
    unittest.main()
