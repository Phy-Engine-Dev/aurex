import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aurex.tools import create_registry
from aurex.tools.content import (
    PLAR_READ_BODY_TOOL,
    PLAR_READ_TITLE_TOOL,
    plar_read_body,
    plar_read_title,
)
from aurex.tools.registry import ToolError


SID = "6a9bd3da5e55336480704f9b"


def _response(*, title="标题", body="第一段\n第二段"):
    return {"Status": 200, "Data": {"Summary": {
        "ID": SID,
        "Subject": title,
        "Description": body,
        # Electrical payload-like fields must never leak from the prose tools.
        "Experiment": {"StatusSave": '{"Elements":[{"Identifier":"C1"}]}', "Wires": [1, 2]},
    }}}


class PlarContentToolTests(unittest.TestCase):
    def runtime(self):
        return SimpleNamespace(user=object(), cache_dir=tempfile.gettempdir())

    def test_title_reader_returns_only_bounded_title_metadata(self):
        runtime = self.runtime()
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=_response(title="一体式加减排序", body="x" * 5000)) as get:
            out = plar_read_title(runtime, {"summary_id": SID})
        get.assert_called_once_with(runtime.user, summary_id=SID, category_value="Experiment")
        self.assertEqual(out["title"], "一体式加减排序")
        self.assertEqual(out["source_field"], "Subject")
        self.assertNotIn("body_text", out)
        self.assertNotIn("Experiment", out)
        self.assertFalse(out["title_truncated"])

    def test_title_reader_never_pages_or_silently_truncates_title(self):
        title = "题" * 4000
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=_response(title=title)) as get:
            out = plar_read_title(self.runtime(), {"summary_id": SID})
        self.assertEqual(out["title"], title)
        self.assertEqual(out["title_characters"], 4000)
        self.assertFalse(out["title_truncated"])
        get.assert_called_once()

    def test_body_reader_reads_a_bounded_window_without_exposing_raw_json(self):
        body = "前缀\n" + "正文内容" * 10000
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=_response(body=body)):
            out = plar_read_body(self.runtime(), {"summary_id": SID, "length": 32})
        self.assertEqual(out["mode"], "read")
        self.assertEqual(out["offset"], 0)
        self.assertEqual(len(out["text"]), 32)
        self.assertTrue(out["has_more"])
        self.assertNotIn("StatusSave", str(out))
        self.assertNotIn("Elements", str(out))

    def test_body_reader_never_treats_content_field_as_prose(self):
        response = {"Status": 200, "Data": {"Summary": {
            "ID": SID,
            "Subject": "无正文",
            "Content": '{"StatusSave":{"Elements":[{"Identifier":"C1"}]}}',
        }}}
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=response):
            out = plar_read_body(self.runtime(), {"summary_id": SID})
        self.assertEqual(out["text"], "")
        self.assertEqual(out["body_characters"], 0)
        self.assertTrue(out["stop"])
        self.assertNotIn("StatusSave", str(out))

        response["Data"]["Summary"]["Description"] = {
            "Content": '{"StatusSave":{"Elements":[{"Identifier":"C2"}]}}'}
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=response):
            nested = plar_read_body(self.runtime(), {"summary_id": SID})
        self.assertEqual(nested["text"], "")
        self.assertNotIn("StatusSave", str(nested))

    def test_body_literal_search_returns_small_context_excerpts(self):
        body = "开头\n目标：R1=10Ω\n中间\n目标：R1=10Ω\n结尾"
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=_response(body=body)):
            out = plar_read_body(self.runtime(), {
                "summary_id": SID, "mode": "search", "pattern": "R1=10Ω", "context_chars": 2,
            })
        self.assertTrue(out["found"])
        self.assertEqual(out["match_count_returned"], 2)
        self.assertEqual([m["text"] for m in out["matches"]], ["标：R1=10Ω\n中", "标：R1=10Ω\n结"])
        self.assertEqual(out["matches"][0]["start"], body.index("R1=10Ω"))
        self.assertTrue(out["search_complete"])
        self.assertTrue(out["stop"])
        self.assertEqual(out["stop_reason"], "all_matches_returned")

    def test_body_regex_search_is_bounded_and_reports_invalid_pattern(self):
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=_response(body="R1=10Ω; R2=20Ω")):
            out = plar_read_body(self.runtime(), {
                "summary_id": SID, "mode": "regex", "pattern": r"R\d+=\d+Ω",
            })
            self.assertEqual(out["match_count_returned"], 2)
            with self.assertRaisesRegex(ToolError, "Invalid regular expression"):
                plar_read_body(self.runtime(), {"summary_id": SID, "mode": "regex", "pattern": "["})

    def test_regex_dialect_rejects_backtracking_constructs_before_scanning_body(self):
        body = "a" * 100_000 + "!"
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=_response(body=body)):
            for pattern in (r"(a+)+$", r"a*a*a*b", r"a|b", r"a{1,4}", r"(?!x)y"):
                with self.subTest(pattern=pattern), self.assertRaisesRegex(ToolError, "Unsafe regular expression"):
                    plar_read_body(self.runtime(), {"summary_id": SID, "mode": "regex", "pattern": pattern})
            safe = plar_read_body(self.runtime(), {
                "summary_id": SID, "mode": "regex", "pattern": r"^a+!$", "context_chars": 0,
            })
        self.assertTrue(safe["found"])
        self.assertEqual(safe["matches"][0]["match_characters"], len(body))
        self.assertLessEqual(len(safe["matches"][0]["text"]), 2400)
        self.assertFalse(safe["matches"][0]["match_fully_shown"])

        with mock.patch("aurex.tools.content.plar.get_summary",
                        return_value=_response(body="prefix=useful value; suffix")):
            delimited = plar_read_body(self.runtime(), {
                "summary_id": SID, "mode": "regex", "pattern": r"prefix=[^;]+;",
            })
        self.assertEqual(delimited["matches"][0]["text"], "prefix=useful value; suffix")

    def test_literal_search_paginates_and_reports_completion(self):
        body = " / ".join(f"R1={value}Ω" for value in range(6))
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=_response(body=body)):
            first = plar_read_body(self.runtime(), {
                "summary_id": SID, "mode": "search", "pattern": "R1=", "max_matches": 2,
            })
            second = plar_read_body(self.runtime(), {
                "summary_id": SID, "mode": "search", "pattern": "R1=", "max_matches": 8,
                "start_offset": first["next_search_offset"],
            })
        self.assertEqual(first["match_count_returned"], 2)
        self.assertTrue(first["has_more_matches"])
        self.assertFalse(first["search_complete"])
        self.assertFalse(first["stop"])
        self.assertEqual(second["match_count_returned"], 4)
        self.assertTrue(second["search_complete"])
        self.assertTrue(second["stop"])

    def test_unicode_casefold_search_maps_matches_to_original_offsets(self):
        body = "x Straße y"
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=_response(body=body)):
            out = plar_read_body(self.runtime(), {
                "summary_id": SID, "mode": "search", "pattern": "STRASSE",
                "case_sensitive": False, "context_chars": 0,
            })
        self.assertEqual(out["matches"][0]["text"], "Straße")
        self.assertEqual((out["matches"][0]["start"], out["matches"][0]["end"]), (2, 8))

    def test_missing_body_search_has_explicit_stop_signal(self):
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=_response(body="只有正文")):
            out = plar_read_body(self.runtime(), {
                "summary_id": SID, "mode": "search", "pattern": "不存在的词",
            })
        self.assertFalse(out["found"])
        self.assertTrue(out["stop"])
        self.assertEqual(out["stop_reason"], "search_pattern_not_found")
        self.assertIn("Do not repeat", out["next_action"])

    def test_repeated_read_is_allowed_and_reexecutes_api(self):
        response = _response(body="可重复读取的正文")
        with mock.patch("aurex.tools.content.plar.get_summary", return_value=response) as get:
            first = plar_read_body(self.runtime(), {"summary_id": SID, "length": 8})
            second = plar_read_body(self.runtime(), {"summary_id": SID, "length": 8})
        self.assertEqual(first, second)
        self.assertEqual(get.call_count, 2)

    def test_omitted_category_resolves_discussion_once_instead_of_repeating_404(self):
        missing = {"Status": 404, "Data": {}}
        with mock.patch("aurex.tools.content.plar.get_summary",
                        side_effect=[missing, _response(body="讨论正文")]) as get:
            out = plar_read_body(self.runtime(), {
                "summary_id": SID, "mode": "read", "length": 20,
            })
        self.assertEqual(out["category"], "Discussion")
        self.assertTrue(out["category_inferred"])
        self.assertEqual(out["text"], "讨论正文")
        self.assertEqual([call.kwargs["category_value"] for call in get.call_args_list],
                         ["Experiment", "Discussion"])

        with mock.patch("aurex.tools.content.plar.get_summary",
                        return_value=missing) as explicit:
            with self.assertRaisesRegex(ToolError, "status 404"):
                plar_read_body(self.runtime(), {
                    "summary_id": SID, "category": "Experiment",
                })
        explicit.assert_called_once()

    def test_schema_and_registry_expose_narrow_tools(self):
        names = {tool.name for tool in create_registry().list()}
        self.assertIn("plar_read_title", names)
        self.assertIn("plar_read_body", names)
        self.assertNotIn("plar_get_experiment_context", names)
        self.assertNotIn("plar_get_status_save", names)
        self.assertEqual(PLAR_READ_TITLE_TOOL["parameters"]["required"], ["summary_id"])
        self.assertEqual(PLAR_READ_BODY_TOOL["parameters"]["properties"]["mode"]["enum"], ["read", "search", "regex"])
        self.assertEqual(PLAR_READ_BODY_TOOL["parameters"]["properties"]["length"]["default"], 4000)

    def test_missing_login_and_bad_ids_fail_closed(self):
        with self.assertRaisesRegex(ToolError, "logged-in"):
            plar_read_title(SimpleNamespace(user=None), {"summary_id": SID})
        with self.assertRaisesRegex(ToolError, "24 hexadecimal"):
            plar_read_body(self.runtime(), {"summary_id": "short"})


if __name__ == "__main__":
    unittest.main()
