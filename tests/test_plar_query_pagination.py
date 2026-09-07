import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import jsonschema

from aurex.tools.plar_tools import (
    PLAR_QUERY_TOOL,
    plar_oldest_by_user,
    plar_query_experiments,
)
from aurex.tools.registry import ToolError


class QueryPaginationTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(user=object())

    def test_default_cursor_is_forwarded_without_extra_fetch(self):
        with patch("aurex.tools.plar_tools.plar.query_experiments", return_value=[{"ID": "next"}]) as api:
            result = plar_query_experiments(self.runtime, {"sort": "Default", "from_skip": "last", "skip": 0})
        self.assertEqual([item["id"] for item in result], ["next"])
        api.assert_called_once()
        self.assertEqual(api.call_args.kwargs["from_skip"], "last")
        self.assertEqual(api.call_args.kwargs["skip"], 0)
        self.assertEqual(api.call_args.kwargs["sort"], "Default")

    def test_popularity_offset_and_range_are_not_forced_to_zero(self):
        with patch("aurex.tools.plar_tools.plar.query_experiments", return_value=[]) as api:
            self.assertEqual(plar_query_experiments(self.runtime, {
                "sort": "Popularity", "skip": 24, "from_skip": "last", "days": 0,
            }), [])
        api.assert_called_once()
        self.assertEqual(api.call_args.kwargs["skip"], 24)
        self.assertEqual(api.call_args.kwargs["from_skip"], "last")
        self.assertEqual(api.call_args.kwargs["days"], 0)

    def test_stable_case_insensitive_dedup_does_not_mutate_page_or_seen(self):
        page = [{"ID": "A", "Subject": "first"}, {"ID": "a", "Subject": "duplicate"},
                {"ID": "B"}, {"Id": "C"}]
        before = copy.deepcopy(page)
        seen = [" b "]
        with patch("aurex.tools.plar_tools.plar.query_experiments", return_value=page) as api:
            result = plar_query_experiments(self.runtime, {"seen_ids": seen})
        api.assert_called_once()
        self.assertEqual([item["id"] for item in result], ["A", "C"])
        self.assertEqual(result[0]["subject"], "first")
        self.assertEqual(page, before)
        self.assertEqual(seen, [" b "])

    def test_repeated_page_is_not_reported_as_end(self):
        with patch("aurex.tools.plar_tools.plar.query_experiments", return_value=[{"ID": "a"}]) as api:
            with self.assertRaisesRegex(ToolError, "no progress.*not proof"):
                plar_query_experiments(self.runtime, {"seen_ids": ["a"]})
        api.assert_called_once()

    def test_empty_page_is_valid_end_and_refresh_may_repeat(self):
        with patch("aurex.tools.plar_tools.plar.query_experiments", return_value=[]):
            self.assertEqual(plar_query_experiments(self.runtime, {"seen_ids": ["a"]}), [])
        with patch("aurex.tools.plar_tools.plar.query_experiments", return_value=[{"ID": "a"}]):
            self.assertEqual(plar_query_experiments(self.runtime, {})[0]["id"], "a")

    def test_bad_seen_ids_fail_before_api(self):
        for invalid in (None, "a", {"a": True}, [1], [" "], ["a" * 129], ["a"] * 4097):
            with self.subTest(value=repr(invalid)[:40]):
                with patch("aurex.tools.plar_tools.plar.query_experiments") as api:
                    with self.assertRaisesRegex(ToolError, "seen_ids"):
                        plar_query_experiments(self.runtime, {"seen_ids": invalid})
                api.assert_not_called()

    def test_missing_id_fails_explicitly(self):
        with patch("aurex.tools.plar_tools.plar.query_experiments", return_value=[{"Subject": "no id"}]):
            with self.assertRaisesRegex(ToolError, "without a usable ID"):
                plar_query_experiments(self.runtime, {})

    def test_schema_documents_both_pagination_modes_and_preserves_list_api(self):
        schema = PLAR_QUERY_TOOL["parameters"]
        jsonschema.validate({"category": "Experiment", "sort": "Popularity", "skip": 24, "seen_ids": ["a"]}, schema)
        for term in ("Default", "from_skip", "Popularity", "skip", "seen_ids"):
            self.assertIn(term, PLAR_QUERY_TOOL["description"])
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({"category": "Experiment", "seen_ids": "a"}, schema)


class OldestPaginationTests(unittest.TestCase):
    def scan(self, pages, *, max_pages=10):
        with patch("aurex.tools.plar_tools.plar.query_experiments", side_effect=pages) as api:
            result = plar_oldest_by_user(SimpleNamespace(user=object()), {
                "user_id": "a" * 24, "category": "Experiment", "take": 2, "max_pages": max_pages,
            })
        return result, api

    def test_repeated_full_page_stops_incomplete_without_scanning_limit(self):
        page = [{"ID": "a", "CreationDate": 200}, {"ID": "b", "CreationDate": 100}]
        result, api = self.scan([page, page])
        self.assertEqual(api.call_count, 2)
        self.assertEqual(result["item"]["id"], "b")
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["stop_reason"], "pagination_no_progress")
        self.assertEqual(result["unique_items_scanned"], 2)
        self.assertEqual(result["duplicate_items_ignored"], 2)

    def test_overlapping_pages_with_progress_are_allowed(self):
        result, api = self.scan([
            [{"ID": "a", "CreationDate": 300}, {"ID": "b", "CreationDate": 200}],
            [{"ID": "b", "CreationDate": 200}, {"ID": "c", "CreationDate": 100}],
            [],
        ])
        self.assertEqual(api.call_count, 3)
        self.assertFalse(result["incomplete"])
        self.assertEqual(result["item"]["id"], "c")
        self.assertEqual(result["unique_items_scanned"], 3)
        self.assertEqual(result["duplicate_items_ignored"], 1)
        self.assertEqual(api.call_args_list[2].kwargs["from_skip"], "c")

    def test_cursor_not_advancing_despite_new_front_item_is_incomplete(self):
        result, api = self.scan([
            [{"ID": "a", "CreationDate": 300}, {"ID": "b", "CreationDate": 200}],
            [{"ID": "c", "CreationDate": 100}, {"ID": "b", "CreationDate": 200}],
        ])
        self.assertEqual(api.call_count, 2)
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["stop_reason"], "pagination_no_progress")
        self.assertEqual(result["item"]["id"], "c")

    def test_missing_id_does_not_claim_complete_scan(self):
        result, api = self.scan([[{"ID": "a", "CreationDate": 100}, {"CreationDate": 50}]])
        self.assertEqual(api.call_count, 1)
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["stop_reason"], "missing_pagination_id")

    def test_max_pages_limit_is_not_expanded(self):
        result, api = self.scan([[{"ID": "a", "CreationDate": 200}, {"ID": "b", "CreationDate": 100}]], max_pages=1)
        self.assertEqual(api.call_count, 1)
        self.assertTrue(result["incomplete"])
        self.assertEqual(result["stop_reason"], "max_pages")


if __name__ == "__main__":
    unittest.main()
