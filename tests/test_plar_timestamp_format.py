import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aurex.tools.plar_tools import _compact_comment, _compact_qe_item, plar_get_comments


class VerifiedTimestampTests(unittest.TestCase):
    def test_comment_exact_milliseconds_have_both_named_timezones(self):
        raw = {"ID": "comment", "Timestamp": 1775660641953, "Content": "hello"}
        before = copy.deepcopy(raw)
        with patch("aurex.tools.plar_tools.plar.get_comments", return_value=[raw]):
            result = plar_get_comments(SimpleNamespace(user=object()), {
                "target_type": "Experiment", "target_id": "a" * 24,
            })[0]
        self.assertEqual(result["Timestamp"], 1775660641953)
        self.assertEqual(result["ts_ms"], 1775660641953)
        self.assertEqual(result["timestamp_utc"], "2026-04-08T15:04:01.953+00:00")
        self.assertEqual(result["timestamp_Asia_Shanghai"], "2026-04-08T23:04:01.953+08:00")
        self.assertEqual(raw, before)

    def test_experiment_creation_date_is_raw_plus_formatted(self):
        raw = {"ID": "experiment", "CreationDate": 1752917284273, "UpdateDate": 1775660641953,
               "SortingDate": 1775660641953}
        result = _compact_qe_item(raw)
        self.assertEqual(result["CreationDate"], raw["CreationDate"])
        self.assertEqual(result["creation_date"], raw["CreationDate"])
        self.assertEqual(result["creation_date_utc"], "2025-07-19T09:28:04.273+00:00")
        self.assertEqual(result["creation_date_Asia_Shanghai"], "2025-07-19T17:28:04.273+08:00")
        self.assertNotIn("sorting_date_utc", result)
        self.assertNotIn("update_date_utc", result)

    def test_integral_float_retains_original_numeric_value(self):
        value = float(1775660641953)
        result = _compact_comment({"ID": "c", "Timestamp": value, "Content": "x"})
        self.assertIsInstance(result["Timestamp"], float)
        self.assertEqual(result["timestamp_utc"], "2026-04-08T15:04:01.953+00:00")

    def test_invalid_or_sentinel_values_do_not_fabricate_dates(self):
        for value in (None, True, False, "1775660641953", "bad", 0, -1, 1.5,
                      float("nan"), float("inf"), 10**100):
            with self.subTest(value=value):
                result = _compact_comment({"ID": "c", "Timestamp": value, "Content": "x"})
                self.assertIs(result["Timestamp"], value)
                self.assertNotIn("timestamp_utc", result)
                self.assertNotIn("timestamp_Asia_Shanghai", result)
                experiment = _compact_qe_item({"ID": "e", "CreationDate": value})
                self.assertIs(experiment["CreationDate"], value)
                self.assertNotIn("creation_date_utc", experiment)
                self.assertNotIn("creation_date_Asia_Shanghai", experiment)

    def test_alias_timestamps_are_not_assumed_to_be_milliseconds(self):
        for alias in ("Time", "CreateTime", "CreatedAt", "Created"):
            with self.subTest(alias=alias):
                result = _compact_comment({"ID": "c", alias: 1775660641953, "Content": "x"})
                self.assertNotIn("timestamp_utc", result)
                self.assertNotIn("timestamp_Asia_Shanghai", result)

    def test_explicit_utc_plus_eight_date_can_cross_midnight(self):
        result = _compact_comment({"ID": "c", "Timestamp": 1775690400000, "Content": "x"})
        self.assertEqual(result["timestamp_utc"], "2026-04-08T23:20:00.000+00:00")
        self.assertEqual(result["timestamp_Asia_Shanghai"], "2026-04-09T07:20:00.000+08:00")


if __name__ == "__main__":
    unittest.main()
