import os
import sys
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

import tools as tools_mod


class TestHotSearch(unittest.TestCase):
    def test_hot_experiments_uses_popularity_and_days(self):
        calls = []

        def _fake_qe(user, *, category, take, skip, from_skip, days=None, sort=None, user_id=None):
            calls.append(
                {
                    "category": category,
                    "take": take,
                    "skip": skip,
                    "from_skip": from_skip,
                    "days": days,
                    "sort": sort,
                    "user_id": user_id,
                }
            )
            return [{"ID": "x", "Subject": "S", "Category": category}]

        with mock.patch.object(tools_mod, "query_experiments", side_effect=_fake_qe):
            hits = tools_mod.search_recent_experiments(user=object(), query="热门实验 14天", max_results=2)

        self.assertEqual(len(hits), 1)
        self.assertEqual(calls[0]["category"], "Experiment")
        self.assertEqual(calls[0]["days"], 14)
        self.assertEqual(calls[0]["sort"], "Popularity")

    def test_hot_category_list_parses(self):
        calls = []

        def _fake_qe(user, *, category, take, skip, from_skip, days=None, sort=None, user_id=None):
            calls.append({"category": category, "days": days, "sort": sort})
            return [{"ID": category, "Subject": "S"}]

        with mock.patch.object(tools_mod, "query_experiments", side_effect=_fake_qe):
            hits = tools_mod.search_recent_experiments(user=object(), query="hot category=User/Discussion days=7", max_results=5)

        self.assertEqual([c["category"] for c in calls], ["User", "Discussion"])
        self.assertTrue(all(c["days"] == 7 for c in calls))
        self.assertTrue(all(c["sort"] == "Popularity" for c in calls))
        self.assertEqual(len(hits), 2)


if __name__ == "__main__":
    unittest.main()

