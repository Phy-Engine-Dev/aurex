import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aurex.tools.plar_tools import PLAR_SUMMARY_TOOL, plar_get_summary


class ExactSummaryToolTests(unittest.TestCase):
    def test_type3_discussion_returns_text_and_downloads_cover_only_when_requested(self):
        sid = "6a9bd3da5e55336480704f9b"
        response = {"Status": 200, "Data": {"Summary": {
            "ID": sid, "Subject": "aurex v3 即将发布", "Description": "正文",
            "Category": "Discussion", "Type": 3, "Tags": ["Type-3", "问与答"],
            "Image": 0, "CreationDate": 1788597210274,
            "Stars": 12, "Supports": 4, "Visits": 88, "Comments": 3,
            "User": {"ID": "a" * 24, "Nickname": "Author"},
        }}}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch("aurex.tools.plar_tools.plar.get_summary", return_value=response) as api, \
             mock.patch("aurex.tools.plar_tools.download_public_image", return_value={
                 "url": "http://cdn/final.jpg", "path": directory + "/cover.jpg",
                 "mime_type": "image/jpeg", "bytes": 12}) as download:
            runtime = SimpleNamespace(user=object(), cache_dir=directory)
            plain = plar_get_summary(runtime, {"summary_id": sid, "category": "Discussion"})
            pictured = plar_get_summary(runtime, {"summary_id": sid, "category": "Discussion",
                                                   "with_image": True})
        self.assertEqual(api.call_count, 2)
        download.assert_called_once()
        self.assertEqual(plain["title"], "aurex v3 即将发布")
        self.assertNotIn("body_text", plain)
        self.assertEqual(plain["body"], {
            "available": True, "characters": 2, "reader": "plar_read_body",
        })
        self.assertEqual(plain["author"], {"id": "a" * 24, "nickname": "Author"})
        self.assertEqual(plain["classification"]["type"], 3)
        self.assertEqual(plain["classification"]["tags"], ["Type-3", "问与答"])
        self.assertEqual(plain["metrics"], {
            "stars": 12, "supports": 4, "visits": 88, "comments": 3,
        })
        self.assertEqual(plain["creation_date_ms"], 1788597210274)
        self.assertIn("creation_date_utc", plain)
        self.assertNotIn("Experiment", plain)
        self.assertNotIn("StatusSave", str(plain))
        self.assertTrue(plain["cover_available"])
        self.assertEqual(plain["cover_count"], 1)
        self.assertNotIn("url", plain["cover_sources"][0])
        self.assertNotIn("images", plain)
        self.assertEqual(pictured["images"][0]["path"], directory + "/cover.jpg")
        self.assertIs(pictured["external_write_performed"], False)
        self.assertIn("with_image", PLAR_SUMMARY_TOOL["parameters"]["properties"])

    def test_cover_discovery_is_bounded_and_raw_urls_stay_out_of_metadata(self):
        sid = "6a9bd3da5e55336480704f9b"
        response = {"Status": 200, "Data": {"Summary": {
            "ID": sid, "Subject": "封面边界", "Description": "正文",
            "Images": ([{"URL": "https://example.org/" + "x" * 3000}] +
                       [{"URL": f"https://example.org/{index}.jpg"} for index in range(30)]),
        }}}
        with mock.patch("aurex.tools.plar_tools.plar.get_summary", return_value=response):
            out = plar_get_summary(SimpleNamespace(user=object(), cache_dir=tempfile.gettempdir()),
                                   {"summary_id": sid})
        self.assertEqual(out["cover_count"], 8)
        self.assertLessEqual(len(out["cover_sources"]), 4)
        self.assertNotIn("https://example.org", str(out))


if __name__ == "__main__":
    unittest.main()
