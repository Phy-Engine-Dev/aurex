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
        self.assertEqual(plain["body_text"], "正文")
        self.assertEqual(plain["author"], {"id": "a" * 24, "nickname": "Author"})
        self.assertEqual(plain["classification_and_state_raw"]["Type"], 3)
        self.assertTrue(plain["cover_available"])
        self.assertNotIn("images", plain)
        self.assertEqual(pictured["images"][0]["path"], directory + "/cover.jpg")
        self.assertIs(pictured["external_write_performed"], False)
        self.assertIn("with_image", PLAR_SUMMARY_TOOL["parameters"]["properties"])


if __name__ == "__main__":
    unittest.main()
