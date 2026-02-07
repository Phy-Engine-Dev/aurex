import os
import sys
import unittest


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from text import contains_mention, strip_leading_mention


class TestText(unittest.TestCase):
    def test_contains_mention_allows_cjk_after_handle(self):
        self.assertTrue(contains_mention("@aurex你好", mention_tag="@aurex"))
        self.assertTrue(contains_mention("hello @aurex你好", mention_tag="@aurex"))

    def test_contains_mention_rejects_ascii_word_continuation(self):
        self.assertFalse(contains_mention("@aurex2 hi", mention_tag="@aurex"))
        self.assertFalse(contains_mention("@aurex_ hi", mention_tag="@aurex"))

    def test_strip_leading_mention(self):
        self.assertEqual(strip_leading_mention("@aurex hello", mention_tag="@aurex"), "hello")
        self.assertEqual(strip_leading_mention("@aurex你好", mention_tag="@aurex"), "你好")
        self.assertEqual(strip_leading_mention("＠aurex  hi", mention_tag="@aurex"), "hi")


if __name__ == "__main__":
    unittest.main()

