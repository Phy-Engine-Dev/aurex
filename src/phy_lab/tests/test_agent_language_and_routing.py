import os
import sys
import unittest
from types import SimpleNamespace


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from agent import (  # type: ignore
    _dominant_lang_is_zh,
    _extract_content_ref_anywhere,
    _effective_system_prompt,
    _looks_like_content_intro_request,
    _extract_user_work_list_target_name,
    _looks_like_simulation_request,
    _normalize_embedded_plar_refs,
    _looks_like_user_work_list_request,
)


class TestAgentLanguageAndRouting(unittest.TestCase):
    def test_dominant_lang_zh(self):
        self.assertTrue(_dominant_lang_is_zh("MapMaths发布的实验结果有哪些？"))
        self.assertTrue(_dominant_lang_is_zh("你好"))
        self.assertFalse(_dominant_lang_is_zh("Please list MapMaths experiments"))

    def test_system_prompt_reinforces_language(self):
        cfg = SimpleNamespace(agent=SimpleNamespace(system_prompt="BASE", reply_once=True))
        zh = _effective_system_prompt(cfg=cfg, user_text="你好，解释一下串联电路")
        self.assertIn("你的输出必须使用中文", zh)
        en = _effective_system_prompt(cfg=cfg, user_text="Explain a series circuit")
        self.assertIn("Language (must follow)", en)

    def test_extract_user_work_list_target_name(self):
        self.assertEqual(
            _extract_user_work_list_target_name("MapMaths发布的实验结果有哪些？"),
            "MapMaths",
        )
        self.assertEqual(
            _extract_user_work_list_target_name("列出 MapMaths 的实验"),
            "MapMaths",
        )
        self.assertEqual(
            _extract_user_work_list_target_name("@MapMaths 的作品有哪些"),
            "MapMaths",
        )

    def test_looks_like_user_work_list_request(self):
        self.assertTrue(_looks_like_user_work_list_request("MapMaths发布的实验结果有哪些？"))
        self.assertFalse(_looks_like_user_work_list_request("告诉我紫兰斋的第一个作品是什么"))

    def test_looks_like_simulation_request(self):
        # "模拟" alone is ambiguous; require circuit-ish hints.
        self.assertFalse(_looks_like_simulation_request("电车实验模拟器，引发了我对电车难题的思考"))
        self.assertTrue(_looks_like_simulation_request("帮我模拟一下这个电路，看看电压电流"))
        self.assertTrue(_looks_like_simulation_request("请仿真一下这个电路"))

    def test_extract_content_ref_anywhere(self):
        sid = "0123456789abcdef01234567"
        s = _normalize_embedded_plar_refs(f"总结一下这篇文章：<experiment={sid}>标题</experiment>")
        self.assertIn(f"experiment:{sid}", s)
        self.assertEqual(_extract_content_ref_anywhere(s), ("Experiment", sid))
        self.assertEqual(_extract_content_ref_anywhere(f"discussion:{sid}"), ("Discussion", sid))
        self.assertEqual(_extract_content_ref_anywhere(sid), (None, sid))
        self.assertIsNone(_extract_content_ref_anywhere(f"uid:{sid} 的作品"))

    def test_looks_like_content_intro_request_with_summary_words(self):
        sid = "0123456789abcdef01234567"
        self.assertTrue(_looks_like_content_intro_request(f"总结一下 experiment:{sid} 在说什么"))


if __name__ == "__main__":
    unittest.main()
