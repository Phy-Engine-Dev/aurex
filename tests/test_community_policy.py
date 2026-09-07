import json
from pathlib import Path
import tempfile
import unittest

from aurex.config import ConfigError, ContextPolicyConfig, load_config, parse_context_policy, save_config


class CommunityPolicyTests(unittest.TestCase):
    def test_defaults_are_retrieval_policy_not_task_budgets(self):
        cfg = ContextPolicyConfig().resolved()
        self.assertEqual(cfg.community_recent_hours, 24)
        self.assertEqual(cfg.community_max_comments, 100)
        self.assertEqual(cfg.community_recent_comments, 20)
        self.assertTrue(cfg.auto_compact)

    def test_profiles_can_override_all_community_window_fields_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps({'context': {'community_recent_hours': 2.5,
                'profile': 'wall', 'profiles': {'wall': {'community_recent_hours': 48,
                    'community_max_comments': 200, 'community_recent_comments': 30}}}}))
            cfg = load_config(str(path))
            self.assertEqual(cfg.context.community_recent_hours, 2.5)
            selected = cfg.context.resolved()
            self.assertEqual((selected.community_recent_hours, selected.community_max_comments,
                              selected.community_recent_comments), (48, 200, 30))
            save_config(cfg, str(path))
            self.assertEqual(load_config(str(path)).context, cfg.context)

    def test_zero_hours_is_allowed_and_unsafe_or_mistyped_retrieval_settings_are_rejected(self):
        self.assertEqual(parse_context_policy({'community_recent_hours': 0}).community_recent_hours, 0)
        invalid = [{'community_recent_hours': -1}, {'community_recent_hours': True},
                   {'community_recent_hours': float('inf')}, {'community_recent_hours': float('nan')},
                   {'community_recent_hours': '24'}, {'community_max_comments': 501},
                   {'community_max_comments': 0}, {'community_max_comments': True},
                   {'community_recent_comments': 0}, {'community_recent_comments': 101},
                   {'profiles': {'bad': {'community_recent_hours': -1}}}]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ConfigError):
                parse_context_policy(raw)


if __name__ == '__main__':
    unittest.main()
