import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


from aurex.config import AurexConfig, ConfigError, load_config  # noqa: E402


class TestResolvePath(unittest.TestCase):
    def test_task_timeout_defaults_to_1800_and_cannot_exceed_hard_cap(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "aurex.json")
            with open(path, "w", encoding="utf-8") as output:
                output.write('{"agent":{"task_timeout_sec":17}}')
            self.assertEqual(load_config(path).agent.task_timeout_sec, 17)
            with open(path, "w", encoding="utf-8") as output:
                output.write('{"agent":{"task_timeout_sec":1801}}')
            with self.assertRaises(ConfigError):
                load_config(path)
        self.assertEqual(AurexConfig().agent.task_timeout_sec, 1800)

    def test_resolve_path_falls_back_to_cwd_when_config_relative_missing(self):
        cfg = AurexConfig()
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, ".config"), exist_ok=True)
            cfg_path = os.path.join(td, ".config", "aurex.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write("{}\n")

            # Create a repo-root-like folder that exists only under cwd, not under ".config/".
            src_dir = os.path.join(td, "third-parties", "Phy-Engine", "src")
            os.makedirs(src_dir, exist_ok=True)

            old_cwd = os.getcwd()
            try:
                os.chdir(td)
                got = cfg.resolve_path("third-parties/Phy-Engine/src", config_path=cfg_path)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(os.path.abspath(src_dir), got)

    def test_resolve_path_prefers_cwd_when_parent_structure_matches_better(self):
        cfg = AurexConfig()
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, ".config"), exist_ok=True)
            cfg_path = os.path.join(td, ".config", "aurex.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write("{}\n")

            # Only the cwd has ".aurex/cache" present; build dir doesn't exist yet.
            os.makedirs(os.path.join(td, ".aurex", "cache"), exist_ok=True)

            old_cwd = os.getcwd()
            try:
                os.chdir(td)
                got = cfg.resolve_path(".aurex/cache/phy-engine-build", config_path=cfg_path)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(os.path.abspath(os.path.join(td, ".aurex", "cache", "phy-engine-build")), got)


if __name__ == "__main__":
    unittest.main()
