"""Lifecycle commands must not unexpectedly replace an active model service."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/aurex-vllm.sh'


class VLLMLifecycleTests(unittest.TestCase):
    def invoke(self, action):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log = root / 'commands'
            podman = root / 'podman'
            podman.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$AUREX_TEST_COMMANDS"\n'
                              'if [ "$1" = inspect ]; then echo true; fi\n')
            podman.chmod(0o755)
            curl = root / 'curl'
            curl.write_text('#!/bin/sh\nexit 0\n')
            curl.chmod(0o755)
            result = subprocess.run(['bash', str(SCRIPT), action], text=True, capture_output=True,
                env={**os.environ, 'PATH': folder + os.pathsep + os.environ['PATH'], 'AUREX_TEST_COMMANDS': str(log)})
            return result, log.read_text().splitlines() if log.exists() else []

    def test_status_is_read_only(self):
        result, calls = self.invoke('status')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(calls)
        self.assertTrue(all(line.split()[0] in {'ps', 'inspect'} for line in calls))

    def test_start_does_not_replace_running_container(self):
        result, calls = self.invoke('start')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('already running', result.stdout)
        self.assertTrue(all(line.startswith('inspect aurex-vllm ') for line in calls))

    def test_stop_targets_only_aurex_container(self):
        result, calls = self.invoke('stop')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ['stop --time 60 aurex-vllm'])

    def test_unknown_action_has_no_side_effects(self):
        result, calls = self.invoke('unexpected')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])
