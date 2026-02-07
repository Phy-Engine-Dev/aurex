import os
import sys
import types
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from plar import query_experiments


class _DummyUserOk:
    def __init__(self):
        self.calls = []

    def query_experiments(self, **kwargs):
        self.calls.append(kwargs)
        return {"Data": {"$values": [{"ID": "x"}]}}


class _DummyUserRaises:
    def __init__(self):
        self.token = "tok"
        self.auth_code = "auth"

    def query_experiments(self, **kwargs):
        raise RuntimeError("boom")


class TestQueryExperiments(unittest.TestCase):
    def test_wrapper_call_sends_tags_array(self):
        user = _DummyUserOk()
        got = query_experiments(user, category="Experiment", take=1)
        self.assertEqual(got, [{"ID": "x"}])
        self.assertEqual(len(user.calls), 1)
        self.assertEqual(user.calls[0]["tags"], [])
        self.assertIsNone(user.calls[0]["exclude_tags"])

    def test_falls_back_to_direct_http_on_wrapper_error(self):
        user = _DummyUserRaises()

        captured = {}

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"Data": {"$values": [{"ID": "y"}]}}

        def _post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return _Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.post = _post  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            got = query_experiments(user, category="Discussion", take=2, skip=3, from_skip=None)

        self.assertEqual(got, [{"ID": "y"}])
        self.assertEqual(captured["url"], "https://physics-api-cn.turtlesim.com/Contents/QueryExperiments")
        self.assertEqual(captured["json"]["Query"]["Category"], "Discussion")
        self.assertEqual(captured["json"]["Query"]["Tags"], [])
        self.assertIsNone(captured["json"]["Query"]["ExcludeTags"])
        self.assertEqual(captured["headers"]["x-API-Token"], "tok")
        self.assertEqual(captured["headers"]["x-API-AuthCode"], "auth")

    def test_returns_empty_when_data_is_null(self):
        class _User:
            def query_experiments(self, **kwargs):
                return {"Status": 200, "Message": "", "Data": None}

        got = query_experiments(_User(), category="Experiment", take=1)
        self.assertEqual(got, [])

    def test_raises_on_non_200_status(self):
        class _User:
            def query_experiments(self, **kwargs):
                return {"Status": 401, "Message": "Unauthorized", "Data": None}

        with self.assertRaises(Exception) as ctx:
            query_experiments(_User(), category="Experiment", take=1)
        self.assertIn("status=401", str(ctx.exception))

    def test_direct_http_shape_matches_plweb2_and_take_capped(self):
        class _User:
            token = "tok"
            auth_code = "auth"

        captured = {}

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"Status": 200, "Message": "", "Data": {"$values": []}}

        def _post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Resp()

        import types

        fake_requests = types.ModuleType("requests")
        fake_requests.post = _post  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            got = query_experiments(_User(), category="Experiment", take=200, skip=-1, from_skip="")

        self.assertEqual(got, [])
        q = captured["json"]["Query"]
        self.assertEqual(q["Take"], 24)
        self.assertEqual(q["Skip"], 0)
        self.assertIsNone(q["From"])
        self.assertEqual(q["Tags"], [])
        self.assertIsNone(q["ExcludeTags"])
        self.assertIsNone(q["ExcludeLanguages"])


if __name__ == "__main__":
    unittest.main()
