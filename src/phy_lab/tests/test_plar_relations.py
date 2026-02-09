import os
import sys
import types
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from plar import get_relations


class _UserNoGetRelations:
    token = "t"
    auth_code = "a"


class TestPlarRelations(unittest.TestCase):
    def test_get_relations_maps_display_type_names(self):
        captured = {"payload": None}

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"Status": 200, "Data": {"$values": []}}

        def _post(_url, json=None, headers=None, timeout=None):
            captured["payload"] = json
            return _Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.post = _post  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            get_relations(_UserNoGetRelations(), user_id="u1", display_type="Banned", take=1)

        self.assertIsInstance(captured["payload"], dict)
        self.assertEqual(captured["payload"]["DisplayType"], 2)

    def test_get_relations_accepts_numeric_strings(self):
        captured = {"payload": None}

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"Status": 200, "Data": {"$values": []}}

        def _post(_url, json=None, headers=None, timeout=None):
            captured["payload"] = json
            return _Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.post = _post  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            get_relations(_UserNoGetRelations(), user_id="u1", display_type="4", take=1)

        self.assertIsInstance(captured["payload"], dict)
        self.assertEqual(captured["payload"]["DisplayType"], 4)


if __name__ == "__main__":
    unittest.main()

