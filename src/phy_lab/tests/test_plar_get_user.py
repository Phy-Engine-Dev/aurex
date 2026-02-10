import os
import sys
import types
import unittest
from unittest import mock


TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PHY_LAB_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PHY_LAB_DIR not in sys.path:
    sys.path.insert(0, PHY_LAB_DIR)

from plar import PLARError, get_user_by_name


class TestGetUser(unittest.TestCase):
    def test_wrapper_get_user_by_name(self):
        class _User:
            def get_user_by_name(self, name):
                return {"Status": 200, "Message": "", "Data": {"User": {"ID": "u1", "Nickname": name}}}

        got = get_user_by_name(_User(), name="abc")
        self.assertEqual(got["User"]["ID"], "u1")

    def test_wrapper_get_user_by_name_strips_at_prefix(self):
        captured = {"name": None}

        class _User:
            def get_user_by_name(self, name):
                captured["name"] = name
                return {"Status": 200, "Message": "", "Data": {"User": {"ID": "u1", "Nickname": name}}}

        got = get_user_by_name(_User(), name="@abc")
        self.assertEqual(got["User"]["Nickname"], "abc")
        self.assertEqual(captured["name"], "abc")

    def test_wrapper_get_user_by_name_retries_common_mium_to_nium_typo(self):
        calls = {"names": []}

        class _User:
            def get_user_by_name(self, name):
                calls["names"].append(name)
                if name == "Neptumium":
                    return {"Status": 404, "Message": "NotFound", "Data": None}
                if name == "Neptunium":
                    return {"Status": 200, "Message": "", "Data": {"User": {"ID": "u9", "Nickname": name}}}
                return {"Status": 404, "Message": "NotFound", "Data": None}

        got = get_user_by_name(_User(), name="Neptumium")
        self.assertEqual(got["User"]["ID"], "u9")
        self.assertEqual(got["User"]["Nickname"], "Neptunium")
        self.assertEqual(calls["names"][:2], ["Neptumium", "Neptunium"])

    def test_direct_http_get_user_by_name(self):
        class _User:
            token = "tok"
            auth_code = "auth"

        captured = {}

        class _Resp:
            def raise_for_status(self):
                return None

            def json(self):
                return {"Status": 200, "Message": "", "Data": {"User": {"ID": "u2", "Nickname": "n"}}}

        def _post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.post = _post  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            got = get_user_by_name(_User(), name="n")
        self.assertEqual(captured["url"], "https://physics-api-cn.turtlesim.com/Users/GetUser")
        self.assertEqual(captured["json"], {"Name": "n"})
        self.assertEqual(got["User"]["ID"], "u2")

    def test_get_user_raises_on_non_200(self):
        class _User:
            def get_user_by_name(self, name):
                return {"Status": 404, "Message": "NotFound", "Data": None}

        with self.assertRaises(PLARError) as ctx:
            get_user_by_name(_User(), name="x")
        self.assertIn("status=404", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
