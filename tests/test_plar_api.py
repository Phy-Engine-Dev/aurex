import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import plar


class TestUserApis(unittest.TestCase):
    def test_get_user_by_name_wrapper_strips_at(self):
        captured = {"name": None}

        class User:
            def get_user_by_name(self, name):
                captured["name"] = name
                return {"Status": 200, "Message": "", "Data": {"User": {"ID": "u1", "Nickname": name}}}

        data = plar.get_user_by_name(User(), name="@abc")
        self.assertEqual(data["User"]["ID"], "u1")
        self.assertEqual(captured["name"], "abc")

    def test_get_user_by_name_wrapper_retries_mium_to_nium(self):
        called = {"names": []}

        class User:
            def get_user_by_name(self, name):
                called["names"].append(name)
                if name == "Neptumium":
                    return {"Status": 404, "Message": "NotFound", "Data": None}
                if name == "Neptunium":
                    return {"Status": 200, "Message": "", "Data": {"User": {"ID": "u9", "Nickname": name}}}
                return {"Status": 404, "Message": "NotFound", "Data": None}

        data = plar.get_user_by_name(User(), name="Neptumium")
        self.assertEqual(data["User"]["ID"], "u9")
        self.assertEqual(called["names"][:2], ["Neptumium", "Neptunium"])

    def test_get_user_by_name_direct_http_posts_name(self):
        class User:
            token = "tok"
            auth_code = "auth"

        captured = {}

        class Resp:
            status_code = 200

            def json(self):
                return {"Status": 200, "Message": "", "Data": {"User": {"ID": "u2", "Nickname": "n"}}}

        def post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.post = post  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            data = plar.get_user_by_name(User(), name="n")

        self.assertEqual(captured["url"], "https://physics-api-cn.turtlesim.com/Users/GetUser")
        self.assertEqual(captured["json"], {"Name": "n"})
        self.assertEqual(captured["headers"]["x-API-Token"], "tok")
        self.assertEqual(data["User"]["ID"], "u2")

    def test_get_user_by_name_raises_on_non_200(self):
        class User:
            def get_user_by_name(self, _name):
                return {"Status": 404, "Message": "NotFound", "Data": None}

        with self.assertRaises(plar.PLARError) as ctx:
            plar.get_user_by_name(User(), name="x")
        self.assertIn("status=404", str(ctx.exception))


class TestQueryAndRelations(unittest.TestCase):
    def test_query_experiments_wrapper_sends_tags_array(self):
        class User:
            def __init__(self):
                self.calls = []

            def query_experiments(self, **kwargs):
                self.calls.append(kwargs)
                return {"Data": {"$values": [{"ID": "x"}]}}

        u = User()
        got = plar.query_experiments(u, category="Experiment", take=1)
        self.assertEqual(got, [{"ID": "x"}])
        self.assertEqual(u.calls[0]["tags"], [])
        self.assertIsNone(u.calls[0]["exclude_tags"])

    def test_query_experiments_direct_http_shape_and_take_cap(self):
        class User:
            token = "tok"
            auth_code = "auth"

        captured = {}

        class Resp:
            status_code = 200

            def json(self):
                return {"Status": 200, "Message": "", "Data": {"$values": []}}

        def post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            captured["timeout"] = timeout
            return Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.post = post  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            got = plar.query_experiments(User(), category="Experiment", take=200, skip=-1, from_skip="")

        self.assertEqual(got, [])
        self.assertEqual(captured["url"], "https://physics-api-cn.turtlesim.com/Contents/QueryExperiments")
        q = captured["json"]["Query"]
        self.assertEqual(q["Take"], 24)
        self.assertEqual(q["Skip"], 0)
        self.assertIsNone(q["From"])
        self.assertIsNone(q["Tags"])
        self.assertIsNone(q["ExcludeTags"])
        self.assertIsNone(q["ExcludeLanguages"])

    def test_query_experiments_direct_http_days_and_sort_keep_strings(self):
        class User:
            token = "tok"
            auth_code = "auth"

        captured = {}

        class Resp:
            status_code = 200

            def json(self):
                return {"Status": 200, "Message": "", "Data": {"$values": []}}

        def post(_url, json=None, headers=None, timeout=None):
            captured["json"] = json
            return Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.post = post  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            plar.query_experiments(User(), category="Experiment", take=5, days=14, sort="Popularity")

        q = captured["json"]["Query"]
        self.assertEqual(q["Days"], "14")
        self.assertEqual(q["Sort"], "Popularity")

    def test_get_relations_maps_display_type_names(self):
        class User:
            token = "t"
            auth_code = "a"

        captured = {}

        class Resp:
            status_code = 200

            def json(self):
                return {"Status": 200, "Data": {"$values": []}}

        def post(_url, json=None, headers=None, timeout=None):
            captured["payload"] = json
            return Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.post = post  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            plar.get_relations(User(), user_id="u1", display_type="Banned", take=1)

        self.assertEqual(captured["payload"]["DisplayType"], 2)

    def test_get_relations_accepts_numeric_strings(self):
        class User:
            token = "t"
            auth_code = "a"

        captured = {}

        class Resp:
            status_code = 200

            def json(self):
                return {"Status": 200, "Data": {"$values": []}}

        def post(_url, json=None, headers=None, timeout=None):
            captured["payload"] = json
            return Resp()

        fake_requests = types.ModuleType("requests")
        fake_requests.post = post  # type: ignore[attr-defined]

        with mock.patch.dict(sys.modules, {"requests": fake_requests}):
            plar.get_relations(User(), user_id="u1", display_type="4", take=1)

        self.assertEqual(captured["payload"]["DisplayType"], 4)


class TestStatusSaveAndPublish(unittest.TestCase):
    def test_get_status_save_falls_back_to_wrapper_when_helper_fails(self):
        fake_physicslab = types.ModuleType("physicsLab")

        class Category:
            class _C:
                def __init__(self, value):
                    self.value = value

            Experiment = _C("Experiment")
            Discussion = _C("Discussion")

        class OpenMode:
            load_by_plar_app = 2

        class Experiment:
            def __init__(self, *_a, **_kw):
                raise RuntimeError("boom")

        fake_physicslab.Category = Category
        fake_physicslab.OpenMode = OpenMode
        fake_physicslab.Experiment = Experiment

        status = {"Elements": [{"ModelID": "Resistor"}], "Wires": []}

        class User:
            def get_experiment(self, *_a, **_kw):
                return {"Status": 200, "Message": "", "Data": {"StatusSave": json.dumps(status)}}

        import plar.api as api_mod

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(api_mod, "ensure_physicslab_importable", lambda **_kw: None):
                with mock.patch.dict(sys.modules, {"physicsLab": fake_physicslab}):
                    got = plar.get_status_save(
                        User(),
                        summary_id="sid",
                        category_value="Experiment",
                        cache_dir=td,
                        ttl_sec=0,
                    )

        self.assertEqual(got["Elements"][0]["ModelID"], "Resistor")

    def test_upload_sav_raises_on_non_200_submit(self):
        fake_physicslab = types.ModuleType("physicsLab")

        class Category:
            class _C:
                def __init__(self, value):
                    self.value = value

            Experiment = _C("Experiment")
            Discussion = _C("Discussion")

        class OpenMode:
            load_by_filepath = 1

        class Experiment:
            def __init__(self, *_a, **_kw):
                pass

            def edit_publish_info(self, **_kw):
                return self

            def _Experiment__upload(self, *_a, **_kw):
                return ({"Status": 403, "Message": "invalid", "Data": None}, {"Summary": {"Image": 0}})

        fake_physicslab.Category = Category
        fake_physicslab.OpenMode = OpenMode
        fake_physicslab.Experiment = Experiment

        import plar.api as api_mod

        with mock.patch.object(api_mod, "ensure_physicslab_importable", lambda **_kw: None):
            with mock.patch.dict(sys.modules, {"physicsLab": fake_physicslab}):
                with self.assertRaises(plar.PLARError) as ctx:
                    plar.upload_sav_as_experiment(
                        user=object(),
                        sav_path="x.sav",
                        title="t",
                        introduction="i",
                        cache_dir="cache",
                        category_value="Experiment",
                        tags=None,
                    )
        self.assertIn("status=403", str(ctx.exception))

    def test_upload_sav_unwraps_user_proxy(self):
        fake_physicslab = types.ModuleType("physicsLab")

        class Category:
            class _C:
                def __init__(self, value):
                    self.value = value

            Experiment = _C("Experiment")
            Discussion = _C("Discussion")

        class OpenMode:
            load_by_filepath = 1

        captured = {"upload_user": None, "confirm": None}

        class RealUser:
            def confirm_experiment(self, summary_id, category, image_counter):
                captured["confirm"] = (summary_id, category.value, image_counter)

        class LockedUser:
            def __init__(self, inner):
                self._user = inner

        class Experiment:
            def __init__(self, *_a, **_kw):
                pass

            def edit_publish_info(self, **_kw):
                return self

            def _Experiment__upload(self, user, *_a, **_kw):
                captured["upload_user"] = user
                return (
                    {"Status": 200, "Message": "", "Data": {"Summary": {"ID": "sid"}}},
                    {"Summary": {"Image": 0}},
                )

        fake_physicslab.Category = Category
        fake_physicslab.OpenMode = OpenMode
        fake_physicslab.Experiment = Experiment

        import plar.api as api_mod

        inner = RealUser()
        locked = LockedUser(inner)
        with mock.patch.object(api_mod, "ensure_physicslab_importable", lambda **_kw: None):
            with mock.patch.dict(sys.modules, {"physicsLab": fake_physicslab}):
                info = plar.upload_sav_as_experiment(
                    user=locked,
                    sav_path="x.sav",
                    title="t",
                    introduction="i",
                    cache_dir="cache",
                    category_value="Experiment",
                    tags=None,
                )

        self.assertIs(captured["upload_user"], inner)
        self.assertEqual(captured["confirm"], ("sid", "Experiment", 0))
        self.assertEqual(info["summary_id"], "sid")


if __name__ == "__main__":
    unittest.main()

