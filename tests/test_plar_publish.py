import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from aurex.publishing import (PublicationError, approve_publication, authorize_session,
    bind_task_actions, inspect_publication_source, publication_authorization, publish_approved,
    _chinese_text)
from aurex.tools.plar_tools import plar_publish_experiment, PLAR_PUBLISH_EXPERIMENT_TOOL
from aurex.tools.registry import ToolError
from plar.api import submit_original_experiment, upload_experiment_cover, confirm_original_experiment
from plar.official_publish_api import hdl_source_carrier_status


SUMMARY_ID = "6a9b5c31ec746130fc33154f"


class PublicationLedgerTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = self.temp.name
        root = Path(self.cache)
        self.sav = root / "new-design.sav"
        self.cover = root / "system-cover.jpg"
        self.evidence = root / "verified.json"
        self.evidence.write_text('{"verified":true,"cases":8}', encoding="utf8")
        Image.new("RGB", (512, 512), "white").save(self.cover, "JPEG")
        self.source = {"Type": 0, "Experiment": {"ID": None, "Type": 0,
            "StatusSave": json.dumps({"Elements": [{"Identifier": "r1", "ModelID": "Resistor",
                "Position": "0.0123456,0.9876543,-2E-3", "Rotation": "0,180,0", "Properties": {"电阻": 10}}],
                "Wires": [], "UnknownState": "preserve exactly"}, ensure_ascii=False),
            "CameraSave": '{"Mode":0,"VisionCenter":"0,1.08,0","TargetRotation":"90,0,0","Distance":2.7}',
            "Version": 2404}, "Summary": {"ID": None, "ContentID": None, "Price": 0, "Image": 0}}
        self.write_source()
        self.user = SimpleNamespace(user_id="test-user", token="SECRET_TOKEN", auth_code="SECRET_AUTH")

    def write_source(self):
        self.sav.write_text(json.dumps(self.source, ensure_ascii=False), encoding="utf8")

    def test_chinese_language_ratio_ignores_fenced_hdl_but_not_english_prose(self):
        source = "module demo;\n" + "assign signal = other_signal;\n" * 100 + "endmodule"
        text = "这是中文实验说明，包含验证结果、能力边界和使用限制。\n\n```verilog\n" + source + "\n```"
        self.assertEqual(_chinese_text(text, "introduction", 16000, 20), text)
        with self.assertRaisesRegex(PublicationError, "must use Chinese prose"):
            _chinese_text("中文 " + "This remains English prose. " * 20, "introduction", 16000, 20)

    def test_publication_rejects_more_than_5000_elements(self):
        template = self.source["Experiment"]["StatusSave"]
        status = json.loads(template)
        original = status["Elements"][0]
        status["Elements"] = [{**original, "Identifier": f"r{i}"} for i in range(5001)]
        self.source["Experiment"]["StatusSave"] = json.dumps(status, ensure_ascii=False)
        self.write_source()
        with self.assertRaisesRegex(PublicationError, "1..5000 elements"):
            inspect_publication_source(self.cache, str(self.sav))

    def approve(self, **override):
        info = inspect_publication_source(self.cache, str(self.sav))
        args = dict(session_id="session-riscv", run_id="run-a", user_id="test-user", sav_path=str(self.sav),
            title="教学电路验证", introduction="这是经过本地验证的教学实验，公开说明能力边界，不声称等同于完整处理器。",
            evidence_paths=[str(self.evidence)], review={"approved": True, "server_validated": True,
                "thinking": False, 'publish_requested': True,
                "source_sha256": info["sha256"], "summary": "已逐项核对源文件与验证记录。"},
            cover_path=str(self.cover), cover_manifest={"source_sha256": info["sha256"],
                "cover_sha256": hashlib.sha256(self.cover.read_bytes()).hexdigest(), "rendered_all": True,
                "total_elements": info["elements"], "visible_elements": info["elements"], "clipped_ids": [],
                "view": {"yaw": 45, "pitch": 60, "projection": "orthographic", "fit": "all"}})
        args.update(override)
        return approve_publication(self.cache, **args)

    def setup_approval(self):
        self.bind()
        return self.approve()

    def bind(self, **kwargs):
        args = dict(task_id='run-a', session_id='session-riscv', source='admin',
                    original_user_request='请验证并发布这个教学实验', explicit_publish_requested=True,
                    purpose='riscv_teaching_subset')
        args.update(kwargs)
        return bind_task_actions(self.cache, **args)

    def publish(self, receipt_id, **override):
        args = dict(session_id="session-riscv", run_id="run-a", user=self.user, approval_id=receipt_id)
        args.update(override)
        return publish_approved(self.cache, **args)

    def network(self):
        submit = mock.patch("aurex.publishing.plar_api.submit_original_experiment", return_value={
            "summary_id": SUMMARY_ID, "image_counter": 1,
            "_cover_credential": {"Policy": "SECRET_POLICY", "Authorization": "SECRET_COVER_AUTH"}})
        cover = mock.patch("aurex.publishing.plar_api.upload_experiment_cover")
        confirm = mock.patch("aurex.publishing.plar_api.confirm_original_experiment")
        return submit, cover, confirm

    def test_no_scope_no_direct_model_approval_and_exactly_two_slots(self):
        with self.assertRaises(PublicationError):
            self.approve()
        with self.assertRaises(ToolError):
            plar_publish_experiment(None, {"approved": True, "cover_path": str(self.cover)})
        self.assertNotIn("cover_path", PLAR_PUBLISH_EXPERIMENT_TOOL["parameters"]["properties"])
        authorize_session(self.cache, "session-riscv", "riscv_teaching_subset")
        authorize_session(self.cache, "session-555", "555_state_table")
        with self.assertRaises(PublicationError):
            self.approve()  # Old two-session authorizations cannot bypass new task identity/intent.
        self.bind()
        self.assertEqual(publication_authorization(self.cache, "session-riscv", task_id='run-a')["limit"], 1)
        for sid, purpose in (("third", "riscv_teaching_subset"), ("session-riscv", "555_state_table"), ("other", "spam")):
            with self.subTest(sid=sid), self.assertRaises(PublicationError):
                authorize_session(self.cache, sid, purpose)

    def test_cancel_between_http_stages_preserves_receipt_and_never_starts_confirm(self):
        approval = self.setup_approval()
        patches = self.network()
        cancelled = False
        def check():
            if cancelled:
                raise RuntimeError('cancelled')
        def submitted(*args, **kwargs):
            nonlocal cancelled
            cancelled = True
            return {'summary_id': SUMMARY_ID, 'image_counter': 1,
                    '_cover_credential': {'Policy': 'SECRET', 'Authorization': 'SECRET'}}
        with patches[0] as submit, patches[1] as cover, patches[2] as confirm:
            submit.side_effect = submitted
            with self.assertRaisesRegex(RuntimeError, 'cancelled'):
                self.publish(approval['approval_id'], check_cancel=check)
            self.assertEqual(publication_authorization(self.cache, 'session-riscv', task_id='run-a')['state'], 'image_pending')
            cover.assert_not_called()
            confirm.assert_not_called()
            cancelled = False
            result = self.publish(approval['approval_id'], check_cancel=check)
            self.assertTrue(result['published'])
            submit.assert_called_once()
            cover.assert_called_once()
            confirm.assert_called_once()

    def test_community_publication_first_line_is_bound_requester_but_admin_has_no_mention(self):
        uid = '1234567890abcdef12345678'
        self.bind(source='community', requester_user_id=uid, requester_nickname='提问者',
                  target={'type': 'Experiment', 'id': SUMMARY_ID})
        approval = self.approve()
        patches = self.network()
        with patches[0] as submit, patches[1], patches[2]:
            self.publish(approval['approval_id'])
            body = submit.call_args.kwargs['introduction']
            self.assertEqual(body.splitlines()[0], f'<user={uid}>@提问者</user>：')
            self.assertTrue(body.startswith(f'<user={uid}>@提问者</user>：\n\n'))
            self.assertEqual(body.count('<user='), 1)
        # A different task in the same conversation receives its own one-shot slot.
        self.bind(task_id='run-b', source='admin')
        second = self.approve(run_id='run-b')
        patches = self.network()
        with patches[0] as submit, patches[1], patches[2]:
            self.publish(second['approval_id'], run_id='run-b')
            self.assertNotIn('<user=', submit.call_args.kwargs['introduction'])
        with self.assertRaises(PublicationError):
            self.approve()  # Original task cannot create a second publication.

    def test_community_missing_identity_and_dry_run_cannot_approve(self):
        self.bind(source='community')
        with self.assertRaisesRegex(PublicationError, 'requester user ID'):
            self.approve()
        self.bind(task_id='dry-task', dry_run=True)
        with self.assertRaisesRegex(PublicationError, 'Dry-run'):
            self.approve(run_id='dry-task')

    def test_raw_full_source_integrity_and_single_publication_cover_before_confirm(self):
        approval = self.setup_approval()
        patches = self.network()
        events = []
        with patches[0] as submit, patches[1] as cover, patches[2] as confirm:
            submit.side_effect = lambda *a, **kw: (events.append("submit") or {
                "summary_id": SUMMARY_ID, "image_counter": 1,
                "_cover_credential": {"Policy": "SECRET_POLICY", "Authorization": "SECRET_COVER_AUTH"}})
            cover.side_effect = lambda **kw: events.append("cover")
            confirm.side_effect = lambda *a, **kw: events.append("confirm")
            result = self.publish(approval["approval_id"])
            again = self.publish(approval["approval_id"])
        self.assertEqual(events, ["submit", "cover", "confirm"])
        self.assertTrue(result["published"])
        self.assertEqual(result, again)
        self.assertNotIn("SECRET", json.dumps(result))
        self.assertEqual(submit.call_args.kwargs["source"], self.source)
        self.assertEqual(submit.call_args.kwargs["source"]["Experiment"]["CameraSave"], self.source["Experiment"]["CameraSave"])
        self.assertEqual(cover.call_args.kwargs["cover"], self.cover.read_bytes())
        self.assertEqual(confirm.call_args.kwargs["summary_id"], SUMMARY_ID)
        self.assertEqual(json.loads(self.sav.read_text()), self.source)
        self.assertEqual(os.stat(Path(self.cache, ".publication", "ledger.sqlite3")).st_mode & 0o777, 0o600)
        with sqlite3.connect(Path(self.cache, ".publication", "ledger.sqlite3")) as db:
            self.assertIsNone(db.execute("SELECT credential FROM task_publications").fetchone()[0])
        with self.assertRaises(PublicationError):
            self.approve()

    def test_approval_is_bound_to_real_session_run_and_account(self):
        approval = self.setup_approval()
        patches = self.network()
        with patches[0] as submit, patches[1], patches[2]:
            for override in ({"session_id": "other"}, {"run_id": "other"}, {"user": SimpleNamespace(user_id="other")}, {"approval_id": "0" * 32}):
                with self.subTest(override=override), self.assertRaises(PublicationError):
                    self.publish(approval["approval_id"], **override)
            submit.assert_not_called()

    def test_source_evidence_and_cover_changes_invalidate_review_without_network(self):
        for attribute in ("sav", "evidence", "cover"):
            with self.subTest(attribute=attribute):
                approval = self.setup_approval()
                file = getattr(self, attribute)
                original = file.read_bytes()
                file.write_bytes(original + b" ")
                patches = self.network()
                with patches[0] as submit, patches[1], patches[2], self.assertRaises(PublicationError):
                    self.publish(approval["approval_id"])
                submit.assert_not_called()
                file.write_bytes(original)

    def test_submit_uncertainty_is_persisted_and_never_retried(self):
        approval = self.setup_approval()
        patches = self.network()
        with patches[0] as submit, patches[1] as cover, patches[2] as confirm:
            submit.side_effect = TimeoutError("SECRET_NETWORK_DETAILS")
            result = self.publish(approval["approval_id"])
            again = self.publish(approval["approval_id"])
        self.assertEqual(submit.call_count, 1)
        cover.assert_not_called()
        confirm.assert_not_called()
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(again["state"], "unknown")
        self.assertNotIn("SECRET", json.dumps(result))

    def test_crash_after_submit_started_requires_reconciliation_not_new_submit(self):
        approval = self.setup_approval()
        with sqlite3.connect(Path(self.cache, ".publication", "ledger.sqlite3")) as db:
            db.execute("UPDATE task_publications SET state='submitting'")
        patches = self.network()
        with patches[0] as submit, patches[1], patches[2]:
            result = self.publish(approval["approval_id"])
        submit.assert_not_called()
        self.assertEqual(result["state"], "unknown")

    def test_cover_failure_never_confirms_and_retry_uses_same_slot(self):
        approval = self.setup_approval()
        patches = self.network()
        with patches[0] as submit, patches[1] as cover, patches[2] as confirm:
            cover.side_effect = [TimeoutError("SECRET"), None]
            first = self.publish(approval["approval_id"])
            confirm.assert_not_called()
            second = self.publish(approval["approval_id"])
        self.assertEqual(first["state"], "image_pending")
        self.assertEqual(first["summary_id"], SUMMARY_ID)
        self.assertTrue(second["published"])
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(cover.call_count, 2)
        self.assertEqual(confirm.call_count, 1)
        self.assertNotIn("SECRET", json.dumps(first))

    def test_confirm_failure_retry_only_confirms_same_summary_id(self):
        approval = self.setup_approval()
        patches = self.network()
        with patches[0] as submit, patches[1] as cover, patches[2] as confirm:
            confirm.side_effect = [TimeoutError("SECRET"), None]
            first = self.publish(approval["approval_id"])
            second = self.publish(approval["approval_id"])
        self.assertEqual(first["state"], "confirm_pending")
        self.assertTrue(second["published"])
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(cover.call_count, 1)
        self.assertEqual([c.kwargs["summary_id"] for c in confirm.call_args_list], [SUMMARY_ID, SUMMARY_ID])

    def test_concurrent_call_cannot_duplicate_submission_or_mark_active_submit_unknown(self):
        approval = self.setup_approval()
        started, release = threading.Event(), threading.Event()
        patches = self.network()
        outcomes = []
        def submit_once(*args, **kwargs):
            started.set()
            self.assertTrue(release.wait(3))
            return {"summary_id": SUMMARY_ID, "image_counter": 1,
                    "_cover_credential": {"Policy": "p", "Authorization": "a"}}
        with patches[0] as submit, patches[1], patches[2]:
            submit.side_effect = submit_once
            thread = threading.Thread(target=lambda: outcomes.append(self.publish(approval["approval_id"])))
            thread.start()
            self.assertTrue(started.wait(3))
            try:
                with self.assertRaisesRegex(PublicationError, "already in progress"):
                    self.publish(approval["approval_id"])
            finally:
                release.set()
                thread.join(3)
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(len(outcomes), 1)
        self.assertTrue(outcomes[0]["published"])

    def test_must_have_server_validation_fixed_all_element_cover_and_chinese_metadata(self):
        self.bind()
        for override in ({"title": "English only"}, {"introduction": "too short"},
                         {"introduction": "中 " + "This is an English publication. " * 10},
                         {"review": {"approved": True, "server_validated": False, "thinking": False}}, {"evidence_paths": []},
                         {"cover_manifest": {}}, {"cover_path": str(self.evidence)}):
            with self.subTest(override=override), self.assertRaises(PublicationError):
                self.approve(**override)
        with self.assertRaises(PublicationError):
            self.approve(title="<experiment=deadbeef>垃圾广告</experiment>")

    def test_source_rejects_missing_non_electrical_type_existing_id_nonfinite_and_bad_wires(self):
        original = copy.deepcopy(self.source)
        mutations = [lambda s: s["Experiment"].pop("Type"), lambda s: s["Experiment"].update(Type=False),
            lambda s: s["Experiment"].update(Type=3), lambda s: s["Summary"].update(ID=SUMMARY_ID),
            lambda s: s["Summary"].update(Price=5), lambda s: s["Experiment"].update(StatusSave="{}")]
        for mutation in mutations:
            self.source = copy.deepcopy(original)
            mutation(self.source)
            self.write_source()
            with self.assertRaises(PublicationError):
                inspect_publication_source(self.cache, str(self.sav))
        for position in ("NaN,0,0", "0,Infinity,0", "1,2", "not,a,number"):
            self.source = copy.deepcopy(original)
            status = json.loads(self.source["Experiment"]["StatusSave"])
            status["Elements"][0]["Position"] = position
            self.source["Experiment"]["StatusSave"] = json.dumps(status)
            self.write_source()
            with self.assertRaises(PublicationError):
                inspect_publication_source(self.cache, str(self.sav))
        self.source = copy.deepcopy(original)
        status = json.loads(self.source["Experiment"]["StatusSave"])
        status["Wires"] = [{"Source": "r1", "Target": "missing", "SourcePin": 0, "TargetPin": 1}]
        self.source["Experiment"]["StatusSave"] = json.dumps(status)
        self.write_source()
        with self.assertRaises(PublicationError):
            inspect_publication_source(self.cache, str(self.sav))

    def test_unsafe_path_duplicate_json_and_old_approval_revocation(self):
        link = Path(self.cache, "link.sav")
        link.symlink_to(self.sav)
        with self.assertRaises(PublicationError):
            inspect_publication_source(self.cache, str(link))
        self.sav.write_text('{"Experiment":{},"Experiment":{}}')
        with self.assertRaises(PublicationError):
            inspect_publication_source(self.cache, str(self.sav))
        self.write_source()
        first = self.setup_approval()
        second = self.approve(title="重新审核的教学电路")
        self.assertNotEqual(first["approval_id"], second["approval_id"])
        with self.assertRaises(PublicationError):
            self.publish(first["approval_id"])

    def test_oversized_hdl_type3_is_text_only_and_skips_cover_upload(self):
        status = hdl_source_carrier_status()
        camera = {"Mode": 2, "Distance": 2.75, "VisionCenter": "0,1.08,0",
                  "TargetRotation": "90,0,0"}
        self.source = {"Type": 3, "Experiment": {"ID": None, "Type": 3, "Components": 3,
            "StatusSave": json.dumps(status, ensure_ascii=False), "CameraSave": json.dumps(camera), "Version": 2503},
            "Summary": {"ID": None, "ContentID": None, "Price": 0, "Type": 3,
                "Tags": ["Type-3", "高中", "教学实验"]},
            "InternalName": "Aurex HDL 源码载体"}
        self.write_source()
        self.bind()
        info = inspect_publication_source(self.cache, str(self.sav))
        appendix = "## 已验证 HDL 源码\n\n```verilog\nmodule top; endmodule\n```"
        model_intro = "这是经过验证的教学设计，正文说明了验证范围和能力边界。\n\n```verilog\nmodule top; endmodule\n```"
        approval = self.approve(cover_path=None, cover_manifest=None, trusted_appendix=appendix,
            introduction=model_intro,
            review={"approved": True, "server_validated": True, "thinking": False, "publish_requested": True,
                    "source_sha256": info["sha256"], "summary": "已核对完整源码与验证报告。"})
        patches = self.network()
        with patches[0] as submit, patches[1] as cover, patches[2] as confirm:
            submit.return_value = {"summary_id": SUMMARY_ID, "image_counter": 0}
            result = self.publish(approval["approval_id"])
        self.assertTrue(result["published"])
        self.assertIsNone(result["cover_sha256"])
        self.assertEqual(submit.call_args.kwargs["cover_bytes"], 0)
        self.assertIn(appendix, submit.call_args.kwargs["introduction"])
        self.assertEqual(submit.call_args.kwargs["introduction"].count("module top; endmodule"), 1)
        cover.assert_not_called()
        confirm.assert_called_once_with(self.user, summary_id=SUMMARY_ID, image_counter=0)
        self.assertFalse(Path(self.cache, ".publication", approval["approval_id"] + ".jpg").exists())


class RawPublicationAPITests(unittest.TestCase):
    def test_submit_keeps_raw_state_uses_free_experiment_and_positive_cover_slot(self):
        source = {"Type": 0, "Experiment": {"ID": None, "Type": 0, "StatusSave": " RAW_STATE ", "CameraSave": " RAW_CAMERA "},
                  "Summary": {"ID": None, "User": {}, "Price": 123, "Tags": ["精选"]}, "Extra": [1, 2]}
        original = copy.deepcopy(source)
        user = SimpleNamespace(user_id="author", token="SECRET_TOKEN", auth_code="SECRET_AUTH", nickname="测试作者")
        response = {"Status": 200, "Data": {
            "Summary": {"ID": SUMMARY_ID}, "Token": {"Policy": "SECRET_POLICY", "Authorization": "SECRET_IMAGE"}}}
        captured = {}
        def submit(actual_user, payload):
            captured.update(copy.deepcopy(payload))
            return response
        with mock.patch("plar.api.official_publish_api.submit_experiment", side_effect=submit) as official:
            result = submit_original_experiment(user, source=source, title="中文标题", introduction="中文简介", cover_bytes=1024)
        body = captured
        self.assertEqual(body["Workspace"]["Experiment"], source["Experiment"])
        self.assertIsNone(body["Workspace"]["Summary"])
        self.assertEqual(body["Summary"]["Category"], "Experiment")
        self.assertEqual(body["Summary"]["Price"], 0)
        self.assertEqual(body["Summary"]["Tags"], ["Type-0"])
        self.assertEqual(body["Summary"]["User"]["ID"], "author")
        self.assertEqual(body["Request"], {"FileSize": 1024, "Extension": ".jpg"})
        official.assert_called_once()
        self.assertEqual(source, original)
        self.assertEqual(result["summary_id"], SUMMARY_ID)

    def test_cover_and_confirm_delegate_to_official_sdk_with_exact_image_slot(self):
        user = SimpleNamespace(token="t", auth_code="a")
        with mock.patch("plar.api.official_publish_api.upload_image", return_value={"Status": 200}) as upload, \
             mock.patch("plar.api.official_publish_api.confirm_experiment", return_value={"Status": 200}) as confirm:
            upload_experiment_cover(user=user, cover=b"jpeg",
                credential={"Policy": "p", "Authorization": "a", "URL": "http://evil"})
            confirm_original_experiment(user, summary_id=SUMMARY_ID)
        upload.assert_called_once_with(user, "p", "a", b"jpeg")
        confirm.assert_called_once_with(user, SUMMARY_ID, 1)

    def test_fixed_type3_submission_has_title_body_and_no_image_request(self):
        status = hdl_source_carrier_status()
        camera = {"Mode": 2, "Distance": 2.75, "VisionCenter": "0,1.08,0",
                  "TargetRotation": "90,0,0"}
        source = {"Type": 3, "Experiment": {"ID": None, "Type": 3, "Components": 3,
            "StatusSave": json.dumps(status, ensure_ascii=False), "CameraSave": json.dumps(camera), "Version": 2503},
            "Summary": {"ID": None, "ContentID": None, "Type": 3,
                "Tags": ["Type-3", "高中", "教学实验"]},
            "InternalName": "Aurex HDL 源码载体"}
        user = SimpleNamespace(user_id="author", token="token", auth_code="auth", nickname="作者")
        response = {"Status": 200, "Data": {"Summary": {"ID": SUMMARY_ID}}}
        captured = {}
        def submit(actual_user, payload):
            captured.update(copy.deepcopy(payload))
            return response
        with mock.patch("plar.api.official_publish_api.submit_experiment", side_effect=submit):
            result = submit_original_experiment(user, source=source, title="源码实验", introduction="中文正文和完整源码", cover_bytes=0)
        body = captured
        self.assertNotIn("Request", body)
        self.assertNotIn("Anonymous", body["Summary"])
        self.assertEqual(body["Summary"]["Tags"], ["Type-3", "高中", "教学实验"])
        self.assertEqual(body["Summary"]["Description"], ["中文正文和完整源码"])
        self.assertEqual(body["Workspace"]["Experiment"]["Type"], 3)
        self.assertEqual(body["Workspace"]["InternalName"], "Aurex HDL 源码载体")
        self.assertEqual(len(status["Elements"]), 3)
        self.assertEqual(result["image_counter"], 0)
        self.assertNotIn("_cover_credential", result)


if __name__ == "__main__":
    unittest.main()
