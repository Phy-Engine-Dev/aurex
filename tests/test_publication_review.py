import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from aurex.publication_review import review_and_publish
from aurex.publishing import bind_task_actions, publication_authorization
from aurex.sessiondb import SessionDB
from aurex.tools.registry import ToolError
from aurex.vllm_client import ModelReply


class ReviewClient:
    def __init__(self):
        self.config = SimpleNamespace(max_output_tokens=4096)
        self.decision = {"approved": True, 'publish_requested': True, "title": "已核验的教学子集电路",
            "introduction": "通过固定五组教学程序完成测试，支持列明的指令子集。本实验不是完整RV32I合规实现，也不声称已经证明综合等价。",
            "summary": "源文件及独立验证记录一致，介绍明确说明教学范围。"}
        self.finish = "stop"
        self.requests = []
        self.mutation = None
        self.input_tokens = 1000

    def capacity(self):
        return 32768

    def count(self, messages, tools=None):
        return self.input_tokens

    def chat(self, messages, **options):
        self.requests.append((messages, options))
        options["on_delta"]("reasoning", "PRIVATE_REVIEW_REASONING")
        options["on_delta"]("text", json.dumps(self.decision, ensure_ascii=False))
        if self.mutation:
            self.mutation()
        return ModelReply(json.dumps(self.decision, ensure_ascii=False), "PRIVATE_REVIEW_REASONING", [], {"total_tokens": 200}, self.finish)


class PublicationReviewTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = self.temp.name
        root = Path(self.cache)
        self.sav = root / "staged_sav" / "export.sav"
        self.sav.parent.mkdir()
        self.source = {"Type": 0, "Experiment": {"ID": None, "Type": 0,
            "StatusSave": json.dumps({"Elements": [{"Identifier": "r1", "ModelID": "Resistor", "Position": "0,0,0",
                "Rotation": "0,180,0", "Properties": {"电阻": 10}}], "Wires": []}, ensure_ascii=False),
            "CameraSave": '{"Mode":0,"Distance":2.7,"VisionCenter":"0,1.08,0","TargetRotation":"90,0,0"}'},
            "Summary": {"ID": None, "ContentID": None, "Price": 0}}
        self.sav.write_text(json.dumps(self.source, ensure_ascii=False), encoding="utf8")
        self.source_hash = hashlib.sha256(self.sav.read_bytes()).hexdigest()
        self.cover_path = root / "system-cover.jpg"
        Image.new("RGB", (512, 512), "white").save(self.cover_path, "JPEG")
        self.cover = {"cover_path": str(self.cover_path), "images": [{"path": str(self.cover_path), "mime_type": "image/jpeg"}],
            "cover_manifest": {"source_sha256": self.source_hash, "cover_sha256": hashlib.sha256(self.cover_path.read_bytes()).hexdigest(),
                "rendered_all": True, "total_elements": 1, "visible_elements": 1, "clipped_ids": [],
                "view": {"yaw": 45, "pitch": 60, "projection": "orthographic", "fit": "all"}}}
        folder = root / "hdl" / "verification-fixture"
        folder.mkdir(parents=True)
        self.rtl = folder / "cpu.sv"
        self.rtl.write_text("module aurex_rv32i_teaching; endmodule\n")
        hashes = {self.rtl.name: hashlib.sha256(self.rtl.read_bytes()).hexdigest()}
        bundle = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.report_path = folder / "verification.json"
        self.report = {"verification_id": "fixture-id", "profile": "rv32i_teaching_v1", "verified": True,
            "top": "aurex_rv32i_teaching", "report_path": str(self.report_path), "source_sha256": bundle,
            "source_files_sha256": hashes, "sources_paths": [str(self.rtl)],
            "compile": {"exit_code": 0, "failure": None}, "simulation": {"exit_code": 0, "failure": None},
            "scope": "Fixed five-program teaching subset checks, not full compliance"}
        self.report_path.write_text(json.dumps(self.report))
        self.manifest_path = Path(str(self.sav) + ".export.json")
        self.manifest = {"schema": "aurex.hdl-export.v1", "source_sha256": bundle,
            "source_files_sha256": hashes, "sav_sha256": self.source_hash, "sav_path": str(self.sav),
            "strict_export": True, "binary_logic_export": True,
            "export_verilog_sha256": hashlib.sha256(self.rtl.read_text().encode("utf-8")).hexdigest(),
            "verification_id": "fixture-id", "verification_report_path": str(self.report_path),
            "top": "aurex_rv32i_teaching"}
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.db = SessionDB(str(root / "sessions.sqlite3"))
        self.sid = self.db.session("review-session")
        self.rid = self.db.begin(self.sid, "review", "review-run")
        self.user = SimpleNamespace(user_id="account", token="SECRET_TOKEN", auth_code="SECRET_AUTH")
        self.runtime = SimpleNamespace(cache_dir=self.cache, task_id=self.rid, session_id=self.sid, user=self.user)
        self.args = {"sav_path": str(self.sav), "title": "申请发布的教学实验",
            "introduction": "请发布这一份教学子集电路，验证文件位于服务器本地，必须按照真实能力介绍。",
            "evidence_paths": [str(self.report_path), str(self.manifest_path)]}
        self.client = ReviewClient()
        self.events = []
        self.cover_mock = mock.patch("aurex.publication_review.circuits.publication_cover", return_value=self.cover).start()
        self.submit_mock = mock.patch("aurex.publishing.plar_api.submit_original_experiment", return_value={
            "summary_id": "6a9b5c31ec746130fc33154f", "image_counter": 1,
            "_cover_credential": {"Policy": "SECRET", "Authorization": "SECRET"}}).start()
        self.upload_mock = mock.patch("aurex.publishing.plar_api.upload_experiment_cover").start()
        self.confirm_mock = mock.patch("aurex.publishing.plar_api.confirm_original_experiment").start()
        self.addCleanup(mock.patch.stopall)

    def authorize(self, purpose="riscv_teaching_subset"):
        bind_task_actions(self.cache, task_id=self.rid, session_id=self.sid, source='admin',
            original_user_request='请验证并发布这个教学实验', explicit_publish_requested=True, purpose=purpose)

    def call(self, args=None):
        return review_and_publish(self.runtime, self.args if args is None else args, self.client, self.db,
                                  self.sid, self.runtime.task_id, lambda kind, data: self.events.append((kind, data)))

    def test_deterministic_validation_publishes_without_second_model_call(self):
        self.authorize()
        result = self.call()
        self.assertTrue(result["published"])
        self.assertEqual(self.client.requests, [])
        self.assertIs(result["cover_pixels_in_context"], False)
        self.assertIs(result["publication_review"], False)
        self.assertEqual(result["images"], self.cover["images"], "Cover stays available to UI without model attachment")
        self.assertEqual(self.submit_mock.call_args.kwargs["title"], self.args["title"])
        self.assertEqual(self.submit_mock.call_args.kwargs["introduction"], self.args["introduction"])
        self.assertNotIn("PRIVATE_REVIEW_REASONING", json.dumps(self.submit_mock.call_args.kwargs))
        self.assertNotIn("PRIVATE_REVIEW_REASONING", json.dumps(self.db.messages(self.sid)))
        self.assertFalse(any(k in {"model_start", "model_end", "reasoning_delta", "publication_review"} for k, _ in self.events))
        self.assertTrue(any(k == "publication_validated" and d["model_review"] is False for k, d in self.events))
        self.cover_mock.assert_called_once_with(self.runtime, str(self.sav))
        self.upload_mock.assert_called_once()
        self.confirm_mock.assert_called_once()
        with self.db.connect() as db:
            sources = db.execute("SELECT content FROM documents WHERE title LIKE 'Publication original PLSAV %'").fetchall()
        self.assertEqual(sources[0]["content"], self.sav.read_text())

    def test_explicit_false_still_generates_and_uploads_fixed_cover(self):
        self.authorize()
        result = self.call({**self.args, "with_image": False})
        self.assertTrue(result["published"])
        self.assertEqual(self.client.requests, [])
        self.cover_mock.assert_called_once()
        self.upload_mock.assert_called_once()
        self.assertIs(result["cover_pixels_in_context"], False)

    def test_explicit_true_is_compatible_but_does_not_start_model_review(self):
        self.authorize()
        result = self.call({**self.args, "with_image": True})
        self.assertEqual(self.client.requests, [])
        self.assertIs(result["cover_pixels_in_context"], False)
        self.cover_mock.assert_called_once_with(self.runtime, str(self.sav))
        self.upload_mock.assert_called_once()
        self.assertTrue(any(k == "publication_validated" for k, _ in self.events))

    def test_oversized_hdl_review_publishes_text_only_type3_without_cover(self):
        from aurex.tools.phy_engine import _celestial_hdl_source_template
        self.source = _celestial_hdl_source_template()
        self.sav.write_text(json.dumps(self.source, ensure_ascii=False, separators=(",", ":")), encoding="utf8")
        self.source_hash = hashlib.sha256(self.sav.read_bytes()).hexdigest()
        self.manifest.update(sav_sha256=self.source_hash,
            publication_fallback={"schema": "aurex.hdl-source-celestial-fallback.v1",
                "reason": "physical_gate_element_limit_exceeded", "max_direct_elements": 5000,
                "gate_elements": 92000, "gate_wires": 100000,
                "discarded_gate_plsav_sha256": "a" * 64, "template_type": 3,
                "interactive_circuit": False, "allowed_followup": "comments_only"})
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.authorize()
        self.submit_mock.return_value = {"summary_id": "6a9b5c31ec746130fc33154f", "image_counter": 0}
        result = self.call({**self.args, "with_image": False})
        self.assertTrue(result["published"])
        self.assertEqual(result["images"], [])
        self.assertIsNone(result["cover_sha256"])
        self.cover_mock.assert_not_called()
        self.upload_mock.assert_not_called()
        self.confirm_mock.assert_called_once_with(self.user,
            summary_id="6a9b5c31ec746130fc33154f", image_counter=0)
        published = self.submit_mock.call_args.kwargs
        self.assertEqual(published["cover_bytes"], 0)
        self.assertIn("## 已验证 HDL 源码", published["introduction"])
        self.assertIn("module aurex_rv32i_teaching", published["introduction"])
        self.assertEqual(self.client.requests, [])

    def test_oversized_hdl_rejects_with_image(self):
        from aurex.tools.phy_engine import _celestial_hdl_source_template
        self.source = _celestial_hdl_source_template()
        self.sav.write_text(json.dumps(self.source, ensure_ascii=False, separators=(",", ":")), encoding="utf8")
        self.source_hash = hashlib.sha256(self.sav.read_bytes()).hexdigest()
        self.manifest.update(sav_sha256=self.source_hash,
            publication_fallback={"schema": "aurex.hdl-source-celestial-fallback.v1",
                "reason": "physical_gate_element_limit_exceeded", "max_direct_elements": 5000,
                "gate_elements": 5001, "gate_wires": 0,
                "discarded_gate_plsav_sha256": "b" * 64, "template_type": 3,
                "interactive_circuit": False, "allowed_followup": "comments_only"})
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.authorize()
        with self.assertRaisesRegex(ToolError, "不能生成、读取或上传截图"):
            self.call({**self.args, "with_image": True})
        self.cover_mock.assert_not_called()
        self.submit_mock.assert_not_called()

    def test_with_image_rejects_truthy_strings_numbers_and_null_before_review(self):
        self.authorize()
        for value in ("true", "false", 1, 0, None, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ToolError, "with_image必须是布尔值"):
                self.call({**self.args, "with_image": value})
        self.cover_mock.assert_not_called()
        self.assertFalse(self.client.requests)
        self.submit_mock.assert_not_called()

    def test_explicit_visual_request_does_not_grant_publication_authority(self):
        with self.assertRaises(ToolError):
            self.call({**self.args, "with_image": True})
        self.cover_mock.assert_not_called()
        self.assertFalse(self.client.requests)
        self.submit_mock.assert_not_called()

    def test_text_only_review_cannot_bypass_cover_bytes_or_full_component_coverage(self):
        self.authorize()
        manifest = copy.deepcopy(self.cover["cover_manifest"])
        for changes, error in (({"cover_sha256": "0" * 64}, "封面字节"),
                               ({"visible_elements": 0}, "every source element"),
                               ({"clipped_ids": ["r1"]}, "every source element"),
                               ({"rendered_all": False}, "every source element")):
            self.cover["cover_manifest"] = {**manifest, **changes}
            with self.subTest(changes=changes), self.assertRaisesRegex(ToolError, error):
                self.call({**self.args, "with_image": False})
        self.submit_mock.assert_not_called()
        self.upload_mock.assert_not_called()

    def test_publish_tool_schema_defaults_to_text_and_refuses_nonboolean_image_flag(self):
        import jsonschema
        from aurex.tools.plar_tools import PLAR_PUBLISH_EXPERIMENT_TOOL
        schema = PLAR_PUBLISH_EXPERIMENT_TOOL["parameters"]
        self.assertIs(schema["properties"]["with_image"]["default"], False)
        self.assertNotIn("with_image", schema["required"])
        for args in (self.args, {**self.args, "with_image": False}, {**self.args, "with_image": True}):
            jsonschema.validate(args, schema)
        for value in ("true", 1, None):
            with self.subTest(value=value), self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({**self.args, "with_image": value}, schema)

    def test_without_authority_stops_before_cover_or_model(self):
        with self.assertRaises(ToolError):
            self.call()
        self.cover_mock.assert_not_called()
        self.assertFalse(self.client.requests)
        self.submit_mock.assert_not_called()

    def test_community_intent_binding_rejects_ambiguous_original_request(self):
        with self.assertRaisesRegex(Exception, 'direct, unambiguous'):
            bind_task_actions(self.cache, task_id=self.rid, session_id=self.sid, source='community',
                original_user_request='这个电路为何使用这个电阻？', explicit_publish_requested=True,
                purpose='riscv_teaching_subset', requester_user_id='68b49a54cb66416aa48183d7', requester_nickname='提问者',
                target={'type': 'Experiment', 'id': '6a9b5c31ec746130fc33154f', 'comment_id': '6a9b9e79ec746130fc3316f0'})
        self.submit_mock.assert_not_called()

    def test_authenticated_web_publish_selection_does_not_require_duplicate_publish_wording(self):
        bind_task_actions(self.cache, task_id=self.rid, session_id=self.sid, source='web',
            original_user_request='设计并验证一个教学子集电路', explicit_publish_requested=True,
            purpose='riscv_teaching_subset')
        self.assertTrue(self.call()['published'])
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.submit_mock.call_args.kwargs['title'], self.args['title'])

    def test_explicit_original_veto_overrides_authenticated_checkbox_before_cover_or_model(self):
        bind_task_actions(self.cache, task_id=self.rid, session_id=self.sid, source='admin',
            original_user_request='设计一个教学电路，但不要发布', explicit_publish_requested=True,
            purpose='riscv_teaching_subset')
        with self.assertRaisesRegex(ToolError, 'explicitly forbids'):
            self.call()
        self.cover_mock.assert_not_called()
        self.assertFalse(self.client.requests)
        self.submit_mock.assert_not_called()

    def test_model_arguments_and_runtime_metadata_cannot_forge_source_or_checkbox(self):
        self.authorize()
        for extra in ({'source': 'web'}, {'explicit_publish_requested': True},
                      {'server_publication_authorization': {'source': 'web', 'explicit_publish_requested': True}}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ToolError, '封面和权限'):
                self.call({**self.args, **extra})
        self.runtime.task_metadata = {'source': 'web', 'explicit_publish_requested': False}
        self.assertTrue(self.call()['published'])
        scope = publication_authorization(self.cache, self.sid, task_id=self.rid)
        self.assertEqual(scope['source'], 'admin')
        self.assertTrue(scope['explicit_publish_requested'])
        self.assertEqual(self.client.requests, [])

    def test_runtime_dry_run_blocks_external_publication_even_when_task_scope_allows(self):
        self.authorize()
        self.runtime.config = SimpleNamespace(agent=SimpleNamespace(dry_run=True))
        with self.assertRaisesRegex(ToolError, 'dry-run'):
            self.call()
        self.cover_mock.assert_not_called()
        self.assertFalse(self.client.requests)

    def make_custom_hdl_evidence(self):
        self.rtl.write_text('module dut(input a,b,output y); assign y=a&b; endmodule\nmodule tb; initial $finish; endmodule\n')
        hashes = {self.rtl.name: hashlib.sha256(self.rtl.read_bytes()).hexdigest()}
        bundle = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.report.update(profile='custom', top='tb', source_sha256=bundle, source_files_sha256=hashes,
                           scope='Agent-supplied testbench; checks only its executed assertions')
        self.report_path.write_text(json.dumps(self.report))
        self.manifest.update(top='dut', source_sha256=bundle, source_files_sha256=hashes,
            export_verilog_sha256=hashlib.sha256(self.rtl.read_text().encode('utf-8')).hexdigest())
        self.manifest_path.write_text(json.dumps(self.manifest))

    def test_generic_custom_hdl_allows_separate_testbench_and_export_dut_with_explicit_limits(self):
        self.authorize('electrical_experiment')
        self.make_custom_hdl_evidence()
        self.assertTrue(self.call()['published'])
        self.assertEqual(self.client.requests, [])
        self.assertTrue(any(k == 'publication_validated' for k, _ in self.events))

    def test_generic_hdl_refuses_timeout_and_nonexistent_export_top(self):
        self.authorize('electrical_experiment')
        self.make_custom_hdl_evidence()
        self.report['simulation'] = {'exit_code': 0, 'failure': 'timeout'}
        self.report_path.write_text(json.dumps(self.report))
        with self.assertRaises(ToolError):
            self.call()
        self.make_custom_hdl_evidence()
        self.report['simulation'] = {'exit_code': 0, 'failure': None}
        self.report_path.write_text(json.dumps(self.report))
        self.manifest['top'] = 'nonexistent'
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ToolError, 'DUT顶层'):
            self.call()
        self.submit_mock.assert_not_called()

    def test_model_cannot_choose_cover_or_session_identifiers(self):
        self.authorize()
        with self.assertRaisesRegex(ToolError, "封面和权限"):
            self.call({**self.args, "cover_path": str(self.cover_path)})
        self.runtime.session_id = "wrong"
        with self.assertRaisesRegex(ToolError, "标识不匹配"):
            self.call()
        self.cover_mock.assert_not_called()

    def test_undocumented_555_proof_is_rejected_not_published_from_verified_flag(self):
        self.authorize("555_state_table")
        with self.assertRaisesRegex(ToolError, "555任务需要唯一"):
            self.call()
        self.cover_mock.assert_not_called()

    def test_cancel_during_local_validation_does_not_submit(self):
        self.authorize()
        cancelled = False
        def check():
            if cancelled:
                raise RuntimeError('cancelled')
        original_cover = self.cover
        def cancel(*_args):
            nonlocal cancelled
            cancelled = True
            return original_cover
        self.runtime.check_cancel = check
        self.cover_mock.side_effect = cancel
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            self.call()
        self.submit_mock.assert_not_called()
        self.assertEqual(publication_authorization(self.cache, self.sid, task_id=self.rid)['state'], 'authorized')

    def test_bad_rtl_report_export_binding_rejects_before_cover(self):
        self.authorize()
        report_before, manifest_before = self.report_path.read_bytes(), self.manifest_path.read_bytes()
        cases = [(self.report_path, {**self.report, "verified": False}),
            (self.report_path, {**self.report, "simulation": {"exit_code": 0, "failure": "timeout"}}),
            (self.report_path, {**self.report, "profile": "custom"}),
            (self.manifest_path, {**self.manifest, "sav_sha256": "0" * 64}),
            (self.manifest_path, {**self.manifest, "strict_export": False}),
            (self.manifest_path, {**self.manifest, "source_sha256": "0" * 64})]
        for path, value in cases:
            self.report_path.write_bytes(report_before)
            self.manifest_path.write_bytes(manifest_before)
            path.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ToolError):
                self.call()
        self.assertFalse(self.client.requests)
        self.cover_mock.assert_not_called()
        self.submit_mock.assert_not_called()

    def test_source_changed_after_verification_or_forged_report_location_is_refused(self):
        self.authorize()
        original = self.rtl.read_bytes()
        self.rtl.write_bytes(original + b"// changed")
        with self.assertRaisesRegex(ToolError, "测试后发生变化"):
            self.call()
        self.rtl.write_bytes(original)
        forged = Path(self.cache, "copied.json")
        forged.write_bytes(self.report_path.read_bytes())
        with self.assertRaisesRegex(ToolError, "服务端hdl_simulate"):
            self.call({**self.args, "evidence_paths": [str(forged), str(self.manifest_path)]})
        self.assertFalse(self.client.requests)

    def test_review_client_settings_are_ignored_but_invalid_metadata_never_publishes(self):
        self.authorize()
        self.client.finish = "length"
        with self.assertRaises(ToolError):
            self.call({**self.args, 'title': 'English only'})
        self.assertEqual(self.client.requests, [])
        self.submit_mock.assert_not_called()

    def test_model_context_budget_no_longer_blocks_deterministic_publication(self):
        self.authorize()
        self.client.input_tokens = 32768
        self.assertTrue(self.call()['published'])
        self.assertEqual(self.client.requests, [])
        with self.db.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 3)

    def test_evidence_changed_during_cover_generation_cannot_get_approval(self):
        self.authorize()
        original = self.report_path.read_bytes()
        def mutate_evidence(*_args):
            self.report_path.write_bytes(original + b" ")
            return self.cover
        self.cover_mock.side_effect = mutate_evidence
        with self.assertRaisesRegex(ToolError, "发布校验期间发生变化"):
            self.call()
        self.submit_mock.assert_not_called()

    def test_cover_source_hash_mismatch_fails_before_model(self):
        self.authorize()
        self.cover["cover_manifest"]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ToolError, "封面与发布源文件"):
            self.call()
        self.assertFalse(self.client.requests)
        self.submit_mock.assert_not_called()

    def test_resume_confirm_uses_original_receipt_without_revalidation_or_submit(self):
        self.authorize()
        self.confirm_mock.side_effect = [TimeoutError(), None]
        first = self.call()
        self.assertEqual(first["state"], "confirm_pending")
        self.runtime.task_id = self.rid
        second = self.call(args={})
        self.assertTrue(second["published"])
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.cover_mock.call_count, 1)
        self.assertEqual(self.submit_mock.call_count, 1)
        self.assertEqual(self.upload_mock.call_count, 1)
        self.assertEqual(self.confirm_mock.call_count, 2)
        self.assertTrue(any(k == "publication_resume" and d["original_run_id"] == "review-run" for k, d in self.events))

    def test_unknown_submit_is_never_reapproved_or_resubmitted_on_next_run(self):
        self.authorize()
        self.submit_mock.side_effect = TimeoutError()
        first = self.call()
        self.assertEqual(first["state"], "unknown")
        self.runtime.task_id = self.rid
        second = self.call(args={})
        self.assertEqual(second["state"], "unknown")
        self.assertEqual(self.client.requests, [])
        self.assertEqual(self.submit_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
