"""Evidence-gate tests use a plain DC source/resistor fixture, not a 555 design."""
import json
from pathlib import Path
import unittest

import test_analog_evidence as fixtures
import test_publication_review as review_fixtures
from aurex.publication_review import _analog_task, _electrical_task, _evidence
from aurex import publishing
from aurex.tools.registry import ToolError


class AnalogPublicationGateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.AnalogEvidenceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def evidence(self, trace=True):
        if trace:
            self.fixture.add_trace()
        report, path = self.fixture.record()
        source = publishing.inspect_publication_source(self.fixture.cache, str(self.fixture.sav_path))
        records = _evidence(self.fixture.cache, [path])
        return report, path, source, records

    def test_real_half_second_sequence_is_necessary_but_never_functional_proof(self):
        report, path, source, records = self.evidence()
        checks = _analog_task(self.fixture.cache, source, records)
        self.assertFalse(checks['functional_verification'])
        self.assertTrue(checks['generic_numerical_and_export_verified'])
        self.assertEqual(checks['actual_stop_s'], .5)
        self.assertEqual(checks['sample_count'], 2)
        self.assertEqual(checks['complete_native_spec'], self.fixture.spec)
        self.assertEqual(checks['required_final_table'].count('| V1 |'), 1)
        self.assertEqual(checks['required_final_table'].count('| R1 |'), 1)
        self.assertIn('NOT every raw waveform point', checks['limitation'])
        self.assertEqual(len(checks['sampled_node_statistics']), 2)
        # No hardcoded oscillation mode: the publication text must stay within
        # the executed evidence and cannot claim an unmeasured target function.
        self.assertTrue(all(not node['midpoint_crossing_brackets'] for node in checks['sampled_node_statistics']))

    def test_generic_analog_exposes_actual_trace_statistics_without_target_mode_requirement(self):
        report, path, source, records = self.evidence()
        checks = _electrical_task(self.fixture.cache, source, records)
        self.assertEqual(checks['kind'], 'analog_electrical_experiment')
        self.assertEqual(checks['actual_stop_s'], .5)
        self.assertEqual(checks['sample_times_s'], [.25, .5])
        self.assertEqual(checks['sample_count'], 2)
        self.assertIn('sampled_node_statistics', checks)
        self.assertFalse(checks['functional_verification'])

    def test_generic_static_analog_explicitly_has_no_waveform_evidence(self):
        report, path, source, records = self.evidence(trace=False)
        checks = _electrical_task(self.fixture.cache, source, records)
        self.assertIsNone(checks['actual_stop_s'])
        self.assertEqual(checks['sample_count'], 0)
        self.assertEqual(checks['sample_times_s'], [])
        self.assertIn('no waveform', checks['statistics_scope'])

    def test_static_and_endpoint_only_evidence_are_not_sufficient(self):
        report, path, source, records = self.evidence(trace=False)
        with self.assertRaisesRegex(ToolError, '瞬态序列'):
            _analog_task(self.fixture.cache, source, records)
        self.fixture.make_transient()
        report, path = self.fixture.record()
        with self.assertRaisesRegex(ToolError, '瞬态序列'):
            _analog_task(self.fixture.cache, source, _evidence(self.fixture.cache, [path]))

    def test_changed_raw_measurement_or_source_cannot_use_old_verified_flag(self):
        report, path, source, records = self.evidence()
        original = self.fixture.state_path.read_bytes()
        self.fixture.state_path.write_bytes(original + b' ')
        with self.assertRaisesRegex(ToolError, '复验失败'):
            _analog_task(self.fixture.cache, source, records)
        self.fixture.state_path.write_bytes(original)
        with self.assertRaisesRegex(ToolError, '不是同一源文件'):
            _analog_task(self.fixture.cache, {**source, 'sha256': '0' * 64}, records)

    def test_fake_verified_schema_does_not_replace_original_server_report(self):
        report, path, source, records = self.evidence()
        fake = Path(self.fixture.cache) / 'fake.json'
        fake.write_text(json.dumps(report))
        with self.assertRaisesRegex(ToolError, '原始证据复验失败'):
            _analog_task(self.fixture.cache, source, _evidence(self.fixture.cache, [str(fake)]))

    def test_server_appends_complete_final_table_without_second_model_review(self):
        report, path, source, records = self.evidence()
        review = review_fixtures.PublicationReviewTests()
        review.setUp()
        self.addCleanup(review.doCleanups)
        review.cache = self.fixture.cache
        review.runtime.cache_dir = self.fixture.cache
        cover_path = Path(self.fixture.cache) / 'system-cover.jpg'
        cover_path.write_bytes(review.cover_path.read_bytes())
        review.cover['cover_path'] = str(cover_path)
        review.cover['images'] = [{'path': str(cover_path), 'mime_type': 'image/jpeg'}]
        review.cover['cover_manifest'].update(source_sha256=source['sha256'],
                                              total_elements=source['elements'], visible_elements=source['elements'])
        review.args.update(sav_path=source['path'], evidence_paths=[path, report['trace_table']['path']])
        review.authorize('555_state_table')
        result = review.call()
        self.assertTrue(result['published'])
        introduction = review.submit_mock.call_args.kwargs['introduction']
        self.assertIn(Path(report['table']['path']).read_text().strip(), introduction)
        self.assertEqual(introduction.count('| V1 |'), 1)
        self.assertEqual(introduction.count('| R1 |'), 1)
        self.assertEqual(review.client.requests, [])
        self.assertFalse(_analog_task(self.fixture.cache, source, records)['functional_verification'])
        with review.db.connect() as db:
            archived = db.execute("SELECT content FROM documents WHERE title LIKE 'Publication evidence %'").fetchall()
        self.assertTrue(any(report['trace_table']['path'] in row['content'] or 'time' in row['content'] for row in archived))


if __name__ == '__main__':
    unittest.main()
