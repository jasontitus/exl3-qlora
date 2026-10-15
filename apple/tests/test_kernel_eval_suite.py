import copy
import hashlib
import json
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exl3_mlx import kernel_eval_suite as k


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.h = '1' * 64
        self.b = {'attempts': 1, 'max_output_tokens': 1024}
        self.s = {'format': 'kernel-eval-suite-v1', 'id': 'test', 'tasks': [dict(id='t', sha256=self.h, prompt_sha256=self.h, protocol_sha256=self.h)], 'adapters': [dict(id='base', sha256=self.h), dict(id='trained', sha256='2'*64)], 'budget': self.b, 'results': []}
        self.evidence = self.save('evidence.json', {'actual': 'synthetic test only'})

    def save(self, name, obj):
        raw = json.dumps(obj).encode()
        (self.root/name).write_bytes(raw)
        return {'path': name, 'sha256': hashlib.sha256(raw).hexdigest()}

    def row(self, adapter='base'):
        stages = {s: {'status': 'passed', 'evidence': [self.evidence]} for s in k.STAGES}
        stages['timing']['measurements'] = dict(workload_sha256=self.h, device='test', cache_state='warm', unit='ms', samples=[1., 2., 3.])
        return dict(format='kernel-eval-result-v1', task_id='t', adapter_id=adapter, task_sha256=self.h, adapter_sha256=self.h if adapter=='base' else '2'*64, prompt_sha256=self.h, protocol_sha256=self.h, budget=copy.deepcopy(self.b), attempt=1, tracks={'strict': stages})

    def run_rows(self, *rows):
        self.s['results'] = [self.save(f'r{i}.json', row) for i,row in enumerate(rows)]
        return k.build_report(self.s, k.Reader(self.root))

    def test_pending_not_ranked(self):
        report = self.run_rows()
        self.assertIsNone(report['winner'])
        self.assertEqual(report['summaries'][0]['stages']['strict']['numerical'], {'pending': 1})

    def test_same_workload_correct_timing(self):
        self.assertEqual(len(self.run_rows(self.row(), self.row('trained'))['latency_comparisons']), 1)

    def test_cache_mismatch_not_paired(self):
        r = self.row('trained'); r['tracks']['strict']['timing']['measurements']['cache_state'] = 'cold'
        self.assertEqual(self.run_rows(self.row(), r)['latency_comparisons'], [])

    def test_failed_numerical_cannot_have_timing(self):
        r = self.row(); r['tracks']['strict']['numerical']['status'] = 'failed'
        with self.assertRaises(ValueError): self.run_rows(r)

    def test_refused_is_not_failure(self):
        r = self.row(); r['tracks']['strict'] = {'source': {'status': 'refused', 'evidence': [self.evidence]}}
        report = self.run_rows(r)
        self.assertEqual(report['summaries'][0]['stages']['strict']['source'], {'refused': 1})

    def test_duplicate_attempt(self):
        with self.assertRaises(ValueError): self.run_rows(self.row(), self.row())

    def test_budget_and_identity(self):
        for field, value in [('budget', {'attempts': 1, 'max_output_tokens': 512}), ('prompt_sha256', '3'*64), ('attempt', True)]:
            r = self.row(); r[field] = value
            with self.assertRaises(ValueError): self.run_rows(r)

    def test_normalized_not_raw_success(self):
        r = self.row(); r['tracks']['normalized'] = {'raw_format': {'status': 'passed', 'evidence': [self.evidence]}}
        with self.assertRaises(ValueError): self.run_rows(r)

    def test_normalized_can_follow_raw_failure(self):
        r = self.row(); norm = copy.deepcopy(r['tracks']['strict']); del norm['generation']; del norm['raw_format']
        r['tracks'] = {'strict': {'generation': r['tracks']['strict']['generation'], 'raw_format': {'status': 'failed', 'evidence': [self.evidence]}}, 'normalized': norm}
        report = self.run_rows(r)
        self.assertEqual(report['summaries'][0]['stages']['strict']['numerical'], {'pending': 1})
        self.assertEqual(report['summaries'][0]['stages']['normalized']['numerical'], {'passed': 1})

    def test_drift_and_forged_digest(self):
        reader = k.Reader(self.root); reader.read(self.evidence)
        (self.root/'evidence.json').write_text('{}')
        with self.assertRaises(ValueError): reader.recheck()
        with self.assertRaises(ValueError): reader.read(self.evidence)

    def test_cli_preserves_existing_output(self):
        suite = self.root/'suite.json'; suite.write_text(json.dumps(self.s))
        out = self.root/'report.json'
        k.main(['--suite', str(suite), '--output', str(out)])
        original = out.read_bytes()
        with self.assertRaises(FileExistsError): k.main(['--suite', str(suite), '--output', str(out)])
        self.assertEqual(out.read_bytes(), original)

    def test_huge_budget_and_combined_slots(self):
        self.s['budget']['attempts'] = 10**9
        with self.assertRaises(ValueError): self.run_rows()
        self.s['budget']['attempts'] = 100
        self.s['tasks'] = [dict(self.s['tasks'][0], id=f't{i}') for i in range(30)]
        with self.assertRaises(ValueError): self.run_rows()

    def test_recommendation_respects_common_support(self):
        good = self.row(); bad = self.row('trained')
        bad['tracks']['strict'] = {'generation': {'status': 'passed', 'evidence': [self.evidence]}, 'raw_format': {'status': 'failed', 'outcome_class': 'model_output', 'evidence': [self.evidence]}}
        r = self.run_rows(good, bad)
        self.assertEqual(r['recommendation']['adapters'], ['base'])
        self.assertTrue(r['recommendation']['coverage_complete'])
        self.assertIn('base', k.render_markdown(r))
        self.assertEqual(self.run_rows(good)['recommendation']['status'], 'insufficient_evidence')

    def test_infrastructure_excluded_and_cap_resolved(self):
        bad = self.row('trained')
        for status in ('failed', 'refused'):
            bad['tracks']['strict'] = {'generation': {'status': status, 'outcome_class': 'infrastructure', 'evidence': [self.evidence]}}
            self.assertEqual(self.run_rows(self.row(), bad)['recommendation']['status'], 'insufficient_evidence')
        bad['tracks']['strict'] = {'generation': {'status': 'failed', 'outcome_class': 'model_output', 'evidence': [self.evidence]}}
        self.assertEqual(self.run_rows(self.row(), bad)['recommendation']['adapters'], ['base'])
        bad['tracks']['strict'] = {'source': {'status': 'refused', 'outcome_class': 'model_output', 'evidence': [self.evidence]}}
        self.assertEqual(self.run_rows(self.row(), bad)['recommendation']['status'], 'insufficient_evidence')
        bad['tracks']['strict']['generation'] = {'status': 'passed', 'evidence': [self.evidence]}
        self.assertEqual(self.run_rows(self.row(), bad)['recommendation']['adapters'], ['base'])

    def test_descriptive_not_qualified_timing(self):
        r = self.run_rows(self.row(), self.row('trained'))
        self.assertFalse(r['latency_comparisons'][0]['timing_qualified'])
        self.assertEqual(r['recommendation']['status'], 'tie')

    def test_nan_bool_timing(self):
        for bad in (True, float('nan'), 0, -1):
            r = self.row(); r['tracks']['strict']['timing']['measurements']['samples'] = [bad]
            with self.assertRaises(ValueError): self.run_rows(r)

    def workload_rows(self, row, count=12):
        old = row['tracks']['strict']['timing']['measurements']
        values = [dict(copy.deepcopy(old), workload_sha256=f'{i + 1:064x}', status='passed') for i in range(count)]
        row['tracks']['strict']['timing']['measurements'] = values
        return values

    def test_twelve_workloads_keep_slowdowns_and_identity(self):
        left, right = self.row(), self.row('trained')
        xs, ys = self.workload_rows(left), self.workload_rows(right)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x['samples'] = [1., 2., 3.]
            y['samples'] = [float(i + 1)] * 3
        report = self.run_rows(left, right)
        ratios = report['latency_comparisons']
        self.assertEqual(len(ratios), 12)
        self.assertEqual([x['workload_sha256'] for x in ratios], [x['workload_sha256'] for x in xs])
        self.assertEqual([x['left_over_right_median_latency'] for x in ratios], [2. / (i + 1) for i in range(12)])
        self.assertTrue(all(not x['timing_qualified'] for x in ratios))
        self.assertEqual(report['timing_comparison_coverage'][0]['compared_workloads'], 12)
        self.assertIsNone(report['winner'])

    def test_multi_workload_failure_preserved_without_survivor_comparison(self):
        left, right = self.row(), self.row('trained')
        self.workload_rows(left)
        values = self.workload_rows(right)
        values[-1] = {k: v for k, v in values[-1].items() if k != 'samples'}
        values[-1].update(status='failed', reason='Synthetic failed workload')
        right['tracks']['strict']['timing']['status'] = 'failed'
        report = self.run_rows(left, right)
        self.assertEqual(report['latency_comparisons'], [])
        self.assertEqual(report['timing_comparison_coverage'][0]['right_statuses'], {'passed': 11, 'failed': 1})
        self.assertEqual(report['results'][1]['tracks']['strict']['timing']['measurements'][-1], values[-1])
        right['tracks']['strict']['timing']['status'] = 'passed'
        with self.assertRaisesRegex(ValueError, 'unfinished/failed'):
            self.run_rows(right)

    def test_multi_workload_missing_and_context_mismatch_visible(self):
        left, right = self.row(), self.row('trained')
        self.workload_rows(left)
        values = self.workload_rows(right, 11)
        values[-1]['cache_state'] = 'cold'
        report = self.run_rows(left, right)
        self.assertEqual(len(report['latency_comparisons']), 10)
        coverage = report['timing_comparison_coverage'][0]
        self.assertEqual((coverage['left_workloads'], coverage['right_workloads'], coverage['matched_workloads']), (12, 11, 10))
        self.assertFalse(coverage['same_declared_workloads'])

    def test_legacy_timing_matches_single_explicit_workload(self):
        right = self.row('trained')
        self.workload_rows(right, 1)[0]['workload_sha256'] = self.h
        self.assertEqual(len(self.run_rows(self.row(), right)['latency_comparisons']), 1)
        left = self.row()
        left['tracks']['strict']['timing']['measurements']['status'] = 'passed'
        with self.assertRaisesRegex(ValueError, 'legacy'):
            self.run_rows(left)

    def test_multi_workload_bounds_duplicates_and_invalid_samples(self):
        for value in ([], [None], [dict(status=True)]):
            row = self.row()
            row['tracks']['strict']['timing']['measurements'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_rows(row)
        row = self.row()
        values = self.workload_rows(row, 65)
        with self.assertRaisesRegex(ValueError, 'count'):
            self.run_rows(row)
        del values[64:]
        values[1] = copy.deepcopy(values[0])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.run_rows(row)
        for sample in (True, 0, -1, float('nan')):
            row = self.row()
            self.workload_rows(row, 1)[0]['samples'] = [sample]
            with self.subTest(sample=sample), self.assertRaises(ValueError):
                self.run_rows(row)

    def test_partial_timing_cannot_bypass_numerical_or_invent_failed_samples(self):
        row = self.row()
        values = self.workload_rows(row)
        row['tracks']['strict']['timing']['status'] = 'pending'
        row['tracks']['strict']['numerical']['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'passed numerical'):
            self.run_rows(row)
        row['tracks']['strict']['numerical']['status'] = 'passed'
        values[-1]['status'] = 'refused'
        with self.assertRaisesRegex(ValueError, 'fields'):
            self.run_rows(row)
        values[-1].pop('samples')
        self.assertEqual(self.run_rows(row)['results'][0]['tracks']['strict']['timing']['measurements'][-1]['status'], 'refused')

    def test_saved_actual_transpose_twelve_workloads(self):
        path = Path(__file__).parent / 'fixtures/kernel_timing_transpose_base.json'
        fixture = json.loads(path.read_text())
        row = self.row()
        row['tracks']['strict']['timing']['measurements'] = fixture['measurements']
        row['tracks']['strict']['timing']['evidence'] = [{'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]
        values = self.run_rows(row)['results'][0]['tracks']['strict']['timing']['measurements']
        self.assertEqual(len(values), 12)
        self.assertEqual([len(v['samples']) for v in values], [30] * 12)
        for value, expected in zip(values, fixture['expected_medians_ms']):
            self.assertAlmostEqual(statistics.median(value['samples']), expected, places=12)
        self.assertEqual(fixture['provenance'][0]['sha256'], 'c23e31bc7e244e46676424dc0245217abc8fe6f5461f867fbe267e9ed293661c')

    def test_timing_record_budget_fails_closed_without_truncation(self):
        left, right = self.row(), self.row('trained')
        self.workload_rows(left, 3)
        self.workload_rows(right, 3)
        with patch.object(k, 'MAX_TIMING_RECORDS', 4):
            report = self.run_rows(left, right)
            self.assertEqual(len(report['latency_comparisons']), 3)
            self.assertEqual(len(report['timing_comparison_coverage']), 1)
        with patch.object(k, 'MAX_TIMING_RECORDS', 3):
            with self.assertRaisesRegex(ValueError, 'record budget exceeded'):
                self.run_rows(left, right)
            suite = self.root / 'over-budget-suite.json'
            suite.write_text(json.dumps(self.s))
            output = self.root / 'must-not-publish.json'
            with self.assertRaisesRegex(ValueError, 'record budget exceeded'):
                k.main(['--suite', str(suite), '--output', str(output)])
            self.assertFalse(output.exists())
        # Coverage-only records count too; no passing sample can bypass this cap.
        left['tracks']['strict']['timing']['status'] = 'pending'
        with patch.object(k, 'MAX_TIMING_RECORDS', 0):
            with self.assertRaisesRegex(ValueError, 'record budget exceeded'):
                self.run_rows(left, right)
        self.assertEqual(k.MAX_TIMING_RECORDS, 65536)


if __name__ == '__main__': unittest.main()
