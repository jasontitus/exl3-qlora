import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == '__main__': unittest.main()
