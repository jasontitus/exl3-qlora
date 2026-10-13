import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reasoning_data',ROOT/'tools/reasoning_data.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class EvidenceGateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tests')
        self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        evidence=self.path/'source.txt';evidence.write_text('Historical author: faster only for short contexts.\n')
        license=self.path/'LICENSE';license.write_text('Synthetic test fixture license metadata.\n')
        self.r={'id':'fixture','repo':'https://github.com/ml-explore/mlx','commit':'1'*40,'parent':'2'*40,'lineage':'fixture-change',
                'task_type':'reject_optimization','outcome_class':'rejected','prompt':'Should the reported result apply to all contexts?',
                'completion':'No; the evidence is limited to short contexts.','rationale_basis':'source_reported',
                'source_evidence':[{'path':str(evidence),'sha256':m.sha(evidence.read_bytes()),'quote':'faster only for short contexts.','url':'https://github.com/example/fixture/commit/'+'1'*40}],
                'licenses':[{'path':str(license),'sha256':m.sha(license.read_bytes())}],
                'review':{'status':'approved','reviewer':'test-only','evidence_fidelity':True,'no_fabricated_observations':True,'task_answer_separation':True,'lineage_screened':True},
                'protection_review':{'status':'cleared','reason':'Synthetic fixture'},'validation':{'local_execution':'not_run'}}
    def test_valid_source_backed_rejection(self):self.assertEqual(len(m.validate(self.r,set())),2)
    def test_rejected_and_correctness_examples_cannot_bypass_protection(self):
        for outcome in ('positive','tradeoff','correctness','rejected','neutral','unknown'):
            for field in ('commit','parent'):
                r=copy.deepcopy(self.r);r['outcome_class']=outcome
                with self.assertRaisesRegex(ValueError,'Protected'):m.validate(r,{r[field]})
    def test_review_required(self):
        for field in ('evidence_fidelity','no_fabricated_observations','task_answer_separation','lineage_screened'):
            r=copy.deepcopy(self.r);r['review'][field]=False
            with self.assertRaisesRegex(ValueError,'review check'):m.validate(r,set())
        self.r['review']['status']='pending'
        with self.assertRaisesRegex(ValueError,'semantic review'):m.validate(self.r,set())
    def test_quote_and_hash_tamper(self):
        r=copy.deepcopy(self.r);r['source_evidence'][0]['quote']='invented performance result'
        with self.assertRaisesRegex(ValueError,'quote absent'):m.validate(r,set())
        Path(self.r['source_evidence'][0]['path']).write_text('Changed source')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):m.validate(self.r,set())
    def test_no_claimed_execution_without_receipt(self):
        self.r['validation']['local_execution']='performance'
        with self.assertRaisesRegex(ValueError,'requires receipt'):m.validate(self.r,set())
    def test_evidence_path_boundary(self):
        self.r['source_evidence'][0]['path']='/private/tmp/unrelated-secret'
        with self.assertRaisesRegex(ValueError,'outside kernel-library'):m.validate(self.r,set())
    def test_reserved_control_markers_rejected(self):
        for field in ('prompt','completion'):
            r=copy.deepcopy(self.r);r[field]+='<｜end▁of▁sentence｜>'
            with self.assertRaisesRegex(ValueError,'reserved tokenizer'):m.validate(r,set())
    def test_json_pointer_escaped_quote(self):
        p=self.path/'data.json';p.write_text(json.dumps({'source':{'text':'line1\nline2'}}))
        e=self.r['source_evidence'][0];e.update(path=str(p),sha256=m.sha(p.read_bytes()),quote='line1\nline2',json_pointer='/source/text')
        m.validate(self.r,set())
    def test_missing_license_and_semantic_review(self):
        r=copy.deepcopy(self.r);r['licenses']=[]
        with self.assertRaisesRegex(ValueError,'License provenance'):m.validate(r,set())
        self.r['protection_review']['status']='unknown'
        with self.assertRaisesRegex(ValueError,'Semantic protection'):m.validate(self.r,set())
    def test_export_counts_lineages_not_views(self):
        r2=copy.deepcopy(self.r);r2.update(id='second',prompt='What evidence is still missing?')
        p=self.path/'records.json';p.write_text(json.dumps([self.r,r2]));report=m.export([p],self.path/'export')
        self.assertEqual((report['records'],report['distinct_commits'],report['distinct_lineages']),(2,1,1))
        with self.assertRaisesRegex(ValueError,'overwrite'):m.export([p],self.path/'export')
    def test_duplicate_payload_rejected(self):
        r2=copy.deepcopy(self.r);r2['id']='different-label'
        p=self.path/'records.json';p.write_text(json.dumps([self.r,r2]))
        with self.assertRaisesRegex(ValueError,'Duplicate prompt'):m.export([p],self.path/'export')
    def test_reserved_ledger_checksum(self):
        self.assertGreater(len(m.reserved_ids()),100)
        with self.assertRaisesRegex(ValueError,'ledger changed'):m.reserved_ids(expected_sha='0'*64)

if __name__=='__main__':unittest.main()
