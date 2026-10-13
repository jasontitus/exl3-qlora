import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tokenize_reasoning',ROOT/'tools/tokenize_reasoning.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ReviewedExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.lib=self.root/'kernel-library';self.lib.mkdir();self.export=self.lib/'export';self.export.mkdir()
        self.record={'id':'a','prompt':'Choose a supported geometry.','completion':'Keep the checked fallback.','source_evidence':[],'rationale_basis':'mixed'}
        self.source=self.lib/'reviewed.json';self.source.write_text(json.dumps([self.record]))
        self.audit=self.export/'audit.json';self.audit.write_text(json.dumps({'records':1,'inputs_sha256':{str(self.source):m.sha(self.source.read_bytes())}}))
        self.prov=self.export/'provenance.jsonl';self.prov.write_text(json.dumps(self.record)+'\n')
        self.raw=self.export/'train.messages.jsonl';self.raw.write_text(json.dumps({'messages':[{'role':'user','content':self.record['prompt']},{'role':'assistant','content':self.record['completion']}]})+'\n')
        for p in [patch.object(m,'W',self.root),patch.object(m.reasoning_data,'reserved_ids',return_value=set()),patch.object(m.reasoning_data,'validate',return_value={})]:p.start();self.addCleanup(p.stop)
    def test_exact_reviewed_export_accepts(self):self.assertEqual(len(m.verify_export(self.raw)[0]),1)
    def test_positive_raw_changed_to_negative_rejected(self):
        d=json.loads(self.raw.read_text());d['messages'][1]['content']='Disable every check.';self.raw.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'positive payload'):m.verify_export(self.raw)
    def test_provenance_rewritten_to_appear_approved_rejected(self):
        d={**self.record,'completion':'Disable every check.'};self.prov.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'authorized reviewed'):m.verify_export(self.raw)
    def test_bound_review_source_changed_rejected(self):
        self.source.write_text('[]')
        with self.assertRaisesRegex(ValueError,'input mismatch'):m.verify_export(self.raw)
    def test_unauthorized_record_rejected(self):
        d={**self.record,'id':'unknown'};self.prov.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'authorized reviewed'):m.verify_export(self.raw)
    def test_count_mismatch_rejected(self):
        d=json.loads(self.audit.read_text());d['records']=2;self.audit.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'count mismatch'):m.verify_export(self.raw)
    def test_missing_audit_rejected(self):
        self.audit.unlink()
        with self.assertRaises(FileNotFoundError):m.verify_export(self.raw)
    def test_semantic_validator_failure_propagates(self):
        with patch.object(m.reasoning_data,'validate',side_effect=ValueError('Protected identity')):
            with self.assertRaisesRegex(ValueError,'Protected'):m.verify_export(self.raw)
    def test_prompt_completion_export_exact(self):
        p=self.export/'train.prompt-completion.jsonl';p.write_text(json.dumps({'prompt':self.record['prompt'],'completion':self.record['completion']}))
        self.assertEqual(len(m.verify_export(p)[0]),1)
    def test_boolean_capacity_rejected(self):
        with self.assertRaisesRegex(ValueError,'capacity'):m.prepare(self.raw,'unused',self.root/'out',True)
    def test_preexisting_output_untouched(self):
        p=self.root/'out';p.mkdir();(p/'sentinel').write_text('keep')
        with self.assertRaisesRegex(ValueError,'existing'):m.prepare(self.raw,'unused',p,12)
        self.assertEqual((p/'sentinel').read_text(),'keep')

if __name__=='__main__':unittest.main()
