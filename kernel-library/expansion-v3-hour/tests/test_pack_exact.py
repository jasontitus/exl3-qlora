import copy
import importlib.util
import hashlib
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pack_exact',ROOT/'tools/pack_exact.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from exact_edit_renderer import unified_patch,render_command


def blob(text):
    raw=text.encode();return {'content':text,'sha256':hashlib.sha256(raw).hexdigest(),'blob':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'mode':'100644'}

class ExactGateTests(unittest.TestCase):
    def setUp(self):
        self.s={'files':{'kernel.metal':{'role':'changed','before':blob('old'),'after':blob('new')}}}
        patch=unified_patch({'kernel.metal':b'old'},{'kernel.metal':b'new'})
        self.raw={'messages':[{'role':'assistant','tool_calls':[{'type':'function','function':{'name':'exec_command','arguments':{'cmd':render_command(patch)}}}]}]}
        self.receipt={'patch_sha256':hashlib.sha256(patch.encode()).hexdigest()}
    def test_exact_missing_newline_replay(self):m.verify_source(self.s);self.assertTrue(m.verify_edit(self.s,self.raw,self.receipt)['exact_scoped_replay'])
    def test_content_tamper(self):
        self.s['files']['kernel.metal']['after']['content']='different'
        with self.assertRaisesRegex(ValueError,'Git blob mismatch'):m.verify_source(self.s)
    def test_blob_oid_tamper(self):
        self.s['files']['kernel.metal']['after']['blob']='0'*40
        with self.assertRaisesRegex(ValueError,'Git blob mismatch'):m.verify_source(self.s)
    def test_code_omission_rejected(self):
        self.receipt['omitted_documentation']={'kernel.metal':{'reason':'convenient'}}
        with self.assertRaisesRegex(ValueError,'Non-documentation'):m.verify_edit(self.s,self.raw,self.receipt)
    def test_shell_execution_rejected(self):
        self.raw['messages'][0]['tool_calls'][0]['function']['arguments']['cmd']='echo unsafe'
        with self.assertRaisesRegex(ValueError,'Only the declared'):m.verify_edit(self.s,self.raw,self.receipt)
    def test_second_call_rejected(self):
        self.raw['messages'][0]['tool_calls']*=2
        with self.assertRaisesRegex(ValueError,'one exact'):m.verify_edit(self.s,self.raw,self.receipt)
    def test_changed_target_rejected(self):
        self.s['files']['kernel.metal']['after']=blob('unexpected')
        with self.assertRaises(ValueError):m.verify_edit(self.s,self.raw,self.receipt)

if __name__=='__main__':unittest.main()

class ReviewReceiptTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        from unittest.mock import patch
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.patch=patch.object(m,'WORK',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.lib=self.root/'kernel-library';self.lib.mkdir()
        self.entry={'id':'case','paths':{},'review':{'status':'approved'}}
        for name in ('source','raw','prepared','replay'):
            p=self.lib/(name+'.json');p.write_text('{}');self.entry['paths'][name]=str(p)
        self.receipt={'schema':'exact-edit-independent-review-v1','status':'approved','reviewer':'independent-peer','cases':[{'id':'case','inputs_sha256':{p:m.sha(Path(p).read_bytes()) for p in self.entry['paths'].values()}}]}
        self.review_path=self.lib/'review.json';self.publish()
    def publish(self):
        import json
        self.review_path.write_text(json.dumps(self.receipt));self.entry['review'].update(receipt=str(self.review_path),sha256=m.sha(self.review_path.read_bytes()))
    def test_bound_review_accepts(self):self.assertEqual(m.verify_review(self.entry,{})['reviewer'],'independent-peer')
    def test_status_alone_rejected(self):
        self.receipt={'status':'approved'};self.publish()
        with self.assertRaisesRegex(ValueError,'review receipt'):m.verify_review(self.entry,{})
    def test_other_case_rejected(self):
        self.receipt['cases'][0]['id']='other';self.publish()
        with self.assertRaisesRegex(ValueError,'not authorized'):m.verify_review(self.entry,{})
    def test_duplicate_authorization_rejected(self):
        self.receipt['cases']*=2;self.publish()
        with self.assertRaisesRegex(ValueError,'Duplicate'):m.verify_review(self.entry,{})
    def test_each_input_tamper_rejected(self):
        for name,p in self.entry['paths'].items():
            Path(p).write_text('{"tamper":true}')
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'identity mismatch'):m.verify_review(self.entry,{})
            Path(p).write_text('{}')
    def test_missing_input_binding_rejected(self):
        self.receipt['cases'][0]['inputs_sha256'].pop(self.entry['paths']['raw']);self.publish()
        with self.assertRaisesRegex(ValueError,'identity mismatch'):m.verify_review(self.entry,{})
    def test_hash_mutation_between_review_and_read_rejected(self):
        hashes={};m.verify_review(self.entry,hashes);p=Path(self.entry['paths']['raw']);p.write_text('{"new":1}')
        with self.assertRaisesRegex(ValueError,'during binding'):m.bound(p,hashes)
    def test_review_hash_tamper_rejected(self):
        self.review_path.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Review identity'):m.verify_review(self.entry,{})
    def test_outside_input_rejected(self):
        p=self.root/'outside.json';p.write_text('{}');self.entry['paths']['raw']=str(p)
        with self.assertRaisesRegex(ValueError,'outside library'):m.verify_review(self.entry,{})

class RawExportIntegrationTests(ReviewReceiptTests):
    def test_native_raw_export_roundtrips_without_metadata(self):
        import json
        from unittest.mock import patch
        source={'case_id':'case','repository':'https://github.com/example/repo','head':{'commit':'a'*40},'parent':{'commit':'b'*40},'commits':['a'*40,'b'*40],
                'protected':False,'input_metadata':{'historical_protection_review':{'status':'cleared'}},'stable_patch_id':'c'*40,'outcome':'source_only',
                'files':{'kernel.metal':{'role':'changed','before':blob('old'),'after':blob('new')}}}
        command=render_command(unified_patch({'kernel.metal':b'old'},{'kernel.metal':b'new'}))
        raw={'tools':[],'messages':[{'role':'assistant','tool_calls':[{'function':{'name':'exec_command','arguments':{'cmd':command}}}]}]}
        prepared={'input_ids':[1,2,3],'loss_start':1}
        replay={'patch_sha256':hashlib.sha256(unified_patch({'kernel.metal':b'old'},{'kernel.metal':b'new'}).encode()).hexdigest()}
        for name,value in [('source',source),('raw',raw),('prepared',prepared),('replay',replay)]:Path(self.entry['paths'][name]).write_text(json.dumps(value))
        self.receipt['cases'][0]['inputs_sha256']={p:m.sha(Path(p).read_bytes()) for p in self.entry['paths'].values()};self.publish()
        self.entry.update(curriculum_tier='correctness',lineage='change:a')
        index=self.lib/'index.json';index.write_text(json.dumps({'cases':[self.entry]}))
        model=self.root/'model';model.mkdir();(model/'config.json').write_text(json.dumps({'vocab_size':10}))
        output=self.lib/'out'
        def tokenize(row,tok):
            if set(row)!={'tools','messages'}:raise ValueError('Native tool row rejects metadata')
            return prepared.copy()
        with patch.object(m,'implementation_inputs',return_value={}),patch.object(m,'reserved_ids',return_value=set()),patch.object(m,'load_tokenizer',return_value=None),patch.object(m,'tokenizer_identity',return_value={}),patch.object(m,'tokenize_row',side_effect=tokenize),patch.object(m,'ALLOWED_REPOS',{'https://github.com/example/repo'}),patch.object(m.checks,'cross_split_code_report',return_value={'automatic_blocker':False,'near_duplicate_pairs':[]}):
            result=m.build([index],model,output,12)
            exported=json.loads((output/'train.jsonl').read_text())
            self.assertEqual(tokenize(exported,None),prepared)
            self.assertEqual(result['count'],1)
            self.assertEqual(set(exported),{'tools','messages'})
            metadata=json.loads((output/'case.provenance.json').read_text());self.assertEqual(metadata['source_id'],'change:a')

class CurriculumTierTests(unittest.TestCase):
    def test_unmeasured_is_opt_in(self):
        self.assertFalse(m.core_tier('source_grounded_unmeasured','source_only'))
        self.assertTrue(m.core_tier('source_grounded_unmeasured','source_only',True))
    def test_unmeasured_not_relabelled(self):
        self.assertFalse(m.core_tier('source_grounded_unmeasured','reported_improvement'))
        with self.assertRaisesRegex(ValueError,'negative outcome'):m.core_tier('source_grounded_unmeasured','negative',True)
    def test_measured_tier_needs_source_claim(self):
        for tier in ('optimization','measured_tradeoff'):
            with self.subTest(tier=tier),self.assertRaisesRegex(ValueError,'positive source'):m.core_tier(tier,'source_only')
    def test_correctness_not_counted_as_speed(self):self.assertTrue(m.core_tier('correctness','source_only'))
    def test_rejected_strategy_cannot_apply(self):
        for tier in ('rejected_optimization','negative','neutral'):
            with self.subTest(tier=tier),self.assertRaisesRegex(ValueError,'non-training'):m.core_tier(tier,'source_only',True)
