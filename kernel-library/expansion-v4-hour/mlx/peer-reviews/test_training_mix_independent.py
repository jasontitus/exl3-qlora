"""Independent CPU rejection tests; mock token math, exercise publication bindings."""
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

W=Path(__file__).resolve().parents[4]
P=W/'kernel-library/expansion-v4-hour/tools/build_training_mix.py'
s=importlib.util.spec_from_file_location('review_training_mix',P);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
H=lambda b:hashlib.sha256(b).hexdigest()

def encoded(raw,*_):
 n=int(raw['messages'][0]['content']);return {'input_ids':[1,n+2,999],'loss_start':2}

def data(path,obj):path.write_text(json.dumps(obj)+'\n')
def lines(path,rows):path.write_bytes(b''.join(m.canon(x)+b'\n'for x in rows))

class Mix(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name);self.exact=root/'exact';self.exact.mkdir();self.output=root/'output';self.model=root/'model';self.model.mkdir();data(self.model/'config.json',{'vocab_size':10000});self.diag=root/'diag';self.diag.mkdir()
  self.identity={'tokenizer.json':'synthetic'};self.raw=[{'messages':[{'role':'user','content':str(i)}]}for i in range(20)];self.pre=[encoded(x)for x in self.raw];self.meta=[]
  for i in range(20):self.meta.append({'id':f'c{i}','split':'train'if i<4 else'test','repo':'https://github.com/ml-explore/mlx','commit':f'{i:040x}','change_lineage':f'lineage{i}','split_family':'llama-heldout'if i>=4 else'mlx','raw_sha256':H(m.canon(self.raw[i])),'prepared_sha256':H(m.canon(self.pre[i])),'supervised_tokens':1})
  lines(self.exact/'train.raw.jsonl',self.raw[:4]);lines(self.exact/'train.jsonl',self.pre[:4]);lines(self.exact/'test.raw.jsonl',self.raw[4:]);lines(self.exact/'test.jsonl',self.pre[4:]);lines(self.exact/'provenance.jsonl',self.meta);(self.exact/'valid.jsonl').write_bytes(b'');data(self.exact/'cross-split-code-overlap.json',{'exact_pairs':[]});self.audit={'capacity':12288,'truncated':0,'heldout_prepared_exact':True,'train_examples':4,'inputs_sha256':{},'tokenizer':self.identity};self.rehash()
  self.testbytes=(self.exact/'test.jsonl').read_bytes()
  for target,value in [('IMPLEMENTATION_HASHES',{}),('TEST_SHA',H(self.testbytes)),('tokenizer_identity',lambda *_:self.identity),('load_tokenizer',lambda *_:object()),('tokenize_row',encoded),('validate_example',lambda *_args,**_kwargs:None)]:
   p=patch.object(m,target,value);p.start();self.addCleanup(p.stop)
 def rehash(self):
  self.audit['artifacts_sha256']={p.name:H(p.read_bytes())for p in self.exact.iterdir()if p.name!='audit.json'};data(self.exact/'audit.json',self.audit)
 def runbuild(self,reasoning=()):return m.build(self.exact,reasoning,self.model,self.output)
 def test_success_preserves_exact_test_bytes(self):
  out=self.runbuild();self.assertEqual(out['counts'],{'train':4,'valid':0,'test':16});self.assertEqual((self.output/'test.prepared.jsonl').read_bytes(),self.testbytes)
 def test_unbound_consumed_provenance_rejected(self):
  del self.audit['artifacts_sha256']['provenance.jsonl'];data(self.exact/'audit.json',self.audit)
  with self.assertRaisesRegex(ValueError,'required artifact'):self.runbuild()
  self.assertFalse(self.output.exists())
 def test_tampered_raw_rejected(self):
  (self.exact/'train.raw.jsonl').write_bytes(b'{}\n')
  with self.assertRaisesRegex(ValueError,'artifact drift'):self.runbuild()
 def test_repaired_outer_checksum_cannot_change_frozen_test(self):
  (self.exact/'test.jsonl').write_bytes(self.testbytes+b'\n');self.rehash()
  with self.assertRaisesRegex(ValueError,'heldout'):self.runbuild()
 def test_nonempty_validation_rejected(self):
  (self.exact/'valid.jsonl').write_bytes(b'{}\n');self.rehash()
  with self.assertRaisesRegex(ValueError,'Validation'):self.runbuild()
 def test_raw_provenance_disagreement_rejected(self):
  self.meta[0]['raw_sha256']='0'*64;lines(self.exact/'provenance.jsonl',self.meta);self.rehash()
  with self.assertRaisesRegex(ValueError,'identity mismatch'):self.runbuild()
 def test_drift_during_tokenizer_load_rejected(self):
  def mutate(*_):(self.exact/'train.raw.jsonl').write_bytes(b'changed');return object()
  with patch.object(m,'load_tokenizer',mutate),self.assertRaisesRegex(ValueError,'drift'):self.runbuild()
  self.assertFalse(self.output.exists())
 def test_config_drift_during_load_rejected(self):
  def mutate(*_):data(self.model/'config.json',{'vocab_size':42});return object()
  with patch.object(m,'load_tokenizer',mutate),self.assertRaisesRegex(ValueError,'changed'):self.runbuild()
 def test_no_overwrite(self):
  self.output.mkdir()
  with self.assertRaisesRegex(ValueError,'fresh'):self.runbuild()
 def test_known_heldout_lineage_rejected(self):
  row={'messages':[{'role':'user','content':'22'}]};record={'id':'diagnostic','repo':'https://github.com/ml-explore/mlx','commit':'f'*40,'lineage':'LINEAGE4'};lines(self.diag/'provenance.jsonl',[record]);p=self.diag/'provenance.jsonl';bound={str(p):H(p.read_bytes())}
  with patch.object(m,'verify_export',lambda *_:([row],b'',bound)),self.assertRaisesRegex(ValueError,'protected lineage'):self.runbuild([self.diag])
 def test_commit_case_alias_and_lineage_cap(self):
  rs=[{'id':'a','repo':'https://github.com/A/B','commit':'a'*40,'lineage':'same'}, {'id':'b','repo':'https://github.com/a/b/','commit':'b'*40,'lineage':'SAME'}, {'id':'c','repo':'https://github.com/A/B','commit':'a'*40,'lineage':'other'}, {'id':'d','repo':'https://github.com/A/B','commit':'c'*40,'lineage':'new'}]
  selected=m.select_diagnostics(rs,set(),20);self.assertEqual([r['id']for r in selected],['a','d'])
 def test_new_commit_preference_and_bound(self):
  rs=[{'id':'a','repo':'https://github.com/a/b','commit':'a'*40,'lineage':'old'},{'id':'z','repo':'https://github.com/a/b','commit':'b'*40,'lineage':'new'}]
  self.assertEqual([r['id']for r in m.select_diagnostics(rs,{m.key(rs[0])},1)],['z']);self.assertEqual(m.select_diagnostics(rs,set(),0),[])

if __name__=='__main__':unittest.main()
