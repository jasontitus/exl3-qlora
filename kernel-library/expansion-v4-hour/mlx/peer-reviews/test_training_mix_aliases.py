"""Independent CPU rejection checks for known-family selection overlays."""
import unittest
from unittest.mock import patch
from pathlib import Path
from test_training_mix_independent import Mix,H,data,lines,m

class Aliases(Mix):
 def setup_diags(self):
  self.ds=[{'id':f'd{i}','repo':'https://github.com/example/a','commit':ch*40,'lineage':f'orig{i}'}for i,ch in enumerate('abc')]
  self.dr=[{'messages':[{'role':'user','content':str(40+i)}]}for i in range(3)]
  lines(self.diag/'provenance.jsonl',self.ds);p=self.diag/'provenance.jsonl'
  mock=patch.object(m,'verify_export',lambda *_:(self.dr,b'',{str(p):H(p.read_bytes())}));mock.start();self.addCleanup(mock.stop)
  self.alias=self.diag/'aliases.json';self.aliasdoc={'schema':'diagnostic-known-family-aliases-v1','groups':[{'id':'linked','members':[{'repo':d['repo'],'commit':d['commit']}for d in self.ds[:2]]}],'inputs_sha256':{}};data(self.alias,self.aliasdoc)
 def build_alias(self):return m.build(self.exact,[self.diag],self.model,self.output,self.alias)
 def test_alias_selects_one_family_preserves_original(self):
  self.setup_diags()
  extra=[{'messages':[{'role':'user','content':str(24+i)}]}for i in range(4)];ep=[m.tokenize_row(r,None)for r in extra]
  em=[{'id':f'ex{i}','split':'train','repo':'https://github.com/source/b','commit':f'{100+i:040x}','change_lineage':f'ex{i}','split_family':'other','raw_sha256':H(m.canon(r)),'prepared_sha256':H(m.canon(p)),'supervised_tokens':1}for i,(r,p)in enumerate(zip(extra,ep))]
  lines(self.exact/'train.raw.jsonl',self.raw[:4]+extra);lines(self.exact/'train.jsonl',self.pre[:4]+ep);lines(self.exact/'provenance.jsonl',self.meta[:4]+em+self.meta[4:]);self.audit['train_examples']=8;self.rehash()
  out=self.build_alias();self.assertEqual(out['diagnostic_rows'],2)
  import json
  meta=[json.loads(l)for l in(self.output/'provenance.jsonl').read_text().splitlines()];di=[x for x in meta if x.get('kind')=='diagnostic'];self.assertEqual([x['id']for x in di],['d0','d2']);self.assertEqual(di[0]['record']['original_lineage'],'orig0');self.assertEqual(di[0]['lineage'],'known-family:linked')
 def test_alias_heldout_bridge_rejected(self):
  self.setup_diags();self.aliasdoc['groups'][0]['members'].append({'repo':self.meta[4]['repo'],'commit':self.meta[4]['commit']});data(self.alias,self.aliasdoc)
  with self.assertRaisesRegex(ValueError,'connects diagnostic to heldout'):self.build_alias()
  self.assertFalse(self.output.exists())
 def test_intersecting_families_rejected(self):
  self.setup_diags();self.aliasdoc['groups'].append({'id':'unmerged','members':[self.aliasdoc['groups'][0]['members'][1]]});data(self.alias,self.aliasdoc)
  with self.assertRaisesRegex(ValueError,'Overlapping'):self.build_alias()
 def test_alias_evidence_drift_rejected(self):
  self.setup_diags();p=self.diag/'evidence';p.write_text('old');self.aliasdoc['inputs_sha256'][str(p)]=H(p.read_bytes());data(self.alias,self.aliasdoc);p.write_text('new')
  with self.assertRaisesRegex(ValueError,'source drift'):self.build_alias()
 def test_wrong_alias_schema_rejected(self):
  self.setup_diags();self.aliasdoc['schema']='wrong';data(self.alias,self.aliasdoc)
  with self.assertRaisesRegex(ValueError,'alias schema'):self.build_alias()

if __name__=='__main__':unittest.main()
