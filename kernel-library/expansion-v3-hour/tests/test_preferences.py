import copy,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import export_preferences as p
class PreferenceExportTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=ROOT/'tests');self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
  self.row=json.loads((ROOT/'reasoning-core/records-reviewed-v2.json').read_text())[0]
 def run_export(self,rows=None):
  f=self.path/'records.json';f.write_text(json.dumps(rows or [self.row]));return p.export(f,self.path/'out')
 def test_roundtrip_and_synthetic_label(self):
  r=self.run_export();self.assertEqual(r['pairs'],1)
  row=json.loads((self.path/'out/preferences.jsonl').read_text());self.assertEqual(set(row),{'prompt','chosen','rejected'});self.assertEqual(row['chosen'][0]['content'],self.row['completion']);self.assertEqual(row['rejected'][0]['content'],self.row['verifier_negative']['text'])
  self.assertIn('synthetic',r['scope'])
 def test_requires_negative_review(self):
  self.row['review'].pop('negative_rubric_review')
  with self.assertRaisesRegex(ValueError,'review required'):self.run_export()
 def test_rejects_fabricated_observation_label(self):
  self.row['verifier_negative']['origin']='observed_model_failure'
  with self.assertRaisesRegex(ValueError,'provenance'):self.run_export()
 def test_duplicate_id_rejected(self):
  with self.assertRaisesRegex(ValueError,'Duplicate'):self.run_export([self.row,copy.deepcopy(self.row)])
 def test_identical_answers_rejected(self):
  self.row['verifier_negative']['text']=self.row['completion']
  with self.assertRaisesRegex(ValueError,'rejected answer'):self.run_export()
 def test_marker_rejected(self):
  self.row['verifier_negative']['text']+='<｜end▁of▁sentence｜>'
  with self.assertRaisesRegex(ValueError,'marker'):self.run_export()
 def test_missing_rubric_rejected(self):
  self.row['rubric']['required']=[]
  with self.assertRaisesRegex(ValueError,'Rubric'):self.run_export()
if __name__=='__main__':unittest.main()
