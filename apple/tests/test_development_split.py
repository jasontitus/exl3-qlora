import copy,hashlib,json,pathlib,unittest
import os
import exl3_mlx.development_split as m

def rows(n=8):
 return [{'train_row':i,'id':str(i),'repo':'r'+str(i%2),'commit':str(i),'kind':'diagnostic' if i%3==0 else 'exact_edit','row_sha256':hashlib.sha256(str(i).encode()).hexdigest(),'serialized_tokens':100+i*500,'supervised_tokens':50+i*100,'allocation_labels':['broad-repo-bucket'],'patch_ids':[str(i)],'development_exclusion_reasons':[]} for i in range(n)]
class Tests(unittest.TestCase):
 def test_deterministic_and_partition(self):
  r=rows();a=m.propose(r,[],3);b=m.propose(r,[],3);self.assertEqual(a,b);self.assertEqual(sorted(a['development_rows']+a['future_train_rows']),list(range(8)))
 def test_parent_transitive_whole(self):
  r=rows();edges=[{'rows':[0,1],'reason':'parent'},{'rows':[1,2],'reason':'known port'}];a=m.propose(r,edges,3);self.assertTrue({0,1,2}.issubset(a['development_rows']) or {0,1,2}.issubset(a['future_train_rows']))
 def test_allocation_does_not_merge(self):self.assertEqual(len(m.groups(rows(),[])),8)
 def test_patch_duplicate_auto_merge(self):
  r=rows();r[1]['patch_ids']=r[0]['patch_ids'];g=m.groups(r,[]);self.assertTrue(any(x['rows']==[0,1] for x in g))
 def test_unresolved_family_excluded(self):
  r=rows();r[0]['development_exclusion_reasons']=['unresolved port'];a=m.propose(r,[{'rows':[0,1],'reason':'parent'}],3);self.assertTrue({0,1}.issubset(a['future_train_rows']))
 def test_reject_loss_field(self):
  r=rows();r[0]['loss']=.1
  with self.assertRaises(ValueError):m.propose(r,[],3)
 def test_strict_scalar(self):
  for key,value in [('train_row',False),('serialized_tokens',100.0),('supervised_tokens',True)]:
   r=rows();r[0][key]=value
   with self.assertRaises(ValueError):m.propose(r,[],3)
 def test_unattainable_does_not_split(self):
  r=rows(4);edges=[{'rows':[0,i],'reason':'family'} for i in range(1,4)]
  with self.assertRaises(ValueError):m.propose(r,edges,2)
 def test_invalid_mask(self):
  r=rows();r[0]['supervised_tokens']=101
  with self.assertRaises(ValueError):m.propose(r,[],3)
 @unittest.skipUnless(os.environ.get("DEVELOPMENT_SPLIT_EVIDENCE_DIR"), "optional sealed local evidence replay")
 def test_actual_replay(self):
  p=pathlib.Path(os.environ["DEVELOPMENT_SPLIT_EVIDENCE_DIR"]);x=json.loads((p/'split-input.json').read_text());old=json.loads((p/'proposal.json').read_text());new=m.propose(x['rows'],x['edges'],41)
  self.assertEqual(new['development_rows'],old['development_rows']);self.assertEqual(new['summary'],old['summary']);self.assertEqual(new['crossing_known_edges'],0)
if __name__=='__main__':unittest.main()
