"""CPU patch replay in disposable copies only; no repository scripts or builds."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import check_sources as checker

class Bundle(unittest.TestCase):
 def setUp(self):self.manifest=json.loads((checker.HERE/'manifest.json').read_text())
 def copy(self,lane,root):
  for rel,row in self.manifest['lanes'][lane]['files'].items():
   dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(row['source_before'],dest)
 def test_complete_replay_both_roots(self):
  for lane,row in self.manifest['lanes'].items():
   with self.subTest(lane=lane),tempfile.TemporaryDirectory() as d:
    root=Path(d).resolve();self.copy(lane,root);checker.check(lane,root,'before')
    patch=checker.HERE/row['patch']
    for flags in (['--check'],[]):subprocess.run(['/usr/bin/git','-c','core.hooksPath=/dev/null','apply',*flags,str(patch)],cwd=root,check=True,capture_output=True)
    checker.check(lane,root,'after')
    for rel,entry in row['files'].items():self.assertEqual((root/rel).read_bytes(),Path(entry['source_after']).read_bytes())
 def test_changed_base_refuses(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();lane='mlx-serve-host';self.copy(lane,root);p=root/'src/exact_generation.zig';data=p.read_bytes();p.write_bytes(b'X'+data[1:])
   with self.assertRaisesRegex(ValueError,'content differs'):checker.check(lane,root,'before')
 def test_symlink_refuses(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();lane='mlx-serve-host';self.copy(lane,root);p=root/'src/exact_generation.zig';target=root/'original.zig';p.rename(target);p.symlink_to(target)
   with self.assertRaisesRegex(ValueError,'Aliased'):checker.check(lane,root,'before')
 def test_already_applied_refuses_before(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();lane='mlx-serve-host';self.copy(lane,root);entry=self.manifest['lanes'][lane]['files']['src/exact_generation.zig'];shutil.copyfile(entry['source_after'],root/'src/exact_generation.zig')
   with self.assertRaises(ValueError):checker.check(lane,root,'before')

if __name__=='__main__':unittest.main()
