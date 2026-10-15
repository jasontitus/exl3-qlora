"""Local evidence-dependent replay tests; not a standalone copied-tree test suite."""
import importlib.util,json,shutil,subprocess,tempfile,unittest
from pathlib import Path
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('checker',H/'check_sources.py');c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
class Bundle(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name).resolve();self.m=json.loads((H/'manifest.json').read_text())
  for rel,v in self.m['files'].items():
   if v['before_sha256'] is not None:
    p=self.root/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(v['source_before'],p)
 def tearDown(self):self.tmp.cleanup()
 def test_full_patch_replay(self):
  c.check(self.root,'before')
  subprocess.run(['git','apply','--check',str(H/self.m['patch'])],cwd=self.root,check=True,capture_output=True)
  subprocess.run(['git','apply',str(H/self.m['patch'])],cwd=self.root,check=True,capture_output=True)
  c.check(self.root,'after')
  for rel,v in self.m['files'].items():self.assertEqual((self.root/rel).read_bytes(),Path(v['source_after']).read_bytes())
 def test_changed_base_refuses(self):
  p=self.root/'src/arch/deepseek_v41/graph.zig';p.write_bytes(p.read_bytes()+b'\n')
  with self.assertRaises(ValueError):c.check(self.root,'before')
 def test_new_file_collision_refuses(self):
  p=self.root/'src/arch/nax_capability.zig';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'')
  with self.assertRaises(ValueError):c.check(self.root,'before')
 def test_parent_alias_refuses(self):
  p=self.root/'src/arch/deepseek_v41';q=self.root/'moved';p.rename(q);p.symlink_to(q,target_is_directory=True)
  with self.assertRaises(ValueError):c.check(self.root,'before')
 def test_fifo_refuses_without_blocking(self):
  import os
  p=self.root/'fifo';os.mkfifo(p)
  with self.assertRaises(ValueError):c.capture(p)
if __name__=='__main__':unittest.main()
