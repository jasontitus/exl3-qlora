import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('panel_import',HERE/'import_evidence.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ImportTests(unittest.TestCase):
    def request(self):
        return {'payload_manifest':'/model/manifest.json','inputs_sha256':{
            '/model/manifest.json':'1'*64,'/model/tokenizer.json':'2'*64,
            '/model/tokenizer_config.json':'3'*64,'/runtime/libmlx.dylib':'4'*64}}
    def test_equal_and_distinct_resources(self):
        rows={k:self.request() for k in ('basezeroB','old458','final188')}
        rows['final188']['execution']={'native_box_ceiling_gib':104}
        self.assertEqual(m.check_shared_identities(rows),m.shared_identity(rows['old458']))
    def test_each_identity_mismatch(self):
        for key in self.request()['inputs_sha256']:
            rows={k:self.request() for k in ('basezeroB','old458','final188')}
            rows['final188']['inputs_sha256'][key]='f'*64
            with self.assertRaises(ValueError):m.check_shared_identities(rows)
    def test_missing_or_ambiguous(self):
        r=self.request();del r['inputs_sha256']['/model/tokenizer.json']
        with self.assertRaises(ValueError):m.shared_identity(r)
        r=self.request();r['inputs_sha256']['/foreign/libmlx.dylib']='5'*64
        with self.assertRaises(ValueError):m.shared_identity(r)
    def test_actual_composition_rejects_late_map_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve(); mapfile=p/'map.json'
            mapfile.write_bytes((HERE/'common-replay/replays.json').read_bytes())
            original=m._read; changed=False
            def read(path,limit):
                nonlocal changed
                if path.name=='design-v2.json' and not changed:
                    changed=True;mapfile.write_bytes(mapfile.read_bytes()+b' ')
                return original(path,limit)
            with patch.object(m,'_read',read):
                with self.assertRaisesRegex(ValueError,'Late import source drift'):
                    m.run(HERE.parents[3],mapfile,p/'out')
            self.assertTrue(changed)
            self.assertFalse((p/'out/suite.json').exists())

    def test_replay_map_uses_bounded_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve();mapfile=p/'map.json';mapfile.write_text('{}')
            original=m._read
            def read(path,limit):
                if path==mapfile:raise ValueError('bounded map read reached')
                return original(path,limit)
            with patch.object(m,'_read',read):
                with self.assertRaisesRegex(ValueError,'bounded map read reached'):
                    m.run(p,mapfile,p/'out')

if __name__=='__main__':unittest.main()
