"""Independent receipt consistency regressions. All subprocess/compiler/GPU work mocked."""
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

R=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('independent_expression_author_fixture',R/'diagnostic-eval/tests/test_expression_launch.py')
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)


class ReceiptBoundaries(unittest.TestCase):
    def setUp(self):
        self.fixture=t.Tests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)

    def test_negative_control_error_is_not_intended_numerical_failure(self):
        f=self.fixture;request=f.controls();item=request['controls']['negative'];r=t.d.read(item['receipt']);r['error']='failure after GPU dispatch'
        Path(item['receipt']).write_text(json.dumps(r));item['sha256']=t.d.digest(item['receipt'])
        with self.assertRaises(ValueError):t.m.validate_controls(t.d,request,f.plan,{})

    def test_control_incomplete_numerical_status_rejected(self):
        f=self.fixture;request=f.controls();item=request['controls']['negative'];r=t.d.read(item['receipt']);r['numerical']='incomplete'
        Path(item['receipt']).write_text(json.dumps(r));item['sha256']=t.d.digest(item['receipt'])
        with self.assertRaises(ValueError):t.m.validate_controls(t.d,request,f.plan,{})

    def context(self, admission):
        f=self.fixture
        with patch.object(t.b,'load',return_value=t.m),patch.object(t.m,'load_launcher',return_value=t.d),patch.object(t.m,'guard'),patch.object(t.m,'tokenizer',return_value=f.tok),patch.object(t.d,'load_module',return_value=t.builder):
            return t.b.context(admission,t.d.digest(admission))

    def test_observed_process_contract_rejected_on_each_drift(self):
        f=self.fixture;result=f.run_mock();admission=Path(result['collections']['base']['admission']['path']);launch_path=f.out/'launch.json';original=t.d.read(launch_path)
        # First require the author's realistic fixture succeeds unchanged.
        self.context(admission)
        variants=[('argv',['wrong-program','--wrong-mode']),('cwd','/unrelated'),('environment',{'MLX_ENABLE_TF32':'1'})]
        for key,value in variants:
            for process in ['base-'+t.m.PROBES[0],'base-collect']:
                changed=json.loads(json.dumps(original));changed['processes'][process][key]=value;launch_path.write_text(json.dumps(changed))
                with self.subTest(field=key,process=process),self.assertRaises(ValueError):self.context(admission)
        launch_path.write_text(json.dumps(original))

    def test_exact_child_keys_required(self):
        f=self.fixture;result=f.run_mock();admission=Path(result['collections']['base']['admission']['path']);launch_path=f.out/'launch.json';launch=t.d.read(launch_path)
        launch['processes']['unrelated-collect']=launch['processes'].pop('base-collect');launch_path.write_text(json.dumps(launch))
        with self.assertRaises(ValueError):self.context(admission)
