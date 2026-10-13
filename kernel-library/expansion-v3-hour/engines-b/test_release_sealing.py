"""Isolated sealing negative tests; never modifies live release assets."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1]/'tools/seal_release.py'


class SealReleaseTests(unittest.TestCase):
    def run_rejection(self, review, match, optimized=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inp = root/'changed.json'; inp.write_text('changed')
            if review.get('inputs_sha256') is None:
                review['inputs_sha256'] = {str(inp): '0'*64}
            (root/'final-aggregate-independent-review.json').write_text(json.dumps(review))
            code = 'import importlib.util,pathlib; s=importlib.util.spec_from_file_location("seal",'+repr(str(SCRIPT))+');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.ROOT=pathlib.Path('+repr(str(root))+');m.main()'
            args = [sys.executable]+(['-O'] if optimized else [])+['-c', code]
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(match,result.stderr)
            self.assertFalse((root/'dataset-card.json').exists())
            self.assertFalse((root/'release.json').exists())

    def test_stale_input_rejected(self):
        self.run_rejection({'status':'passed_scoped_preparation_audit','blockers':[]},'Stale independent review')

    def test_stale_input_rejected_under_python_O(self):
        self.run_rejection({'status':'passed_scoped_preparation_audit','blockers':[]},'Stale independent review',True)

    def test_failed_status_under_python_O(self):
        self.run_rejection({'status':'failed','blockers':[]},'Independent review did not pass',True)

    def test_blocker_under_python_O(self):
        self.run_rejection({'status':'passed_scoped_preparation_audit','blockers':['unresolved']},'Independent review has blockers',True)

    def test_wrong_reasoning_count_under_python_O(self):
        self.run_rejection({'status':'passed_scoped_preparation_audit','blockers':[],'inputs_sha256':{},'reasoning':{'records':68,'supervised_tokens':9023}},'Reasoning count mismatch',True)

    def test_wrong_preference_count_under_python_O(self):
        self.run_rejection({'status':'passed_scoped_preparation_audit','blockers':[],'inputs_sha256':{},'reasoning':{'records':69,'supervised_tokens':9023},'preferences':{'pairs':24}},'Preference count mismatch',True)

if __name__=='__main__':unittest.main()
