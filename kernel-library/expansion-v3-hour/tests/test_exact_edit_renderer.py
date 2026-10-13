import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import exact_edit_renderer as m

class NewlineTests(unittest.TestCase):
    def check(self,before,after):
        b={'kernel.metal':before};a={'kernel.metal':after}
        patch=m.unified_patch(b,a);command=m.render_command(patch)
        self.assertEqual(m.parse_command(command),patch)
        result=m.replay(command,b,a)
        self.assertTrue(result['exact_scoped_replay'])
        return patch
    def test_both_missing(self):self.assertEqual(self.check(b'a\nb',b'a\nc').count('No newline'),2)
    def test_old_missing(self):self.assertEqual(self.check(b'a\nb',b'a\nc\n').count('No newline'),1)
    def test_new_missing(self):self.assertEqual(self.check(b'a\nb\n',b'a\nc').count('No newline'),1)
    def test_newline_only_change(self):self.check(b'a\nb',b'a\nb\n');self.check(b'a\nb\n',b'a\nb')
    def test_context_last_line_missing(self):self.check(b'a\nb\nc',b'A\nb\nc')
    def test_new_file_missing(self):self.check(None,b'kernel code')
    def test_crlf_preserved(self):self.check(b'a\r\nb\r\n',b'a\r\nc\r\n')
    def test_empty_after(self):self.check(b'abc',b'')
    def test_unicode(self):self.check('a\nπ'.encode(),'b\nπ'.encode())
    def test_literal_marker_line(self):self.check(b'a\n\\ No newline at end of file\n',b'b\n\\ No newline at end of file\n')
    def test_no_source_rewrite(self):
        r={'files':{'x.metal':{'before':{'content':'old','mode':'100644'},'after':{'content':'new','mode':'100644'}}}}
        b,a=m.full_commit_images(r);self.assertEqual(b['x.metal'],b'old');self.assertEqual(a['x.metal'],b'new')
    def test_modes_cannot_silently_disappear(self):
        for before_mode,after_mode in [('100644','100755'),('100755','100644'),('120000','120000')]:
            r={'files':{'x.py':{'before':{'content':'same','mode':before_mode},'after':{'content':'same','mode':after_mode}}}}
            with self.assertRaisesRegex(ValueError,'modes'):m.full_commit_images(r)
        r={'files':{'x.py':{'before':None,'after':{'content':'new','mode':'100755'}}}}
        with self.assertRaisesRegex(ValueError,'modes'):m.full_commit_images(r)
    def test_unsafe_paths(self):
        with self.assertRaises(ValueError):m.unified_patch({'../x':b'a'},{'../x':b'b'})

if __name__=='__main__':unittest.main()
