import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "inventory", Path(__file__).parents[1] / "inventory.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PartialMirrorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "mirror.git"
        subprocess.run(
            ["git", "init", "--bare", str(self.repo)], check=True, capture_output=True
        )
        self.tree = self.git("mktree", data="").strip()

    def git(self, *args, data=None):
        return subprocess.check_output(
            ["git", "--git-dir", str(self.repo), *args], input=data, text=True
        )

    def commit(self, message, parent=None):
        raw = f"tree {self.tree}\n"
        if parent:
            raw += f"parent {parent}\n"
        raw += (
            "author Test <test@example.com> 1700000000 -0730\ncommitter Test <test@example.com> 1700000000 -0730\n\n"
            + message
        )
        return self.git(
            "hash-object", "-t", "commit", "-w", "--stdin", data=raw
        ).strip()

    def test_missing_parent_root_survives_without_fetch(self):
        missing = "1" * 40
        root = self.commit("Metal 4 precise title\n\nFull message body.\n", missing)
        rows, result = module.scan_records(self.repo, [root])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][:2], (root, [missing]))
        self.assertEqual(rows[0][2], "2023-11-14T14:43:20-07:30")
        self.assertEqual(rows[0][3], "Metal 4 precise title\n\nFull message body.")

    def test_partial_and_direct_roots_are_unique(self):
        normal = self.commit("NAX normal\n")
        self.git("update-ref", "refs/heads/main", normal)
        partial = self.commit("M5 partial\n", "2" * 40)
        rows, result = module.scan_records(self.repo, [normal, partial, normal])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual({r[0] for r in rows}, {normal, partial})
        self.assertEqual(len(rows), 2)

    def test_complete_history_no_fallback(self):
        parent = self.commit("Old body\n")
        head = self.commit("Metal 4 head\n", parent)
        rows, result = module.scan_records(self.repo, [head])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][:2], (head, [parent]))

    def test_raw_metadata_keeps_parent_at_shallow_boundary(self):
        parent = self.commit("Original parent\n")
        head = self.commit("Metal 4 shallow root\n", parent)
        (self.repo / "shallow").write_text(head + "\n")
        self.assertEqual(module.scan_records(self.repo, [head])[0][0][1], [])
        self.assertEqual(module.root_metadata(self.repo, head)[1], [parent])

    def test_noncommit_objects_not_root_candidates(self):
        head = self.commit("NAX head\n")
        blob = self.git("hash-object", "-w", "--stdin", data="Metal 4 blob").strip()
        checked = self.git(
            "cat-file",
            "--batch-check=%(objectname) %(objecttype)",
            data=f"{head}\n{blob}\n{self.tree}\n",
        )
        roots = [
            line.split()[0] for line in checked.splitlines() if line.endswith(" commit")
        ]
        self.assertEqual(roots, [head])


if __name__ == "__main__":
    unittest.main()
