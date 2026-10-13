import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

SPEC = importlib.util.spec_from_file_location(
    "collector", Path(__file__).parents[1] / "tools" / "collect_candidates.py"
)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def commit(n, detail=False):
    row = {
        "sha": f"{n:040x}",
        "parents": [{"sha": f"{n + 10000:040x}"}],
        "commit": {
            "message": "Metal performance: 2x faster",
            "author": {},
            "committer": {},
        },
    }
    if detail:
        row["files"] = [
            {
                "filename": "src/fast.metal",
                "status": "modified",
                "patch": "@@ -1 +1 @@\n-a\n+b",
            }
        ]
    return row


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "sources.json"
        self.store = self.root / "store"
        self.write_sources(1)

    def tearDown(self):
        self.temp.cleanup()

    def write_sources(self, n):
        self.source.write_text(
            json.dumps(
                {
                    "repositories": [
                        {
                            "id": f"r{i}",
                            "repository": f"owner/r{i}",
                            "url": f"https://github.com/owner/r{i}",
                        }
                        for i in range(n)
                    ]
                }
            )
        )

    def test_inventory_before_round_robin_details_and_unique_counts(self):
        self.write_sources(12)
        calls = []

        def request(url, timeout):
            calls.append(url)
            path = urlsplit(url).path
            if path.count("/") == 3:
                body = {"default_branch": "main"}
            elif path.endswith("/commits"):
                body = [commit(n) for n in range(1, 7)]
            else:
                body = commit(int(path.rsplit("/", 1)[1], 16), detail=True)
            return 200, {}, json.dumps(body).encode()

        report = C.collect(self.source, self.store, requester=request)
        self.assertEqual(len(calls), 50)
        self.assertTrue(all("/commits/" not in url for url in calls[:24]))
        first_detail_repos = [urlsplit(url).path.split("/")[3] for url in calls[24:36]]
        self.assertEqual(first_detail_repos, [f"r{i}" for i in range(12)])
        self.assertTrue(
            all(2 <= row["details_fetched"] <= 3 for row in report["repositories"])
        )
        self.assertEqual(report["unique_candidate_commits_in_store"], 72)
        self.assertEqual(report["added"]["candidate_records"], 98)
        for receipt in report["requests"]:
            self.assertEqual(
                hashlib.sha256(Path(receipt["raw_path"]).read_bytes()).hexdigest(),
                receipt["raw_sha256"],
            )

    def test_dedup_and_pinned_history_and_gap_anchors(self):
        round_number, calls = 0, []

        def request(url, timeout):
            calls.append(url)
            path, query = urlsplit(url).path, parse_qs(urlsplit(url).query)
            if path.endswith("/r0"):
                return 200, {}, b'{"default_branch":"main"}'
            if "/commits/" in path:
                return 200, {}, C.encode(commit(int(path.rsplit("/", 1)[1], 16), True))
            page = int(query["page"][0])
            values = [commit(100 if round_number else 1)] if page == 1 else [commit(2)]
            return 200, {"Link": '<next>; rel="next"'}, C.encode(values)

        first = C.collect(
            self.source, self.store, max_commits=1, max_details=1, requester=request
        )
        detail = next((self.store / "commits").glob("*/*/detail.json"))
        before = detail.read_bytes()
        round_number = 1
        second = C.collect(
            self.source, self.store, max_commits=1, max_details=0, requester=request
        )
        status = second["repositories"][0]
        self.assertEqual(
            status["history"]["pinned_head"], first["repositories"][0]["latest_head"]
        )
        self.assertEqual(status["history"]["next_page"], 3)
        self.assertEqual(
            status["latest_gaps"],
            [
                {
                    "pinned_new_head": f"{100:040x}",
                    "previous_head": f"{1:040x}",
                    "next_page": 2,
                    "per_page": 1,
                    "status": "backfill_pending",
                }
            ],
        )
        third = C.collect(
            self.source, self.store, max_commits=1, max_details=0, requester=request
        )
        self.assertEqual(third["repositories"][0]["latest_gaps"], status["latest_gaps"])
        self.assertEqual(detail.read_bytes(), before)
        self.assertTrue(
            any("sha=" + f"{1:040x}" in url and "page=2" in url for url in calls)
        )

    def test_rate_failure_explicit_and_unattempted_inventory(self):
        self.write_sources(3)
        report = C.collect(
            self.source,
            self.store,
            requester=lambda *args: (
                403,
                {"X-RateLimit-Remaining": "0"},
                b'{"message":"API rate limit"}',
            ),
        )
        self.assertEqual(report["requests_made"], 1)
        self.assertTrue(report["rate_limit_observed"])
        self.assertEqual(
            [r["inventory_attempted"] for r in report["repositories"]],
            [True, False, False],
        )
        self.assertTrue(
            all(
                r["status"] == "inventory_failed" and r["errors"]
                for r in report["repositories"]
            )
        )

    def test_protection_and_heuristic_scope(self):
        row = C.normalize_commit("ggml-org/llama.cpp", commit(1, True), detail=True)
        self.assertTrue(row["protected_final_heldout"])
        self.assertTrue(row["protected_overlap_check_pending"])
        self.assertFalse(row["training_eligible"])
        self.assertFalse(row["reported_metrics_reproduced"])
        self.assertEqual(
            row["source_reported_metric_quotes"], ["Metal performance: 2x faster"]
        )
        self.assertTrue(C.classify("fix correctness", ["x.metal"])["candidate"])
        self.assertFalse(C.classify("improve unrelated CPU performance")["candidate"])

    def test_exclusive_never_replaces(self):
        path = self.root / "receipt.json"
        C.exclusive(path, b"original")
        with self.assertRaises(FileExistsError):
            C.exclusive(path, b"replacement")
        self.assertEqual(path.read_bytes(), b"original")
        self.assertEqual(list(self.root.glob(".pending-*")), [])

    def test_sources_reject_unsafe_and_duplicate_identities(self):
        for repo in ("../escape", "owner/../escape", "owner/repo?x=1"):
            self.source.write_text(
                json.dumps(
                    {
                        "repositories": [
                            {
                                "id": "safe",
                                "repository": repo,
                                "url": "https://github.com/" + repo,
                            }
                        ]
                    }
                )
            )
            with self.assertRaises(ValueError):
                C.load_sources(self.source)
        self.write_sources(1)
        data = json.loads(self.source.read_text())
        data["repositories"].append(dict(data["repositories"][0], id="different"))
        self.source.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            C.load_sources(self.source)
        self.assertNotEqual(C.repo_key("a--b/c"), C.repo_key("a/b--c"))

    def test_bounds_and_strict_json(self):
        for budget in (0, 51, True):
            with self.assertRaises(ValueError):
                C.collect(self.source, self.store, request_budget=budget)
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError):
                C.strict_json(raw)

    def test_partial_files_and_wrong_sha_recorded(self):
        def request(url, timeout):
            path = urlsplit(url).path
            if path.endswith("/r0"):
                return 200, {}, b'{"default_branch":"main"}'
            if path.endswith("/commits"):
                return 200, {}, C.encode([commit(1)])
            return 200, {}, C.encode(commit(2, True))

        report = C.collect(self.source, self.store, requester=request)
        self.assertEqual(report["added"]["details"], 0)
        self.assertIn("Detail SHA differs", str(report["repositories"][0]["errors"]))
        row = commit(1, True)
        row["files"][0].pop("patch")
        parsed = C.normalize_commit("owner/r0", row, detail=True)
        self.assertEqual(
            parsed["missing_or_unverified"]["missing_patch_paths"], ["src/fast.metal"]
        )
        self.assertTrue(parsed["missing_or_unverified"]["patches_may_be_truncated"])


if __name__ == "__main__":
    unittest.main()
