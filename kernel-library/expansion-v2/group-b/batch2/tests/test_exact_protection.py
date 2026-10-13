import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / 'exact_protection.py'
spec = importlib.util.spec_from_file_location('gate', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
A, B, C, D = ('a'*40, 'b'*40, 'c'*40, 'd'*40)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.index = m.reserved_index({'review': {'coverage': [{'id': 'mlx:1', 'head': A, 'base': B}]}})

    def test_manual_clearance_cannot_override_reserved_head(self):
        calls = []
        c = {'commit': A, 'parent': C, 'historical_protection_review': {'status': 'cleared'}}
        got = m.review_after_identity_gate(c, self.index, lambda c: calls.append(c))
        self.assertEqual(got['status'], 'protected_exact_identity')
        self.assertFalse(calls)

    def test_parent_and_related_match_reserved_base(self):
        for c in ({'commit': C, 'parent': B}, {'commit': C, 'parent': D, 'related_commits': [B]}):
            self.assertTrue(m.identity_matches(c, self.index))

    def test_nonmatch_reaches_manual_review_once(self):
        calls = []
        got = m.review_after_identity_gate({'commit': C, 'parent': D}, self.index, lambda c: calls.append(c) or 'pending')
        self.assertEqual(got, 'pending')
        self.assertEqual(len(calls), 1)

    def test_missing_or_malformed_reserved_metadata_fails_closed(self):
        for d in ({}, {'review': {'coverage': []}}, {'review': {'coverage': [{'id': 'x','head': A[:8], 'base': B}]}}):
            with self.assertRaises(ValueError):m.reserved_index(d)

    def test_non_git_task_rows_are_explicitly_not_identity_evidence(self):
        value = {'review': {'coverage': [{'id':'mlx:1','head':A,'base':B}, {'id':'protected-task:hc','units':15}, {'id':'hc-original-winner','units':8}]}}
        self.assertEqual(set(m.reserved_index(value)), {A,B})
        with self.assertRaises(ValueError):
            m.reserved_index({'review':{'coverage':[{'id':'mlx:1','units':1}]}})

    def test_malformed_candidate_identity_fails_closed(self):
        for c in ({'commit': C[:8], 'parent': D}, {'commit': C,'parent': D,'related_commits': [None]}):
            with self.assertRaises(ValueError):m.identity_matches(c,self.index)


if __name__ == '__main__':unittest.main()
