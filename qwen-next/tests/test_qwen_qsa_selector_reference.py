import dataclasses
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools'))
import qwen_qsa_selector_reference as s


def fixture(length=19, start=0, batch=1, heads=2, dim=3):
    rng = np.random.default_rng(19)
    return (rng.normal(size=(batch, length-start, heads, dim)).astype('f4'),
            rng.normal(size=(batch, length//4, dim)).astype('f4'),
            np.tile(np.arange(start, length, dtype='i8'), (batch, 1)),
            np.full(batch, length, dtype='i8'))


class Selector(unittest.TestCase):
    def test_scalar_oracle_and_causality(self):
        for length in range(1, 26):
            args = fixture(length)
            q, k, pos, _ = args
            out = s.select(*args, budget=8, query_tile=3).array()
            for row, p in enumerate(pos[0]):
                scores = []
                for block in range((p+1)//4):
                    score = sum(max(float(np.dot(q[0,row,head], k[0,block])), 0) for head in range(2)) / np.sqrt(3)
                    scores.append((score, block))
                chosen = [block for _, block in sorted(scores, key=lambda x: (-x[0], x[1]))[:2]]
                expected = [4*c+j for c in chosen for j in range(4)] + list(range(((p+1)//4)*4, p+1))
                actual = out[0,row][out[0,row]>=0].tolist()
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual), len(set(actual)))
                self.assertTrue(all(t <= p for t in actual))

    def test_tiles_and_replay(self):
        args = fixture(47, batch=2)
        first = s.select(*args, budget=12, query_tile=1)
        for tile in (2, 7, 16, 64):
            other = s.select(*args, budget=12, query_tile=tile)
            np.testing.assert_array_equal(first.array(), other.array())
            np.testing.assert_array_equal(first.array(), s.replay(first, *args, budget=12, query_tile=tile))
            self.assertLessEqual(other.peak_score_elements, min(tile,47)*2*(47//4))
        with self.assertRaises(ValueError):
            first.array()[0,0,0] = 9

    def test_full_budget_threshold(self):
        # Last rows at visible lengths2051 and2052: full coverage then binding selection.
        args = fixture(2052, start=2048, heads=1, dim=1)
        args[0].fill(1)
        args[1][:] = np.arange(513)[:,None]
        routes = s.select(*args).array()[0]
        self.assertEqual((routes[2]>=0).sum(), 2051)
        self.assertEqual(set(routes[2][routes[2]>=0]), set(range(2051)))
        self.assertEqual((routes[3]>=0).sum(), 2048)
        self.assertEqual(set(routes[3][routes[3]>=0]), set(range(4,2052)))
        self.assertEqual(routes.shape[1], 2080)

    def test_compact_layout_not_torch_slot_layout(self):
        a=fixture(19)
        out=s.select(*a,budget=8).array()
        self.assertEqual(out[0,0,0],0)
        self.assertTrue((out[0,0,1:]==-1).all())
        # Torch reference uses ksel=2, so its first tail slot would be8.
        self.assertEqual(out[0,0,8],-1)

    def test_tie_order(self):
        args = fixture(16, start=15)
        args[0].fill(0)
        self.assertEqual(s.select(*args,budget=4).array()[0,0,:4].tolist(), [0,1,2,3])

    def test_padding_and_independent_batches(self):
        args = list(fixture(13,batch=2))
        args[3][1]=7
        args[2][1,7:]=-1
        args[1][1,1:]=1e10  # unavailable completed blocks must never win
        result=s.select(*args,budget=4).array()
        self.assertTrue((result[1,7:]==-1).all())
        self.assertTrue((result[1,:7] < 7).all())
        self.assertEqual(set(result[1,6][result[1,6]>=0]),set(range(7)))

    def test_replay_drift(self):
        args=fixture()
        capture=s.select(*args,budget=8)
        bad=list(args);bad[0]=args[0].copy();bad[0][0,0,0,0]+=1
        with self.assertRaises(ValueError):s.replay(capture,*bad,budget=8)
        with self.assertRaises(ValueError):s.replay(dataclasses.replace(capture,payload=b'x'+capture.payload[1:]),*args,budget=8)
        with self.assertRaises(ValueError):s.replay(capture,*args,budget=4)

    def test_rejections(self):
        for value in (True,0,3,5,2052,8.0):
            with self.subTest(budget=value),self.assertRaises(ValueError):s.select(*fixture(),budget=value)
        for value in (False,0,65):
            with self.subTest(tile=value),self.assertRaises(ValueError):s.select(*fixture(),query_tile=value)
        for variant in ('nan','dtype','positions','padding','pool','length','boolpos'):
            a=list(fixture())
            if variant=='nan':a[0][0,0,0,0]=np.nan
            if variant=='dtype':a[0]=a[0].astype('f8')
            if variant=='positions':a[2][0,0]=-2
            if variant=='padding':a[2][0,1]=-1
            if variant=='pool':a[1]=a[1][:,:-1]
            if variant=='length':a[3][0]=0
            if variant=='boolpos':a[2]=a[2].astype(bool)
            with self.subTest(variant=variant),self.assertRaises(ValueError):s.select(*a)

    def test_resource_refusal_before_scores(self):
        a=(np.zeros((2,4096,1,1),'f4'),np.zeros((2,1024,1),'f4'),np.tile(np.arange(4096,dtype='i8'),(2,1)),np.array([4096,4096],dtype='i8'))
        with self.assertRaisesRegex(ValueError,'output byte'):s.select(*a)
        a=(np.zeros((1,64,4,128),'f4'),np.zeros((1,1024,128),'f4'),np.arange(4032,4096,dtype='i8')[None],np.array([4096],dtype='i8'))
        # Exact1MiB dot tile permitted; smaller tile must produce same routes.
        np.testing.assert_array_equal(s.select(*a,query_tile=64).array(),s.select(*a,query_tile=1).array())

if __name__=='__main__':unittest.main()
