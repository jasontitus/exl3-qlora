import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import qwen_conv_reference as c

class ConvReferenceTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(702);self.x=self.rng.normal(size=(2,3,7));self.w=self.rng.normal(size=(3,4));self.h=self.rng.normal(size=(2,3,3));self.b=self.rng.normal(size=3)
    def test_pinned_semantic_source(self):self.assertEqual(c.provenance()['sha256'],c.SOURCE_SHA256)
    def test_independent_scalar_convolution(self):
        y,h=c.forward(self.x,self.w,self.h,self.b);joined=np.concatenate([self.h,self.x],axis=2);expected=np.empty_like(y)
        for batch in range(2):
            for ch in range(3):
                for t in range(7):
                    z=self.b[ch]+sum(joined[batch,ch,t+j]*self.w[ch,j] for j in range(4));expected[batch,ch,t]=z/(1+np.exp(-z))
        np.testing.assert_allclose(y,expected,atol=2e-14,rtol=2e-14);np.testing.assert_array_equal(h,self.x[:,:,-3:])
    def test_full_vs_chunks_and_cross_boundary_adjoint(self):
        y,end=c.forward(self.x,self.w,self.h,self.b);dy=self.rng.normal(size=y.shape);de=self.rng.normal(size=end.shape)
        full=c.vjp(self.x,self.w,self.h,dy,de,self.b)
        for cut in (1,2,3,6):
            ya,ha=c.forward(self.x[:,:,:cut],self.w,self.h,self.b);yb,hb=c.forward(self.x[:,:,cut:],self.w,ha,self.b)
            np.testing.assert_allclose(np.concatenate([ya,yb],axis=2),y,atol=1e-14);np.testing.assert_array_equal(hb,end)
            gb=c.vjp(self.x[:,:,cut:],self.w,ha,dy[:,:,cut:],de,self.b);ga=c.vjp(self.x[:,:,:cut],self.w,self.h,dy[:,:,:cut],gb['history'],self.b)
            np.testing.assert_allclose(np.concatenate([ga['x'],gb['x']],axis=2),full['x'],atol=1e-13)
            np.testing.assert_allclose(ga['history'],full['history'],atol=1e-13)
            for k in ('weight','bias'):np.testing.assert_allclose(ga[k]+gb[k],full[k],atol=1e-13)
    def test_all_inputs_directional_difference(self):
        dy=self.rng.normal(size=self.x.shape);de=self.rng.normal(size=self.h.shape);g=c.vjp(self.x,self.w,self.h,dy,de,self.b);args=[self.x,self.w,self.h,self.b]
        for index,key in enumerate(('x','weight','history','bias')):
            d=self.rng.normal(size=args[index].shape);eps=1e-6;vals=[]
            for sign in (1,-1):
                a=[v.copy() for v in args];a[index]+=sign*eps*d;y,h=c.forward(a[0],a[1],a[2],a[3]);vals.append(np.sum(y*dy)+np.sum(h*de))
            self.assertAlmostEqual((vals[0]-vals[1])/(2*eps),np.sum(g[key]*d),delta=2e-8)
    def test_future_independence_and_no_mutation(self):
        saved=self.h.copy();y,_=c.forward(self.x,self.w,self.h);x=self.x.copy();x[:,:,4:]+=100;z,_=c.forward(x,self.w,self.h)
        np.testing.assert_array_equal(z[:,:,:4],y[:,:,:4]);np.testing.assert_array_equal(saved,self.h)
    def test_kernel_one_and_no_bias(self):
        w=self.w[:,:1].copy();h=self.h[:,:,:0].copy();y,end=c.forward(self.x,w,h);g=c.vjp(self.x,w,h,np.ones_like(y))
        self.assertEqual(end.shape,(2,3,0));self.assertIsNone(g['bias']);self.assertEqual(g['history'].shape,h.shape)
    def test_reject_bad_inputs(self):
        for x,w,h,b in [(self.x.astype('float32'),self.w,self.h,self.b),(self.x,self.w,self.h[:,:,:2],self.b),(self.x,self.w,self.h,np.ones(4)),(np.zeros((2,3,33)),self.w,self.h,self.b)]:
            with self.assertRaises(ValueError):c.forward(x,w,h,b)
        with self.assertRaises(ValueError):c.vjp(self.x,self.w,self.h,np.zeros((1,3,7)))
        bad=self.x.copy();bad[0,0,0]=np.nan
        with self.assertRaises(ValueError):c.forward(bad,self.w,self.h)
if __name__=='__main__':unittest.main()
