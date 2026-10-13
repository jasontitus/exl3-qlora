"""Tiny composed CPU oracle: both raw convolution and recurrent state adjoints."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import qwen_conv_reference as conv
import qwen_gdn_reference as gdn


def forward(x, w, h, state, decay, beta):
    packed, end = conv.forward(x, w, h)
    qkv = gdn.split_postconv_qkv(packed.transpose(0, 2, 1), num_k_heads=1,
                               num_v_heads=2, key_dim=2, value_dim=2)
    y, state_end = gdn.recurrent_forward(*qkv, decay, beta,
                                        initial_state=state, precision='float64')
    return y, end, state_end, qkv


def backward(x, w, h, state, decay, beta, dy, dh, ds):
    _, _, _, qkv = forward(x, w, h, state, decay, beta)
    r = gdn.recurrent_input_adjoint(*qkv, decay, beta, dy,
                                   initial_state=state, d_final_state=ds)
    packed = np.concatenate([r[k].reshape(x.shape[0], x.shape[2], -1)
                             for k in ('q', 'k', 'v')], axis=-1).transpose(0, 2, 1)
    c = conv.vjp(x, w, h, packed, dh)
    return c, r


class CompositionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(918)
        self.x = rng.normal(size=(1, 8, 7)) * .3
        self.w = rng.normal(size=(8, 4)) * .3
        self.h = rng.normal(size=(1, 8, 3)) * .3
        self.state = rng.normal(size=(1, 2, 2, 2)) * .3
        self.decay = np.full((1, 7, 2), -.1)
        self.beta = np.full((1, 7, 2), .4)
        self.dy = np.zeros((1, 7, 2, 2))
        self.dy[:, -1] = rng.normal(size=(1, 2, 2))

    def test_full_vs_chunked_late_loss_and_both_state_adjoints(self):
        x,w,h,s,g,b = self.x,self.w,self.h,self.state,self.decay,self.beta
        y,he,se,_ = forward(x,w,h,s,g,b)
        full,rf = backward(x,w,h,s,g,b,self.dy,np.zeros_like(he),np.zeros_like(se))
        for cut in (1,2,4,6):
            ya,ha,sa,_ = forward(x[:,:,:cut],w,h,s,g[:,:cut],b[:,:cut])
            yb,hb,sb,_ = forward(x[:,:,cut:],w,ha,sa,g[:,cut:],b[:,cut:])
            np.testing.assert_allclose(np.concatenate([ya,yb],axis=1),y,atol=1e-14)
            cb,rb = backward(x[:,:,cut:],w,ha,sa,g[:,cut:],b[:,cut:],self.dy[:,cut:],np.zeros_like(hb),np.zeros_like(sb))
            ca,ra = backward(x[:,:,:cut],w,h,s,g[:,:cut],b[:,:cut],self.dy[:,:cut],cb['history'],rb['initial_state'])
            np.testing.assert_allclose(np.concatenate([ca['x'],cb['x']],axis=2),full['x'],atol=1e-12)
            np.testing.assert_allclose(ca['weight']+cb['weight'],full['weight'],atol=1e-12)
            np.testing.assert_allclose(ca['history'],full['history'],atol=1e-12)
            np.testing.assert_allclose(ra['initial_state'],rf['initial_state'],atol=1e-12)
        self.assertGreater(np.linalg.norm(full['x'][:,:,:1]),1e-8)

    def test_composed_directional_derivatives(self):
        args=[self.x,self.w,self.h,self.state,self.decay,self.beta]
        c,r=backward(*args,self.dy,np.zeros_like(self.h),np.zeros_like(self.state))
        grads=[c['x'],c['weight'],c['history'],r['initial_state'],r['g'],r['beta']]
        rng=np.random.default_rng(42)
        for i,grad in enumerate(grads):
            direction=rng.normal(size=args[i].shape); losses=[]
            for sign in (1,-1):
                trial=[a.copy() for a in args];trial[i]+=sign*1e-6*direction
                losses.append(float(np.sum(forward(*trial)[0]*self.dy)))
            self.assertAlmostEqual((losses[0]-losses[1])/2e-6,float(np.sum(grad*direction)),delta=2e-8)

if __name__ == '__main__':
    unittest.main()
