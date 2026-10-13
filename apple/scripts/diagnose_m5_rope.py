"""Diagnose existing float32 RoPE against independent component references."""
import json
import numpy as np
import mlx.core as mx
from exl3_mlx.attention import rope
c={'qk_rope_head_dim':8,'rope_theta':10000,'compress_rope_theta':160000,'rope_scaling':{'original_max_position_embeddings':65536,'factor':16,'beta_fast':32,'beta_slow':1}}
x=np.random.default_rng(2).normal(size=(2,3,4,16)).astype(np.float32);pos=np.array([0,7,600],np.float32)
f=1/(10000**(mx.arange(0,8,2,dtype=mx.float32)/8));a=mx.array(pos)[:,None]*f[None,:];fn=np.array(f);an=np.array(a)
y=np.array(rope(mx.array(x),mx.array(pos),c));ref=x.copy();tail=x[...,8:];aa=tail[...,::2];bb=tail[...,1::2];co=np.cos(an.astype(np.float64))[None,:,None,:];si=np.sin(an.astype(np.float64))[None,:,None,:];ref[...,8::2]=aa*co-bb*si;ref[...,9::2]=aa*si+bb*co
inverse={}
for compressed in [False,True]:
 back=np.array(rope(rope(mx.array(x),mx.array(pos),c,compressed=compressed),mx.array(pos),c,compressed=compressed,inverse=True));inverse[str(compressed)]=float(np.max(np.abs(back-x)))
print(json.dumps({'frequencies_gpu':fn.tolist(),'frequencies_exact':(1/(10000**(np.arange(0,8,2)/8))).tolist(),'angles_gpu':an.tolist(),'same_angles_float64_trig_max_abs':float(np.max(np.abs(y-ref))),'inverse_max_abs':inverse,'qualification':'Component diagnosis, not full-model or long-context acceptance.'},indent=2))
