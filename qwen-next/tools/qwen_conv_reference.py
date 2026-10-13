"""Tiny FP64 causal depthwise conv+SiLU/history oracle; no MLX or GPU.

Packed q|k|v channels use [batch, channels, time], weights [channels, K].
History is the MINIMAL K-1 preceding raw inputs, oldest first. The upstream
inference state stores K columns; discard its oldest column before passing it
here. No history detachment: VJP includes the returned-history cotangent.
This mathematical oracle does not implement dtype casts, full model, serving
state ABI, document packing, or production long-context convolution.
"""
import hashlib
from pathlib import Path
import numpy as np

SOURCE = Path('/Users/jasontitus/experiments/exl3-qlora/exllamav3/training/gdn.py')
SOURCE_SHA256 = 'e1de1f6112715a7868388972872a5f37b0d8198d98246527650ce791d9b33a57'

INFERENCE_SOURCE = Path('/Users/jasontitus/experiments/exl3-qlora/exllamav3/modules/gated_delta_net_fn/conv1d.py')
INFERENCE_SHA256 = 'c750304f06fa7b970556202337792a1d698dda060b2985071692a364c72c219b'

def provenance():
    actual=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if actual!=SOURCE_SHA256:raise ValueError('Semantic source changed')
    if hashlib.sha256(INFERENCE_SOURCE.read_bytes()).hexdigest()!=INFERENCE_SHA256:raise ValueError('Inference semantic source changed')
    return {'inference_source':str(INFERENCE_SOURCE),'inference_sha256':INFERENCE_SHA256,'source':str(SOURCE),'sha256':actual,'scope':'FP64 bounded CPU oracle; no serving/GPU qualification'}

def _check(x,w,h,b):
    for name,a in [('x',x),('weight',w),('history',h)]+([] if b is None else [('bias',b)]):
        if not isinstance(a,np.ndarray) or a.dtype!=np.float64 or not np.isfinite(a).all():raise ValueError(name+' must be finite FP64 array')
    if x.ndim!=3 or w.ndim!=2:raise ValueError('Expected [B,C,T] and [C,K]')
    B,C,T=x.shape;K=w.shape[1]
    if not (1<=B<=2 and 1<=C<=64 and 1<=T<=32 and 1<=K<=8):raise ValueError('Outside tiny fixture bounds')
    if w.shape[0]!=C or h.shape!=(B,C,K-1) or (b is not None and b.shape!=(C,)):raise ValueError('Incompatible dimensions')
    return B,C,T,K

def _sigmoid(z):
    out=np.empty_like(z);pos=z>=0
    out[pos]=1/(1+np.exp(-z[pos]));e=np.exp(z[~pos]);out[~pos]=e/(1+e)
    return out

def forward(x,weight,history,bias=None):
    """Return activated convolution and minimal raw-input history; no mutation."""
    B,C,T,K=_check(x,weight,history,bias)
    joined=np.concatenate((history,x),axis=-1);z=np.zeros_like(x)
    with np.errstate(over='ignore',invalid='ignore'):
        for j in range(K):z+=joined[:,:,j:j+T]*weight[None,:,j,None]
        if bias is not None:z+=bias[None,:,None]
    if not np.isfinite(z).all():raise ValueError('Convolution overflow')
    y=z*_sigmoid(z)
    end=joined[:,:,-(K-1):].copy() if K>1 else joined[:,:,:0].copy()
    return y,end

def vjp(x,weight,history,dy,dend=None,bias=None):
    """Continuous FP64 gradients including history handoff; tiny oracle only."""
    B,C,T,K=_check(x,weight,history,bias)
    if not isinstance(dy,np.ndarray) or dy.dtype!=np.float64 or dy.shape!=x.shape or not np.isfinite(dy).all():raise ValueError('Invalid output cotangent')
    if dend is None:dend=np.zeros_like(history)
    if not isinstance(dend,np.ndarray) or dend.dtype!=np.float64 or dend.shape!=history.shape or not np.isfinite(dend).all():raise ValueError('Invalid history cotangent')
    forward(x,weight,history,bias)  # same validation and finite-arithmetic contract
    joined=np.concatenate((history,x),axis=-1);z=np.zeros_like(x)
    for j in range(K):z+=joined[:,:,j:j+T]*weight[None,:,j,None]
    if bias is not None:z+=bias[None,:,None]
    s=_sigmoid(z);dz=dy*(s+z*s*(1-s));dj=np.zeros_like(joined);dw=np.zeros_like(weight)
    for j in range(K):
        dj[:,:,j:j+T]+=dz*weight[None,:,j,None]
        dw[:,j]=np.sum(dz*joined[:,:,j:j+T],axis=(0,2))
    if K>1:dj[:,:,-(K-1):]+=dend
    result={'x':dj[:,:,K-1:].copy(),'history':dj[:,:,:K-1].copy(),'weight':dw,'bias':dz.sum(axis=(0,2)) if bias is not None else None}
    if any(v is not None and not np.isfinite(v).all() for v in result.values()):raise ValueError('Adjoint overflow')
    return result
