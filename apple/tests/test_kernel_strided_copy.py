import struct
import pytest
from exl3_mlx.kernel_strided_copy import dispatch,geometry,reference

def test_bits_and_padding():
    p=[2,3,5,4,1,3]
    words=[0x7f7fffff,0,0x80000000,1,0x007fffff,0x00800000,0x80000001,0x807fffff,0x80800000,0xff7fffff,0,0,0,0,0,0,0,0,0]
    x=struct.pack('<19I',*words)
    y=struct.unpack('<23I',reference(p,x))
    for r in range(2):
        for c in range(3):assert y[3+c*4+r]==words[1+r*5+c]
    assert y[0]==0x7fc12345

@pytest.mark.parametrize('p',[[True,3,3,1,0,0],[0,3,3,1,0,0],[2,3,2,2,0,0],[65535]*6])
def test_geometry_refuses(p):
    with pytest.raises(ValueError):geometry(p)

def test_dispatch():
    assert dispatch({'schema':'strided-copy-dispatch-v1','layout':'xy','threads':[16,16]},[17,33,41,23,1,3])['threadgroups']==[3,2]
    with pytest.raises(ValueError):dispatch({'schema':'strided-copy-dispatch-v1','layout':'linear','threads':[16,16]},[17,33,41,23,1,3])
    with pytest.raises(ValueError):dispatch({'schema':'strided-copy-dispatch-v1','layout':'xy','threads':[32,32]},[17,33,41,23,1,3])

def test_exact_lengths():
    with pytest.raises(ValueError):reference([1,1,1,1,0,0],b'1234')
