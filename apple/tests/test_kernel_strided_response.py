import json
import pytest
from exl3_mlx import kernel_strided_response as p
from exl3_mlx import kernel_full_files as rms

def raw(**changes):
    row={'format':p.FORMAT,'rationale':'Preserve exact bits.','files':{'kernel.metal':'// unchanged is allowed\n','dispatch.json':json.dumps({'schema':'strided-copy-dispatch-v1','layout':'xy','threads':[16,16]})+'\n'}}
    row.update(changes)
    return json.dumps(row).encode()

def test_exact_bytes():
    out=p.parse(raw());assert out['kernel.metal']==b'// unchanged is allowed\n'
    with pytest.raises(ValueError):rms.parse(raw())

def test_old_protocol_refused():
    with pytest.raises(ValueError):p.parse(raw(format='kernel-full-files-v1'))

@pytest.mark.parametrize('spec',[{'threads_per_row':32},{'schema':'strided-copy-dispatch-v1','layout':'xy','threads':[True,16]},{'schema':'strided-copy-dispatch-v1','layout':'xy','threads':[16.0,16]},{'schema':'strided-copy-dispatch-v1','layout':'xy','threads':[32,32]},{'schema':'strided-copy-dispatch-v1','layout':'shell','threads':[1,1]}])
def test_dispatch_refusal(spec):
    with pytest.raises(ValueError):p.parse(raw(files={'kernel.metal':'x\n','dispatch.json':json.dumps(spec)+'\n'}))

def test_paths_and_duplicate_keys():
    with pytest.raises(ValueError):p.parse(raw(files={'../kernel.metal':'x\n','dispatch.json':'{}\n'}))
    with pytest.raises(ValueError):p.parse(raw().replace(b'"format":',b'"format":"duplicate","format":',1))

def test_bound_and_no_repair():
    with pytest.raises(ValueError):p.parse(raw(files={'kernel.metal':'missing newline','dispatch.json':'{}\n'}))
    with pytest.raises(ValueError):p.parse(b' '* (128*1024+1))
