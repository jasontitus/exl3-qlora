"""Independent byte-extraction oracle; synthetic CPU fixtures only."""
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('independent_reader', Path(__file__).parents[1]/'tools/affine_row_reader.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

@pytest.mark.parametrize('bits',[2,3,4,5,6,8])
@pytest.mark.parametrize('group',[32,64,128])
@pytest.mark.parametrize('dtype,scalar_bytes',[('F16',2),('BF16',2),('F32',4)])
def test_permuted_duplicate_reads_and_exact_budget(tmp_path,bits,group,dtype,scalar_bytes):
    prefix='layer.proj'
    components=[('scales',dtype,128//group,scalar_bytes),('weight','U32',128*bits//32,4),('biases',dtype,128//group,scalar_bytes)]
    payload=b'';header={};expected={row:{} for row in range(4)}
    for number,(kind,dt,width,size) in enumerate(components):
        start=len(payload)
        for row in range(4):
            raw=bytes((row*37+number*53+j*11)%256 for j in range(width*size))
            payload+=raw;expected[row][kind]=raw
        header[prefix+'.'+kind]={'dtype':dt,'shape':[4,width],'data_offsets':[start,len(payload)]}
    raw=json.dumps(header).encode();raw+=b' '*(-len(raw)%8)
    (tmp_path/'part.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+payload)
    (tmp_path/'config.json').write_text(json.dumps({'quantization':{'bits':bits,'group_size':group,'mode':'affine'}}))
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{name:'part.safetensors' for name in header}}))
    budget=sum(len(v) for row in (1,3) for v in expected[row].values())
    identity=m.bind_rows(tmp_path,prefix=prefix,rows=[3,1,3],max_read_bytes=budget)
    reader=m.AffineRowReader(tmp_path,identity,expected_sha256=m.identity_sha256(identity),max_read_bytes=budget)
    result=reader.read([1,3,1,3])
    assert result['payloads']==(expected[1],expected[3])
    assert result['request_to_unique']==(0,1,0,1)
    assert result['receipt']['payload_bytes_read']==budget
    for span in result['receipt']['plan']['spans']:
        assert span['sha256']==hashlib.sha256(expected[span['row']][span['kind']]).hexdigest()
    with pytest.raises(ValueError):m.bind_rows(tmp_path,prefix=prefix,rows=[3,1],max_read_bytes=budget-1)
    with pytest.raises(ValueError,match='not bound'):reader.read([0])
