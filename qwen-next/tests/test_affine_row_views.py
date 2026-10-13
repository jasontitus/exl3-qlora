import copy
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from affine_row_views import plan_rows


def fixture(bits=4):
    n, k, g = 3, 128, 32
    specs, offset = {}, 0
    for kind,dtype,width,size in [('weight','U32',k*bits//32,4),('scales','BF16',k//g,2),('biases','BF16',k//g,2)]:
        end=offset+n*width*size
        specs['a.'+kind]={'dtype':dtype,'shape':[n,width],'data_offsets':[offset,end]}
        offset=end
    return specs,{'quantization':{'bits':bits,'group_size':g,'mode':'affine'}},offset+64


@pytest.mark.parametrize('bits',[2,3,4,5,6,8])
def test_complete_rows_and_duplicate_mapping(bits):
    h,c,size=fixture(bits)
    p=plan_rows(h,c,'a',[2,0,2],data_start=64,file_bytes=size)
    assert p['rows']==[2,0] and p['request_to_unique']==[0,1,0]
    assert p['columns']==128
    assert p['total_read_bytes']==2*(128*bits//8+16)
    raw=bytes(i%256 for i in range(size))
    for span in p['spans']:
        spec=h['a.'+span['kind']]
        stride=(spec['data_offsets'][1]-spec['data_offsets'][0])//3
        assert raw[span['offset']:span['offset']+span['bytes']]==raw[64+spec['data_offsets'][0]+span['row']*stride:64+spec['data_offsets'][0]+(span['row']+1)*stride]


@pytest.mark.parametrize('rows',[[],[True],[-1],[3],[1.0]])
def test_invalid_rows(rows):
    h,c,size=fixture()
    with pytest.raises(ValueError): plan_rows(h,c,'a',rows,data_start=64,file_bytes=size)


def test_bounds_and_conflict():
    h,c,size=fixture()
    for kw in ({'max_read_bytes':1},{'max_rows':1}):
        with pytest.raises(ValueError): plan_rows(h,c,'a',[0,1],data_start=64,file_bytes=size,**kw)
    c['quantization_config']={'bits':8}
    with pytest.raises(ValueError): plan_rows(h,c,'a',[0],data_start=64,file_bytes=size)


def test_overlap_and_dtype():
    for mutate in [lambda h:h['a.biases'].update(data_offsets=h['a.scales']['data_offsets']),lambda h:h['a.weight'].update(dtype='F32')]:
        h,c,size=fixture();mutate(h)
        with pytest.raises(ValueError):plan_rows(h,c,'a',[0],data_start=64,file_bytes=size)


def test_explicit_override_and_no_partial_override():
    h,c,size=fixture(8);c['quantization']['bits']=4
    c['quantization']['a']={'bits':8,'group_size':32,'mode':'affine'}
    assert plan_rows(h,c,'a',[0],data_start=64,file_bytes=size)['bits']==8
    c['quantization']['a']={'bits':8}
    with pytest.raises(ValueError):plan_rows(h,c,'a',[0],data_start=64,file_bytes=size)


def test_pinned_real_trunk_header_geometry_only():
    root=Path(__file__).resolve().parents[1]/'sources'
    header=json.loads((root/'model-trunk-00002-of-00002.safetensors.header.json').read_text())
    config=json.loads((root/'config.json').read_text())
    # Synthetic header origin for geometry test only; no real payload is present.
    start=64
    size=start+max(v['data_offsets'][1] for v in header.values() if isinstance(v,dict) and 'data_offsets' in v)
    count=0
    for name,spec in header.items():
        if not name.endswith('.weight'):continue
        prefix=name[:-7]
        if prefix+'.scales' not in header or prefix+'.biases' not in header:continue
        plan=plan_rows(header,config,prefix,[0,spec['shape'][0]-1],data_start=start,file_bytes=size)
        assert plan['total_read_bytes'] <= 8<<20
        assert all(s['offset']+s['bytes']<=size for s in plan['spans'])
        count+=1
    assert count>100
