"""CPU-only bounded affine row-span planning; no reads, decoding or model support.

Consumes a single already validated safetensors header and explicit file geometry.
Index ownership and payload identity must be checked by the eventual file reader.
"""
from __future__ import annotations

BITS = (2, 3, 4, 5, 6, 8)
GROUPS = (32, 64, 128)
SIZES = {'U32':4, 'F16':2, 'BF16':2, 'F32':4}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def plan_rows(header, config, prefix, rows, *, data_start, file_bytes,
              max_rows=4096, max_read_bytes=8 << 20):
    """Return deduplicated packed/scales/bias row byte spans in request order."""
    require(isinstance(header, dict) and isinstance(config, dict), 'Expected mappings')
    require(isinstance(prefix, str) and prefix, 'Expected tensor prefix')
    require(integer(data_start, 8) and integer(file_bytes, data_start), 'Invalid file geometry')
    require(integer(max_rows, 1) and integer(max_read_bytes, 1), 'Invalid limits')
    require(isinstance(rows, (list, tuple)) and 0 < len(rows) <= max_rows,
            'Expected bounded nonempty row requests')
    q = config.get('quantization')
    other = config.get('quantization_config')
    require(isinstance(q, dict), 'Missing quantization metadata')
    require(other is None or other == q, 'Conflicting quantization metadata')
    selected = q.get(prefix, q)
    require(isinstance(selected, dict), 'Invalid per-weight quantization')
    bits, group = selected.get('bits'), selected.get('group_size')
    require(type(bits) is int and bits in BITS and type(group) is int and group in GROUPS,
            'Unsupported bits/group size')
    require(selected.get('mode') == 'affine', 'Only affine encoding supported')
    tensors = {}
    for kind in ('weight', 'scales', 'biases'):
        name = prefix + '.' + kind
        spec = header.get(name)
        require(isinstance(spec, dict) and set(spec) == {'dtype','shape','data_offsets'},
                'Missing or malformed tensor: ' + name)
        shape, offsets, dtype = spec['shape'], spec['data_offsets'], spec['dtype']
        require(isinstance(shape, list) and len(shape) == 2 and all(integer(d,1) for d in shape),
                'Expected nonempty matrix')
        require(isinstance(dtype,str) and dtype in SIZES, 'Unsupported dtype')
        require(isinstance(offsets,list) and len(offsets)==2 and all(integer(d) for d in offsets),
                'Invalid offsets')
        lo, hi = offsets
        require(lo <= hi <= file_bytes-data_start and hi-lo == shape[0]*shape[1]*SIZES[dtype],
                'Tensor range/shape mismatch')
        require(dtype == 'U32' if kind == 'weight' else dtype in ('F16','BF16','F32'),
                'Invalid packed or affine metadata dtype')
        tensors[kind] = spec
    w, s, b = (tensors[k] for k in ('weight','scales','biases'))
    require(w['shape'][1]*32 % bits == 0, 'Partial packed row')
    columns = w['shape'][1]*32//bits
    require(columns % group == 0, 'Partial affine group')
    expected = [w['shape'][0],columns//group]
    require(s['shape']==b['shape']==expected and s['dtype']==b['dtype'], 'Scale/bias geometry mismatch')
    intervals = sorted(t['data_offsets'] for t in tensors.values())
    require(all(a[1] <= b[0] for a,b in zip(intervals,intervals[1:])), 'Overlapping tensor payloads')
    require(all(integer(row) and row < w['shape'][0] for row in rows), 'Row outside matrix')
    unique = list(dict.fromkeys(rows))
    strides = {k:v['shape'][1]*SIZES[v['dtype']] for k,v in tensors.items()}
    total = len(unique)*sum(strides.values())
    require(total <= max_read_bytes, 'Requested rows exceed byte budget')
    spans = [{'row':row,'kind':kind,'offset':data_start+spec['data_offsets'][0]+row*strides[kind],
              'bytes':strides[kind],'dtype':spec['dtype']}
             for row in unique for kind,spec in tensors.items()]
    return {'schema':'affine-row-spans-v1','rows':unique,'request_to_unique':[unique.index(r) for r in rows],
            'bits':bits,'group_size':group,'columns':columns,'total_read_bytes':total,'spans':spans,
            'qualification':'Span planning only; no index ownership, payload identity, decode or numerical validation.'}
