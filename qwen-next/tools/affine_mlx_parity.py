"""Tiny, explicit MLX GPU qualification; no checkpoint/model loading or training."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

from affine_reference import AffineReference, verify_spec_sources

ROOT = Path(__file__).resolve().parents[1]
MLX_ROOT = Path('/Users/jasontitus/experiments/exl3-qlora/apple/.venv/lib/python3.12/site-packages/mlx')
TOLERANCES = {'float32': (2e-5, 2e-5), 'float16': (0.004, 0.004), 'bfloat16': (0.03, 0.03)}
RELATIVE_L2_LIMITS = {'float32': 0.0001, 'float16': 0.01, 'bfloat16': 0.05}


def require_shape(actual, expected, label):
    if tuple(actual) != tuple(expected):
        raise ValueError(f'{label}: output shape {actual} != {expected}')


def single_output(value):
    """Handle documented list result without iterating an array's batch axis."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError('Expected one VJP output')
        return value[0]
    return value


def matrix_comparison(observed, target, name):
    import numpy as np
    observed, target = np.asarray(observed), np.asarray(target)
    require_shape(observed.shape, target.shape, 'matrix')
    atol, rtol = TOLERANCES[name]
    norm = float(np.linalg.norm(target))
    delta = float(np.linalg.norm(observed-target))
    relative = delta / norm if norm else (0.0 if delta == 0 else float('inf'))
    return {'passed':bool(np.allclose(observed,target,atol=atol,rtol=rtol) and relative <= RELATIVE_L2_LIMITS[name]),
            'relative_l2':relative, 'relative_l2_limit':RELATIVE_L2_LIMITS[name],
            'max_abs_error':float(np.max(np.abs(observed-target))),
            'comparison':'Predeclared elementwise and relative-L2 gates versus binary64 matrix'}


def fixture(bits, group, batch):
    if bits not in (2, 3, 4, 5, 6, 8) or group not in (32, 64, 128) or batch not in (1, 3):
        raise ValueError('Unsupported bounded fixture')
    k, n = 2 * group, 32
    mask = (1 << bits) - 1
    codes = [[(o * 13 + i * 7 + i // 3) & mask for i in range(k)] for o in range(n)]
    packed = []
    for row in codes:
        whole = sum(value << (i * bits) for i, value in enumerate(row))
        packed.append([(whole >> shift) & 0xffffffff for shift in range(0, k * bits, 32)])
    scales = [[(1 if o % 2 else -1) / 1024, 0 if o % 5 == 0 else 1 / 512] for o in range(n)]
    biases = [[(o % 3 - 1) / 16, (o % 5 - 2) / 32] for o in range(n)]
    x = [[((i * 3 + b * 7) % 17 - 8) / 32 for i in range(k)] for b in range(batch)]
    dy = [[((o * 7 + b * 3) % 13 - 6) / 16 for o in range(n)] for b in range(batch)]
    return packed, scales, biases, x, dy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dispatch', action='store_true', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError('Refusing to overwrite receipt')
    checked = verify_spec_sources(MLX_ROOT)
    import importlib.metadata
    import numpy as np
    import mlx.core as mx
    if importlib.metadata.version('mlx') != '0.32.3':
        raise ValueError('Unqualified MLX version')
    if not mx.metal.is_available():
        raise RuntimeError('Metal unavailable')
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(256 << 20)
    mx.set_cache_limit(16 << 20)
    results = []
    for bits in (2, 3, 4, 5, 6, 8):
        for group in (32, 64, 128):
            for name, (atol, rtol) in TOLERANCES.items():
                for batch in (1, 3):
                    row = {'bits':bits,'group':group,'dtype':name,'batch':batch,'atol':atol,'rtol':rtol}
                    try:
                        packed, scales, biases, x, dy = fixture(bits, group, batch)
                        ref = AffineReference(packed, scales, biases, in_features=group*2, bits=bits, group_size=group)
                        dtype = getattr(mx, name)
                        wq = mx.array(packed, dtype=mx.uint32)
                        s, z, inp, cot = [mx.array(v, dtype=dtype) for v in (scales,biases,x,dy)]
                        call = lambda value: mx.quantized_matmul(value,wq,s,z,transpose=True,group_size=group,bits=bits)
                        y, (dx,) = mx.vjp(call, [inp], [cot])
                        y = single_output(y)
                        transpose = mx.quantized_matmul(cot,wq,s,z,transpose=False,group_size=group,bits=bits)
                        decoded = mx.dequantize(wq,s,z,group_size=group,bits=bits)
                        mx.eval(y,dx,transpose,decoded)
                        actual = [np.array(a.astype(mx.float32)) for a in (y,dx,transpose,decoded)]
                        expected = [np.array(ref.forward(x)), np.array(ref.transpose(dy)), np.array(ref.transpose(dy)), np.array(ref.decode_rows(count=32))]
                        checks = {}
                        for label, observed, target in zip(('forward','input_vjp','transpose','decode'), actual, expected):
                            require_shape(observed.shape, target.shape, label)
                            if label == 'decode':
                                target = target.astype(np.float32)
                                if name == 'float16':
                                    target = target.astype(np.float16).astype(np.float32)
                                elif name == 'bfloat16':
                                    raw = target.view(np.uint32)
                                    target = ((raw + np.uint32(0x7fff) + ((raw >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
                                passed = np.array_equal(observed,target)
                                comparison = 'exact dtype-rounded values; host BF16 round-to-nearest-even'
                            else:
                                checks[label] = matrix_comparison(observed,target,name)
                                continue
                            checks[label] = {'passed':bool(passed), 'comparison':comparison, 'max_abs_error':float(np.max(np.abs(observed-target)))}
                        row.update(checks=checks, passed=all(c['passed'] for c in checks.values()), output_dtype=str(y.dtype), input_gradient_dtype=str(dx.dtype))
                    except Exception as error:
                        row.update(passed=False,error=repr(error))
                    results.append(row)
                    mx.clear_cache()
    report = {'scope':'Tiny synthetic affine native MLX forward and input-only VJP; no checkpoint, adapter training or model support claim',
              'mlx_version':importlib.metadata.version('mlx'),'python':platform.python_version(),'device':mx.metal.device_info(),
              'reference':'Binary64 mathematical affine matrix; low-precision results use predeclared dtype tolerances, not bit-exact parity',
              'memory_limit_bytes':256<<20,'peak_mlx_bytes':mx.get_peak_memory(), 'cases':len(results),
              'passed':all(r['passed'] for r in results),'results':results,
              'inputs_sha256':{str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                               str(ROOT/'tools/affine_reference.py'):hashlib.sha256((ROOT/'tools/affine_reference.py').read_bytes()).hexdigest()},
              'mlx_source_sha256':checked}
    args.receipt.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('cases','passed','peak_mlx_bytes')}))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
