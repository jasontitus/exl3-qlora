"""CPU arithmetic checks for reviewed explanations, not execution of source kernels."""
import hashlib,json,platform
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;B=R.parent/'engines-b';rng=np.random.default_rng(19619);rows=[]
for D in [32,64,128,256]:
 for magnitude in [0.,1e-6,1e-3,1.]:
  x=rng.standard_normal((8,D))*magnitude;eps=1e-6
  reference=x/np.sqrt(np.sum(x*x,axis=-1,keepdims=True)+eps)
  repaired=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps/D)/np.sqrt(D)
  wrong=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)/np.sqrt(D)
  err=float(np.max(np.abs(reference-repaired)));bad=float(np.max(np.abs(reference-wrong)))
  assert err<1e-14
  if magnitude==1e-3:assert bad>0.01
  rows.append({'probe':'sum_vs_mean_epsilon','D':D,'magnitude':magnitude,'corrected_max_abs_error':err,'old_formula_max_abs_error':bad})
for D in [32,128,256]:
 exact=np.float32(D**-0.5);rendered=np.float32(format(float(exact),'.7g'));same=exact.tobytes()==rendered.tobytes()
 assert same==(D==256)
 rows.append({'probe':'seven_significant_digit_float32_scale','D':D,'memory_operand':float(exact),'rendered_constant':float(rendered),'same_float32_bits':same})
top=np.float32(2**24-1);u=np.float32(np.float32(top+np.float32(.5))*np.float32(1/(2**24)));cap=np.nextafter(np.float32(1),np.float32(0));fixed=np.minimum(u,cap)
assert u==1 and 0<fixed<1
with np.errstate(divide='ignore',invalid='ignore'):
 bad=-np.log(-np.log(u));good=-np.log(-np.log(fixed))
assert np.isinf(bad) and np.isfinite(good)
rows.append({'probe':'float32_uniform_endpoint','old_uniform':float(u),'capped_uniform':float(fixed),'old_race_is_infinite':bool(np.isinf(bad)),'capped_race_is_finite':bool(np.isfinite(good))})
sources=[B/'drafts'/cid/'source.json' for cid in ['vllm-metal-a57d71e5fbf7-hour-v3','mtplx-404084baedaa-hour-v3','tensorfold-6128a0c131f9-hour-v3']]
report={'passed':True,'checks':len(rows),'python':platform.python_version(),'numpy':np.__version__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_records_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'qualification':'Standalone CPU arithmetic corroboration of three reasoning claims. Not MLX/Metal/engine execution, model parity, or measured optimization performance. Does not upgrade complete cases to numerical-validation tier.','results':rows}
(R/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(rows))
