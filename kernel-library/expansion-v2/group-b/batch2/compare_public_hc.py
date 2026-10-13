"""Finite CPU source comparison, never a proof of semantic independence."""
import difflib
import hashlib
import json
import re
import shutil
from pathlib import Path
from refined_units import function_range

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[3]
H = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
D = lambda v: json.dumps(v, indent=2, ensure_ascii=False) + '\n'
prior = WORK / 'kernel-library/integration-v1/group-b/drafts/mtplx-7628402337f8.source.json'
record = json.loads(prior.read_text())
meta_path = ROOT.parent / 'evidence/reserved-task-contracts.json'
metadata = json.loads(meta_path.read_text())
task = next(x['data'] for x in metadata['rows'] if x['data'].get('id') == 'kernel-pilot:hc')
(root := ROOT / 'evidence/public-hc').mkdir(exist_ok=True)
units = []
for revision, function in [(task['base_commit'], 'kernel void kernel_qwen4_hc_gate_mix('), (task['reference_commit'], 'kernel void kernel_qwen4_hc_gate_mix_f16_pf(')]:
    source = ROOT.parent / 'evidence/public-hc' / (revision + '.metal')
    assert H(source) == task['provenance']['source_sha256'][revision]
    target = root / source.name
    shutil.copyfile(source, target)
    raw = source.read_bytes()
    for needle in ['static inline float qwen4_sigmoid(', 'static inline float qwen4_silu(', function]:
        a, b = function_range(raw.decode(), needle)
        units.append({'revision': revision, 'name': needle, 'start': a, 'end': b, 'sha256': hashlib.sha256(raw[a:b]).hexdigest(), 'source_sha256': H(source), 'text': raw[a:b].decode()})

def tokens(s):
    return re.findall(r'[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]', s)

protected = [tokens(x['text']) for x in units]
raw_example = json.loads((prior.parent / 'mtplx-7628402337f8.raw.json').read_text())
inputs = {'complete_rendered_prompt': json.dumps(raw_example['messages'][:-1], sort_keys=True), 'all_added_source_lines': '\n'.join(x[1:] for x in record['diff'].splitlines() if x.startswith('+') and not x.startswith('+++'))}
metrics = {}
for name, text in inputs.items():
    t = tokens(text)
    metrics[name] = {'tokens': len(t), 'maximum_contiguous_lexical_match': max(difflib.SequenceMatcher(None, p, t, autojunk=False).find_longest_match().size for p in protected), 'shared_32grams': sum(len(set(tuple(t[i:i+32]) for i in range(max(0,len(t)-31))) & set(tuple(p[i:i+32]) for i in range(max(0,len(p)-31)))) for p in protected)}
result = {'format':'finite-public-hc-source-comparison-v1','candidate_commit':record['head']['commit'],'candidate_source_record':str(prior),'candidate_source_record_sha256':H(prior),'candidate_raw_example_sha256':H(prior.parent/'mtplx-7628402337f8.raw.json'),'protected_task_metadata_sha256':H(meta_path),'public_sources':[{'revision':v,'url':f'https://raw.githubusercontent.com/ivanfioravanti/ds4-metal/{v}/metal/qwen4.metal','path':str((root/(v+'.metal')).relative_to(ROOT)),'sha256':H(root/(v+'.metal')),'matches_captured_task_pin':True} for v in [task['base_commit'],task['reference_commit']]],'protected_units':[{k:v for k,v in u.items() if k!='text'} for u in units],'lexical_diagnostic':metrics,'findings':['The high-level SiLU/up-projection/sigmoid/stream-reduction operation overlaps. Neither a different repository nor additional norm/down work proves operation-family independence.','Reserved source accepts F32 normalized inputs and raw low-rank values with F16 projection weights; it computes F32 SiLU/up accumulation and F32 sigmoid/product/shuffle reduction, returning FP32 bits. Reference prefetches eight terms per lane with explicit non-FMA order.','MTPLX candidate consumes F16/BF16 streams and mixers, rounds projection intermediates and elementwise operations to that type, obtains sigmoid values from the installed MLX lowering at all16-bit input patterns, and uses the installed GemvWide template. Its output and intermediate ABI are different from the reserved FP32-output task.','MTPLX serves verify widths2–8, including3; the reserved fixtures include T3. There is no token-shape disjointness claim. Its up projection/stream sum is related HC training material.','Public files exactly match captured reserved source pins. Lexical comparison is a limited diagnostic, not proof of unrelated ancestry or absence of all semantic leakage. No remote protected implementation or local upstream code was executed.'],'recommendation':'Admit as related HC source-reported training only if independent reviewer agrees that this is a distinct dtype/ABI/implementation task; quarantine any HC-family held-out efficacy interpretation. The shared family must remain visible.','execution_verified':False,'semantic_independence_certified':False}
(ROOT/'evidence/hc-public-contract-comparison.json').write_text(D(result))
print(D({k:result[k] for k in ['candidate_commit','lexical_diagnostic','recommendation']}))
