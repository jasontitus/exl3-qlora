from pathlib import Path
import json
R=Path(__file__).resolve().parent;m=json.loads((R/'manifest-review-candidate.json').read_text());conditions={
'cb477466a2a1':'The captured engine falls back on3-bit expert layers, and the down-projection reduction leaves lanes idle at the recorded K=640 geometry. Preserve packed-bit interpretation and distinguish reduction-order tolerance from bitwise equality.',
'e92459219ce3':'The supported Ultra single-slot verification request has several query rows, but the existing per-row-equivalent fused MoE route is not selected. Preserve other device/batch routes and the explicit operator override.',
'080a62e2f2d7':'GQA12 verification widths fall off the vector attention route under the old GQA6 assumption. Keep every row’s causal key range and avoid preempting a preferred fused NAX kernel. Default activation must stay within the measured architecture envelope.',
'9a5467375008':'Dense BF16 KV verification currently builds a large mask or gathers a union of rows despite a fused quantized route. Preserve production reachability, correct offset/strided views and the untouched quantized kernel’s source identity.',
'3dc1dab54e7b':'Verification recurrence, norm/gate and rollback capture require several dependent dispatches. Preserve all captured outputs and stored-width rounding, reject malformed shapes, and retain a clean fallback on devices that refuse the needed threadgroup width.',
'9a5c98126fb8':'Sorted expert-row prefill has redundant input movement and separate activation work. Preserve expert assignment, padding and per-configuration correctness; unsupported or numerically divergent configurations must keep a known route.',
'c24c70ab8ead':'GDN decode and short verification pay separate prework/recurrence dispatches. Preserve the exact rollback state and norm/gate behavior, including the Hadamard-dependent route that is outside the chosen fusion.',
'987c40030406':'EXL3 prefill fails during evaluated NAX compilation on macOS27 even though construction succeeds. Keep a functioning fallback and preserve Hadamard-block boundaries when tuning decode/prefill work. Report changed-reduction numerical limits explicitly.',
'6755ff2426b7':'A quantized long-context KV view rebuilds an entire layer in BF16 although each attention dispatch reads only a key window. Bound temporary memory while preserving absolute causal/mask coordinates, softmax carry and the dense fallback.',
'b89991a4e6ab':'Packed-cache attention suffers a route cliff at16 query rows and can repeatedly stage the same cache row for wider requests. Choose routes from supported width/reuse regimes and keep memory admission consistent; do not require universal bitwise generation identity.',
'25a4994db9cc':'Sliding-window attention with learned sink logits falls outside the fused prefill route. Preserve sink semantics, ring-buffer views and sink-free behavior while accounting for temporary attention memory.',
'2d58ab608455':'Long-cache verification repeats the packed K/V page walk per row. Preserve each row’s decode arithmetic, causal split boundaries and the fallback when decode templates or grouping are unsupported.',
'8cbdd7624f27':'Compressor position-table preparation incurs tens of thousands of tiny copy encoders for a large prompt. Preserve conversion values, bounds and opt-out behavior while reducing launch overhead.',
'81f8534e3b85':'Routed MoE prefill dispatch should compact its work without changing selected-expert input/output addressing. The source supplies no elapsed-time measurement for this edit; keep that limit explicit.',
'17c6726e8388':'A single-tile MoE route still gathers input rows even though the supported tile can access them directly. Preserve multi-tile and architecture-specific fallback plans and unchanged arithmetic.',
'5ecf1fe88676':'Block MoE dispatch launches work for inactive expert tiles. Preserve the maximum capacity bound and exact full-grid outputs while allowing host/kernel metadata to describe live work.',
'dff5dfc57308':'MoE padding and gathering write more rows than the selected matmul reads. Preserve live-row correctness, tail canaries and the existing8-row behavior.',
'd2b331112d7f':'Sparse live expert rows need an appropriate supported tile height. The supplied author comparison includes a companion gather-trimming change in both arms, so its gains are conditional and must not be attributed to this commit alone.',
'4828e1637469':'Draft-attention query preparation writes a duplicate query buffer unused downstream. Preserve the live grouped-query representation, host binding order and input immutability.'}
for c in m['candidates']:
 s=c['commit'][:12]
 if s in conditions:
  msg=(R/'sourcepacks'/c['id']/'commit-message.txt').read_text();c['annotations']['constraints']={'text':conditions[s],'basis':'source','evidence':[{'source':'commit_message','quote':msg},{'source':'diff-'+s,'quote':(R/'sourcepacks'/c['id']/'commit.diff').read_text()}]}
 if s=='b64d3e0689a6':
  c['curriculum_eligibility']='quarantined_protected_operation';c['historical_protection_review']['status']='pending';c['historical_protection_review']['reason']='Compressor fusion reproduces existing FP8/FP4 phases; reserved FP8-rounding operation overlap has not been resolved by a public source-contract comparison. Excluded from exact and reasoning training; preserved for review.'
(R/'manifest-review-candidate.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n')
x=json.loads((R/'reasoning-records-v3.json').read_text());x['excluded_records']=[r['id']for r in x['records']if r['commit'].startswith('b64d3e0689a6')];x['records']=[r for r in x['records']if not r['commit'].startswith('b64d3e0689a6')];(R/'reasoning-records-final.json').write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');print(len(x['records']),'reasoning eligible')
