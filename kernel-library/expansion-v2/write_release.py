"""Publish a human-readable handoff only after independent aggregate acceptance."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent
p=R/'corpus-12288-v3';a=R/'corpus-32768-v3'
small=json.loads((p/'published-summary.json').read_text());large=json.loads((a/'published-summary.json').read_text())
audit=R/'aggregate-independent-audit-v3.json'
if not audit.exists():raise SystemExit('Independent aggregate audit missing; no release')
review=json.loads(audit.read_text())
if review.get('status')!='passed_scoped_audit':raise ValueError('Independent audit has not passed')
if {c['corpus'] for c in review['corpora']}!={str(p),str(a)}:raise ValueError('Audit corpus mismatch')
for name,expected in review['inputs_sha256'].items():
 if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=expected:raise ValueError('Post-audit input drift: '+name)
if any((d/n).exists() for d in (p,a) for n in ('SUPERSEDED.json','REJECTED.json')):raise ValueError('Unusable corpus marker')
rows=[]
for repo,n in small['by_repository'].items():rows.append('| '+repo.removeprefix('https://github.com/')+' | '+str(n)+' | '+str(large['by_repository'][repo])+' |')
text=f'''# Metal optimization training corpus — reviewed release 3

The practical corpus contains **{small['counts']['train']} distinct complete historical edits**, up from 24, with **{small['total_supervised_tokens']:,} supervised tokens**. The longer archive contains **{large['counts']['train']}** edits, including the practical set. Both preserve the same **16 final-held-out llama.cpp examples byte for byte**. Counts are changes, not independent optimization families.

Use [corpus-12288-v3](corpus-12288-v3/) for the next training preparation. [corpus-32768-v3](corpus-32768-v3/) retains longer complete edits; preparing these rows does not establish that 32K training fits or works. Maximum serialized item lengths are {small['lengths']['max']:,} and {large['lengths']['max']:,} tokens respectively. No edit was silently truncated.

Each directory contains standard JSONL messages and a native `exec_command` tool schema. The assistant gives source-grounded rationale and a complete `git apply` edit. Prompts retain complete relevant source units and exact original edit anchors. Full before/after source, parent and head revisions, licenses, source measurements and hash-bound documentation omissions remain in provenance. These are reconstructed single-edit examples, not recorded multi-turn agent trajectories.

The existing `exl3_mlx.workflow train --data <corpus directory>` flow reads `train.jsonl`; its separate `test.jsonl` is reserved for evaluation. Keep the supplied split. Do not randomly repartition connected history or use the final-held-out set to select hyperparameters. The new corpus has not been substituted into the running DeepSeek job, and this release does not launch another GPU job.

## Coverage

| Source | Practical ≤12K | Archive ≤32K |
|---|---:|---:|
'''+ '\n'.join(rows)+f'''
| ggml-org/llama.cpp | 16 evaluation only | 16 evaluation only |

The practical training set has {small['by_task_kind']['kernel']} kernel tasks and {small['by_task_kind']['host_dispatch']} host dispatch/workspace/scheduling tasks. Auxiliary conversion-only and loader-only changes are retained in collection evidence, not inflated into these counts.

## What passed

- Complete non-documentation edit replay against pinned source images, stable-patch deduplication, native tool serialization and exact tokenizer/mask checks.
- Independent aggregate audit: [receipt](aggregate-independent-audit-v3.json).
- The old 24 practical training payloads and 16 held-out payloads are preserved. Exact and threshold-based cross-split checks found no blockers; this does not prove semantic family independence.
- Automatic reserved-head/base rejection against 157 captured identities, pinned metadata and post-build input-drift checks. Ten focused regression tests passed independently; earlier extraction/planner tests also passed.
- Both sides of two standalone TensorFold Metal source pairs compiled under Metal4: [compiler receipt](compiler-probes/compiler-receipt.json). This is a limited compiler check, not host-engine or runtime qualification.

## Review corrections and remaining gates

Reviews removed a neutral TensorFold change from positive examples, corrected Sushi's superseded speed attribution, corrected one protected source-only MLX record, and removed an exact protected GQA commit discovered in staging. Old partial and superseded builds remain as audit evidence; do not train `preflight-*`, `corpus-*-v1` or `corpus-*-v2`. Use only the version3 paths above.

Performance evidence remains **author-reported** except where an explicit local receipt says otherwise. The corpus is source-reviewed and replay-verified; it is not a set of universally compiled, numerically verified or locally benchmarked kernels. Dependency-heavy kernels and host dispatch changes require their native build/workload harness. Do not treat a successful patch application as numerical correctness.

Related HC examples are explicitly related-family training material. They cannot support HC-family-independent efficacy claims. Unknown historical lineage, semantic equivalence and possible pretraining exposure remain limitations.

Once the existing DeepSeek run and its fixed evaluation finish, choose the next run using this practical corpus and its actual lengths. Evaluate held-out generated patches with compilation, numerical correctness and timing; report code, rationale and EOS metrics separately. Improved teacher-forced loss alone does not demonstrate better kernel optimization.

The daily source collection cadence is unchanged. [Source-admission review](source-admission-independent-review.json), [practical summary](corpus-12288-v3/published-summary.json), [archive summary](corpus-32768-v3/published-summary.json), and each corpus's `exclusions.json` record the evidence and attrition.
'''
(R/'README.md').write_text(text)
receipt={'format':'metal-corpus-release-v1','timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary_corpus':str(p),'archive_corpus':str(a),'train_count':small['counts']['train'],'archive_train_count':large['counts']['train'],'heldout_count':16,'qualification':'source-reviewed, exact-replay and tokenizer audited; runtime validation incomplete','training_launched':False,'files_sha256':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'train.jsonl',p/'test.jsonl',a/'train.jsonl',audit,R/'README.md',R/'build_combined.py',R/'tools/reserved_identity.py']}}
(R/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
(R.parent/'reports/latest-training-corpus.json').write_text(json.dumps({'release':str(R/'release.json'),'sha256':hashlib.sha256((R/'release.json').read_bytes()).hexdigest(),'primary_corpus':str(p),'archive_corpus':str(a),'train_count':small['counts']['train'],'archive_train_count':large['counts']['train'],'qualification':receipt['qualification']},indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='files_sha256'},indent=2))
