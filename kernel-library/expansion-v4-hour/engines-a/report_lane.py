"""Summarize held CPU curation artifacts; no model execution or payload mutation."""
import collections
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
H = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
index = json.loads((R / 'case-index-reviewed-v2.json').read_text())
records = json.loads((R / 'reasoning-records-v2.json').read_text())['records']
cases = index['cases']
core = [x for x in cases if x.get('recommended_core')]
opt = [x for x in cases if x.get('exact_edit_eligible') and not x.get('recommended_core')]
identity = lambda x: (x['repo'].lower().rstrip('/'), x['commit'])
inputs = ['manifest-reviewed.json', 'documentation-omissions.json', 'reviewed-length-audit.json', 'reasoning-records-v2.json', 'candidate-screen.json', 'protection-dedup-review.json', 'source-pins-batch1.json', 'source-pins-batch2.json']
screen = json.loads((R / 'candidate-screen.json').read_text())
histories = []
for f, expected in screen['sources_sha256'].items():
    p = Path(f)
    assert H(p) == expected
    v = json.loads(p.read_text()) if p.suffix == '.json' else [json.loads(l) for l in p.read_text().splitlines()]
    histories.append({'path': f, 'sha256': expected, 'metadata_rows': len(v)})
report = {
    'schema': 'engine-kernel-curation-lane-v4-v1',
    'lane': 'engines-a',
    'scope': 'Historical public-source edit imitation and evidence/diagnostic tasks; no captured agent rollout, local kernel execution, or independent speed qualification.',
    'history_metadata': histories,
    'history_coverage': 'Bounded metadata screening of these histories, followed by purposive source inspection; not exhaustive review of every historical change or follow-up regression.',
    'prior_identity_screen_count': screen['excluded_identity_count'],
    'selected_distinct_source_changes': len(cases),
    'whole_commit_source_reconstruction_candidates': len(json.loads((R / 'manifest-reviewed.json').read_text())['candidates']),
    'exact_scopes': index['scopes'],
    'default_core_exact_32k': len(core),
    'opt_in_unmeasured_exact_32k': len(opt),
    'core_reasoning_views': len(records),
    'reasoning_distinct_source_changes': len({identity(x) for x in records}),
    'reasoning_perspectives': dict(collections.Counter(x['task_perspective'] for x in records)),
    'default_core_exact_or_reasoning_distinct_changes': len({identity(x) for x in core + records}),
    'all_including_unmeasured_exact_or_reasoning_distinct_changes': len({identity(x) for x in core + records + opt}),
    'default_core_view_count_32k': len(core) + len(records),
    'count_caveat': '46 task views cover 29 source changes, not 46 independent optimization concepts. Related follow-up revisions and explicit ports require the supplemental linked-change ledger for split/exposure decisions.',
    'exact_attrition': [{'id': x['id'], 'reason': x.get('exact_edit_exclusion_reason'), 'protection_status': x.get('protection_review', {}).get('status'), 'serialized_ids': x.get('total_serialized_ids')} for x in cases if not x.get('exact_edit_eligible')],
    'token_definition': 'Total serialized IDs include the actual DeepSeek native chat/tool template and final EOS; supervised count is len(input_ids)-loss_start. No truncation. 12K means 12,288 IDs; archival 32K means 32,768 IDs.',
    'edit_contract': 'Single native exec_command call applies an exact Git patch to pinned parent files. All changed source/host/build/test bytes remain in the target; only hash-reviewed .md/.rst changes may be omitted. Context contains complete selected baseline units plus hindsight-selected anchors, not an independently complete build environment.',
    'reasoning_contract': '19 tasks intentionally show historical outcomes. Six reconstructed prospective prompts show baseline code and observed symptoms without the post-change solution or achieved timing; their inferred test plans are not historical tool observations.',
    'protection': 'Existing llama.cpp holdout and captured ktune identities remain protected. DS4 b4da host tile-count repair has a finite public operation-contract comparison and independent review; it is related Q8 work, not evidence of Q8-family independence. mlx-serve909 remains pending and excluded.',
    'validation': {'cpu_exact_replay': 'passed for every admitted edit', 'actual_tokenizer': 'passed for every admitted edit and exported reasoning view', 'local_compile': 'not_run', 'local_kernel_correctness': 'not_run', 'local_benchmark': 'not_run', 'gpu_or_model_execution': 'not_run'},
    'selection_policy': 'Source-reported speed/memory improvements and desired compute-linked correctness repairs form the default core. Three unmeasured optimization edits require explicit opt-in. Neutral, regressed, invalidated or failed hypotheses appear only as critical reasoning tasks or archives, not default positive edit targets.',
    'inputs_sha256': {str((R / f).resolve()): H(R / f) for f in inputs},
    'peer_review_status': 'All24 eligible exact payloads and all25 revised reasoning views independently approved by adversarial_code; receipts are bound in case-index-reviewed-v2.json.'
}
(R / 'integration-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

by_prefix = lambda p: next(x for x in cases if x['id'].startswith(p))
links = []
def evidence(prefix, quote):
    c = by_prefix(prefix)
    p = R / 'sourcepacks' / c['id'] / 'commit-message.txt'
    assert quote in p.read_text()
    return {'path': str(p), 'sha256': H(p), 'url': c['repo'] + '/commit/' + c['commit'], 'quote': quote}
def add(id_, members, relation, ev, recommendation, limitation):
    links.append({'id': id_, 'members': members, 'relation': relation, 'evidence': ev, 'recommendation': recommendation, 'limitation': limitation})
def member(prefix):
    c = by_prefix(prefix)
    return {'id': c['id'], 'repository': c['repo'].lower().rstrip('/'), 'commit': c['commit'], 'patch_lineage': c['lineage']}
add('mimo-192-prefill-explicit-cherry-pick', [member('sushi-84f2'), {'repository': 'unknown upstream checkout', 'commit': '070199c3c041b422d8bd92606313e820581e65ef'}], 'explicit_cherry_pick', [evidence('sushi-84f2', '(cherry picked from commit 070199c3c041b422d8bd92606313e820581e65ef)')], 'If the referenced upstream revision is collected, union split lineage and suppress a second independent-change/concept count; retain distinct implementation identities until exact patch comparison.', 'Upstream identity not reconstructed here; no claim that every surrounding file is identical.')
add('packed-kv-matmul2d-port', [member('sushi-8e991'), {'repository': 'https://github.com/ddalcu/mlx-serve', 'commit': None, 'unit': 'msv_qkv_mpp'}, {'repository': 'https://github.com/incoai/splash', 'commit': None, 'unit': 'q8 attention tile'}], 'source_declared_port_and_design_ancestry', [evidence('sushi-8e991', "Port upstream mlx-serve's kv8 matmul2d attention kernel (msv_qkv_mpp, structure\nafter Inco Splash's q8 attention tile) and generalize it to separate key and\nvalue widths")], 'Keep a shared implementation-family edge for future split review. The DK/DV generalization is a real follow-up task, but is not an independent invention of packed-KV matmul2d attention.', 'No upstream commit SHA is identified in this source; do not equate a different attention commit by keyword alone.')
add('ds4-grouped-moe-decode-campaign', [member('ds4-fe457'), member('ds4-54a'), member('ds4-3c0')], 'source_declared_followup_specialization', [evidence('ds4-fe457', 'The grouped kernels give each expert to the threadgroup of the\nfirst (row, slot) pair in its list'), evidence('ds4-54a', 'Specialize the grouped MoE mid pass on its pair count'), evidence('ds4-3c0', 'Same shape as the mid pass: a template on the pair count')], 'Place all three changes and their diagnostic views in the same future split. Exposure may alternate the grouping, mid specialization and down specialization tasks without treating them as three independent technique families.', 'Reported gains are successive baselines; do not multiply or add them as independent improvements.')
add('ds4-routed-tensorops-repair-and-rejection', [member('ds4-3d14'), member('ds4-d4f')], 'later_contradiction_and_rollback', [evidence('ds4-3d14', 'TensorOps for up+down remains stable.'), evidence('ds4-d4f', 'Semantic evals showed that gate, up, and down TensorOps routed-MoE variants can each move the model into bad continuations')], 'Teach the earlier hypothesis only through critical reasoning with the later contradiction; the later complete removal is the admitted repair target. Keep the two source views together.', 'Neither author evaluation has been reproduced locally; this linkage preserves the chronology instead of asserting universal numerical causation.')
ledger = {'schema': 'supported-implementation-links-v1', 'scope': 'Supplemental strong source-supported links, not an exhaustive semantic deduplication proof. No sealed payload or existing split was changed.', 'normalization': 'Repository URLs lowercased with trailing slash removed; commit hashes remain exact.', 'groups': links, 'view_rule': 'Exact and reasoning views of one patch already share patch lineage. Use these additional edges when preparing future splits/exposure; generic shared motifs alone are not merged.', 'inputs_sha256': report['inputs_sha256']}
(R / 'linked-change-ledger.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'core32k': len(core), 'reasoning': len(records), 'union_changes': report['default_core_exact_or_reasoning_distinct_changes'], 'strong_link_groups': len(links)}))
