"""Supplemental, read-only implementation-family review; no payload mutation."""
from pathlib import Path
import collections
import hashlib
import json

ROOT = Path(__file__).resolve().parent
HOUR = ROOT.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inventory_path = ROOT / 'inventory.json'
inventory = json.loads(inventory_path.read_text())
rows = inventory['rows']
exact = {r['commit']: r for r in rows if r['view'] != 'new_reasoning'}
inputs = dict(inventory['inputs_sha256'])
inputs[str(inventory_path)] = sha(inventory_path)

def record(prefix):
    found = [r for commit, r in exact.items() if commit.startswith(prefix)]
    if len(found) != 1:
        raise ValueError(prefix)
    return found[0]

def evidence(prefix, quote):
    row = record(prefix)
    path = Path(row['source'])
    source = json.loads(path.read_text())['source']
    assert quote in source['head']['message']
    return {'path': str(path), 'sha256': sha(path), 'json_pointer': '/source/head/message',
            'quote': quote, 'url': row['repository'] + '/commit/' + row['commit']}

def text_evidence(path, quote, url):
    path = Path(path).resolve()
    assert quote in path.read_text()
    inputs[str(path)] = sha(path)
    return {'path': str(path), 'sha256': sha(path), 'quote': quote, 'url': url}

def member(prefix):
    row = record(prefix)
    return {'key': row['key'], 'commit': row['commit'], 'repository': row['repository'],
            'views': [{'id': r['id'], 'view': r['view']} for r in rows if r['key'] == row['key']]}

groups = [
    {'family_id': 'gdn-qk-l2norm-via-rmsnorm-epsilon-scaling',
     'relationship': 'explicit_upstream_fix_adaptation',
     'members': [member('a57d71e5fbf7')],
     'external_anchor': {'repository': 'https://github.com/ml-explore/mlx-lm',
                         'symbol': 'normalize_qk', 'commit': None,
                         'identity_limit': 'The author names the upstream implementation but not its revision; no exact upstream commit is asserted.'},
     'evidence': [evidence('a57d71e5fbf7', "The epsilon is scaled by `inv_scale**2`, which is 1e-6 divided by\n`head_dim`, matching what mlx-lm's `normalize_qk` does since the fix\nlanded upstream.")],
     'interpretation': 'This is an implementation of an explicitly credited upstream correction, not an independent discovery of the epsilon rule. No matching upstream epsilon-fix commit is an exact row in the inspected new/prior set; MLX-LM735a43 changes recurrent-state precision and is a different edit.',
     'split_rule': 'Keep this edit, its diagnostic view and any later captured normalize_qk upstream fix in the same implementation family. Resolve the upstream revision before adding that port as a supposedly independent test.',
     'exposure_rule': 'Count separate code ports as separate implementation exercises, but cap/schedule exposure by this shared fix family; do not count two conceptual discoveries.'},
    {'family_id': 'sushi-quantized-window-kv-cherry-pick',
     'relationship': 'explicit_cherry_pick', 'members': [member('6755ff2426b7')],
     'external_anchor': {'commit': '00d1132bcab3bf897f301e42103f7360d5449d6a', 'repository': None,
                         'identity_limit': 'Original repository/ref is not named by the captured message.'},
     'evidence': [evidence('6755ff2426b7', '(cherry picked from commit 00d1132bcab3bf897f301e42103f7360d5449d6a)')],
     'interpretation': 'Exact source history declares a cherry-pick. The original commit is not another inspected dataset row.',
     'split_rule': 'Union any future row carrying the original full Git SHA or a verified equivalent patch with this family.',
     'exposure_rule': 'Treat the original and cherry-pick as alternate implementations/views unless substantive extra changes are separately documented.'},
    {'family_id': 'ds4-glm-padded-prefill-row-cherry-pick',
     'relationship': 'explicit_cherry_pick',
     'members': [{'repository': 'https://github.com/antirez/ds4',
                  'commit': '582d809fd300', 'identity_limit': 'Full commit is bound by the reasoning provenance below.',
                  'views': [{'id': r['id'], 'view': r['view'], 'key': r['key']} for r in rows if r['commit'].startswith('582d809fd300')]}],
     'external_anchor': {'commit': '4e166b1fb491d81009f978569cb4efbb1a1cbefb', 'repository': None},
     'evidence': [text_evidence(HOUR/'engines-a/sourcepacks/ds4-582d809fd300-hour-v3/commit-message.txt',
        '(cherry picked from commit 4e166b1fb491d81009f978569cb4efbb1a1cbefb)',
        'https://github.com/antirez/ds4/commit/' + next(r['commit'] for r in rows if r['commit'].startswith('582d809fd300')))],
     'interpretation': 'The active dataset view is diagnostic reasoning; the overlarge/unreviewed bounded exact task remains excluded.',
     'split_rule': 'Keep future original/cherry-pick edit and reasoning views together.', 'exposure_rule': 'One fix family, not two independent bug discoveries.'},
    {'family_id': 'omlx-to-mlxserve-qsa-nax-occupancy-port',
     'relationship': 'explicit_kernel_port', 'members': [member('bfe518f77e4d')],
     'external_anchor': {'repository': 'https://github.com/jundot/omlx', 'revision_prefix': 'd6b2b92',
                         'path': 'qsa_nax.py', 'source_issue': 'https://github.com/jundot/omlx/pull/4020'},
     'evidence': [evidence('bfe518f77e4d', "This ports that kernel (qsa_nax.py @ d6b2b92):")],
     'interpretation': 'The older training row explicitly ports oMLX occupancy-tuned QSA; a future original-source row cannot supply an independent concept or family-disjoint evaluation.',
     'split_rule': 'Join the original oMLX kernel and mlx-serve adaptation after resolving the short revision to a full source identity.',
     'exposure_rule': 'Preserve engine-specific ABI/dispatch learning, but report one credited kernel-design family.'},
    {'family_id': 'sushi-mimo-f16-p-lockstep-ablation-correction',
     'relationship': 'related_implementation_stages_with_corrected_attribution',
     'members': [member('3ba7272b3305'), member('1dd27c838a07')],
     'evidence': [evidence('1dd27c838a07', 'The f16 P landed in 3ba7272. Its docs table credited the f16 term alone with -18..-25% on a global layer. That\nreading came from a kernel that also had the two changes in this commit. Measured interleaved in one process\n(qL 4096, kv8), f16 P alone buys only -3..-8%. These changes are where the rest comes from.')],
     'interpretation': 'These are distinct code changes in one implementation/ablation story. The later source corrects the earlier gain attribution; do not teach the superseded18–25% claim as the isolated FP16-P benefit.',
     'split_rule': 'Keep both edits and attribution-review reasoning in the same split.',
     'exposure_rule': 'Retain both stages when teaching numerical tradeoffs and ablation, but sample them as one related experiment family, with the correction retained.'},
    {'family_id': 'mtplx-sorted-row-map-swiglu-kernel-extension-gates',
     'relationship': 'explicit_same_kernel_extension_and_version_gate',
     'members': [member('aa68928d1e2e'), member('ae2746a50260'), member('072610551789')],
     'evidence': [evidence('072610551789', 'The tensor-unit kernel that serves Flash-Next\'s fused\ngate/up now also takes separate gate and up weights'),
                  evidence('ae2746a50260', 'min_rows() now reads the installed MLX with the row guard\'s own version\nrule: 4,096 routed rows before 0.32.3, 16,384 from 0.32.3 on.')],
     'interpretation': 'The source and direct parent chain bind original kernel→version-dependent admission threshold→separate SwitchGLU layout extension. Three useful edit tasks, one linked kernel lineage.',
     'split_rule': 'Union these three commits and their reasoning views for future splits; preserve version and workload conditions per task.',
     'exposure_rule': 'Cap repeated exposures within this family, while retaining the distinct dispatch/layout lessons.'},
    {'family_id': 'splash-live-row-tile-height-gather-trim-ablation',
     'relationship': 'companion_optimization_with_shared_ablation',
     'members': [member('d2b331112d7f'), member('dff5dfc57308')],
     'evidence': [],
     'interpretation': 'Tile-height selection and gather-extent trimming are separate edits whose reported comparison conditions depend on their combination. The trim commit directly follows the tile-height commit.',
     'split_rule': 'Keep both changes and any ablation reasoning together; a trained companion edit is exposure to the held-out implementation family.',
     'exposure_rule': 'Report two edit exercises, one related ablation family; do not sum isolated speed claims across conditions.'},
]
# Exact companion wording varies by line wrapping; preserve a complete bound paragraph.
splash = json.loads(Path(record('d2b331112d7f')['source']).read_text())['source']['head']['message']
paragraphs = splash.split('\n\n')
matching = [p for p in paragraphs if 'gather' in p.lower() and any(w in p.lower() for w in ['both', 'trim', 'arm'])]
assert matching
groups[-1]['evidence'] = [evidence('d2b331112d7f', matching[-1])]

# Fill canonical full identity for the reasoning-only cherry-pick.
for group in groups:
    for node in group['members']:
        if node.get('commit') == '582d809fd300':
            node['commit'] = next(r['commit'] for r in rows if r['commit'].startswith('582d809fd300'))
    for ev in group['evidence']:
        inputs[ev['path']] = ev['sha256']

report = {
    'schema': 'supplemental-linked-implementation-families-v1',
    'reviewer': 'quality_research', 'status': 'reviewed_supplement_not_payload_mutation',
    'scope': '57 new-core32K exact edits,97 prior-v2 training edits and69 new historical reasoning views. Prior held-out rows are excluded from this content review. Repository identities normalized to lowercase without trailing slash.',
    'inventory_summary': inventory['summary'],
    'same_repo_commit_multi_view_groups': len(inventory['same_change_views']),
    'new_exact_prior_exact_commit_duplicates': 0,
    'different_commit_same_stable_patch_groups': inventory['same_stable_patch_different_change'],
    'same_git_sha_different_repo_groups': inventory['same_git_sha_different_repo'],
    'implementation_families': groups,
    'generic_motifs_not_merged': [
        {'cases': ['mlx-789be76e27a3-hour-v3', 'tensorfold-8bb6e2857f45-hour-v3'], 'reason': 'Both gate a kernel against the actual pipeline thread limit, but operate different kernels; no explicit copied implementation or shared experiment is established.'},
        {'cases': ['mlx-lm-735a43b2754e-hour-v3', 'sushi-3adb8002840a-hour-v3-reasoning'], 'reason': 'Both discuss recurrent-state precision. Shared FP32-state policy alone does not prove this exact MLX-LM change was ported to Sushi; retain broad motif metadata without declaring duplicate code.'},
        {'cases': ['omlx-b6026ce26468-hour-v3', 'mlx-vlm-868596185d3f-hour-v3'], 'reason': 'Both concern attention-mask semantics but repair different conditions and call sites; no source-grounded port link.'},
        {'cases': ['mlx-9cb9c06fb2c8-hour-v3', 'mlx-0840c42a7aab-hour-v3'], 'reason': 'Related large-array reduction overflow motif, but distinct indexing/dispatch failures. Do not merge solely by repository or title.'},
    ],
    'recommendations': [
        'Use repo+full-commit for change identity, stable-patch identity for exact/cherry-pick duplicates, and this explicit linked-family overlay for related implementations. Keep these three counts distinct.',
        'Hard-union exact duplicates, declared ports/cherry-picks and the identified shared implementation/ablation families before creating future train/dev/test splits. Resolve unspecified external revisions when those originals are actually added.',
        'Keep exact edits, diagnostic reasoning, rejection/tradeoff tasks and corrected evidence about a change in the same split. Do not count task views as independent examples of generalization.',
        'Maintain separate motif tags for generic ideas such as wider accumulators, shape guards, fusion and cache reuse. Do not collapse all MLX-dependent kernels into one implementation family.',
        'Sample one or a capped number of views per change/family per epoch; alternate ports or diagnostic views across epochs. Preserve source-group protections in addition to these narrower families.',
        'Do not rewrite released rows or claim this overlay retroactively made earlier evaluations family-disjoint. This is a supplemental grouping recommendation for future datasets and honest reporting.'
    ],
    'limitations': [
        'The port relationships are historical source claims and exact source citations, not independent execution results.',
        'Stable patch identity does not catch renamed/reimplemented ports; absence of collisions is not absence of semantic overlap.',
        'Literal reference checks and sampled source comparisons are finite; this ledger is not an exhaustive provenance or originality proof.',
        'The167 normalized repo/commit changes are not167 independent algorithms or optimization concepts.'
    ],
    'inputs_sha256': inputs,
}
for path, digest in inputs.items():
    assert sha(Path(path)) == digest, path
report['builder_sha256'] = sha(Path(__file__))
(ROOT/'implementation-family-ledger.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
print('PASS',len(groups),'supported families',len(inputs),'hash-bound inputs')
