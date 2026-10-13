"""Seal reviewed data assets, without launching training or changing old datasets."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    review = read('final-aggregate-independent-review.json')
    require(review['status'] == 'passed_scoped_preparation_audit', 'Independent review did not pass')
    require(not review['blockers'], 'Independent review has blockers')
    for name, digest in review['inputs_sha256'].items():
        require(sha(Path(name)) == digest, 'Stale independent review: '+name)
    require(review['reasoning']['records'] == 69 and review['reasoning']['supervised_tokens'] == 9023,
            'Reasoning count mismatch')
    require(review['preferences']['pairs'] == 25, 'Preference count mismatch')
    require(read('reasoning-library-v1/audit.json')['records'] == 69, 'Reasoning audit mismatch')
    require(read('preference-library-v1/audit.json')['pairs'] == 25, 'Preference audit mismatch')
    optional_rows = (ROOT/'optional-unmeasured-32768/train.jsonl').read_text().splitlines()
    require(len(optional_rows) == 10, 'Optional archive count mismatch')
    exact = {}
    for capacity in (12288, 32768):
        directory = f'corpus-{capacity}-v1'
        audit = read(directory + '/audit.json')
        for name, digest in audit['artifacts_sha256'].items():
            require(sha(ROOT / directory / name) == digest, 'Changed exact artifact: '+name)
        require(audit['train_examples'] == review['exact'][str(capacity)]['train'], 'Exact count mismatch')
        require(audit['test_examples'] == review['exact'][str(capacity)]['test'], 'Heldout count mismatch')
        exact[str(capacity)] = {
            k: audit[k] for k in ('train_examples', 'test_examples', 'by_repository',
                                  'by_curriculum_tier', 'by_origin', 'supervised_tokens',
                                  'max_serialized_tokens', 'truncated')
        }
        exact[str(capacity)].update(directory=directory, prepared_train=directory+'/train.jsonl',
                                   raw_train=directory+'/train.raw.jsonl')
    card = {
        'schema': 'metal-optimization-dataset-card-v3',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'purpose': 'Training agents to diagnose, edit, and verify Apple Metal inference and training kernels, including host dispatch and numerical correctness repairs.',
        'exact_edit_datasets': exact,
        'capacity_relationship': 'All 111 examples at 12K are identical rows within the 154-example 32K set. Capacities are preparation limits, not measured training capability.',
        'reasoning': {'records':69, 'distinct_repo_commits':64, 'supervised_tokens':9023,
                      'prepared_train':'reasoning-library-v1/prepared-12288/train.jsonl',
                      'raw_train':'reasoning-library-v1/train.messages.jsonl',
                      'scope':'Evidence-grounded historical diagnosis and verification planning; source statements and inferred explanations distinguished. Not observed agent rollouts.'},
        'preferences': {'pairs':25, 'distinct_repo_commits':20,
                        'path':'preference-library-v1/preferences.jsonl',
                        'scope':'Reviewed curator-authored chosen/rejected diagnostic contrasts. Rejected answers never included in positive SFT. Current Apple trainer has no qualified DPO implementation; optional future asset.'},
        'optional_unmeasured': {'records':10, 'directory':'optional-unmeasured-32768',
                               'default_included':False, 'scope':'Complete edits without established measured performance benefit; opt-in only.'},
        'format': 'Exact historical edits represented as native agent tool calls and messages, with complete retained edit targets, source snapshots, build/host context and replay receipts. Raw transport and prepared tokenizer-specific files are separate.',
        'counting': {'exact_and_reasoning_unique_repo_commits':167,
                     'warning':'Counts are changes and task views, not independent discoveries. Linked ports, cherry-picks, chains and repeated reasoning views are recorded in linked-changes/implementation-family-ledger.json.'},
        'source_coverage': 'All twelve requested repositories are tracked. Eleven contribute core training edits; llama.cpp remains the protected held-out source. This is bounded reviewed expansion, not exhaustive mining of all history or every mlx.fast operation.',
        'validation': {
            'preparation':'Source/revision/hash binding, exact edit replay, tokenization/loss masks, protected split screening and independent aggregate review passed.',
            'heldout':'All 16 historical held-out token sequences and loss masks preserved exactly.',
            'tests':'96 helper tests passed; final-helper-tests.log records the run.',
            'compiler':'Subset only: see compiler-probes receipts for exact files/configurations and failures. Compiler acceptance is not GPU pipeline or runtime qualification.',
            'analytical':'20 NumPy assertions corroborate three numerical reasoning claims; not execution of upstream engines.',
            'runtime':'No new GPU training, numerical kernel parity or performance campaign executed in this hour. Author-reported speedups remain author-reported.'},
        'curriculum': 'Use curriculum-design.json: group exposure by change/implementation family, compare exact-only with reasoning-enriched SFT under matched budgets, reserve fresh tasks for reasoning evaluation. Do not pool negative answers into SFT or interpret comment/header NLL as kernel improvement.',
        'limitations': [
            'Not every example is a standalone MSL kernel or a demonstrated speedup; explicit tiers include correctness and measured tradeoffs.',
            'Reasoning and preference examples are training material, not a held-out reasoning evaluation.',
            'Many diagnostic prompts expose historical outcomes or explanations. They train evidence interpretation and verification planning, not proof of discovering an unseen optimization from baseline-only context.',
            'Prepared IDs bind the existing DeepSeek tokenizer. Other models require fresh templating and tokenization.',
            'Per-source licenses/notices are retained in provenance; the collection does not grant a blanket replacement license.',
            'Useful-model claims require generated held-out edits, compilation, numerical parity and workload-specific performance checks.'
        ]
    }
    write(ROOT/'dataset-card.json', card)
    files = set()
    for directory in ['corpus-12288-v1','corpus-32768-v1','reasoning-library-v1','preference-library-v1','optional-unmeasured-32768']:
        files.update(p for p in (ROOT/directory).rglob('*') if p.is_file())
    files.update(ROOT/p for p in ['dataset-card.json','candidate-inventory.json','curriculum-design.json',
                                'final-aggregate-independent-review.json','final-helper-tests.log','validation-index.json','complete-item-capacities.json',
                                'linked-changes/implementation-family-ledger.json','linked-changes/ledger-selfcheck.json',
                                'engines-a/history-coverage-final.json','engines-a/integration-report.json',
                                'engines-b/integration-final-reviewed-v2.json','mlx/sealed-lane.json'])
    for directory in ['compiler-probes','analytical-probes']:
        files.update((ROOT/directory).glob('*.json'))
        files.update((ROOT/directory).glob('*.py'))
    files.update((ROOT/'tools').glob('*.py'))
    if (ROOT/'release-independent-review.json').exists():
        files.add(ROOT/'release-independent-review.json')
    release = {'schema':'metal-optimization-library-release-v3', 'created_utc':card['created_utc'],
               'dataset_card':'dataset-card.json', 'primary_corpus':'corpus-12288-v1',
               'archive_corpus':'corpus-32768-v1', 'train_count':111, 'archive_train_count':154,
               'new_exact_12k':42,'new_exact_32k':57,'reasoning_count':69,'preference_pairs':25,
               'qualification':'Independently reviewed preparation and provenance; compiler subset only; runtime/model quality unqualified.',
               'files_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}}
    write(ROOT/'release.json',release)
    print(json.dumps({k:v for k,v in release.items() if k!='files_sha256'},indent=2))


if __name__ == '__main__':
    main()
