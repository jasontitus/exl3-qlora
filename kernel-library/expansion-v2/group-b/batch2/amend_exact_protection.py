"""Fail-closed CPU revision publisher: exact identity gate precedes manual review.

Preserves the original held manifest, source, raw rows and receipts. Writes v2
outputs with exclusive creation; source/manual clearance cannot override a match.
"""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
from exact_protection import reserved_index, identity_matches, review_after_identity_gate

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[3]
H = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
D = lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def exclusive(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write('\n')


def main():
    evidence = ROOT / 'evidence/reserved-metadata-exact.json'
    origin = WORK / 'kernel-library/integration-v1/group-a/evidence/reserved-metadata.json'
    if evidence.exists():
        if H(evidence) != H(origin):
            raise ValueError('Existing reserved metadata differs')
    else:
        shutil.copyfile(origin, evidence)
    manifest_path = ROOT / 'manifest-reviewed.json'
    manifest = json.loads(manifest_path.read_text())
    index = reserved_index(json.loads(evidence.read_text()))
    scan, kept, removed, keys = [], [], [], set(manifest['protected_keys'])
    for candidate in manifest['candidates']:
        decision = review_after_identity_gate(candidate, index, lambda c: {'status': 'identity_clear_manual_review_retained'})
        hits = identity_matches(candidate, index)
        scan.append({'case_id': candidate['id'], 'commit': candidate['commit'], 'parent': candidate['parent'], 'related_commits': candidate.get('related_commits', []), 'decision': decision})
        if hits:
            record = json.loads((ROOT / 'drafts' / (candidate['id'] + '.source.json')).read_text())
            removed.append(candidate['id'])
            keys.update('commit:' + x['sha'] for x in hits)
            keys.add('stable_patch:' + record['stable_patch_id'])
        else:
            kept.append(candidate)
    if removed != ['mlx-bbf8a7a54a44-expansion-v2']:
        raise ValueError('Unexpected exact protection results; review before publication')
    blocker = WORK / 'kernel-library/expansion-v2/exact-protection-audit-v2-blocker.json'
    correction = {'format': 'exact-protected-admission-correction-v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'original_manifest_sha256': H(manifest_path), 'original_seal_sha256': H(ROOT/'sealed-batch.json'), 'metadata': str(evidence.relative_to(ROOT)), 'metadata_sha256': H(evidence), 'independent_blocker_receipt': str(blocker), 'independent_blocker_sha256': H(blocker), 'rule': 'Exact reserved head/base/related identity has priority over manual finite operation clearance. Protected candidates are removed before review/admission, and keys prevent re-admission by alias.', 'gate_source_sha256': H(ROOT/'exact_protection.py'), 'gate_tests_sha256': H(ROOT/'tests/test_exact_protection.py'), 'gate_tests_log_sha256': H(ROOT/'exact-protection-tests.log'), 'scan': scan, 'removed': removed, 'supersedes_claim': 'Earlier source-only manual review missed this exact reserved identity; it does not authorize admission.', 'scope': 'Exact identity gate only; semantic and ancestry review remain necessary for nonmatches.'}
    cp = ROOT/'exact-protection-correction-v2.json'
    exclusive(cp, correction)
    manifest['candidates'] = kept
    manifest['protected_keys'] = sorted(keys)
    for candidate in kept:
        candidate['historical_protection_review']['evidence_sha256'][cp.name] = H(cp)
        candidate['historical_protection_review']['evidence_sha256'][str(evidence.relative_to(ROOT))] = H(evidence)
    manifest['protection_amendment'] = {'supersedes_manifest_sha256': H(manifest_path), 'correction': cp.name, 'sha256': H(cp)}
    manifest.pop('manifest_sha256', None)
    manifest['manifest_sha256'] = hashlib.sha256(D(manifest)).hexdigest()
    exclusive(ROOT/'manifest-reviewed-v2.json', manifest)
    docs = json.loads((ROOT/'documentation-omissions.json').read_text())
    for case in removed:
        docs.pop(case, None)
    exclusive(ROOT/'documentation-omissions-v2.json', docs)
    report = json.loads((ROOT/'integration-report.json').read_text())
    for label, status in [('12k', {'admitted_12k'}), ('32k', {'admitted_12k', 'admitted_32k_only'})]:
        cases = [c for c in report['cases'] if c['status'] in status]
        for kind in ['raw', 'prepared']:
            old = [json.loads(x) for x in (ROOT/f'train-{label}.{kind}.jsonl').read_text().splitlines()]
            if len(old) != len(cases):
                raise ValueError('Published row count mismatch')
            values = []
            for case, row in zip(cases, old):
                expected = json.loads((ROOT/'drafts'/f"{case['case_id']}.{kind}.json").read_text())
                if row != expected:
                    raise ValueError('Published row order/payload differs from bound case')
                if case['case_id'] not in removed:
                    values.append(row)
            with (ROOT/f'train-{label}-v2.{kind}.jsonl').open('xb') as f:
                f.write(b''.join(D(x)+b'\n' for x in values))
    paths = ['manifest-reviewed-v2.json', 'documentation-omissions-v2.json', cp.name, str(evidence.relative_to(ROOT)), 'exact_protection.py', 'amend_exact_protection.py', 'tests/test_exact_protection.py', 'exact-protection-tests.log', *[f'train-{label}-v2.{kind}.jsonl' for label in ['12k','32k'] for kind in ['raw','prepared']]]
    # Check every old sealed file before declaring an immutable revision.
    original = json.loads((ROOT/'sealed-batch.json').read_text())
    if not all(H(ROOT/p) == digest for p,digest in original['files_sha256'].items()):
        raise ValueError('An original held file changed')
    exclusive(ROOT/'sealed-batch-v2.json', {'format': 'exact-protection-amended-batch-v1', 'status': 'held', 'positive_12k': 3, 'positive_32k': 5, 'removed': removed, 'exact_reserved_identities_checked': len(index), 'candidate_scan_count': len(scan), 'original_held_files_unchanged': True, 'original_manifest_sha256': H(manifest_path), 'files_sha256': {p:H(ROOT/p) for p in paths}})
    print(json.dumps({'manifest_sha256':H(ROOT/'manifest-reviewed-v2.json'),'docs_sha256':H(ROOT/'documentation-omissions-v2.json'),'seal_sha256':H(ROOT/'sealed-batch-v2.json'),'removed':removed,'counts':{'12k':3,'32k':5},'reserved_identities':len(index)}, indent=2))


if __name__ == '__main__':
    main()
