"""Audit/export evidence-grounded diagnostic SFT; never execute source/tool text.

Different task views of a change share one lineage. This is an evidence and
serialization gate, not an automatic truth/semantic-leakage certificate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]
RESERVED = WORK / 'kernel-library/integration-v1/group-a/evidence/reserved-metadata.json'
RESERVED_SHA = 'f4eef3821f220de8903d19de9452bdd524648215cc99c0af8ee3106b48aab92b'
FROZEN_TEST = WORK / 'kernel-library/expansion-v2/corpus-12288-v3'
ALLOWED_REPOS = {('https://github.com/'+row['repository']).lower() for row in json.loads((WORK/'kernel-library/sources.json').read_text())['repositories']} - {'https://github.com/ggml-org/llama.cpp'}
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
TASKS = {'diagnose', 'select_dispatch', 'interpret_benchmark', 'review_correctness',
         'plan_validation', 'explain_change', 'repair_from_feedback', 'reject_optimization'}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name}: nonempty string required')
    if len(value.encode('utf-8')) > 1024*1024:
        raise ValueError(f'{name}: exceeds1MiB')
    if any(marker in value for marker in ('<｜', '</｜', '｜DSML｜')):
        raise ValueError(f'{name}: reserved tokenizer delimiter')
    return value


def reserved_ids(path=RESERVED, expected_sha=RESERVED_SHA):
    raw = Path(path).read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError('Reserved identity ledger changed')
    rows = json.loads(raw)['review']['coverage']
    identities = {r[k] for r in rows for k in ('head', 'base') if r.get(k)}
    if not identities or any(not HEX40.fullmatch(v) for v in identities):
        raise ValueError('Invalid reserved identities')
    for provenance in FROZEN_TEST.glob('*.provenance.json'):
        row=json.loads(provenance.read_text())
        if row['split']=='test':
            source=row['source']
            for value in source['commits']:
                if not isinstance(value,str) or not HEX40.fullmatch(value):raise ValueError('Invalid heldout Git identity')
                identities.add(value)
    return identities


def evidence_text(item):
    path = Path(item['path']).resolve()
    # Evidence files belong to workspace; never read model weights/secrets via record paths.
    if not path.is_relative_to(WORK / 'kernel-library'):
        raise ValueError('Evidence outside kernel-library')
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError('Evidence file exceeds16MiB')
    if not HEX64.fullmatch(item.get('sha256', '')) or sha(raw) != item['sha256']:
        raise ValueError('Evidence hash mismatch')
    body = raw.decode('utf-8')
    # Optional JSON pointer reads decoded strings rather than matching JSON escapes.
    if 'json_pointer' in item:
        pointer = item['json_pointer']
        if not isinstance(pointer,str) or not pointer.startswith('/'):
            raise ValueError('Invalid JSON pointer')
        value = json.loads(body)
        for part in pointer[1:].split('/'):
            part = part.replace('~1','/').replace('~0','~')
            value = value[int(part)] if isinstance(value,list) else value[part]
        if not isinstance(value,str):
            raise ValueError('Evidence pointer must select text')
        body=value
    quote = text(item.get('quote'), 'evidence quote')
    if quote not in body:
        raise ValueError('Evidence quote absent')
    url=text(item.get('url'),'evidence URL')
    if not url.startswith('https://'):
        raise ValueError('Evidence needs public HTTPS source URL')
    return path, sha(raw)


def validate(record, reserved):
    for name in ('id','repo','lineage','prompt','completion'):
        text(record.get(name),name)
    if record['repo'].lower().rstrip('/') not in ALLOWED_REPOS:
        raise ValueError('Unapproved or held-out repository')
    for name in ('commit','parent'):
        if not isinstance(record.get(name),str) or not HEX40.fullmatch(record[name]):
            raise ValueError('Invalid Git identity')
        if record[name] in reserved:
            raise ValueError('Protected identity: '+record[name])
    if record.get('task_type') not in TASKS:
        raise ValueError('Unsupported reasoning task')
    if record.get('rationale_basis') not in {'source_reported','inferred','mixed'}:
        raise ValueError('Declare source/inference basis')
    if record.get('outcome_class') not in {'positive','tradeoff','correctness','rejected','neutral','unknown'}:
        raise ValueError('Declare outcome class')
    review=record.get('review',{})
    if review.get('status')!='approved' or not review.get('reviewer'):
        raise ValueError('Explicit human/agent semantic review required')
    for field in ('evidence_fidelity','no_fabricated_observations','task_answer_separation','lineage_screened'):
        if review.get(field) is not True:
            raise ValueError('Missing review check: '+field)
    protection=record.get('protection_review',{})
    if protection.get('status')!='cleared' or not protection.get('reason'):
        raise ValueError('Semantic protection review required; exact IDs alone insufficient')
    if not record.get('source_evidence') or len(record['source_evidence'])>32:
        raise ValueError('Require1..32evidence spans')
    evidence={}
    for item in record['source_evidence']:
        path,digest=evidence_text(item);evidence[str(path)]=digest
    if record.get('validation',{}).get('local_execution') not in {'not_run','compile_only','numerical','performance'}:
        raise ValueError('Explicit local execution tier required')
    # A prose claim of local success must never be inferred from historical author results.
    if record['validation']['local_execution']!='not_run':
        receipt=record['validation'].get('receipt')
        if not isinstance(receipt,dict) or set(receipt)!={'path','sha256'}:
            raise ValueError('Local execution tier requires receipt with hash')
        rp=Path(receipt['path']).resolve()
        if not rp.is_relative_to(WORK/'kernel-library') or sha(rp.read_bytes())!=receipt['sha256']:
            raise ValueError('Local receipt identity mismatch')
        evidence[str(rp)]=receipt['sha256']
        # Bound receipt only: semantic validation status remains the reviewers responsibility.
    licenses=record.get('licenses',[])
    if not licenses:
        raise ValueError('License provenance required')
    for item in licenses:
        path=Path(item['path']).resolve()
        if not path.is_relative_to(WORK/'kernel-library') or sha(path.read_bytes())!=item['sha256']:
            raise ValueError('License identity mismatch')
        evidence[str(path)]=item['sha256']
    return evidence


def export(paths, output):
    output=Path(output)
    if output.exists():
        raise ValueError('Refuse overwrite export')
    reserved=reserved_ids(); records=[]; hashes={str(RESERVED):RESERVED_SHA,str(WORK/'kernel-library/sources.json'):sha((WORK/'kernel-library/sources.json').read_bytes())};
    for provenance in FROZEN_TEST.glob('*.provenance.json'):
        if json.loads(provenance.read_text())['split']=='test':hashes[str(provenance)]=sha(provenance.read_bytes())
    seen=set(); seen_payloads=set()
    for name in paths:
        path=Path(name).resolve();raw=path.read_bytes();hashes[str(path)]=sha(raw)
        value=json.loads(raw);rows=value if isinstance(value,list) else value['records']
        for row in rows:
            if row['id'] in seen:raise ValueError('Duplicate record ID')
            seen.add(row['id']);hashes.update(validate(row,reserved))
            payload=sha(json.dumps([row['prompt'],row['completion']],ensure_ascii=False).encode())
            if payload in seen_payloads:raise ValueError('Duplicate prompt/completion')
            seen_payloads.add(payload);records.append(row)
    # Snapshot verification after all validation, before publishing any outputs.
    for path,digest in hashes.items():
        if sha(Path(path).read_bytes())!=digest:raise ValueError('Input drift')
    output.mkdir(parents=True)
    def write_jsonl(name,rows):
        (output/name).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows))
    write_jsonl('train.prompt-completion.jsonl',[{'prompt':r['prompt'],'completion':r['completion']} for r in records])
    write_jsonl('train.messages.jsonl',[{'messages':[{'role':'user','content':r['prompt']},{'role':'assistant','content':r['completion']}]} for r in records])
    write_jsonl('provenance.jsonl',records)
    report={'records':len(records),'distinct_commits':len({(r['repo'],r['commit']) for r in records}),
            'distinct_lineages':len({r['lineage'] for r in records}), 'inputs_sha256':hashes,
            'qualification':'Reviewed historical diagnostic SFT; not recorded agent rollouts, not locally benchmark-qualified. Keep all related views in same split; default one sampled view per lineage per epoch.',
            'protected_exact_identities_checked':len(reserved)}
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',action='append',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(export(a.records,a.output),indent=2))
