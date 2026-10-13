"""Fail closed on captured reserved head/base identities before preparation."""
import hashlib,json,re
from pathlib import Path
SHA=re.compile(r'[0-9a-f]{40}\Z')
def reserved_identities(metadata):
    coverage=metadata['review']['coverage']
    if not isinstance(coverage,list) or not coverage:raise ValueError('Missing reserved coverage')
    result={}
    for row in coverage:
        for field in ('head','base'):
            value=row.get(field)
            if value is not None:
                if not isinstance(value,str) or not SHA.fullmatch(value):raise ValueError('Invalid reserved commit')
                result.setdefault(value,[]).append({'id':row['id'],'field':field})
    if not result:raise ValueError('Empty reserved identities')
    return result

def check_manifests(paths,metadata_path,expected_sha256=None):
    metadata_path=Path(metadata_path);raw=metadata_path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Reserved metadata checksum mismatch')
    reserved=reserved_identities(json.loads(raw));hits=[];checked=0;manifest_hashes={}
    for path in paths:
        path=Path(path).resolve();content=path.read_bytes();manifest_hashes[str(path)]=hashlib.sha256(content).hexdigest();manifest=json.loads(content)
        for row in manifest['candidates']:
            if row.get('outcome') not in {'reported_improvement','verified_improvement'} or row.get('task_kind',{}).get('value') not in {'kernel','host_dispatch'}:continue
            checked+=1
            if any(not isinstance(row.get(f),str) or not SHA.fullmatch(row[f]) for f in ('commit','parent')):raise ValueError('Invalid candidate Git identity')
            for field in ('commit','parent'):
                if row.get(field) in reserved:hits.append({'id':row['id'],'field':field,'sha':row[field],'reserved':reserved[row[field]]})
    report={'metadata':str(metadata_path.resolve()),'metadata_sha256':hashlib.sha256(raw).hexdigest(),'manifests_sha256':manifest_hashes,'helper_path':str(Path(__file__).resolve()),'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reserved_identities':len(reserved),'positive_candidates_checked':checked,'matches':hits,'scope':'Exact positive training head/parent identity only; semantic lineage, ports and unknown reservations still require independent review.'}
    if hits:raise ValueError('Reserved exact identity in positive training candidates: '+json.dumps(hits))
    return report

def verify_snapshot(report):
    files={**report['manifests_sha256'],report['metadata']:report['metadata_sha256'],report['helper_path']:report['helper_sha256']}
    for path,expected in files.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Reserved identity preflight input drift: '+path)
