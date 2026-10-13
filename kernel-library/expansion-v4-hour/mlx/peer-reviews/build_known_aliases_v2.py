"""Derive conservative selection aliases only from pinned reviewed source-family links."""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[3]
ROOT = WORK / 'kernel-library/expansion-v4-hour'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
inputs, families, unknown = {}, [], []

def bind(path, expected=None):
    path = Path(path).resolve(); actual = sha(path)
    if expected is not None and actual != expected: raise ValueError('Input drift: '+str(path))
    if str(path) in inputs and inputs[str(path)] != actual: raise ValueError('Conflicting binding')
    inputs[str(path)] = actual
    return path

def member(m):
    repo=m.get('repo',m.get('repository'));commit=m.get('commit')
    if not isinstance(repo,str) or not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?',repo) or not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):return None
    return {'repo':repo.lower().rstrip('/'),'commit':commit}

def evidence(e):
    p=bind(e['path'],e['sha256']);text=p.read_text()
    if e.get('json_pointer'):
        value=json.loads(text)
        for part in e['json_pointer'].strip('/').split('/'):value=value[part.replace('~1','/').replace('~0','~')]
        text=value
    if e.get('json_field'):
        value=json.loads(text)
        for part in e['json_field'].split('.'):
            value=value[part]
        text=value
    if e.get('quote') and e['quote'] not in text:raise ValueError('Quote not bound: '+str(p))

for rel,key in [('expansion-v3-hour/linked-changes/implementation-family-ledger.json','implementation_families'),('expansion-v4-hour/engines-a/linked-change-ledger-v2.json','groups')]:
    p=bind(WORK/'kernel-library'/rel);d=json.loads(p.read_text())
    for path,expected in d.get('inputs_sha256',{}).items():bind(path,expected)
    for group in d[key]:
        members=[];gid=group.get('family_id',group.get('id'))
        for m in group['members']:
            clean=member(m)
            if clean:members.append(clean)
            else:unknown.append({'source_group':gid,'anchor':m,'reason':'Missing verified repository or complete revision; not used as a selection alias.'})
        if group.get('external_anchor'):
            a=group['external_anchor'];clean=member(a)
            if clean:members.append(clean)
            else:unknown.append({'source_group':gid,'anchor':a,'reason':'Incomplete external identity retained as annotation; no keyword or short-SHA merge.'})
        for e in group.get('evidence',[]):evidence(e)
        if members:families.append({'label':gid,'members':members,'ledger':str(p),'relation':group.get('relationship',group.get('relation'))})
# Explicit PR4572 says it applies PR4567 to quantized GEMM; both full merge
# identities and PR association are retained by this lane's captured inventory.
invp=bind(ROOT/'mlx/inventory-all.json');rows=json.loads(invp.read_text())['records'];pair=[r for r in rows if r['commit'].startswith(('a2a09fd5','90089440'))]
assert len(pair)==2
ms=[]
for r in pair:
    bp=bind(ROOT/'mlx'/r['pr']['body_path']);assert '4567' in bp.read_text() or 'Do the same for gather_qmm.' in bp.read_text()
    ms.append({'repo':'https://github.com/ml-explore/mlx','commit':r['commit']})
families.append({'label':'mlx-dense-quantized-expert-boundary-scheduler','members':ms,'ledger':str(invp),'relation':'explicit_PR4572_extension_of_PR4567'})
# A credited MLX pattern has no pinned upstream revision: do not guess one.
p=bind(ROOT/'engines-b/evidence/tensorfold-a826eb7b31ba-message.txt')
assert "Strip the address space as MLX's steel NAX kernels do." in p.read_text()
unknown.append({'source_group':'tensorfold-a826-nax-address-space-port','known_member':{'repo':'https://github.com/ashhart/tensorfold','commit':'a826eb7b31ba290663453e833463b06233843726'},'anchor':{'repo':'https://github.com/ml-explore/mlx','commit':None,'symbol':'steel NAX kernels address-space stripping'},'reason':'Explicit source credit but no exact upstream revision; not merged with another MLX change by keywords.','source_bindings':{str(p):sha(p)}})
# Union only intersecting explicit member sets, retaining all supporting labels.
components=[]
for g in families:
    keys={(m['repo'],m['commit'])for m in g['members']};labels={g['label']};ledgers={g['ledger']};relations={g['relation']}
    for old in list(components):
        if keys & old['keys']:
            keys |=old['keys'];labels|=old['labels'];ledgers|=old['ledgers'];relations|=old['relations'];components.remove(old)
    components.append(dict(keys=keys,labels=labels,ledgers=ledgers,relations=relations))
groups=[]
for g in components:
    members=[{'repo':r,'commit':c}for r,c in sorted(g['keys'])]
    stable=hashlib.sha256(json.dumps(members,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
    groups.append({'id':'known-source-family-'+stable,'members':members,'source_labels':sorted(g['labels']),'relations':sorted(g['relations']),'source_bindings':{p:inputs[p]for p in sorted(g['ledgers'])}})
groups.sort(key=lambda g:g['id'])
assert len({(m['repo'],m['commit']) for g in groups for m in g['members']})==sum(len(g['members'])for g in groups)
for p,h in inputs.items():assert sha(Path(p))==h
out={'schema':'diagnostic-known-family-aliases-v1','status':'reviewed_explicit_source_links_only','reviewer':'adversarial_code','groups':groups,'unknown_links':unknown,'inputs_sha256':inputs,'builder_sha256':sha(Path(__file__)),'scope':'Selection aliases for known explicit source families, not an exhaustive semantic deduplication or independence proof. GDN is a reviewed shared precision contract, not proven identical patches or ancestry. Distinct complete edit exercises remain distinct commits. No payload/split mutation. Unknown upstream identities remain annotations only.','counts':{'groups':len(groups),'members':sum(len(g['members'])for g in groups),'multi_member_groups':sum(len(g['members'])>1 for g in groups),'unresolved_links':len(unknown)}}
p=HERE/'known-family-aliases-v2.json';p.write_text(json.dumps(out,indent=2)+'\n');print(p,sha(p),out['counts'])
