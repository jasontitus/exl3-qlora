"""CPU token comparison. Provenance/launch certification is a separate required gate."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def summarize(values):
    n=len(values)
    b=math.fsum(x[0] for x in values);a=math.fsum(x[1] for x in values)
    return dict(tokens=n,base_nll_sum=b,adapter_nll_sum=a,base_nll=b/n,adapter_nll=a/n,
                nll_delta=(a-b)/n,base_correct=sum(x[2] for x in values),
                adapter_correct=sum(x[3] for x in values),
                corrected_tokens=sum(not x[2] and x[3] for x in values),
                regressed_tokens=sum(x[2] and not x[3] for x in values))


def compare(prepared,categories,base,adapter):
    require(len(prepared)==len(categories['rows'])==len(base['rows'])==len(adapter['rows'])==16,'Require all16 rows')
    vocab=categories['prediction_vocab_size'];tokenizer_size=categories['tokenizer_size']
    require(type(vocab) is int and vocab>0 and type(tokenizer_size) is int and tokenizer_size>0,'Invalid vocabulary dimensions')
    for p in prepared:
        require(all(type(v) is int and 0<=v<min(vocab,tokenizer_size) for v in p['input_ids']),'Input token outside vocabulary')
    selection=hashlib.sha256(canonical(prepared)).hexdigest()
    for report in (base,adapter):
        require(report['evaluation_context']['selected_examples_sha256']==selection,'Selected full sequences differ')
        require(report['examples']==16,'Wrong example count')
        require(report['evaluation_context']['seq_length']>=max(len(p['input_ids'])-1 for p in prepared),'Context truncation')
    groups=defaultdict(list);per_row=[]
    for i,(p,c,b,a) in enumerate(zip(prepared,categories['rows'],base['rows'],adapter['rows'],strict=True)):
        require(c['index']==i and c['loss_start']==p['loss_start'],'Category row/mask mismatch')
        ids_hash=hashlib.sha256(json.dumps(p['input_ids'],separators=(',',':')).encode()).hexdigest()
        require(c['input_ids_sha256']==ids_hash,'Category sequence mismatch')
        targets=p['input_ids'][p['loss_start']:];cats=c['categories']
        require(len(targets)==len(cats)==c['supervised_tokens'] and len(targets)>0,'Incomplete labels')
        for r in (b,a):
            require(r['target_ids']==targets,'Target tokens differ')
            require(len(r['token_nll'])==len(r['predicted_ids'])==len(targets),'Incomplete token metrics')
            require(all(type(v) in (float,int) and math.isfinite(v) and v>=-1e-6 for v in r['token_nll']),'Invalid NLL')
            require(all(type(v) is int and 0<=v<vocab for v in r['predicted_ids']),'Invalid predicted IDs')
        local=defaultdict(list)
        for j,(target,cat) in enumerate(zip(targets,cats,strict=True)):
            require(isinstance(cat,str) and len(cat.split('/'))==3,'Malformed category')
            value=(b['token_nll'][j],a['token_nll'][j],b['predicted_ids'][j]==target,a['predicted_ids'][j]==target)
            for label in ('all',cat,'family:'+cat.split('/')[0]):
                local[label].append(value);groups[label].append(value)
        result={k:summarize(v) for k,v in local.items()}
        for r,prefix in ((b,'base'),(a,'adapter')):
            require(r['correct_tokens']==result['all'][prefix+'_correct'],'Correct count mismatch')
            require(math.isfinite(r['loss']) and abs(r['loss']-result['all'][prefix+'_nll'])<1e-6,'Row NLL mismatch')
        per_row.append({'index':i,'metrics':result})
    total={k:summarize(v) for k,v in groups.items()}
    require(dict(sorted((k,len(v)) for k,v in groups.items() if k!='all' and not k.startswith('family:')))==categories['counts'],'Category counts mismatch')
    for r,prefix in ((base,'base'),(adapter,'adapter')):
        require(r['supervised_tokens']==total['all']['tokens'],'Global token count mismatch')
        require(math.isfinite(r['loss']) and abs(r['loss']-total['all'][prefix+'_nll'])<1e-5,'Global token-weighted loss mismatch')
    return {'schema':'heldout-pair-token-metrics-v1','status':'metrics_consistent_pending_provenance_audit',
            'aggregate':total,'rows':per_row,'quality_certified':False,
            'limits':['Teacher-forced metrics only, not generation quality or inference speed',
                      'Requires separate exact-base, runtime, completed-launch and checkpoint audit',
                      'Categories use lexical historical labels, not semantic code correctness']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('prepared','categories','base','adapter','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();paths={k:Path(getattr(args,k)) for k in ('prepared','categories','base','adapter')}
    captured={k:p.read_bytes() for k,p in paths.items()}
    prepared=[json.loads(v) for v in captured['prepared'].splitlines() if v.strip()]
    result=compare(prepared,*(json.loads(captured[k]) for k in ('categories','base','adapter')))
    result['inputs_sha256']={str(paths[k].resolve()):hashlib.sha256(data).hexdigest() for k,data in captured.items()}
    result['implementation_sha256']=digest(__file__)
    with Path(args.output).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'tokens':result['aggregate']['all']['tokens']}))

if __name__=='__main__':main()
