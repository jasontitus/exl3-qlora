"""CPU-only supplemental review reporting; original paired metrics remain complete."""
import copy
import hashlib
import json

import pytest

from test_evaluation_audit import audit, checksum, fixture


def review_fixture(data, raw_hash='b' * 64):
    rows=[]
    for i,p in enumerate(data[0]):
        rows.append({'index':i,'case_id':f'case-{i}',
                     'source_review_status':('requires_reference_contract_resolution' if i==1 else 'approved_source_scoped'),
                     'prepared_identity':checksum(p),
                     'supervised_tokens':len(p['input_ids'])-p['loss_start'],
                     'limits_and_findings':'Source-only bounds; no device execution.',
                     'kernel_quality_certified':False})
    counts={s:sum(r['source_review_status']==s for r in rows) for s in {r['source_review_status'] for r in rows}}
    tokens={s:sum(r['supervised_tokens'] for r in rows if r['source_review_status']==s) for s in counts}
    return {'format':'frozen-heldout-source-reference-audit-v1','heldout_sha256':raw_hash,
            'review_sha256':'c'*64,'complete_rows':len(rows),'rows':rows,
            'counts_by_source_status':counts,'tokens_by_source_status':tokens,
            'supervised_tokens':sum(tokens.values())}


def test_complete_primary_unchanged_and_strata_token_weighted():
    data=fixture()
    result=audit.compare_token_evaluations(*data)
    original=copy.deepcopy(result)
    review=review_fixture(data)
    output=audit.source_review_strata(data[0],result,review,prepared_sha256='b'*64)
    assert result==original
    sums=[v['metrics']['all'] for v in output['strata'].values()]
    assert sum(v['tokens'] for v in sums)==result['aggregate']['all']['tokens']
    for key in ('base_nll_sum','adapter_nll_sum','corrected_tokens','regressed_tokens'):
        assert sum(v[key] for v in sums)==pytest.approx(result['aggregate']['all'][key])
    approved=output['strata']['approved_source_scoped']['metrics']['all']
    # Unequal 1-token/3-token rows: require weighted mean, not example mean.
    assert approved['base_nll']==pytest.approx((.25+sum(data[2]['rows'][2]['token_nll']))/4)
    assert output['quality_certified'] is False
    assert len(output['rows'])==3


@pytest.mark.parametrize('mutation',[
    'drop','duplicate_index','swap','unknown','mask','prompt','tokens','global_tokens',
    'counts','boolean_counts','boolean_tokens','quality','no_findings','duplicate_id','hash','format','review_hash',
])
def test_bad_reference_bindings_rejected(mutation):
    data=fixture()
    result=audit.compare_token_evaluations(*data)
    r=review_fixture(data)
    if mutation=='drop':r['rows'].pop()
    elif mutation=='duplicate_index':r['rows'][1]['index']=0
    elif mutation=='swap':r['rows'][0],r['rows'][1]=r['rows'][1],r['rows'][0]
    elif mutation=='unknown':r['rows'][0]['source_review_status']='trusted'
    elif mutation in {'mask','prompt'}:
        p=copy.deepcopy(data[0][0]);p['loss_start']=1 if mutation=='mask' else p['loss_start']
        if mutation=='prompt':p['input_ids'][0]=9
        r['rows'][0]['prepared_identity']=checksum(p)
    elif mutation=='tokens':r['rows'][0]['supervised_tokens']+=1
    elif mutation=='global_tokens':r['supervised_tokens']+=1
    elif mutation=='counts':r['counts_by_source_status']['approved_source_scoped']+=1
    elif mutation=='boolean_counts':r['counts_by_source_status']['requires_reference_contract_resolution']=True
    elif mutation=='boolean_tokens':r['tokens_by_source_status']['requires_reference_contract_resolution']=True
    elif mutation=='quality':r['rows'][0]['kernel_quality_certified']=True
    elif mutation=='no_findings':r['rows'][0]['limits_and_findings']=' '
    elif mutation=='duplicate_id':r['rows'][1]['case_id']=r['rows'][0]['case_id']
    elif mutation=='hash':r['heldout_sha256']='a'*64
    elif mutation=='format':r['format']='other'
    elif mutation=='review_hash':r['review_sha256']='missing'
    with pytest.raises(ValueError):audit.source_review_strata(data[0],result,r,prepared_sha256='b'*64)


def test_file_audit_binds_review_and_keeps_all_primary(tmp_path):
    data=fixture()
    paths={name:tmp_path/(name+'.json') for name in ('prepared','categories','base','adapter','source_review')}
    paths['prepared'].write_text(''.join(json.dumps(p)+'\n' for p in data[0]))
    for name,value in zip(('categories','base','adapter'),data[1:]):paths[name].write_text(json.dumps(value))
    rawhash=hashlib.sha256(paths['prepared'].read_bytes()).hexdigest()
    paths['source_review'].write_text(json.dumps(review_fixture(data,rawhash)))
    out=tmp_path/'result.json'
    result=audit.audit_files(**paths,output=out)
    assert result['examples']==3
    assert result['aggregate']==audit.compare_token_evaluations(*data)['aggregate']
    assert str(paths['source_review']) in result['inputs_sha256']
    assert result['input_roles']['source_review']==str(paths['source_review'])
    assert json.loads(out.read_text())==result


def test_duplicate_json_review_refused_before_publication(tmp_path):
    data=fixture()
    paths={name:tmp_path/(name+'.json') for name in ('prepared','categories','base','adapter','source_review')}
    paths['prepared'].write_text(''.join(json.dumps(p)+'\n' for p in data[0]))
    for name,value in zip(('categories','base','adapter'),data[1:]):paths[name].write_text(json.dumps(value))
    paths['source_review'].write_text('{"format":"a","format":"b"}')
    out=tmp_path/'result.json'
    with pytest.raises(ValueError,match='Duplicate'):audit.audit_files(**paths,output=out)
    assert not out.exists()
