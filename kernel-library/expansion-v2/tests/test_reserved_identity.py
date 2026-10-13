import importlib.util,json
from pathlib import Path
import pytest
p=Path(__file__).resolve().parents[1]/'tools/reserved_identity.py';s=importlib.util.spec_from_file_location('reserved_identity',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def setup_files(tmp_path,commit,parent,**extra):
 meta=tmp_path/'metadata.json';meta.write_text(json.dumps({'review':{'coverage':[{'id':'mlx:4380','head':'a'*40,'base':'b'*40}]}}))
 manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'candidates':[{'id':'case','commit':commit,'parent':parent,'outcome':'reported_improvement','task_kind':{'value':'kernel'},**extra}]}));return manifest,meta
@pytest.mark.parametrize('commit,parent',[('a'*40,'c'*40),('c'*40,'b'*40),('b'*40,'c'*40)])
def test_blocks_exact_even_manual_cleared(tmp_path,commit,parent):
 paths=setup_files(tmp_path,commit,parent,historical_protection_review={'status':'cleared'})
 with pytest.raises(ValueError,match='Reserved exact'):m.check_manifests([paths[0]],paths[1])
def test_distinct_does_not_ban_repository(tmp_path):
 p,meta=setup_files(tmp_path,'c'*40,'d'*40);assert m.check_manifests([p],meta)['matches']==[]
def test_nonpositive_retained_as_ledger(tmp_path):
 p,meta=setup_files(tmp_path,'a'*40,'b'*40,outcome='source_only');assert m.check_manifests([p],meta)['positive_candidates_checked']==0
def test_missing_coverage_fails_closed():
 with pytest.raises((ValueError,KeyError)):m.reserved_identities({'review':{'coverage':[]}})

def test_metadata_replacement_rejected(tmp_path):
 p,meta=setup_files(tmp_path,'c'*40,'d'*40)
 with pytest.raises(ValueError,match='checksum mismatch'):m.check_manifests([p],meta,expected_sha256='0'*64)

def test_manifest_drift_rejected(tmp_path):
 p,meta=setup_files(tmp_path,'c'*40,'d'*40);receipt=m.check_manifests([p],meta);p.write_text('{}')
 with pytest.raises(ValueError,match='input drift'):m.verify_snapshot(receipt)
def test_malformed_candidate_identity_rejected(tmp_path):
 p,meta=setup_files(tmp_path,'truncated','d'*40)
 with pytest.raises(ValueError,match='Invalid candidate'):m.check_manifests([p],meta)
def test_snapshot_pass(tmp_path):
 p,meta=setup_files(tmp_path,'c'*40,'d'*40);m.verify_snapshot(m.check_manifests([p],meta))
