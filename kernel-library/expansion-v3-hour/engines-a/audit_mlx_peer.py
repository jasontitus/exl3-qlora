"""Independent CPU-only payload/replay/token audit of the sibling MLX lane."""
from pathlib import Path
import json,hashlib,sys,os,subprocess
R=Path(__file__).resolve().parent;W=R.parents[2];M=R.parent/'mlx';sys.path[:0]=[str(W/'loss-fix/apple/src'),str(R.parent/'tools')]
import exact_edit_renderer as editor
from exl3_mlx.data import load_tokenizer,tokenize_row
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
index_path=M/'case-index.json';index_raw=index_path.read_bytes();index=json.loads(index_raw);index_sha=hashlib.sha256(index_raw).hexdigest();manifest=json.loads((M/'manifest-reviewed.json').read_text());repos={v['url'].lower().rstrip('/'):v['path']for v in manifest['repositories'].values()};cases=[];t=load_tokenizer('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw')
for row in index['cases']:
 if not row.get('exact_edit_eligible')or row.get('total_serialized_ids',999999)>32768:continue
 paths={k:Path(v)for k,v in row['paths'].items()};hashes={str(p):sha(p)for p in paths.values()}
 for k,p in paths.items():
  if row.get('files_sha256',{}).get(k):assert row['files_sha256'][k]==hashes[str(p)]
 src=json.loads(paths['source'].read_text());raw=json.loads(paths['raw'].read_text());prepared=json.loads(paths['prepared'].read_text());receipt=json.loads(paths['replay'].read_text())
 assert src['case_id']==row['id'];assert src['head']['commit']==row['commit'];assert src['parent']['commit']==row['parent'];assert src['repository'].lower().rstrip('/')==row['repo'].lower().rstrip('/')
 for path,pair in src['files'].items():
  for side,rev in [('before',row['parent']),('after',row['commit'])]:
   part=pair[side];argv=['git','-c','core.hooksPath=/dev/null','--git-dir='+repos[row['repo'].lower().rstrip('/')],'ls-tree','-z',rev,'--',path];tree=subprocess.check_output(argv,env={**os.environ,'GIT_NO_LAZY_FETCH':'1','GIT_NO_REPLACE_OBJECTS':'1'},timeout=30)
   if part is None:assert not tree;continue
   mode,kind,blob=tree.split(b'\t')[0].split();assert mode.decode()==part['mode']and blob.decode()==part['blob']and kind==b'blob';content=part['content'].encode();assert hashlib.sha256(content).hexdigest()==part['sha256'];assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()==part['blob']
 assert tokenize_row(raw,t)==prepared
 assert len(prepared['input_ids'])==row['total_serialized_ids'];assert len(prepared['input_ids'])-prepared['loss_start']==row['supervised_tokens'];assert 0<prepared['loss_start']<len(prepared['input_ids']);assert prepared['input_ids'][-1]==t.eos_token_id
 before,after=editor.full_commit_images(src)
 for p in receipt.get('omitted_documentation',{}):
  assert Path(p).suffix.lower()in['.md','.rst'];before.pop(p);after.pop(p)
 call=raw['messages'][-1]['tool_calls'][0];args=call['function']['arguments'];args=json.loads(args)if isinstance(args,str)else args;assert set(args)=={'cmd'};cmd=args['cmd'];patch=editor.parse_command(cmd);assert hashlib.sha256(patch.encode()).hexdigest()==receipt['patch_sha256']
 replay=editor.replay(cmd,before,after);assert replay['patch_applies']and replay['exact_scoped_replay'];assert replay['files']==receipt['files'];assert replay['compile']==replay['numerical']==replay['performance']=='not_run'
 for p in paths.values():assert sha(p)==hashes[str(p)]
 cases.append({'id':row['id'],'inputs_sha256':hashes,'cpu_checks':{'exact_tokenization':True,'full_mask_and_EOS':True,'independent_git_replay':True,'source_head_parent_repository':True,'pinned_git_tree_modes_and_blob_ids':True,'receipt_not_execution_qualified':True},'curriculum_tier':row.get('curriculum_tier'),'semantic_review':'Reviewed all task/assistant summaries and direct source/PR samples; not a compiler or numerical correctness certificate.'});print(row['id'],'PASS',flush=True)
assert 'mlx.core'not in sys.modules
out={'schema':'exact-edit-independent-review-v1','status':'approved','reviewer':'quality_research','scope':'Source/raw/prepared/replay receipt integrity and independently repeated CPU tokenization+Git apply; source/claim review with explicit task and performance limits. No GPU, compilation or numerical execution.','cases':cases,'index_snapshot_sha256':index_sha,'review_script_sha256':sha(Path(__file__)),'reasoning_review':{'status':'source_samples_reviewed','cases':['mlx-lm-9dcefa5272d2','mlx-lm-a790972f0f84','mlx-lm-480934402db0','mlx-f14e6d2d6d51','mlx-f5cffe9456b7','mlx-vlm-868596185d3f'],'findings':'PR-vs-final-diff caveats correctly preserved; copy-ownership wording and validation default-threshold clarification communicated and repaired. No remaining source-attribution blocker in sampled cases.'}}
(R.parent/'mlx-independent-review.json').write_text(json.dumps(out,indent=2)+'\n');print('checked',len(cases))
