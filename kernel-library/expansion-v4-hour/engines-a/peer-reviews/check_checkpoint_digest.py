"""CPU subprocess tests of the portable checkpoint byte verifier."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];S=R/'remote-m5/checkpoint_digest.py';H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();initial=H(S);cases=[]
with tempfile.TemporaryDirectory(prefix='checkpoint-digest-review-')as td:
 t=Path(td);root=t/'source';root.mkdir();(root/'part.bin').write_bytes(bytes(range(256))*33);(root/'nested').mkdir();(root/'nested/config.json').write_bytes(b'{"test":true}\n');counter=0
 def run(label,where=root,manifest=None,expect=True,output=None):
  global counter
  counter+=1;o=output or t/f'result-{counter}.json';argv=[sys.executable,str(S),str(where),str(o)]+(['--verify',str(manifest)]if manifest else[]);p=subprocess.run(argv,capture_output=True,text=True);assert(p.returncode==0)==expect,(label,p.stderr)
  if not expect and output is None:assert not o.exists(),label
  cases.append({'case':label,'expected_success':expect,'exit_code':p.returncode});return o
 src=run('source_manifest_success');expected=json.loads(src.read_text());assert expected['files']['part.bin']['sha256']==H(root/'part.bin')
 dst=t/'target';dst.mkdir();(dst/'nested').mkdir();(dst/'part.bin').write_bytes((root/'part.bin').read_bytes());(dst/'nested/config.json').write_bytes((root/'nested/config.json').read_bytes());v=run('portable_verify_identical_bytes',dst,src);assert json.loads(v.read_text())['verified_manifest_sha256']==H(src)
 run('nonexistent_root',t/'missing',expect=False);empty=t/'empty';empty.mkdir();run('empty_root',empty,expect=False)
 for name,value in [('empty_object',{}),('null',None),('list',[]),('wrong_schema',{'schema':'other','files':expected['files']}),('empty_files',{'schema':expected['schema'],'files':{}}),('boolean_size',{'schema':expected['schema'],'files':{'part.bin':{'size':True,'sha256':'0'*64}}}),('invalid_hash',{'schema':expected['schema'],'files':{'part.bin':{'size':1,'sha256':'z'*64}}})]:
  p=t/(name+'.json');p.write_text(json.dumps(value));run('reject_'+name,dst,p,False)
 (dst/'part.bin').write_bytes(b'bad payload');run('payload_mismatch',dst,src,False);(dst/'part.bin').write_bytes((root/'part.bin').read_bytes());(dst/'extra').write_text('extra');run('extra_file',dst,src,False);(dst/'extra').unlink();(dst/'nested/config.json').unlink();run('missing_file',dst,src,False);(dst/'nested/config.json').write_bytes((root/'nested/config.json').read_bytes());(dst/'link').symlink_to(dst/'part.bin');run('symlink_rejected',dst,src,False);(dst/'link').unlink()
 before=v.read_bytes();run('receipt_not_overwritten',dst,src,False,v);assert v.read_bytes()==before
 assert H(S)==initial
out={'schema':'checkpoint-digest-independent-review-v1','status':'approved_for_quiescent_regular_file_transfer_integrity','source_path':str(S),'source_sha256':initial,'test_script_sha256':H(Path(__file__)),'cases':cases,'count':len(cases),'checks':['Current verifier rejects nonexistent/empty roots and malformed/falsey manifests.','All actual bytes and portable relative file inventory are compared against the captured manifest.','Changed bytes, missing/extra files, symlinks and existing output receipts fail closed.'],'limitations':['Requires a trusted source manifest and quiescent regular-file source/target. This is transfer integrity, not publisher authentication, semantic model validation or a filesystem snapshot against adversarial concurrent replacement.','Original v1 source-side hashing is a separate running invocation on a known nonempty existing model directory without --verify; this review does not retroactively claim it used the repaired script.'],'gpu_execution':False}
p=Path(__file__).parent/'checkpoint-digest-independent-review.json';p.write_text(json.dumps(out,indent=2)+'\n');print('PASS',len(cases),'source',initial,'receipt',H(p))
