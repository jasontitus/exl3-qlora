"""Explicit reviewed operation contexts; every source/test/build edit stays in target."""
import hashlib

SPECS={
 '389977a5':{'zig/src/families/flashnext/replay.zig':['Run.callAs','Model.windowEncode','Model.mtpLayer']},
 '3673a760':{'tools/zig/flashnext_fuse_xsum.py':None,'zig/tests/flashnext_run.zig':['Run.call','Model.lane']},
 'b0aaf33a':{'zig/src/families/flashnext/replay.zig':['Run.denseRows','Model.mtpLayer'],'zig/src/families/flashnext/tp.zig':['Tp2.argmax']},
 '244989a3':{'zig/kernels/metal/decode/fn_gdn.metal':None,'zig/kernels/metal/decode/fn_lane.metal':None,'zig/src/families/flashnext/dense.zig':None,'zig/src/families/flashnext/gdn.zig':None,'zig/src/families/flashnext/split.zig':['gdnHeads'],'zig/src/families/flashnext/replay.zig':['Run.denseRows','Model.lane','Run.callAs']},
 '4ff9f852':{'zig/kernels/metal/glm_kda_prompt.metal':None,'zig/src/families/glm/prompt.zig':['backbone'],'zig/src/families/glm/kernels.zig':['Kernels']},
 '9dd7655c':{'zig/kernels/metal/core/affine_mm.metal':['am_block','tf_affine_mm'],'zig/src/core/affine_mm.zig':['dense'],'zig/src/families/glm/prompt.zig':['mla']},
 'f31afd9b':{'zig/kernels/metal/glm_sparse_nax.metal':None,'zig/src/families/glm/prompt.zig':['mla','Prompt'],'zig/src/families/glm/kernels.zig':['Kernels']},
 '67343158':{'zig/src/families/glm/prompt.zig':['Prompt','buffers','kdaChunk','mla','backbone']},
 '499a2d69':{'zig/kernels/metal/nemotron_tree.metal':['tf_tree_tail']},
 '5b528e7e':{'zig/src/families/nemotron/weights.zig':['Builder','load']},
 '70c27a5d':{'zig/kernels/metal/core/row_projection.metal':None,'zig/src/core/row_projection.zig':None},
 'ff81a4ef':{'zig/src/families/flashnext/split.zig':None,'zig/src/families/flashnext/tp.zig':['Tp2.agree']},
 '77dd9480':{'vllm_metal/attention/caches/turboquant.py':'python'},
 '5f3759b4':{'vllm_metal/attention/impls/linear.py':'python'},
 'd8cb131f':{'tests/test_gqa_paged_decode.py':None},
}
def bind(b,a,z):return {'start':a,'end':z,'sha256':hashlib.sha256(b[a:z]).hexdigest()}
def apply(key,c,rec,uu,su):
 if key not in SPECS:return
 chosen=[]
 for path,names in SPECS[key].items():
  if names=='python':chosen.extend(u for u in uu if u['path']==path);continue
  for name in names or [None]:
   u={'path':path,'kind':'whole_file' if name is None else 'complete_declaration'}
   for side in ('before','after'):
    b=(rec['files'][path][side] or {'content':''})['content'].encode()
    found=su.named_unit(b,'zig' if path.endswith('.zig') else 'c',name) if name else {'start':0,'end':len(b)}
    u[side]=bind(b,found['start'],found['end'])
   chosen.append(u)
 # Python regression tests remain complete functions for the two C++/Metal wrapper cases.
 if key in {'77dd9480'}:chosen.extend(u for u in uu if u['path'].startswith('tests/'))
 chosen.sort(key=lambda u:(u['path'],u['before']['start']))
 c['context_policy']='reviewed_operation';c['reviewed_units']['units']=chosen
 c['reviewed_units']['dependency_review']='Explicit complete operation declarations or whole core files, with supporting regression-test functions where listed. Every remaining host/kernel/test/build edit retains its full original patch and baseline anchors; complete before/after source blobs remain sidecars. Standalone dependency closure is not claimed.'
 omitted=[]
 for p,pair in rec['files'].items():
  b=(pair['before']or{'content':''})['content'].encode();a=(pair['after']or{'content':''})['content'].encode();bi=ai=0
  for u in [x for x in chosen if x['path']==p]:
   bs,ass=u['before'],u['after']
   if b[bi:bs['start']]!=a[ai:ass['start']]:omitted.append(p)
   bi,ai=bs['end'],ass['end']
  if b[bi:]!=a[ai:]:omitted.append(p)
 c['operation_scope']={'description':'Complete operation context and exact baseline anchors for a whole historical implementation/test edit; no source targets removed to fit length.','omitted_path_reasons':{p:'Non-expanded sections retain original baseline edit anchors, complete target edits, and complete before/after sidecars.' for p in sorted(set(omitted))}}
