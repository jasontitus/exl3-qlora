"""Explicit reviewed complete-file/function context, full target edit preservation."""
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from source_units import named_unit

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[2]
sys.path.insert(0, str(WORK / 'loss-fix/apple/src'))
from exl3_mlx.upstream_data import reconstruct_candidate
from exl3_mlx.agent_edits import agent_example
from exl3_mlx.data import load_tokenizer, tokenize_row

H = lambda b: hashlib.sha256(b).hexdigest()
CANON = lambda d: json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
chosen = {}
scoped_names = {
 '12b5d8db': {'zig/src/families/glm/engine.zig': ['Engine.generate']},
 '4535d28f': {'zig/src/core/affine_mm.zig': ['gatherGlu']},
 '2d698959': {'zig/src/families/glm/prompt.zig': ['init']},
}


def buffer_function(raw):
    # This one explicitly inspected function contains no Zig multiline strings.
    needle = b'    pub fn buffers(r: *Run) !@FieldType(Prompt, "b") {'
    start = raw.index(needle)
    assert raw.count(needle) == 1
    text = raw[start:].decode()
    depth = 0
    for match in re.finditer(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|[{}]', text):
        token = match.group()
        if token == '{': depth += 1
        elif token == '}':
            depth -= 1
            if depth == 0:
                return start, start + len(text[:match.end()].encode())
    raise ValueError('Incomplete buffers function')


m = json.loads((ROOT / 'manifest-fullfiles.json').read_text())
tasks = json.loads((ROOT / 'task_annotations.json').read_text())
tokenizer = load_tokenizer('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw')
audits, documentation = [], {}
for c in m['candidates']:
    short = c['commit'][:8]
    record = json.loads((ROOT / 'sourcepacks' / c['id'] / 'complete-record.json').read_text())
    operation, explanation = tasks[short]
    quote = c['annotations']['rationale']['text']
    for field, text in [('operation',operation), ('architecture',explanation)]:
        c['annotations'][field] = {'text':text,'basis':'source','evidence':[{'source':'commit_message','quote':quote}]}
    units = []
    for name in (scoped_names.get(short) or {p:None for p in chosen[short]}):
        pair = record['files'][name]
        names = scoped_names.get(short, {}).get(name) or [None]
        for unit_name in names:
            unit = {'path':name,'kind':'complete_function' if unit_name or short=='28e6ade7' else 'whole_file'}
            for side in ['before','after']:
                raw = (pair[side] or {'content':''})['content'].encode()
                if unit_name:
                    found = named_unit(raw, 'zig' if name.endswith('.zig') else 'metal', unit_name)
                    a,b = found['start'], found['end']
                else:
                    a,b = buffer_function(raw) if short=='28e6ade7' else (0,len(raw))
                unit[side] = {'start':a,'end':b,'sha256':H(raw[a:b])}
            units.append(unit)
    units.sort(key=lambda u:(u['path'],u['before']['start']))
    omitted = []
    for name,pair in record['files'].items():
        before = (pair['before'] or {'content':''})['content'].encode()
        after = (pair['after'] or {'content':''})['content'].encode()
        bc=ac=0
        for unit in [u for u in units if u['path']==name]:
            b,a=unit['before'],unit['after']
            if before[bc:b['start']] != after[ac:a['start']]: omitted.append(name)
            bc,ac=b['end'],a['end']
        if before[bc:] != after[ac:]: omitted.append(name)
    c['context_policy']='reviewed_operation'
    c['reviewed_units']={'reviewed_by':'root explicit core-operation context review',
      'dependency_review':'Complete core kernel/dispatch files or explicitly named buffers function. Other changed code remains in full target and exact baseline anchors; full sources in sidecars. Not a standalone compilation closure.', 'units':units}
    c['operation_scope']={'description':'Core operation context for a complete historical repository edit. Every implementation, host, test, build and CUDA counterpart edit is preserved in the target.',
      'omitted_path_reasons':{p:'Not expanded as a whole prompt file; all of its exact original edit anchors and target edits are retained, with full before/after sidecars.' for p in sorted(set(omitted))}}
    docs={}
    for name,pair in record['files'].items():
        if Path(name).suffix.lower() in {'.md','.rst'}:
            docs[name]={'before_sha256':H(pair['before']['content'].encode()) if pair['before'] else None,
              'after_sha256':H(pair['after']['content'].encode()),'reason':'Reviewed changelog/recipe prose; commands and historical measurements retained as evidence. All executable/build/test changes retained.'}
    documentation[c['id']]=docs
    dest=ROOT/'drafts'/c['id'];dest.mkdir(parents=True,exist_ok=True)
    try:
        reconstructed=reconstruct_candidate(c,m['repositories'],{})
        row,replay=agent_example(reconstructed,documentation_omissions=docs)
        row=json.loads(CANON(row));tokens=tokenize_row(row,tokenizer)
        (dest/'row.json').write_bytes(CANON(row)+b'\n')
        (dest/'replay.json').write_bytes(CANON(replay)+b'\n')
        n=len(tokens['input_ids'])
        audits.append({'id':c['id'],'outcome':c['outcome'],'serialized_tokens':n,
          'supervised_tokens':n-tokens['loss_start'],'fits_12288':n<=12288,'fits_32768':n<=32768,'replay':'passed','historical_protection':'pending_review'})
    except Exception as error:
        audits.append({'id':c['id'],'status':'error','error':str(error)})
    print(audits[-1],flush=True)
m.pop('manifest_sha256');m['manifest_sha256']=H(CANON(m))
(ROOT/'manifest-scoped-draft.json').write_text(json.dumps(m,indent=2)+'\n')
(ROOT/'documentation-omissions.json').write_text(json.dumps(documentation,indent=2)+'\n')
(ROOT/'length-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
