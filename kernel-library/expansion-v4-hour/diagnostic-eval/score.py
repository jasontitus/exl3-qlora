"""Score frozen diagnostic choices from actual, token-bound generation receipts."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check_bindings(bindings):
    for filename, expected in bindings.items():
        if sha(Path(filename).read_bytes()) != expected:
            raise ValueError('Evaluation input drift: ' + filename)


def classify(text, stop_reason, expected):
    if stop_reason != 'eos':
        return 'truncated', False
    cleaned = text.strip()
    if cleaned not in ('A', 'B', 'C', 'D'):
        return 'malformed', False
    return ('correct' if cleaned == expected else 'incorrect'), cleaned == expected


def score(plan_path, reports, model, adapter):
    from transformers import PreTrainedTokenizerFast
    plan_path = Path(plan_path).resolve()
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    check_bindings(plan['source_bindings'])
    check_bindings(plan['tokenizer_bindings'])
    expected_model_files = {str(Path(model).resolve()/n) for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja')}
    if set(plan['tokenizer_bindings']) != expected_model_files:
        raise ValueError('Wrong tokenizer base')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model),local_files_only=True)
    mapping = {}
    for item in reports:
        task_id, filename = item.split('=',1)
        if task_id in mapping:
            raise ValueError('Duplicate task report')
        mapping[task_id] = Path(filename).resolve()
    if set(mapping) != {t['id'] for t in plan['tasks']}:
        raise ValueError('Require exactly one report for every frozen task')
    expected_adapter = str(Path(adapter).resolve()) if adapter else None
    results = []
    for task in plan['tasks']:
        path = mapping[task['id']]
        receipt_bytes = path.read_bytes()
        receipt = json.loads(receipt_bytes)
        if (receipt.get('prompt_ids') != task['prompt_ids']
            or receipt.get('prompt_messages') != task['messages']
            or receipt.get('max_new_tokens') != plan['decoding']['max_new_tokens']
            or receipt.get('method') != 'greedy full-prefix recomputation'
            or receipt.get('adapter') != expected_adapter
            or receipt.get('tools') is not None):
            raise ValueError('Generation receipt does not match the frozen condition')
        ids = receipt.get('generated_ids')
        if (not isinstance(ids,list) or not 1 <= len(ids) <= plan['decoding']['max_new_tokens']
            or any(type(v) is not int or v < 0 or v >= len(tokenizer) for v in ids)):
            raise ValueError('Invalid generated token IDs')
        text = tokenizer.decode(ids,skip_special_tokens=True)
        raw_text = tokenizer.decode(ids,skip_special_tokens=False)
        stop = 'eos' if ids[-1] == tokenizer.eos_token_id else 'max_tokens'
        if (text != receipt.get('text') or raw_text != receipt.get('raw_text')
            or receipt.get('stop_reason') != stop or receipt.get('eos_token_id') != tokenizer.eos_token_id
            or tokenizer.eos_token_id in ids[:-1]):
            raise ValueError('Text/stop metadata does not agree with recorded tokens')
        unexpected_special = any(v in tokenizer.all_special_ids for v in ids[:-1]) or (ids[-1] in tokenizer.all_special_ids and ids[-1] != tokenizer.eos_token_id)
        category, correct = ('malformed', False) if unexpected_special else classify(text, stop, task['expected'])
        results.append(dict(id=task['id'],family=task['source_family'],expected=task['expected'],
                            text=text,category=category,correct=correct,report=str(path),sha256=sha(receipt_bytes)))
    return dict(schema='metal-diagnostic-decision-score-v1',plan_sha256=sha(raw),adapter=expected_adapter,
                correct=sum(r['correct'] for r in results),total=len(results),
                families=len({r['family'] for r in results}),results=results,
                qualification='Token/text scoring only; trusted launch binding must separately establish runtime, full checkpoint payload, numerical mode and adapter identity. Narrow choice accuracy only. No evidence of free-form reasoning, generated patch correctness or kernel speed. Compare paired tasks; related questions are correlated.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--report',action='append',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise ValueError('Refuse to overwrite scored evidence')
    result=score(args.plan,args.report,args.model,args.adapter)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))
