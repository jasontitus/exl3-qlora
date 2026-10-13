"""Freeze narrow diagnosis probes derived from already protected task families.

No model/GPU execution. These are curator-authored decision tests, not historical
agent trajectories or a benchmark of explanation quality or kernel speed.
"""

import hashlib
import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]
MODEL = Path('/Users/jasontitus/models/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw')
SPECS = [
    ('conv_interval', '560445bf34c8', 'B',
     'A transposed 1D convolution sums input positions i in [0,319] satisfying i*5 <= j < i*5+10. For output j=23, which inclusive input interval preserves exactly the contributing terms?',
     ['[2,3]', '[3,4]', '[4,5]', '[0,319]'],
     'Direct enumeration gives i=3 and i=4. Bounds are ceil((j-K+1)/s) through floor(j/s), clamped to the input range.'),
    ('conv_gap', '560445bf34c8', 'D',
     'A transposed 1D convolution contributes from i in [0,9] only when i*5 <= j < i*5+3. At j=4, which input positions should an optimized loop process?',
     ['Only i=0', 'Only i=1', 'Both i=0 and i=1', 'None; the interval is empty'],
     'There is a gap between the first support [0,3) and second [5,8). Clamping the lower bound down to the upper bound would introduce an invalid contribution.'),
    ('broadcast_index', '13268c533177', 'A',
     'A Metal float4 kernel broadcasts one source row of nb=3 float4 vectors across many rows. The destination is contiguous by row. At global float4 index tpig=7, which source-row float4 index is required?',
     ['1', '7', '2', '3'],
     'The source index is tpig modulo nb, so 7 modulo 3 is 1. Units are float4 vectors, not scalar floats.'),
    ('broadcast_discriminator', '13268c533177', 'C',
     'A row-broadcast kernel must reuse a vector of nb float4 values for every destination row. Two proposed source indices are tpig % nb and tpig & (nb-1). Which correctness test best distinguishes these expressions?',
     ['One row with nb=1 and equal source values', 'One row with nb=8 and equal source values', 'Several rows with nb=3 and distinct source values', 'Several rows with nb=8 and distinct source values'],
     'Modulo equals a bit mask for powers of two. A non-power-of-two width with distinct values exposes incorrect indices, including across row boundaries.'),
    ('sum_missing_collective', 'f4ce81c45e7b', 'C',
     'A reduction uses one 64-thread group containing two 32-lane SIMD groups. Each thread holds a partial sum of 1. Only lane zero of each SIMD group writes its current partial to shared memory; a correct final stage sums those two entries. If the SIMD collective was omitted before the writes, what result is produced?',
     ['64', '32', '2', '1'],
     'Only two lane-zero partials reach shared memory, so the final stage gets 1+1. The missing collective must combine all active lanes in each SIMD group.'),
    ('sum_rounding_contract', 'f4ce81c45e7b', 'B',
     'A floating-point sum replaces a serial reduction with SIMD partial sums, changing addition order. Both implementations include every input exactly once. Which validation requirement is justified without an additional proof of bitwise equivalence?',
     ['Require bit-identical outputs for every input and every order', 'Use declared numerical tolerances and stress cancellation and extreme values', 'Only check that compilation succeeds for a nonempty array', 'Ignore every discrepancy because parallel reductions are always approximate'],
     'Reassociation can change rounding. Numerical tolerance and adversarial value distributions are required; inclusion of all terms alone does not prove bitwise identity.'),
    ('timing_scope', 'ba7ffbb2517f', 'D',
     'An upstream Q3_K optimization report states that token generation improved by 13%, while prompt processing is unaffected because it uses separate matrix-multiplication templates. Which conclusion does that evidence support?',
     ['Prompt-processing throughput improved by 13%', 'Every quantization format improved by 13%', 'Both execution paths must use the edited kernel', 'The report supports the generation path, not a prompt-processing gain'],
     'The author explicitly separates TG and PP execution paths. The stated result is historical source evidence, not a local reproduction or a general result for other paths.'),
    ('timing_attribution', '22885105a6b0', 'A',
     'A source report attributes a 4–5% improvement in total prompt processing to a change in quantized weight loads. It provides no isolated timing of the changed helper. Which statement is warranted?',
     ['The report supports an end-to-end gain; isolated helper speedup remains unknown', 'The helper itself is necessarily exactly 4–5% faster', 'Every model and Apple GPU will see the same gain', 'The report proves numerical parity for all input values'],
     'An end-to-end percentage does not determine the changed component speedup without its runtime share and other costs. Hardware/workload scope and numerical evidence remain separate.'),
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def freeze(output):
    from transformers import PreTrainedTokenizerFast
    output = Path(output)
    if output.exists():
        raise ValueError('Refuse to overwrite a frozen evaluation plan')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(MODEL), local_files_only=True)
    sources, tasks = {}, []
    for task_id, prefix, answer, question, choices, rationale in SPECS:
        source = WORK / 'kernel-library/expansion-v2/corpus-12288-v3' / f'llama-cpp-{prefix}.provenance.json'
        raw = source.read_bytes()
        record = json.loads(raw)
        if record['split'] != 'test':
            raise ValueError('Probe is not from a protected family')
        sources[str(source)] = sha(raw)
        prompt = question + '\n\n' + '\n'.join(f'{label}. {choice}' for label, choice in zip('ABCD', choices, strict=True))
        prompt += '\n\nReturn exactly one capital letter, A, B, C or D. Do not include an explanation.'
        messages = [{'role':'user','content':prompt}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
        tasks.append(dict(id=task_id, source_family=record['id'], source_commit=record['source']['head']['commit'],
                          messages=messages, prompt_ids=list(ids), expected=answer, curator_rationale=rationale,
                          provenance='Curator-authored diagnostic derived from protected source semantics; fixture values may be synthetic.'))
    labels = [t['expected'] for t in tasks]
    assert all(labels.count(label) == 2 for label in 'ABCD')
    # Independently enumerate the mathematical fixture answers before publication.
    assert [i for i in range(320) if i*5 <= 23 < i*5+10] == [3,4]
    assert [i for i in range(10) if i*5 <= 4 < i*5+3] == []
    assert 7 % 3 == 1
    assert any(i % 3 != i & 2 for i in range(9))
    assert all(i % 8 == i & 7 for i in range(24))
    plan = dict(schema='metal-diagnostic-decision-eval-v1', tasks=tasks,
                source_bindings=sources, implementation_sha256=sha(Path(__file__).read_bytes()),
                tokenizer_bindings={str(MODEL/n):sha((MODEL/n).read_bytes()) for n in ('tokenizer.json','tokenizer_config.json','chat_template.jinja')},
                decoding={'method':'greedy','max_new_tokens':8,'scoring':'Strip surrounding whitespace only; exact single capital label; every malformed or truncated response is incorrect.'},
                review_status='pending_independent_review',
                comparison='Same fixed questions and generation budgets for zero base, prior adapter, new final adapter. Never use these scores for model/checkpoint/data selection.',
                limitations=['Eight decisions from five protected source families; correlated questions are not independent tasks.',
                             'Recognition and evidence interpretation only; no evaluation of generated reasoning explanations, kernel implementation or speed.',
                             'Potential ceiling effects; base-perfect results imply no measured improvement headroom.',
                             'Protected sources were previously inspected and scored for code NLL. This is a new diagnostic view, not an untouched new benchmark.'])
    output.mkdir(parents=True)
    for task in tasks:
        (output/(task['id']+'.messages.json')).write_text(json.dumps(task['messages'],indent=2)+'\n')
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return dict(tasks=len(tasks), source_families=len({t['source_family'] for t in tasks}), plan_sha256=sha((output/'plan.json').read_bytes()))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(freeze(args.output),indent=2))
