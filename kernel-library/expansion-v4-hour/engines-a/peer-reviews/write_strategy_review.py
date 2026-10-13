from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parents[2];W=R.parents[1];H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();inputs=[R/'strategy/coding-data-research.json',R/'strategy/training-strategy-draft.json',R/'engines-a/linked-change-ledger.json',W/'loss-fix/apple/artifacts/loss-fix/runtime-v5/src/exl3_mlx/data.py',W/'loss-fix/apple/artifacts/loss-fix/runtime-v5/src/exl3_mlx/tool_data.py',W/'loss-fix/apple/artifacts/loss-fix/runtime-v5/src/exl3_mlx/training.py']
# W is workspace, not kernel-library.
inputs=[Path(str(p).replace('/kernel-library/loss-fix/','/loss-fix/')) for p in inputs]
d={
 'schema':'coding-data-strategy-independent-review-v1',
 'reviewer':'quality_research',
 'status':'draft_design_supported_with_required_final_selection_gate',
 'scope':'Read-only research-to-plan review. No model outputs, held-out scores, GPU, training or serving execution used. This is not launch approval or evidence of held-out improvement.',
 'primary_sources_independently_reopened':[
  {'url':'https://arxiv.org/html/2504.21798v1','supports':'Execution/test-backed agent tasks and successful trajectories; task/repository diversity and limiting repeated easy trajectories. It does not establish our proposed mixture or final-layer Metal-LoRA efficacy.'},
  {'url':'https://arxiv.org/html/2412.21139v1','supports':'Real software tasks with execution environments, trajectories and verifiers; a different domain and agent training regime.'},
  {'url':'https://arxiv.org/html/2504.07164v1','supports':'Executable task environments, synthetic/backtranslated problem statements and verifiers; commit-derived prompts still need answer-leakage review.'},
  {'url':'https://huggingface.co/docs/trl/sft_trainer','supports':'Conversational tool schemas and assistant/completion masking are distinct data contracts. TRL capabilities do not imply support in this custom runtime.'}
 ],
 'supported_decisions':[
  'The plan honestly calls the current data historical single-edit imitation or source-grounded diagnostics, not successful multi-turn execution-feedback trajectories.',
  'runtime-v5 normalizes a terminal supervised span and rejects tool-role observations, nonterminal tool calls and multiple terminal calls. Keep one native exec_command action or one final text response; do not feed invented tool traces.',
  'D <= floor(E/4) gives D/(D+E) <= 20%. With accumulation1 and exactly2N samples from the deterministic epoch sampler, each row has two updates, so this is an update-frequency bound as well as a row bound. It is not a token-gradient budget or an empirically optimal ratio.',
  'The final fixed checkpoint and fresh same-host base/prior/new comparisons support a whole-recipe comparison. Larger data, more updates and changed prose mix prevent attribution to reasoning alone.',
  'Teacher-forced code loss, decision-label accuracy and tiny expression compile/parity checks measure different things. The plan explicitly avoids treating any one as proof of broad generated-kernel quality or speed.'
 ],
 'required_before_plan_seal':[
  {'id':'linked_change_alias_selection','status':'open_at_review','action':'Apply the source-supported port/cherry-pick/follow-up alias graph before the promised one-diagnostic-per-lineage selection. DS4 fe457/54a/3c currently have separate patch IDs despite being one documented campaign;84f references070199;8e declares an upstream port. Bind the alias ledger and actual selected IDs in the final selection receipt. Do not merge all examples merely because their repo/framework is shared.'},
  {'id':'final_mix_and_step_receipt','status':'expected_finalization','action':'Seal actual E/D counts, selected/excluded IDs, token share, planned update share, two-exposure sampler cursor and all raw/prepared/provenance hashes. Recheck the 20% rule after any protection or length exclusion, not before.'},
  {'id':'prospective_vs_retrospective_counts','status':'expected_finalization','action':'Report selected diagnostic perspective counts. Retrospective prompts intentionally include achieved outcomes and cannot test optimization discovery. Reconstructed prospective prompts must contain baseline/symptom, not the final body or claimed achieved speed; source metadata can retain the later evidence outside model input.'}
 ],
 'nonblocking_limits':[
  'The research curriculum gate says complete host/build dependencies; current artifacts preserve complete changed host/build/test files but explicitly do not prove full dependency closure or compileability. Treat dependency closure as a future execution qualification, not an already-passed gate.',
  'Deterministic ID ordering is reproducible but not itself a quality or difficulty ranking. Record the realized operation/mechanism/perspective distribution; do not use new held-out scores to select another mixture.',
  'A correct explanation can teach evidence interpretation even when the historical optimization was rejected. Keep rejected edits out of positive tool-apply targets while allowing explicit reject/abstain diagnostic answers.',
  'The present terminal action format can teach edit syntax and source-conditioned repairs. It cannot teach reacting to real compiler/profiler feedback without separately versioned multi-turn ingestion and observed trajectories.',
  'No universal best diagnostic percentage or edit syntax is established by the independently inspected primary sources. The20% cap is a reasonable declared experimental choice, not a research-derived optimum.'
 ],
 'inputs_sha256':{str(p.resolve()):H(p) for p in inputs},
 'training_or_gpu_execution':False
}
p=Path(__file__).parent/'coding-data-strategy-independent-review.json';p.write_text(json.dumps(d,indent=2)+'\n');print(p,H(p))
