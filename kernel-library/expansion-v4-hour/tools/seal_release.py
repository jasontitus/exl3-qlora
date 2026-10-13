"""Seal reviewed v4 datasets without touching prior releases or training outputs."""
import datetime
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'release.json'
    if output.exists():
        raise FileExistsError('Release already sealed')
    read = lambda p: json.loads(p.read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    small = read(root/'corpus-12288-v1/audit.json')
    large = read(root/'corpus-32768-v1/audit.json')
    mix = read(root/'training-mix-12288-v3/ktune-import.json')
    reasoning = read(root/'reasoning-library-v2/audit.json')
    prior = read(root.parent/'expansion-v3-hour/release.json')
    assert small['train_examples'] == mix['exact_rows'] == 159
    assert large['train_examples'] == 226
    assert mix['counts'] == {'train':198,'valid':0,'test':16}
    assert mix['diagnostic_rows'] == 39 and mix['truncated'] == 0
    protected = '0720ead3d696e6e9215e99cbf18dd26e532def8c309add06bbbe21efe1841ad0'
    for name in ('corpus-12288-v1','corpus-32768-v1'):
        assert sha(root/name/'test.jsonl') == protected
    for name, expected in mix['artifacts_sha256'].items():
        assert Path(name).name == name
        assert sha(root/'training-mix-12288-v3'/name) == expected
    selected = []
    for folder in ('corpus-12288-v1','corpus-32768-v1','reasoning-library-v2','training-mix-12288-v3'):
        selected += [p for p in (root/folder).iterdir() if p.is_file()]
    selected += [root/'mlx/peer-reviews/known-family-aliases-v2.json',
                 root/'mlx/peer-reviews/training-mix-alias-v2-independent-review.json',
                 root/'engines-a/peer-reviews/publication-independent-review.json',
                 root/'engines-a/peer-reviews/publication-post-correction-acknowledgment.json',
                 root/'strategy/coding-data-research.json',Path(__file__).resolve()]
    value = {'schema':'metal-optimization-library-release-v4',
             'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'primary_corpus':'corpus-12288-v1','archive_corpus':'corpus-32768-v1',
             'training_mix':'training-mix-12288-v3','train_count':159,'archive_train_count':226,
             'new_exact_12k':159-prior['train_count'],'new_exact_32k':226-prior['archive_train_count'],
             'new_reasoning_views':reasoning['records'],'mix_exact':159,'mix_diagnostics':39,
             'mix_train_count':198,'heldout_count':16,'planned_updates':396,
             'qualification':'Source/replay/tokenization and scoped independent review only; not blanket compilation, numerical, performance or model-quality qualification. Commit counts are not independent optimization families. No model outputs selected these inputs.',
             'source_history_complete':False,
             'known_family_limits':'Reviewed aliases cover known links only; unresolved upstream ancestry remains recorded.',
             'prior_release_sha256':sha(root.parent/'expansion-v3-hour/release.json'),
             'files_sha256':{str(p.relative_to(root)):sha(p) for p in sorted(selected)}}
    with output.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps({'release':str(output),'sha256':sha(output),'files':len(selected)}))

if __name__ == '__main__':
    main()
