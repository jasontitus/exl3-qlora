"""Describe discovery coverage and exact commit reuse without admitting data."""
import collections
import hashlib
import json
from pathlib import Path

base = Path(__file__).resolve().parent
raw = (base/'inventory.json').read_bytes()
data = json.loads(raw)
groups = collections.defaultdict(list)
for row in data['candidates']:
    groups[row['commit']].append(row)
result = {
    'schema': 'new-metal-discovery-summary-v1',
    'inventory_sha256': hashlib.sha256(raw).hexdigest(),
    'candidate_repository_commit_rows': len(data['candidates']),
    'unique_exact_commit_ids': len(groups),
    'exact_commit_ids_seen_in_multiple_mirrors': sum(len(rows)>1 for rows in groups.values()),
    'message_matches_are_not_training_examples': True,
    'counts_by_mirror': dict(collections.Counter(Path(r['repository_mirror']).name for r in data['candidates'])),
    'groups': [{'commit': commit,
                'mirrors': sorted(r['repository_mirror'] for r in rows),
                'tags': sorted({t for r in rows for t in r['discovery_tags']}),
                'title': rows[0]['message'].splitlines()[0],
                'scope': 'hardware_mention_only' if set(rows[0]['discovery_tags']) <= {'m5'} else 'feature_message_match',
                'status': 'requires_source_review_and_lineage_checks'} for commit, rows in sorted(groups.items())],
    'limits': ['Same commit across mirrors is grouped, preserving all source attribution.',
               'Different commits with equivalent patches are not deduplicated here.',
               'No author-reported performance claim has been reproduced.',
               'Hardware mentions alone do not establish use of a newer Metal feature.',
               'MTPLX had no reachable or captured metadata roots in this scan; llama.cpp not scanned.',
               'No active training, heldout, or sealed corpus file changed.']}
(base/'discovery-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('groups','limits','counts_by_mirror')}))
