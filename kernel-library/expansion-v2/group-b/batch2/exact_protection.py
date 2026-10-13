"""Exact reserved identity gate. Run before source/manual admission review.

This CPU-only gate uses captured metadata, not repository names or semantic
similarity. A manual 'cleared' flag cannot override a matched head/base/related
commit. Semantic/source-unit review remains necessary after this gate passes.
"""
import re

SHA = re.compile(r'[0-9a-f]{40}')


def reserved_index(metadata):
    rows = metadata.get('review', {}).get('coverage')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Missing nonempty reserved coverage metadata')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError('Malformed reserved identity row')
        # This captured schema also lists non-Git operation/winner evidence.
        # It remains a separate semantic gate; do not invent commit identities.
        non_git = row['id'].startswith('protected-task:') or row['id'] in {'hc-original-winner', 'hc-masked-winner'}
        if non_git and 'head' not in row and 'base' not in row:
            continue
        for key in ('head', 'base'):
            value = row.get(key)
            if not isinstance(value, str) or not SHA.fullmatch(value):
                raise ValueError('Reserved head/base must be full lowercase Git SHA')
            result.setdefault(value, []).append({'reserved_id': row['id'], 'reserved_field': key})
    if not result:
        raise ValueError('No Git identities in reserved coverage')
    return result


def identity_matches(candidate, index):
    related = candidate.get('related_commits', [])
    if not isinstance(related, list):
        raise ValueError('related_commits must be a list')
    values = [('commit', candidate.get('commit')), ('parent', candidate.get('parent'))]
    values += [(f'related_commits[{i}]', x) for i, x in enumerate(related)]
    result = []
    for field, sha in values:
        if not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError('Candidate identities must be full lowercase Git SHAs')
        for match in index.get(sha, []):
            result.append({'candidate_field': field, 'sha': sha, **match})
    return result


def review_after_identity_gate(candidate, index, manual_review):
    """A blocked candidate never reaches the finite/manual-review callback."""
    matches = identity_matches(candidate, index)
    if matches:
        return {'status': 'protected_exact_identity', 'matches': matches}
    return manual_review(candidate)
