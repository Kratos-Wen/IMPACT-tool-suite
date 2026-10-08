"""Validate one returned review against the exact distributed source.

python inspect_return.py path/to/review.json returned/reviewed_person.json
Does not require video, GPU, Qt or automatic annotation merging.
"""
import argparse
import json
from review_core import ReviewDocument, digest


def validate_return(source_path, returned_path):
    source = ReviewDocument(source_path)
    returned = ReviewDocument(returned_path)
    issues = returned.validate()
    for key in ('snapshot_id', 'trial_id', 'base_snapshot_sha256'):
        if source.data[key] != returned.data[key]: issues.append('Mismatch: ' + key)
    for key in ('source', 'instances'):
        if source.data[key] != returned.data[key]: issues.append('Frozen metadata modified: ' + key)
    originals = {r['event_uid']: r for r in source.events}
    updates = {r['event_uid']: r for r in returned.events}
    for uid, original in originals.items():
        if uid not in updates: issues.append('Missing original candidate: ' + uid)
        elif original['original'] != updates[uid]['original']: issues.append('Original proposal modified: ' + uid)
    for uid, row in updates.items():
        if uid not in originals and (row.get('origin') != 'human_added' or row.get('original') is not None):
            issues.append('Unrecognized new candidate: ' + uid)
    sw = {w['window_id']: w for w in source.data['windows']}
    rw = {w['window_id']: w for w in returned.data['windows']}
    if set(sw) != set(rw): issues.append('Window replay coverage changed')
    for uid in sw.keys() & rw.keys():
        if {k:v for k,v in sw[uid].items() if k!='review'} != {k:v for k,v in rw[uid].items() if k!='review'}:
            issues.append('Frozen window modified: ' + uid)
    return dict(valid=not issues, issues=issues, trial_id=returned.data['trial_id'],
                reviewer=returned.data.get('reviewer'), snapshot_id=returned.data['snapshot_id'],
                completion=returned.completion(),
                counts={state:sum(r['review']['status']==state for r in returned.events)
                        for state in ('needs_review','reviewed','reviewed_unknown','rejected','duplicate')},
                box_correction_frames=len(returned.data['box_overrides']),
                instance_corrections=len(returned.data['instance_edits']),
                human_accuracy_certified=False)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('returned'); a=p.parse_args()
    result=validate_return(a.source,a.returned)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['valid'] else 1)
