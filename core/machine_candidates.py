"""Convert raw event proposals to editable annotations without inventing identities."""
import copy
import hashlib
from core.annotation_migration import adapt_annotation


def editor_candidates(review, *, frames, fps, profile, instance_hints=None):
    instances = copy.deepcopy(instance_hints or review.get('instances') or {})
    source_events = review.get('events', [])
    referenced = set()
    for row in source_events:
        event = row.get('event', row.get('value', {}))
        referenced.update(x for x in (event.get('object_instance_id'), event.get('instrument_instance_id')) if x is not None)
    ids = {}
    used = set()
    for uid in sorted(set(instances) | referenced):
        proposed = instances.get(uid, {}).get('track_id')
        if type(proposed) is int and proposed >= 0 and proposed not in used:
            ids[uid] = proposed
            used.add(proposed)
    for uid in sorted(set(instances) | referenced):
        if uid not in ids:
            proposed = max(used, default=-1) + 1
            ids[uid] = proposed
            used.add(proposed)
    def category(uid):
        info = instances.get(uid, {})
        return info.get('category') or next(iter(info.get('category_candidates') or []), '')
    events = {'left_hand': [], 'right_hand': []}
    seen = set()
    skipped = []
    for index, row in enumerate(source_events):
        if row.get('refinement_withdrew_candidate'):
            skipped.append(dict(index=index, reason='refinement_withdrew_candidate'))
            continue
        v = row.get('event', row.get('value', {}))
        side = v.get('hand')
        start, onset, end = (v.get(k) for k in ('start_frame', 'onset_frame', 'end_frame'))
        if side not in ('left', 'right') or type(start) is not int or type(end) is not int or not 0 <= start <= end < frames:
            skipped.append(dict(index=index, reason='invalid_hand_or_interval'))
            continue
        if onset is not None and (type(onset) is not int or not start <= onset <= end):
            onset = None
        signature = (side, start, onset, end, v.get('verb'), v.get('object_instance_id'), v.get('instrument_instance_id'))
        if signature in seen:
            skipped.append(dict(index=index, reason='exact_duplicate'))
            continue
        seen.add(signature)
        obj, tool = v.get('object_instance_id'), v.get('instrument_instance_id')
        uid = row.get('event_uid') or hashlib.sha256(repr((review['trial_id'], row.get('source_window_id'), index, signature)).encode()).hexdigest()[:24]
        events[side + '_hand'].append(dict(
            event_id=('L_' if side == 'left' else 'R_') + uid, machine_event_uid=uid,
            start_frame=start, contact_onset_frame=onset, end_frame=end,
            verb=v.get('verb') or '', noun_object_id=ids.get(obj), instrument_object_id=ids.get(tool),
            shared_assembly_ref=bool(v.get('shared_assembly_ref')),shared_assembly_id=v.get('shared_assembly_id'),
            anomaly_labels=[], anomaly_review_state='unreviewed', human_verified=False,
            interaction={'target': category(obj), 'instrument': category(tool)},
            links={'target_track_id': obj, 'tool_track_id': tool, 'subject_track_id': None},
            migration_review={'machine_review_reasons': v.get('review_reasons', []),
                              'instance_identity_requires_review': bool(obj and not category(obj) or tool and not category(tool))},
            annotation_state={'field_state': {key: {'status': 'suggested', 'source': 'machine_candidate'} for key in
                ('verb', 'noun', 'instrument', 'interaction_start', 'functional_contact_onset', 'interaction_end')}}))
    # Unknown instance names remain conspicuous and editable; no category guessed from an ID.
    library = {str(ids[u]): {'label': (category(u) or 'unresolved_object') + '_' + str(ids[u]),
                            'category': category(u) or 'unresolved_object'} for u in ids}
    tracks = {u: {'object_id': ids[u], 'category': category(u) or 'unresolved_object', 'boxes': []} for u in ids}
    result = dict(schema='hoi-annotation', video_id=review['trial_id'], video_path='video.mp4',
                  frame_count=frames, fps=fps, bbox_mode='xyxy', bbox_normalized=False,
                  object_library=library, verb_library={}, tracks=tracks, hoi_events=events,
                  provenance={'machine_only': True, 'model': review.get('model'),
                              'source_updated_at': review.get('updated_at_utc'),
                              'geometry_display': 'human anchors required; automatic geometry withheld',
                              'conversion_exclusions': skipped})
    if 'shared_assembly' in review:result['shared_assembly']=copy.deepcopy(review['shared_assembly'])
    return adapt_annotation(result, profile)
