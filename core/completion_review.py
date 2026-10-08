"""Deterministic, read-only issue locations for the existing HIOR editor.

Missing geometry is a review request, not proof of an annotation error: the
entity may be invisible. This module does not infer unlabeled actions or damage.
"""
from numbers import Integral

TIMES = {
    'interaction_start': 'start',
    'functional_contact_onset': 'onset',
    'interaction_end': 'end',
}


def frame_value(value):
    return int(value) if isinstance(value, Integral) and not isinstance(value, bool) else None


def issue_key(issue):
    return (issue.get('event_id'), issue.get('hand'), issue.get('field'), issue.get('code'))


def issue_order(issue):
    frame = issue.get('frame')
    return (frame if frame is not None else 10 ** 15, str(issue.get('event_id')),
            str(issue.get('hand')), str(issue.get('field')), str(issue.get('code')))


def next_issue_index(issues, last_key=None, last_order=None, direction=1):
    if not issues:
        return -1
    direction = 1 if direction >= 0 else -1
    keys = [issue_key(i) for i in issues]
    if last_key in keys:
        return (keys.index(last_key) + direction) % len(issues)
    if last_order is not None:
        indexes = range(len(issues)) if direction > 0 else range(len(issues) - 1, -1, -1)
        for index in indexes:
            if (issue_order(issues[index]) > last_order if direction > 0
                    else issue_order(issues[index]) < last_order):
                return index
    return 0 if direction > 0 else len(issues) - 1


def collect_review_issues(events, actors, completion_for_hand, object_ids=None, frame_count=None):
    issues = []
    for event in events:
        event_id = event.get('event_id')
        bounds = event.get('frames') or []
        fallback = next((frame_value(v) for v in bounds if frame_value(v) is not None), 0)
        active_count = 0
        for actor in actors:
            hand = actor['id']
            data = (event.get('hoi_data') or {}).get(hand) or {}
            # A blank second hand is valid. A completely unassigned segment is not.
            active = any(data.get(k) is not None for k in TIMES)
            active = active or bool(str(data.get('verb') or '').strip())
            active = active or any(data.get(k) is not None for k in (
                *(() if data.get('shared_assembly_ref') else ('noun_object_id','target_object_id')), 'instrument_object_id', 'tool_object_id'))
            anomaly = str(data.get('anomaly_label') or '').strip().lower()
            active = active or anomaly not in ('', 'normal', 'none')
            if not active:
                continue
            active_count += 1
            state = completion_for_hand(data, hand) or {}
            times = {key: frame_value(data.get(key)) for key in TIMES}
            default_frame = next((times[k] for k in (
                'functional_contact_onset', 'interaction_start', 'interaction_end')
                if times[k] is not None), fallback)

            def add(field, code, message, frame=None):
                position = default_frame if frame is None else frame
                position = max(0, position)
                if isinstance(frame_count, int) and frame_count > 0:
                    position = min(position, frame_count - 1)
                issues.append(dict(event_id=event_id, hand=hand, field=field, code=code,
                                   missing=[message], frame=position))

            for key, label in TIMES.items():
                value = data.get(key)
                if value is None:
                    add(key, 'missing', 'Review missing ' + label,
                        fallback if key == 'interaction_start' else default_frame)
                elif times[key] is None or times[key] < 0:
                    add(key, 'invalid_frame', 'Invalid ' + label + ' frame', fallback)
                elif isinstance(frame_count, int) and frame_count > 0 and times[key] >= frame_count:
                    add(key, 'outside_video', label + ' is outside the video', times[key])
            start, onset, end = (times[k] for k in TIMES)
            if start is not None and end is not None and start > end:
                add('interaction_start', 'reversed_interval', 'Start is after end', start)
            if onset is not None and ((start is not None and onset < start) or
                                      (end is not None and onset > end)):
                add('functional_contact_onset', 'outside_segment', 'Onset is outside start/end', onset)
            verb = str(data.get('verb') or '').strip()
            if not verb:
                add('verb', 'missing', 'Missing verb')
            elif verb.lower() == 'transfer':
                add('verb', 'legacy_transfer', 'Review single-hand TRANSFER: GIVE / RECEIVE / HOLD')
            noun = data.get('noun_object_id', data.get('target_object_id'))
            instrument = data.get('instrument_object_id', data.get('tool_object_id'))
            if noun is None and ('noun' in state.get('missing', []) or verb):
                add('noun_object_id', 'missing', 'Missing target object')
            if noun is not None and instrument is not None and str(noun) == str(instrument):
                add('instrument_object_id', 'same_as_target', 'Instrument and target refer to the same instance')
            if object_ids is not None:
                valid_ids = {str(v) for v in object_ids}
                for key, value in (('noun_object_id', noun), ('instrument_object_id', instrument)):
                    if value is not None and str(value) not in valid_ids:
                        add(key, 'unknown_object', 'Object ID is absent from the object library')
            for field in state.get('suggested_fields', []):
                if data.get(field) is not None and str(data.get(field)).strip():
                    add(field, 'unconfirmed_suggestion', 'Confirm suggested ' + field,
                        times.get(field))
            roles = state.get('bbox_missing_roles') or {}
            if roles:
                for role, labels in roles.items():
                    for label in dict.fromkeys(labels):
                        key = next((k for k, v in TIMES.items() if v == label), None)
                        add('bbox_evidence', role + '_bbox_' + label,
                            'Review ' + role + ' bbox at ' + label,
                            times.get(key))
            else:
                for missing in state.get('bbox_missing', []):
                    add('bbox_evidence', str(missing), 'Review ' + str(missing))
        if not active_count and actors:
            position = max(0, fallback)
            if isinstance(frame_count, int) and frame_count > 0:
                position = min(position, frame_count - 1)
            issues.append(dict(event_id=event_id, hand=actors[0]['id'], field='verb',
                               code='unassigned_segment', missing=['Segment has no assigned hand action'],
                               frame=position))
    # Aliased S/C/E can refer to one frame, but keep distinct fields to explain each issue.
    unique = {issue_key(i): i for i in issues}
    return sorted(unique.values(), key=issue_order)
