"""Composition belongs to a physical instance; merges end independent tracks."""
from copy import deepcopy

SCHEMA = 'shared-assembly-2'


def empty_timeline():
    return dict(schema=SCHEMA, states=[], merges=[])


def _id(value):
    # The independent review window uses string track UIDs.
    if isinstance(value, str) and value.startswith('EGO_T'):
        value = value[5:]
    if type(value) is bool or isinstance(value,float) and not value.is_integer():
        return None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def merge_at(data, object_id):
    return next((m for m in data.get('merges', [])
                 if m['source_object_id'] == object_id), None)


def active_id(data, object_id, frame):
    """Follow confirmed merges, without changing the stored event reference."""
    uid = _id(object_id)
    seen = set()
    while uid is not None:
        if uid in seen:
            raise ValueError('Assembly merge cycle')
        seen.add(uid)
        merge = merge_at(data, uid)
        if merge is None or merge['frame'] > frame:
            return uid
        uid = merge['target_object_id']
    return None


def active_object_ids(data, frame):
    return sorted({s['object_id'] for s in data.get('states', [])
                   if s['frame'] <= frame and active_id(data, s['object_id'], frame) == s['object_id']})


def validate_timeline(data):
    if not isinstance(data, dict) or data.get('schema') not in ('shared-assembly-1', SCHEMA):
        raise ValueError('Invalid assembly timeline schema')
    result = deepcopy(data)
    result['schema'] = SCHEMA
    result.setdefault('merges', [])
    states = result.get('states', [])
    if not isinstance(states, list):
        raise ValueError('States must be a list')
    seen = set()
    for s in states:
        if not isinstance(s, dict):
            raise ValueError('Invalid state record')
        f, uid = s.get('frame'), s.get('object_id')
        if type(f) is not int or f < 0 or type(uid) is not int or uid < 0:
            raise ValueError('Invalid state frame or object ID')
        if (uid, f) in seen:
            raise ValueError('Duplicate state frame for this object ID')
        seen.add((uid, f))
        parts = s.get('components')
        if (not isinstance(parts, list) or (not parts and s.get('composition_review_state') != 'unknown')
                or any(not isinstance(x, str) or not x.strip() or x == 'assembly' for x in parts)):
            raise ValueError('Select at least one concrete component')
        if len(parts) != len(set(parts)):
            raise ValueError('Duplicate components')
        interfaces = s.get('interfaces', {})
        if not isinstance(interfaces, dict):
            raise ValueError('Invalid interfaces')
        for key, slots in interfaces.items():
            if (not isinstance(key, str) or not key or not isinstance(slots, dict)
                    or set(slots) != {'1', '2'}
                    or any(v is not None and type(v) is not bool for v in slots.values())):
                raise ValueError('Each interface needs fixed slots 1 and 2')
    merges = result['merges']
    if not isinstance(merges, list):
        raise ValueError('Merges must be a list')
    ended = set();frame_members={}
    for m in sorted(merges, key=lambda x: x.get('frame', -1) if isinstance(x, dict) else -1):
        if not isinstance(m, dict):
            raise ValueError('Invalid merge record')
        f, source, target = m.get('frame'), m.get('source_object_id'), m.get('target_object_id')
        if type(f) is not int or f < 0 or any(type(x) is not int or x < 0 for x in (source, target)):
            raise ValueError('Invalid merge frame or object ID')
        if source == target or source in ended or target in ended:
            raise ValueError('An assembly can only merge into a different active instance')
        if {source,target}.intersection(frame_members.get(f,set())):
            raise ValueError('Related merges need distinct confirmed frames')
        frame_members.setdefault(f,set()).update((source,target))
        if not any(s['object_id'] == source and s['frame'] < f for s in states):
            raise ValueError('Source assembly must exist before the merge frame')
        if not any(s['object_id'] == target and s['frame'] <= f for s in states):
            raise ValueError('Receiving assembly must exist at the merge frame')
        if any(s['object_id'] == source and s['frame'] >= f for s in states):
            raise ValueError('A merged instance cannot have later independent states')
        ended.add(source)
    # Sort once on edit/load, not on every displayed video frame.
    result['states'] = sorted(states, key=lambda s: (s['frame'], s['object_id']))
    result['merges'] = sorted(merges, key=lambda m: (m['frame'], m['source_object_id']))
    return result


def reference_id(hand, data=None, frame=0):
    """Explicit instance first. Legacy booleans are safe only when unambiguous."""
    uid = _id(hand.get('shared_assembly_id'))
    if uid is not None:
        return uid
    uid = _id(hand.get('noun_object_id', hand.get('target_object_id', hand.get('object_instance_id'))))
    if uid is not None:
        return uid
    if hand.get('shared_assembly_ref') and data is not None:
        candidates = active_object_ids(data, frame)
        if len(candidates) == 1:
            return candidates[0]
    return None


def state_at(data, frame, object_id=None, follow_merges=True):
    if object_id is None:
        ids = active_object_ids(data, frame)
        if len(ids) != 1:
            return None
        object_id = ids[0]
    uid = active_id(data, object_id, frame) if follow_merges else _id(object_id)
    states = [s for s in data.get('states', []) if s['object_id'] == uid and s['frame'] <= frame]
    return deepcopy(max(states, key=lambda s: s['frame'])) if states else None


def put_state(data, state):
    result = validate_timeline(data)
    result['states'] = [s for s in result['states']
                        if (s['object_id'], s['frame']) != (state.get('object_id'), state.get('frame'))]
    result['states'].append(deepcopy(state))
    return validate_timeline(result)


def noun_at(data, frame, object_id=None):
    s = state_at(data, frame, object_id)
    return None if s is None else s['components'][0] if len(s['components']) == 1 else 'assembly'


def next_change(data, frame, object_id=None):
    if object_id is None:
        ids = active_object_ids(data, frame)
        if len(ids) != 1:
            return None
        object_id = ids[0]
    uid = active_id(data, object_id, frame)
    changes = [s['frame'] for s in data.get('states', []) if s['object_id'] == uid and s['frame'] > frame]
    merge = merge_at(data, uid)
    if merge and merge['frame'] > frame:
        changes.append(merge['frame'])
    return min(changes, default=None)


def change_frames(data, object_id, start, end):
    """Only changes on this instance's path, including merge boundary anchors."""
    frames = []
    cursor = start
    while cursor < end:
        f = next_change(data, cursor, object_id)
        if f is None or f > end:
            break
        frames.append(f)
        cursor = f
    return frames


def resolve_object(hand, data, frame):
    uid = reference_id(hand, data, frame)
    return active_id(data, uid, frame) if uid is not None else None


def track_segments(data, object_id, start, end):
    """Inclusive geometry intervals; one HOI event can span a confirmed merge."""
    if type(start) is not int or type(end) is not int or start > end:
        return []
    uid, cursor = active_id(data, object_id, start), start
    if uid is None:
        return []
    rows = []
    while cursor <= end:
        merge = merge_at(data, uid)
        stop = min(end, merge['frame'] - 1) if merge else end
        rows.append(dict(object_id=uid, track_id=f'T_OBJ_{uid}', start_frame=cursor, end_frame=stop))
        if stop == end:
            break
        cursor, uid = stop + 1, merge['target_object_id']
    return rows


def merge_instances(data, source, target, frame, interfaces=None):
    result = validate_timeline(data)
    if (type(source) is not int or type(target) is not int or source == target
            or source not in active_object_ids(result, frame) or target not in active_object_ids(result, frame)):
        raise ValueError('Select two different active assembly IDs')
    a, b = state_at(result, frame, source), state_at(result, frame, target)
    if len(a['components']) < 2 or len(b['components']) < 2:
        raise ValueError('Both instances must be assemblies with at least two components')
    composition = list(dict.fromkeys(b['components'] + a['components']))
    slots = deepcopy(b.get('interfaces', {}))
    for name, values in a.get('interfaces', {}).items():
        slots.setdefault(name, deepcopy(values))
    if interfaces is not None:
        slots = deepcopy(interfaces)
    # Keep superseded predictions as history, never as an active second track.
    superseded = [s for s in result['states'] if s['object_id'] == source and s['frame'] >= frame]
    result['states'] = [s for s in result['states'] if s not in superseded]
    result['merges'].append(dict(frame=frame, source_object_id=source, target_object_id=target,
                                  superseded_states=deepcopy(superseded)))
    # Future receiving states may have been generated before this merge was known.
    for s in result['states']:
        if s['object_id'] == target and s['frame'] > frame and not set(composition).issubset(s['components']):
            s['composition_review_state'] = 'unreviewed'
    result['states'] = [s for s in result['states'] if not (s['object_id'] == target and s['frame'] == frame)]
    result['states'].append(dict(frame=frame, object_id=target, components=composition, interfaces=slots,
                                  composition_review_state='reviewed', source='human_merge'))
    return validate_timeline(result)
