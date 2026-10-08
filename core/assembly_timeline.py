"""Shared, frame-indexed composition; no project taxonomy is embedded here."""
from copy import deepcopy

def validate_timeline(data):
    if not isinstance(data, dict) or data.get('schema') != 'shared-assembly-1':
        raise ValueError('Invalid shared assembly schema')
    states = data.get('states', [])
    if not isinstance(states, list): raise ValueError('States must be a list')
    seen = set()
    for s in states:
        if not isinstance(s,dict):raise ValueError('Invalid state record')
        f = s.get('frame')
        if type(f) is not int or f < 0 or f in seen: raise ValueError('Invalid or duplicate state frame')
        seen.add(f)
        if type(s.get('object_id')) is not int or s['object_id'] < 0: raise ValueError('Invalid object ID')
        parts = s.get('components')
        if not isinstance(parts, list) or not parts or any(not isinstance(x,str) or not x.strip() or x == 'assembly' for x in parts):
            raise ValueError('Select at least one concrete component')
        if len(parts) != len(set(parts)): raise ValueError('Duplicate components')
        interfaces=s.get('interfaces',{})
        if not isinstance(interfaces,dict):raise ValueError('Invalid interfaces')
        for key,slots in interfaces.items():
            if not isinstance(key,str) or not key or not isinstance(slots,dict) or set(slots) != {'1','2'} or any(type(v) is not bool for v in slots.values()):
                raise ValueError('Each interface needs fixed slots 1 and 2')
    return deepcopy(data)

def state_at(data, frame):
    states = [s for s in data.get('states', []) if s['frame'] <= frame]
    return deepcopy(max(states, key=lambda s:s['frame'])) if states else None

def put_state(data, state):
    result = deepcopy(data)
    result['states'] = sorted([s for s in result.get('states',[]) if s['frame'] != state['frame']] + [deepcopy(state)],key=lambda s:s['frame'])
    return validate_timeline(result)

def noun_at(data, frame):
    s = state_at(data,frame)
    return None if s is None else s['components'][0] if len(s['components']) == 1 else 'assembly'

def next_change(data, frame):
    return min((s['frame'] for s in data.get('states',[]) if s['frame'] > frame),default=None)

def resolve_object(hand, data, frame):
    if hand.get('shared_assembly_ref'):
        s = state_at(data,frame)
        return s['object_id'] if s else None
    return hand.get('noun_object_id',hand.get('target_object_id'))
