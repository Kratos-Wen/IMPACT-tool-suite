"""Time-varying counted attributes of stable physical instances."""
from copy import deepcopy

SCHEMA = 'instance-count-states-1'

def empty_states():
    return {'schema': SCHEMA, 'states': []}

def validate_states(data, profile, frame_count=None, object_ids=None):
    if not isinstance(data, dict) or data.get('schema') != SCHEMA or not isinstance(data.get('states'), list):
        raise ValueError('Invalid instance state timeline')
    result=deepcopy(data);seen=set();specs=profile.get('counted_object_states', {})
    for s in result['states']:
        if not isinstance(s,dict):raise ValueError('Invalid instance state')
        uid,f=s.get('object_id'),s.get('frame');kind=s.get('kind');count=s.get('count')
        if type(uid) is not int or uid<0 or type(f) is not int or f<0 or kind not in specs:
            raise ValueError('Invalid state frame, instance ID or kind')
        if frame_count is not None and f>=frame_count:raise ValueError('Instance state outside video')
        if object_ids is not None and uid not in object_ids:raise ValueError('Unknown instance state ID')
        if count is not None and (type(count) is not int or not 0<=count<=specs[kind]['maximum']):
            raise ValueError('Invalid completed count')
        if (uid,f) in seen:raise ValueError('Duplicate instance state frame')
        seen.add((uid,f))
        parent=s.get('parent_object_id')
        if parent is not None and (type(parent) is not int or parent<0 or parent==uid or (object_ids is not None and parent not in object_ids)):
            raise ValueError('Invalid parent instance ID')
    result['states'].sort(key=lambda s:(s['frame'],s['object_id']))
    return result

def state_at(data, uid, frame):
    rows=[s for s in data.get('states',[]) if s['object_id']==uid and s['frame']<=frame]
    return deepcopy(max(rows,key=lambda s:s['frame'])) if rows else None

def put_state(data, state, profile):
    result=validate_states(data,profile)
    result['states']=[s for s in result['states'] if (s['object_id'],s['frame'])!=(state['object_id'],state['frame'])]
    result['states'].append(deepcopy(state))
    return validate_states(result,profile)

def noun_at(data, uid, frame, fallback, profile):
    s=state_at(data,uid,frame)
    if s is None or s['count'] is None:return fallback
    return profile['counted_object_states'][s['kind']]['noun_by_count'][str(s['count'])]

def next_instance_id(registry, boxes, timeline, counter):
    used={int(v) for v in registry.values()}
    used.update(b['id'] for b in boxes if type(b.get('id')) is int and b['id']>=0)
    used.update(s['object_id'] for s in timeline.get('states',[]))
    return max(int(counter),max(used,default=-1)+1)
