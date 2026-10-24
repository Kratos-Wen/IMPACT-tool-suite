"""Explicit role IDs, including an explicit empty choice, are authoritative."""
def instance_reference(event,role,track_objects,label_resolver=None):
    fields=('noun_object_id','target_object_id') if role=='object' else ('instrument_object_id','tool_object_id')
    for field in fields:
        if field in event:
            try:return int(event[field]) if event[field] is not None else None
            except (TypeError,ValueError):return None
    links=event.get('links') or {}
    keys=('target_track_id',) if role=='object' else ('tool_track_id','instrument_track_id')
    for key in keys:
        tid=links.get(key)
        if tid in track_objects and track_objects[tid] is not None:
            try:return int(track_objects[tid])
            except (TypeError,ValueError):return None
    interaction=event.get('interaction') or {}
    if label_resolver:
        names=('target','noun') if role=='object' else ('instrument','tool')
        value=next((interaction.get(k) for k in names if interaction.get(k)),None)
        if value is not None:return label_resolver(value)
    return None
