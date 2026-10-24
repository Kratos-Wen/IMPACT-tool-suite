"""Video-level human validation, bound to semantic annotation content."""
import copy
import hashlib
import json
import math

def annotation_content(payload):
    events = []
    for hand, rows in (payload.get('hoi_events') or {}).items():
        for row in rows:
            event = {k: row.get(k) for k in ('event_id','verb','start_frame','contact_onset_frame','end_frame','noun_object_id','instrument_object_id','shared_assembly_ref','shared_assembly_id')}
            event.update(hand=hand.lower(), anomaly_labels=sorted(row.get('anomaly_labels') or []), required_review_frames=sorted(row.get('required_review_frames') or []))
            events.append(event)
    tracks = []
    for track in (payload.get('tracks') or {}).values():
        boxes = []
        for box in track.get('boxes', []):
            coords = [round(float(v), 6) for v in box.get('bbox', [])]
            if not all(math.isfinite(v) for v in coords):
                raise ValueError('Invalid bbox coordinates')
            boxes.append([box.get('frame'), coords, box.get('label_override')])
        tracks.append({'object_id':track.get('object_id'), 'category':str(track.get('category') or '').lower(),
                       'boxes':sorted(boxes,key=lambda x:json.dumps(x,sort_keys=True))})
    value = {'video_id':payload.get('video_id'), 'fps':round(float(payload.get('fps') or 0),6),
             'frame_count':payload.get('frame_count'), 'frame_size':payload.get('frame_size'),
             'events':sorted(events,key=lambda x:json.dumps(x,sort_keys=True)),
             'tracks':sorted(tracks,key=lambda x:json.dumps(x,sort_keys=True)),
             'objects':{k:v.get('category',v.get('label')) for k,v in (payload.get('object_library') or {}).items()},
             'shared_assembly':payload.get('shared_assembly'), 'object_states':payload.get('object_states')}
    return value

def annotation_signature(payload):
    return hashlib.sha256(json.dumps(annotation_content(payload),sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')).hexdigest()

def checked_record(record, payload):
    result=copy.deepcopy(record) if isinstance(record,dict) else {}
    if result.get('status')=='validated':
        if not result.get('reviewer') or not result.get('validated_at') or result.get('annotation_sha256')!=annotation_signature(payload):
            result.update(status='needs_review',reason='annotation_changed_or_unbound')
    return result
