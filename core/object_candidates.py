"""Optional sparse native-view boxes, pinned to an unchanged task draft."""
import copy,hashlib,json,math
from pathlib import Path

def merge_candidates(payload, document, source_md5):
    report={'added':0,'skipped':0,'reason':''}
    if not isinstance(document,dict):
        return payload,dict(report,reason='unsupported_candidate_format')
    if document.get('schema')!='impact-sparse-object-candidates-1' or document.get('coordinate_space')!='native_video':
        return payload,dict(report,reason='unsupported_candidate_format')
    receipts=(payload.get('provenance') or {}).get('object_candidate_imports',[])
    if any(isinstance(x,dict) and x.get('source_annotations_md5')==document.get('source_annotations_md5') for x in receipts):
        return payload,dict(report,reason='already_imported')
    if document.get('source_annotations_md5')!=source_md5:
        return payload,dict(report,reason='saved_work_takes_priority')
    if (document.get('video_id')!=payload.get('video_id') or document.get('frame_count')!=payload.get('frame_count')
            or document.get('frame_size')!=payload.get('frame_size') or document.get('fps')!=payload.get('fps')):
        return payload,dict(report,reason='video_identity_mismatch')
    rows=document.get('boxes') or []
    if not isinstance(rows,list) or len(rows)>50000:
        return payload,dict(report,reason='candidate_limit_exceeded')
    width,height=payload['frame_size'];library=payload.get('object_library') or {}
    result=copy.deepcopy(payload);occupied=set();events={}
    for track in result.get('tracks',{}).values():
        uid=track.get('object_id')
        if uid is not None:
            occupied.update((str(uid),b.get('frame')) for b in track.get('boxes',[]))
    for hand,entries in result.get('hoi_events',{}).items():
        for event in entries:events.setdefault((hand,str(event.get('event_id'))),[]).append(event)
    for row in rows:
        try:
            uid=str(row['object_id']);frame=row['frame'];bbox=row['bbox'];quality=row['quality']
            if not isinstance(quality,dict):raise ValueError()
            if type(frame) is not int or not 0<=frame<payload['frame_count'] or uid not in library:raise ValueError()
            if (uid,frame) in occupied:raise ValueError()
            if len(bbox)!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in bbox):raise ValueError()
            x1,y1,x2,y2=bbox
            if not (0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError()
            if not quality.get('passed') or quality.get('confidence_min',0)<.85 or quality.get('temporal_iou_min',0)<.65:raise ValueError()
            role=row.get('role');field='noun_object_id' if role=='object' else 'instrument_object_id' if role=='instrument' else None
            linked=[e for e in events.get((row.get('hand'),str(row.get('event_id'))),[])
                    if field and str(e.get(field))==uid and type(e.get('start_frame')) is int and type(e.get('end_frame')) is int
                    and e['start_frame']<=frame<=e['end_frame']]
            if not linked:raise ValueError()
            key='T_OBJ_'+uid
            track=result['tracks'].setdefault(key,dict(object_id=int(uid),category=library[uid].get('category',library[uid].get('label')),boxes=[]))
            if str(track.get('object_id'))!=uid:raise ValueError()
            track['boxes'].append(dict(frame=frame,bbox=list(bbox),source='machine_candidate_native_consensus',human_verified=False))
            occupied.add((uid,frame));report['added']+=1
        except (KeyError,TypeError,ValueError):report['skipped']+=1
    for track in result.get('tracks',{}).values():track.get('boxes',[]).sort(key=lambda b:b['frame'])
    if report['added']:
        result.setdefault('provenance',{}).setdefault('object_candidate_imports',[]).append({
            'source_annotations_md5':source_md5,'coordinate_space':'native_video',
            'source':'native_detector_consensus','machine_boxes_added':report['added']})
    return result,report

def load_adjacent_candidates(annotation_path,payload):
    path=Path(annotation_path).with_name('object_candidates.json')
    if not path.is_file():return payload,{'added':0,'reason':'no_sidecar'}
    try:
        if path.stat().st_size>10*1024*1024:raise ValueError('Candidate file exceeds 10 MiB')
        document=json.loads(path.read_text(encoding='utf-8-sig'))
        digest=hashlib.md5(Path(annotation_path).read_bytes()).hexdigest()
        return merge_candidates(payload,document,digest)
    except (OSError,ValueError,TypeError,KeyError) as exc:
        return payload,{'added':0,'reason':'candidate_file_skipped','detail':str(exc)}
