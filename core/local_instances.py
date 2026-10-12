"""Stable video-local object IDs, with source IDs retained as import aliases."""
from copy import deepcopy
from collections import defaultdict
from core.instance_category import instance_category

SCHEMA='video-local-instances-1'
ROLES={'object_id','noun_object_id','target_object_id','instrument_object_id','tool_object_id',
       'shared_assembly_id','object_instance_id','instrument_instance_id','source_object_id','target_object_id',
       'assembly_default_reference'}
HISTORY={'provenance','migration_review','source_record','raw_response','raw_model_output','machine_original',
         'machine_snapshot','raw_snapshot','source_snapshot','source_object_id'}


def localize_instances(source,profile):
    """Normalize once on import. Reopening and deleting instances never renumbers IDs."""
    data=deepcopy(source)
    if data.get('provenance',{}).get('video_local_instances',{}).get('schema')==SCHEMA:
        return data
    library=data.get('object_library',{});tracks=data.get('tracks',{})
    names={int(uid):instance_category(info.get('category') or info.get('label'),profile)
           for uid,info in library.items()}
    object_tracks={}
    for tid,track in tracks.items():
        uid=track.get('object_id');category=instance_category(track.get('category'),profile)
        if type(uid) is int and category not in ('hand','left_hand','right_hand') and not any(x in tid.lower() for x in ('left_hand','right_hand')):
            names.setdefault(uid,category or 'object');object_tracks[tid]=uid
    assembly_ids={s['object_id'] for s in data.get('shared_assembly',{}).get('states',[]) if len(s.get('components',[]))>1}
    assembly_ids.update(m[k] for m in data.get('shared_assembly',{}).get('merges',[]) for k in ('source_object_id','target_object_id'))
    groups=defaultdict(list);singletons=set(profile.get('singleton_noun_classes',[]))
    for uid,name in names.items():
        if name in singletons and uid not in assembly_ids:groups[name].append(uid)
    roots={uid:uid for uid in names}
    for ids in groups.values():
        for uid in ids:roots[uid]=min(ids)
    active=sorted(set(roots.values()))
    compact=max(active,default=0)>99
    display={uid:i+1 for i,uid in enumerate(active)} if compact else {uid:uid for uid in active}
    numeric={uid:display[root] for uid,root in roots.items()}
    aliases={tid:'T_OBJ_'+str(numeric[uid]) for tid,uid in object_tracks.items()}
    aliases.update({'T_OBJ_'+str(uid):'T_OBJ_'+str(new) for uid,new in numeric.items()})

    def refs(value,context=''):
        if isinstance(value,list):return [refs(x,context) for x in value]
        if not isinstance(value,dict):return value
        updated={}
        for k,v in value.items():
            # Assembly merge IDs are active references; migration source IDs are history.
            if k in HISTORY and context not in ('merges','states'):updated[k]=deepcopy(v)
            elif (k in ROLES or context in ROLES and k in ('value','suggested_value','current_value')) and type(v) is int:
                updated[k]=numeric.get(v,v)
            elif (k.endswith('_track_id') or k=='track_id') and isinstance(v,str):updated[k]=aliases.get(v,v)
            else:updated[k]=refs(v,k if k in ROLES or k in ('merges','states') else context)
        return updated

    result=refs(data);new_library={}
    for old,info in library.items():
        uid=int(old);new=numeric[uid];entry=deepcopy(info)
        entry.update(category=names[uid],label=names[uid])
        new_library.setdefault(str(new),entry)
    for uid,name in names.items():new_library.setdefault(str(numeric[uid]),{'label':name,'category':name})
    result['object_library']=new_library
    new_tracks={}
    for tid,track in tracks.items():
        if tid not in object_tracks:new_tracks[tid]=deepcopy(track);continue
        uid=object_tracks[tid];new_tid=aliases[tid];row=refs(track)
        row['category']=names[uid];row['instance_name']=names[uid]
        merged=new_tracks.setdefault(new_tid,{k:v for k,v in row.items() if k not in ('boxes','inactive_boxes')})
        for field in ('boxes','inactive_boxes'):
            if field in row:merged.setdefault(field,[]).extend(deepcopy(row[field]))
    # Deduplicate identical boxes only. Distinct geometry remains visible for correction.
    for track in new_tracks.values():
        for field in ('boxes','inactive_boxes'):
            if field not in track:continue
            unique={}
            for box in track[field]:
                shape=(box.get('frame'),tuple(box.get('bbox') or []))
                old=unique.get(shape)
                human=bool(box.get('human_verified') or str(box.get('source','')).startswith(('manual','human')))
                if old is None or human:unique[shape]=box
            track[field]=list(unique.values())
    result['tracks']=new_tracks
    reviews={}
    for frame,entities in data.get('frame_review',{}).items():
        rows={};conflicts=set()
        for key,record in entities.items():
            new='O:'+str(numeric.get(int(key[2:]),int(key[2:]))) if key.startswith('O:') else key
            if new in rows and rows[new]!=record:conflicts.add(new)
            else:rows[new]=deepcopy(record)
        reviews[frame]={k:v for k,v in rows.items() if k not in conflicts}
    result['frame_review']=reviews
    result.setdefault('provenance',{})['video_local_instances']={
        'schema':SCHEMA,'original_to_local':{str(k):v for k,v in numeric.items()},
        'source_track_aliases':aliases,'singleton_categories':sorted(singletons),
        'coordinate_values_preserved':True}
    return result
