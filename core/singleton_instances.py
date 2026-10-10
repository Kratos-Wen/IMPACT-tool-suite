"""Apply explicit project instance constraints to unreviewed machine drafts."""
from copy import deepcopy
from collections import defaultdict
from core.instance_category import instance_category

VERSION='singleton-identities-2'
ROLE_IDS={'object_id','noun_object_id','target_object_id','instrument_object_id','tool_object_id','shared_assembly_id','object_instance_id','instrument_instance_id'}
RAW_FIELDS={'raw_response','raw_model_output','machine_original','machine_snapshot','raw_snapshot','source_record','provenance'}


def human_work_present(doc):
    if not doc.get('provenance',{}).get('machine_only') or doc.get('frame_review'):
        return True
    for events in doc.get('hoi_events',{}).values():
        for event in events:
            if event.get('human_verified') or event.get('anomaly_review_state') in ('reviewed','partial'):
                return True
    for track in doc.get('tracks',{}).values():
        for box in track.get('boxes',[]):
            if box.get('human_verified') or any(word in str(box.get('source','')).lower() for word in ('human','manual')):
                return True
    return False


def normalize_machine_instances(doc,profile):
    """Return a copy. Conflicting machine boxes are withheld, never guessed."""
    if human_work_present(doc):
        raise ValueError('Preserve human-edited annotation IDs')
    classes=set(profile.get('singleton_noun_classes',[]))
    fixed=profile.get('singleton_object_ids',{})
    if any(type(fixed.get(name)) is not int or fixed[name]<0 for name in classes):
        raise ValueError('Each singleton needs an explicit nonnegative object ID')
    if len({fixed[name] for name in classes})!=len(classes):
        raise ValueError('Singleton object IDs must be distinct')
    result=deepcopy(doc);library=result.get('object_library',{});tracks=result.get('tracks',{})
    categories=defaultdict(set)
    for uid,entry in library.items():
        categories[int(uid)].add(instance_category(entry.get('category') or entry.get('label'),profile))
    for track in tracks.values():
        if type(track.get('object_id')) is int:
            categories[track['object_id']].add(instance_category(track.get('category'),profile))
    groups=defaultdict(list)
    for uid,names in categories.items():
        if len(names)>1:raise ValueError('Conflicting categories for object ID '+str(uid))
        name=next(iter(names))
        if name in classes:groups[name].append(uid)
    # An assembly composition cannot be inferred from a single-part registry name.
    assembly_ids={s['object_id'] for s in result.get('shared_assembly',{}).get('states',[]) if len(s.get('components',[]))>1}
    if any(assembly_ids.intersection(ids) for ids in groups.values()):
        raise ValueError('Assembly state and single-part category conflict')
    numeric={uid:fixed[name] for name,ids in groups.items() for uid in ids}
    reserved=set(fixed.values());used=set(categories)|reserved
    next_id=max(2000,max(used,default=-1)+1)
    for uid in sorted(categories):
        if uid not in numeric and uid in reserved:
            while next_id in used:next_id+=1
            numeric[uid]=next_id;used.add(next_id);next_id+=1
    aliases={}
    for key,track in tracks.items():
        old=track.get('object_id');new=numeric.get(old,old)
        aliases[key]='T_OBJ_'+str(new) if old in numeric else key
    for old,new in numeric.items():
        if old!=new:aliases.setdefault('T_OBJ_'+str(old),'T_OBJ_'+str(new))
    def references(value):
        if isinstance(value,list):return [references(x) for x in value]
        if not isinstance(value,dict):return value
        updated={}
        for key,item in value.items():
            if key in RAW_FIELDS:updated[key]=deepcopy(item)
            elif key in ROLE_IDS and type(item) is int:updated[key]=numeric.get(item,item)
            elif key in ROLE_IDS and isinstance(item,dict) and type(item.get('value')) is int:
                updated[key]=references(item);updated[key]['value']=numeric.get(item['value'],item['value'])
            elif (key.endswith('_track_id') or key=='track_id') and isinstance(item,str):updated[key]=aliases.get(item,item)
            else:updated[key]=references(item)
        return updated
    result=references(result)
    new_library={}
    for old,entry in library.items():
        uid=int(old);new=numeric.get(uid,uid);name=next(iter(categories[uid]));item=deepcopy(entry)
        if name in groups:item.update(label=name,category=name)
        elif new!=uid and str(item.get('label','')).endswith('_'+str(uid)):
            item['label']=item['label'][:-len(str(uid))]+str(new)
        new_library.setdefault(str(new),item)
    new_tracks={};candidates=defaultdict(list)
    for key,track in tracks.items():
        renamed=aliases[key];item=references(track);rows=item.pop('boxes',[])
        entry=new_tracks.setdefault(renamed,item)
        entry.setdefault('source_track_aliases',[])
        entry['source_track_aliases']=list(dict.fromkeys(entry['source_track_aliases']+(track.get('source_track_aliases') or [key])))
        uid=track.get('object_id');name=next(iter(categories[uid])) if uid in categories else None
        if name in groups:
            entry.update(instance_name=name,global_instance_uid=str(doc['video_id'])+'::'+name,identity_status='project_singleton_constraint',human_verified=False)
        for box in rows:candidates[renamed].append(box)
    conflicts=[]
    for key,track in new_tracks.items():
        frames=defaultdict(list)
        for box in candidates[key]:frames[box['frame']].append(box)
        track['boxes']=[]
        for frame,rows in sorted(frames.items()):
            shapes={tuple(row.get('bbox',[])) for row in rows}
            if len(shapes)==1:track['boxes'].append(rows[0])
            else:conflicts.append(dict(track_id=key,frame=frame,box_candidates=rows))
    result['object_library']=new_library;result['tracks']=new_tracks
    prior=result.setdefault('provenance',{}).get('singleton_identity',{})
    all_numeric={old:numeric.get(int(new),int(new)) for old,new in prior.get('object_id_aliases',{}).items()}
    all_numeric.update({str(old):new for old,new in numeric.items() if old!=new})
    all_aliases={old:aliases.get(new,new) for old,new in prior.get('track_id_aliases',{}).items()}
    all_aliases.update({old:new for old,new in aliases.items() if old!=new})
    result['provenance']['singleton_identity']=dict(version=VERSION,singleton_classes=sorted(classes),
        object_id_aliases=all_numeric,track_id_aliases=all_aliases,
        global_instance_uids={name: str(doc['video_id'])+'::'+name for name in groups},
        withheld_box_conflicts=prior.get('withheld_box_conflicts',[])+conflicts)
    return result


def catalog_registry(trial,catalog,profile):
    """A canonical view of detector IDs; ambiguous labels remain unresolved."""
    aliases={};unresolved=[]
    for uid,entry in catalog.items():
        names={instance_category(n,profile) for n in entry.get('category_candidates',[])}
        if len(names)==1 and next(iter(names)) in profile.get('singleton_noun_classes',[]):
            name=next(iter(names));aliases[uid]=dict(instance_name=name,track_uid=name,
                global_instance_uid=trial+'::'+name,object_id=profile['singleton_object_ids'][name])
        elif names.intersection(profile.get('singleton_noun_classes',[])):
            unresolved.append(uid)
    return dict(version=VERSION,trial=trial,aliases=aliases,ambiguous_track_uids=unresolved,
                predictions_human_verified=False)


def normalize_tracking_frame(frame,registry):
    """Expose canonical IDs on demand, preserving detector candidates as history."""
    result=deepcopy(frame);groups=defaultdict(list);untouched=[]
    for track in result.get('tracks',[]):
        alias=registry['aliases'].get(track.get('track_uid'))
        if alias is None:untouched.append(track);continue
        track['source_track_uid']=track['track_uid'];track['source_track_id']=track.get('track_id')
        track.update(track_uid=alias['track_uid'],track_id=alias['object_id'],global_instance_uid=alias['global_instance_uid'])
        groups[alias['track_uid']].append(track)
    conflicts=[]
    for uid,rows in groups.items():
        visible=[row for row in rows if row.get('visible',row.get('bbox_xyxy') is not None) and row.get('bbox_xyxy') is not None]
        shapes={tuple(row['bbox_xyxy']) for row in visible}
        if len(shapes)<=1:
            chosen=(visible or rows)[0];chosen['source_track_uids']=[row['source_track_uid'] for row in rows]
            untouched.append(chosen)
        else:conflicts.append(dict(track_uid=uid,candidates=rows))
    result['tracks']=untouched;result['identity_conflicts']=conflicts
    return result
