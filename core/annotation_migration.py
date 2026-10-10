"""Lossless geometry import adapter; project semantics come from a private profile."""
import copy, re
from core.anomaly_attributes import export_review, display_value, key

SCHEMA = 'hoi-annotation'

def model_from_trial(trial):
    match=re.search(r'(\d{6,7}_r\d+)_([ab])_([ad])(?:_|\b)',str(trial),re.I)
    return match.group(2).upper() if match else None

def canonical_noun(name, profile):
    value=str(name or '').strip('\ufeff').strip()
    aliases={key(a):b for a,b in profile.get('noun_aliases',{}).items()}
    k=key(value)
    if k in aliases:return aliases[k]
    match=re.fullmatch(r'(.*)_(\d+)',k)
    if match and match.group(1) in aliases:return str(aliases[match.group(1)])+'_'+match.group(2)
    return value

def canonical_verb(value, profile):
    verb=str(value or '').strip('\ufeff').strip().lower()
    return (profile.get('verb_aliases',{}) or {}).get(key(verb),verb)

def _target(event, tracks):
    uid=event.get('noun_object_id',event.get('target_object_id'))
    if uid is None:
        tid=(event.get('links') or {}).get('target_track_id')
        uid=(tracks.get(tid) or {}).get('object_id')
    try:return int(uid)
    except (TypeError,ValueError):return None

def adapt_annotation(source, profile):
    if not isinstance(source,dict) or 'tracks' not in source or 'hoi_events' not in source:
        raise ValueError('Expected tracks and hoi_events in the annotation')
    data=copy.deepcopy(source)
    modern=data.get('schema')==SCHEMA
    library=data.get('object_library',{})
    if not isinstance(library,dict):raise ValueError('Object library must be indexed by ID')
    mappings={key(k):list(v) if isinstance(v,list) else None for k,v in profile.get('legacy_assembly_mappings',{}).items()}
    assembly_ids={}
    for uid,info in library.items():
        if not isinstance(info,dict):continue
        category=key(info.get('category') or info.get('label') or info.get('name'))
        if category not in mappings:
            match=re.fullmatch(r'(.*)_(\d+)',category)
            if match and match.group(1) in mappings:category=match.group(1)
        if category in mappings:assembly_ids[int(uid)]=category
        else:
            for field in ('label','name','category'):
                if field in info:
                    info[field]=canonical_noun(info[field],profile)
                    if field=='category':
                        match=re.fullmatch(r'(.*)_(\d+)',info[field])
                        if match and match[1] in profile.get('noun_aliases',{}).values():info[field]=match[1]
    tracks=data.get('tracks',{})
    if not isinstance(tracks,dict):raise ValueError('Tracks must be indexed by track ID')
    model=model_from_trial(data.get('video_id','')+' '+data.get('video_path',''))
    forbidden=set((profile.get('model_component_rules',{}).get(model,{}) or {}).get('forbidden_components',[]))
    new_states=[];old_events=0
    for hand,events in data.get('hoi_events',{}).items():
        if not isinstance(events,list):raise ValueError('Hand events must be a list')
        for event in events:
            if not isinstance(event,dict):continue
            if not modern or 'anomaly_labels' not in event:
                raw=event.get('anomaly_label',event.get('anomaly_type',''))
                review=event.setdefault('migration_review',{})
                if raw:review['source_anomaly']=copy.deepcopy(raw)
                # Legacy negatives and broad labels do not certify the current taxonomy.
                candidates=(profile.get('legacy_anomaly_candidates',{}) or {}).get(key(raw),[])
                event['anomaly_labels']=list(candidates)
                event['anomaly_review_state']='unreviewed'
                old_events+=1
            else:
                event.update(export_review(display_value(event,profile),profile))
            for obsolete in ('anomaly_label','anomaly_type','version'):
                event.pop(obsolete,None)
            event['has_anomaly']=bool(event['anomaly_labels'] and event['anomaly_review_state'] in ('reviewed','partial'))
            fields=(event.get('annotation_state') or {}).get('field_state',{})
            for field,value in list(fields.items()):
                if isinstance(value,str):fields[field]={'status':value,'source':'imported_candidate'}
            verb=canonical_verb(event.get('verb',''),profile)
            if key(verb) in profile.get('forbidden_verbs',{}):
                event.setdefault('migration_review',{})['source_verb']=verb
                event['verb']=''
            else:event['verb']=verb
            interaction=event.get('interaction',{})
            for field in ('target','noun','instrument','tool'):
                if field in interaction:interaction[field]=canonical_noun(interaction[field],profile)
            uid=_target(event,tracks)
            if uid in assembly_ids and not modern:
                label=assembly_ids[uid];components=[canonical_noun(x,profile) for x in mappings[label] or []]
                rejected=[x for x in components if x in forbidden]
                components=[x for x in components if x not in forbidden]
                event.setdefault('migration_review',{}).update(source_object_id=uid,source_assembly=label,
                    composition_review_required=True,handle_review_required=True,screw_review_required=True)
                if rejected:event['migration_review']['model_conflicts']=rejected
                frame=event.get('start_frame')
                if type(frame) is int and frame>=0:
                    state={'frame':frame,'object_id':uid,'components':list(dict.fromkeys(components)),
                           'composition_review_state':'unreviewed' if components else 'unknown','source_assembly':label,
                           'interfaces':{spec['id']:{'1':None,'2':None} for spec in profile.get('assembly_interfaces',[])}}
                    new_states.append(state)
                event['shared_assembly_ref']=True
                event['shared_assembly_id']=uid
                event['noun_object_id']=event['target_object_id']=uid
                interaction['target']=interaction['noun']=components[0] if len(components)==1 else 'assembly'
    if new_states:
        # Simultaneous contradictory source names remain review issues, never silently resolved.
        grouped={}
        for state in sorted(new_states,key=lambda x:(x['frame'],x['object_id'],x['source_assembly'])):
            k=(state['object_id'],state['frame'])
            if k in grouped and grouped[k]['components']!=state['components']:
                grouped[k].setdefault('conflicting_candidates',[]).append(state['components'])
            else:grouped[k]=state
        # Keep only composition changes; regular HOI starts do not create new state boundaries.
        states=[];last={}
        for state in grouped.values():
            prior=last.get(state['object_id'])
            if prior is None or prior['components']!=state['components'] or state.get('conflicting_candidates'):states.append(state)
            last[state['object_id']]=state
        data['shared_assembly']={'schema':'shared-assembly-2','states':states,'merges':[]}
    # Normalize categories without guessing whether different IDs are the same entity.
    for uid in assembly_ids:
        library[str(uid)].update(label=f'assembly_{uid}',category='assembly')
    from core.assembly_timeline import validate_timeline,empty_timeline,reference_id
    data['shared_assembly']=validate_timeline(data.get('shared_assembly',empty_timeline()))
    for events in data['hoi_events'].values():
        for event in events:
            if event.get('shared_assembly_ref'):
                f=event.get('start_frame')
                uid=reference_id(event,data['shared_assembly'],f if type(f) is int else 0)
                event['shared_assembly_id']=uid
    identity_map={int(k):int(v) for k,v in (data.get('provenance',{}).get('assembly_identity_map',{}) or {}).items()}
    if identity_map:
        role_fields={'noun_object_id','target_object_id','instrument_object_id','tool_object_id','shared_assembly_id'}
        def remap_refs(value,context=''):
            if isinstance(value,list):return [remap_refs(v,context) for v in value]
            if not isinstance(value,dict):return value
            out={}
            for field,item in value.items():
                role=field in role_fields or (context in role_fields and field in ('value','suggested_value','current_value'))
                if role and type(item) is int and item in identity_map and str(item) not in library:out[field]=identity_map[item]
                elif field in ('migration_review','provenance','source_record'):out[field]=item
                else:out[field]=remap_refs(item,field if field in role_fields else context)
            return out
        data=remap_refs(data)
        library=data.get('object_library',{})
        tracks=data.get('tracks',{})
    for track in tracks.values():
        if 'category' in track:track['category']=canonical_noun(track['category'],profile)
    # Active vocabularies do not expose obsolete composite classes or duplicate aliases.
    for field in ('noun_library','object_classes'):
        if isinstance(data.get(field),list):
            data[field]=list(dict.fromkeys('assembly' if key(x) in mappings else canonical_noun(x,profile) for x in data[field]))
    if isinstance(data.get('verb_library'),dict):
        data['verb_library']={k:canonical_verb(v,profile) for k,v in data['verb_library'].items() if isinstance(v,str) and key(v) not in profile.get('forbidden_verbs',{})}
    # Class merges keep distinct physical IDs, even when source instance names collide.
    names=set()
    for uid,info in library.items():
        if not isinstance(info,dict):continue
        label=info.get('label')
        if label in names:
            info['label']=str(info.get('category') or label)+'_id'+str(uid)
        names.add(info.get('label'))
    if not modern:
        data.setdefault('provenance',{})['migration']={'source_schema':source.get('schema',source.get('version','legacy')),
            'event_count':old_events,'geometry_and_instance_ids_preserved':True,
            'original_archive_required':True}
    data.pop('version',None)
    data['schema']=SCHEMA
    # Canonicalize nested class dictionaries without changing IDs or source records.
    from core.noun_aliases import normalize_noun_aliases
    return normalize_noun_aliases(data)
