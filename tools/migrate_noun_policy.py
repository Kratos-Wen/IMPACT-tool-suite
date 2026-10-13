"""Migrate the supplied noun-policy revision without changing annotation geometry."""
import argparse
import copy
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.safe_storage import atomic_json
from core.annotation_migration import model_from_trial

RENAMES={'screw':'screw_long','angle_grinder':'assembly'}
LABEL_FIELDS={'label','category','instance_name','object_name','noun','target','instrument','tool','components'}
HISTORY={'provenance','migration_review','source_record','raw_response','raw_model_output',
         'machine_original','machine_snapshot','raw_snapshot','source_snapshot'}
REF_FIELDS={'noun_object_id','target_object_id','instrument_object_id','tool_object_id',
            'shared_assembly_id','object_instance_id','instrument_instance_id',
            'object_id','source_object_id'}

def rename_labels(value,field=''):
    if isinstance(value,str):return RENAMES.get(value,value) if field in LABEL_FIELDS else value
    if isinstance(value,list):return [rename_labels(item,field) for item in value]
    if isinstance(value,dict):
        return {key:copy.deepcopy(item) if key in HISTORY else rename_labels(item,key) for key,item in value.items()}
    return value

def updated_profile(source):
    result=copy.deepcopy(source)
    for field in ('noun_classes','assembly_components','detection_ignored_categories','singleton_noun_classes','ambiguous_nouns'):
        if field in result:result[field]=list(dict.fromkeys(RENAMES.get(name,name) for name in result[field]))
    aliases=result.setdefault('noun_aliases',{})
    for name,value in list(aliases.items()):aliases[name]=RENAMES.get(value,value)
    aliases.update(RENAMES,screw_long='screw_long',assembly='assembly')
    for field in ('noun_explanations','singleton_object_ids'):
        values=result.get(field,{})
        for old,new in RENAMES.items():
            if old in values:
                old_value=values.pop(old)
                values.setdefault(new,old_value)
    for values in result.get('legacy_assembly_mappings',{}).values():
        if isinstance(values,list):values[:]=[RENAMES.get(name,name) for name in values]
    for name,value in result.get('detection_matching_aliases',{}).items():
        result['detection_matching_aliases'][name]=RENAMES.get(value,value)
    for model,forbidden in (('A','screw_black'),('B','screw_short')):
        rule=result.setdefault('model_component_rules',{}).setdefault(model,{})
        rule['forbidden_components']=list(dict.fromkeys(rule.get('forbidden_components',[])+[forbidden]))
    return result

def references(value):
    found=set()
    if isinstance(value,list):
        for row in value:found.update(references(row))
    elif isinstance(value,dict):
        for field,row in value.items():
            if field in HISTORY:continue
            if field in REF_FIELDS and type(row) is int:found.add(row)
            else:found.update(references(row))
    return found

def migrate_document(source):
    result=rename_labels(source)
    used=references(result.get('hoi_events',{}))|references(result.get('shared_assembly',{}))
    default=result.get('assembly_default_reference')
    if type(default) is int:used.add(default)
    used.update(int(key[2:]) for row in result.get('frame_review',{}).values() for key in row if key.startswith('O:'))
    geometry={track.get('object_id') for track in result.get('tracks',{}).values()
              if track.get('boxes') or track.get('inactive_boxes')}
    removed={uid:copy.deepcopy(info) for uid,info in source.get('object_library',{}).items()
             if info.get('category',info.get('label')) in ('unresolved_object','unsolved_object','angle_grinder')
             and int(uid) not in used|geometry}
    for uid in removed:result['object_library'].pop(uid,None)
    result['tracks']={tid:track for tid,track in result.get('tracks',{}).items()
                      if str(track.get('object_id')) not in removed}
    archived=copy.deepcopy(result.get('provenance',{}).get('noun_policy_migration',{}).get('removed_unused_library_entries',{}))
    archived.update(removed)
    result.setdefault('provenance',{})['noun_policy_migration']={
        'schema':'noun-policy-revision-1','renames':RENAMES,
        'removed_unused_library_entries':archived,'instance_ids_preserved':True,
        'geometry_and_event_timing_preserved':True,
        'policy_source':'User-provided author revision: long screws; model-specific screws; assembly.'}
    ids={int(uid) for uid in result.get('object_library',{})}
    assert references(result.get('hoi_events',{}))<=ids,'Dangling event object reference'
    for tid,track in source.get('tracks',{}).items():
        if track.get('boxes') or track.get('inactive_boxes'):
            assert tid in result['tracks']
            for field in ('object_id','boxes','inactive_boxes'):
                assert track.get(field)==result['tracks'][tid].get(field),'Geometry changed'
    for side,events in source.get('hoi_events',{}).items():
        assert len(events)==len(result['hoi_events'][side])
        for before,after in zip(events,result['hoi_events'][side]):
            for field in REF_FIELDS|{'event_id','verb','start_frame','contact_onset_frame','end_frame'}:
                assert before.get(field)==after.get(field),'Event identity/timing changed'
    pending=[dict(object_id=int(uid),category=info.get('category',info.get('label')),
                  referenced=int(uid) in used,has_geometry=int(uid) in geometry)
             for uid,info in result.get('object_library',{}).items()
             if info.get('category',info.get('label')) in ('unresolved_object','unsolved_object')]
    report=dict(renames=RENAMES,removed_unused_ids=sorted(map(int,removed)),
                unresolved_objects=pending,event_count=sum(map(len,result.get('hoi_events',{}).values())),
                box_count=sum(len(t.get('boxes',[])) for t in result['tracks'].values()))
    return result,report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',required=True)
    parser.add_argument('--stage',required=True)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    task_path=Path(args.task).resolve();root=task_path.parent
    task=json.loads(task_path.read_text(encoding='utf-8-sig'))
    annotation=root/task['resume_annotations']
    profile_path=(root/task['project_profile']).resolve()
    source_bytes=annotation.read_bytes();task_bytes=task_path.read_bytes()
    source=json.loads(source_bytes)
    profile=json.loads(profile_path.read_text(encoding='utf-8-sig'))
    result,report=migrate_document(source)
    profile=updated_profile(profile)
    updated_task=dict(task,project_profile='project_profile.noun-v2.json')
    model=model_from_trial(task.get('trial_id',''))
    forbidden=set(profile.get('model_component_rules',{}).get(model,{}).get('forbidden_components',[]))
    report['model']=model
    report['incompatible_event_refs']=[dict(hand=side,event_id=e.get('event_id'),field=field,object_id=e[field])
        for side,events in result.get('hoi_events',{}).items() for e in events
        for field in ('noun_object_id','instrument_object_id')
        if e.get(field) is not None and result['object_library'][str(e[field])].get('category') in forbidden]
    report['source_sha256']=hashlib.sha256(source_bytes).hexdigest()
    stage=Path(args.stage).resolve();stage.mkdir(parents=True,exist_ok=True)
    atomic_json(stage/'reviewed.json',result,backup=False)
    atomic_json(stage/'project_profile.noun-v2.json',profile,backup=False)
    atomic_json(stage/'task.json',updated_task,backup=False)
    graph=root/(annotation.stem+'.event_graph.json')
    if graph.exists():atomic_json(stage/graph.name,rename_labels(json.loads(graph.read_text())),backup=False)
    if args.apply:
        lock=root/'.impact-recovery'/'video.mp4.json.lock'
        if lock.exists():raise RuntimeError('Close the editor before applying migration.')
        if annotation.read_bytes()!=source_bytes or task_path.read_bytes()!=task_bytes:
            raise RuntimeError('Project changed during preparation; rerun migration.')
        stamp=datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y%m%d-%H%M%S')
        backup=root/'annotation-backups'/('before-noun-v2-'+stamp)
        backup.mkdir(parents=True,exist_ok=False)
        for path in (annotation,annotation.with_name(annotation.name+'.bak'),task_path,graph):
            if path.exists():shutil.copy2(path,backup/path.name)
        shutil.copy2(profile_path,backup/'project_profile.original.json')
        recovery=root/'.impact-recovery'
        for path in recovery.glob('video.mp4.json*'):
            if path.is_file():shutil.copy2(path,backup/path.name)
        try:
            atomic_json(root/'project_profile.noun-v2.json',profile,backup=False)
            atomic_json(annotation,result,backup=False)
            if (stage/graph.name).exists():
                atomic_json(graph,json.loads((stage/graph.name).read_text()),backup=False)
            atomic_json(task_path,updated_task,backup=False)
            # Keep stale recovery snapshots from restoring the old vocabulary.
            for path in recovery.glob('video.mp4.json*'):
                if path.is_file():path.unlink()
        except Exception:
            for path in (annotation,task_path,graph):
                if (backup/path.name).exists():shutil.copy2(backup/path.name,path)
            raise
        report['backup_directory']=str(backup)
        atomic_json(root/'noun-policy-migration-report.json',report,backup=False)
    atomic_json(stage/'migration-report.json',report,backup=False)
    print(json.dumps({k:v for k,v in report.items() if k!='unresolved_objects'},ensure_ascii=False,indent=2))
    print('Unresolved objects retained:',len(report['unresolved_objects']))

if __name__=='__main__':main()
