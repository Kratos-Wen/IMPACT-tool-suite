"""Singleton references, ID collisions, conflicting geometry and protected work."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.singleton_instances import normalize_machine_instances,catalog_registry,normalize_tracking_frame
profile=dict(noun_classes=['anti_vibration_handle','assembly','screw','torx_screwdriver_15','torx_screwdriver_20'],
 singleton_noun_classes=['anti_vibration_handle','torx_screwdriver_15','torx_screwdriver_20'],
 singleton_object_ids={'anti_vibration_handle':1002,'torx_screwdriver_15':1019,'torx_screwdriver_20':1020})
library={str(uid):dict(label=name+'_'+str(uid),category=name) for uid,name in
 [(0,'anti_vibration_handle'),(9,'anti_vibration_handle'),(1002,'assembly'),(11,'screw'),(12,'screw'),(13,'torx_screwdriver_15'),(14,'torx_screwdriver_20')]}
tracks={key:dict(object_id=uid,category=library[str(uid)]['category'],boxes=[])
 for key,uid in [('h0',0),('h9',9),('body',1002),('s1',11),('s2',12),('t15',13),('t20',14)]}
tracks['h0']['boxes']=[dict(frame=0,bbox=[1,2,10,20]),dict(frame=5,bbox=[2,3,11,21])]
tracks['h9']['boxes']=[dict(frame=0,bbox=[1,2,10,20]),dict(frame=5,bbox=[20,30,40,50])]
source=dict(schema='hoi-annotation',video_id='trial_a',frame_count=30,fps=15,provenance=dict(machine_only=True,raw_model_output={'object_id':0}),
 object_library=library,tracks=tracks,shared_assembly=dict(schema='shared-assembly-2',states=[dict(frame=0,object_id=1002,components=['shell','part'])],merges=[]),
 hoi_events={'left_hand':[dict(event_id='L',verb='hold',start_frame=0,contact_onset_frame=0,end_frame=10,noun_object_id=0,instrument_object_id=13,
 links=dict(target_track_id='h0',tool_track_id='t15'),anomaly_labels=[],anomaly_review_state='unreviewed',human_verified=False)],
 'right_hand':[dict(event_id='R',verb='hold',start_frame=0,contact_onset_frame=0,end_frame=10,noun_object_id=9,instrument_object_id=14,
 links=dict(target_track_id='h9',tool_track_id='t20'),anomaly_labels=[],anomaly_review_state='unreviewed',human_verified=False)]})
source['hoi_events']['left_hand'][0]['annotation_state']={'field_suggestions':{'noun_object_id':{'value':0}}}
before=copy.deepcopy(source);out=normalize_machine_instances(source,profile)
assert source==before and normalize_machine_instances(out,profile)==out
assert out['hoi_events']['left_hand'][0]['noun_object_id']==out['hoi_events']['right_hand'][0]['noun_object_id']==1002
assert out['hoi_events']['left_hand'][0]['instrument_object_id']==1019
assert out['hoi_events']['left_hand'][0]['annotation_state']['field_suggestions']['noun_object_id']['value']==1002
assert out['hoi_events']['right_hand'][0]['instrument_object_id']==1020
assert out['object_library']['1002']['label']=='anti_vibration_handle'
legacy_suffix=copy.deepcopy(out)
legacy_suffix['object_library']['1002']['label']='anti_vibration_handle_1'
compatible=normalize_machine_instances(legacy_suffix,profile)
assert compatible['object_library']['1002']['label']=='anti_vibration_handle'
assert compatible['tracks']['T_OBJ_1002']['object_id']==1002
assert out['provenance']['raw_model_output']==source['provenance']['raw_model_output']
assert out['shared_assembly']['states'][0]['object_id']!=1002
assert {out['tracks'][k]['object_id'] for k in ('s1','s2')}=={11,12}
assert out['tracks']['T_OBJ_1002']['boxes']==[dict(frame=0,bbox=[1,2,10,20])]
assert len(out['provenance']['singleton_identity']['withheld_box_conflicts'][0]['box_candidates'])==2
assert all(e['links']['target_track_id']=='T_OBJ_1002' for es in out['hoi_events'].values() for e in es)
other=copy.deepcopy(source);other['video_id']='trial_b';b=normalize_machine_instances(other,profile)
assert b['object_library']['1002']['label']==out['object_library']['1002']['label']
assert b['tracks']['T_OBJ_1002']['global_instance_uid']!=out['tracks']['T_OBJ_1002']['global_instance_uid']
for field in ['human_box','review','human_event','legacy']:
 protected=copy.deepcopy(source)
 if field=='human_box':protected['tracks']['h0']['boxes'][0]['source']='manual_box_add'
 elif field=='review':protected['frame_review']={'0':{'O:0':{'state':'verified'}}}
 elif field=='human_event':protected['hoi_events']['left_hand'][0]['human_verified']=True
 else:protected['provenance']['machine_only']=False
 snapshot=copy.deepcopy(protected)
 try:normalize_machine_instances(protected,profile)
 except ValueError:pass
 else:raise AssertionError('Changed protected work: '+field)
 assert protected==snapshot
registry=catalog_registry('trial_a',{'a':dict(category_candidates=['anti_vibration_handle']),
 'b':dict(category_candidates=['anti_vibration_handle']),
 'c':dict(category_candidates=['anti_vibration_handle','screw'])},profile)
assert len(registry['aliases'])==2 and registry['ambiguous_track_uids']==['c']
frame=dict(frame=0,tracks=[dict(track_uid=u,track_id=i,bbox_xyxy=[1,2,3,4]) for i,u in enumerate(['a','b','c'])])
frame['tracks'][1]['bbox_xyxy']=[10,20,30,40]
view=normalize_tracking_frame(frame,registry)
assert len(view['tracks'])==1 and view['tracks'][0]['track_uid']=='c'
assert len(view['identity_conflicts'][0]['candidates'])==2 and frame['tracks'][0]['track_uid']=='a'
frame['tracks'][1].update(visible=False,bbox_xyxy=None)
view=normalize_tracking_frame(frame,registry)
assert len(view['tracks'])==2 and not view['identity_conflicts']
assert next(t for t in view['tracks'] if t['track_uid']=='anti_vibration_handle')['track_id']==1002
import gzip,tempfile
from tools.read_canonical_tracking import read_chunk
with tempfile.TemporaryDirectory() as folder:
 path=Path(folder)/'chunk.json.gz'
 chunk=dict(trial='trial_a',frames=[frame],instances=[dict(track_uid=u,track_id=i) for i,u in enumerate(['a','b','c'])],
  masks=[dict(frame=0,track_id=0,shape=[2,2],bbox_xyxy=[1,2,3,4],encoding='test',data='x')])
 with gzip.open(path,'wt') as stream:json.dump(chunk,stream)
 mapped=read_chunk(path,registry)
 assert mapped['masks'][0]['track_id']==1002 and mapped['masks'][0]['track_uid']=='anti_vibration_handle'
 assert mapped['masks'][0]['source_track_id']==0 and not mapped['withheld_identity_masks']
 chunk['frames'][0]['tracks'][1].update(visible=True,bbox_xyxy=[10,20,30,40])
 with gzip.open(path,'wt') as stream:json.dump(chunk,stream)
 mapped=read_chunk(path,registry)
 assert not mapped['masks'] and len(mapped['withheld_identity_masks'])==1
print('SINGLETON_ALIASES_COLLISIONS_MULTIPLE_SCREWS_HUMAN_PROTECTION_CONFLICTS_IDEMPOTENCE_PASS')
