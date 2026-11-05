"""Supplement only an exact original draft; never restore human-deleted boxes."""
import copy,hashlib,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.object_candidates import merge_candidates,load_adjacent_candidates
payload=dict(video_id='fixture',frame_count=30,fps=30,frame_size=[100,80],object_library={'4':dict(label='part',category='part')},tracks={'T_OBJ_4':dict(object_id=4,category='part',boxes=[dict(frame=2,bbox=[1,2,10,20],source='manual_box_add')])},hoi_events={'left_hand':[dict(event_id='L_001',start_frame=0,end_frame=20,noun_object_id=4)]})
candidate=dict(frame=10,object_id=4,bbox=[20,20,40,45],hand='left_hand',event_id='L_001',role='object',quality=dict(passed=True,confidence_min=.9,temporal_iou_min=.8))
doc=dict(schema='impact-sparse-object-candidates-1',coordinate_space='native_video',video_id='fixture',frame_count=30,fps=30,frame_size=[100,80],source_annotations_md5='original',boxes=[candidate])
before=copy.deepcopy(payload);out,report=merge_candidates(payload,doc,'original')
assert payload==before and report['added']==1
assert out['tracks']['T_OBJ_4']['boxes'][0]==payload['tracks']['T_OBJ_4']['boxes'][0]
assert out['tracks']['T_OBJ_4']['boxes'][1]['human_verified'] is False
assert out['hoi_events']==payload['hoi_events'] and out['object_library']==payload['object_library']
again,report=merge_candidates(out,doc,'original');assert report['added']==0 and again==out,'Repeated imports do not duplicate boxes'
changed=copy.deepcopy(out);changed['tracks']['T_OBJ_4']['boxes'].pop()
restored,report=merge_candidates(changed,doc,'human_saved_draft');assert restored==changed and report['reason']=='already_imported','A saved human deletion cannot be resurrected'
for change in [dict(frame=2),dict(object_id=99),dict(frame=29),dict(hand='right_hand'),dict(role='background'),dict(bbox=[0,0,101,20]),dict(bbox=[float('nan'),0,1,2]),dict(quality={'passed':True,'confidence_min':.5,'temporal_iou_min':1})]:
    bad=copy.deepcopy(doc);bad['boxes']=[dict(candidate,**change)];result,report=merge_candidates(payload,bad,'original');assert not report['added'] and result==payload
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'reviewed.json';raw=json.dumps(payload).encode();path.write_bytes(raw);doc['source_annotations_md5']=hashlib.md5(raw).hexdigest();path.with_name('object_candidates.json').write_text(json.dumps(doc))
    result,report=load_adjacent_candidates(path,payload);assert report['added']==1
    path.write_text(json.dumps(changed));result,report=load_adjacent_candidates(path,changed);assert result==changed and report['added']==0
print('OBJECT_CANDIDATES_NATIVE_COORDINATES_MANUAL_PRIORITY_SCOPE_QUALITY_SAVE_AND_DELETE_GUARDS_PASS')
