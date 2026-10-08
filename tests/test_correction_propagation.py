import importlib.util,json,ast
from pathlib import Path
root=Path(__file__).resolve().parent.parent
results={}
for version in ('',):
 p=root/version/'core/correction_propagation.py';spec=importlib.util.spec_from_file_location('prop',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 def box(uid,frame,source='auto',**kw):return dict(id=uid,orig_frame=frame,label='screw',source=source,x1=0,y1=0,x2=10,y2=10,**kw)
 raw=[box(0,0,'manual_box_edit'),box(0,1),box(7,1),box(0,3,'manual_box_edit'),box(0,4)]
 hand=box(0,1,'auto_handtrack');hand['label']='Left_hand';raw.append(hand)
 req=m.plan(raw,raw[0],100,104,100);assert req['end']==102
 updated=m.apply(raw,req,[dict(frame=101,x1=2,y1=3,x2=12,y2=13)])
 assert next(b for b in updated if b['id']==7)==raw[2]
 assert next(b for b in updated if b.get('label')=='Left_hand')==hand
 assert next(b for b in updated if b['id']==0 and b['orig_frame']==0)==raw[0]
 assert next(b for b in updated if b['id']==0 and b['orig_frame']==3)==raw[3]
 assert any(b['id']==0 and b['orig_frame']==1 and b['x1']==2 and not b['human_verified'] for b in updated)
 assert raw[1]['x1']==0
 occlusion=m.apply(raw,req,[]);assert not any(b['id']==0 and b['orig_frame']==1 and b.get('label')!='Left_hand' for b in occlusion)
 assert any(b.get('label')=='Left_hand' for b in occlusion)
 for invalid in ([dict(frame=103,x1=0,y1=0,x2=10,y2=10)],[dict(frame=101,x1=5,y1=0,x2=2,y2=10)]):
  try:m.apply(raw,req,invalid)
  except ValueError:pass
  else:raise AssertionError('Invalid predictions accepted')
 for path in ('core/correction_propagation.py','tools/sam2_correction_worker.py','ui/correction_propagation.py','ui/hoi_window.py'):ast.parse((root/version/path).read_text(encoding="utf-8"))
 results[version]=dict(instance_id_zero=True,other_instances_preserved=True,hand_namespace_collision_preserved=True,anchor_preserved=True,stop_before_human_anchor=True,empty_mask_removes_auto_box=True,proposal_not_human_verified=True,source_not_mutated=True,invalid_predictions_rejected=True,syntax=True,model_inference_tested=False)
print(json.dumps(results))
# Backward requests preserve earlier human anchors and exclude the seed frame.
back_anchor=dict(id=42,label='part',orig_frame=10,x1=1,y1=2,x2=10,y2=12,source='manual')
back_boxes=[dict(back_anchor,orig_frame=3),back_anchor,dict(back_anchor,orig_frame=8,source='model',human_verified=False)]
back=m.plan(back_boxes,back_anchor,10,0)
assert back['end']==4 and back['direction']==-1
updated=m.apply(back_boxes,back,[dict(frame=5,x1=2,y1=2,x2=12,y2=14)])
assert any(b['orig_frame']==3 for b in updated) and any(b['orig_frame']==10 for b in updated)
assert not any(b['orig_frame']==8 for b in updated)
print('BACKWARD_PROPAGATION_ANCHOR_BOUNDARIES_PASS')
