"""Hand-only import keeps human evidence and omits conflicting machine hands."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.box_import import automatic,import_box,unique_hand_candidates
from core.bbox_markers import state
from core.frame_review import signature

box=dict(id=1,orig_frame=0,label='Left_hand',x1=1,y1=2,x2=8,y2=9,source='machine_candidate')
assert automatic(box) and import_box(box,True,'hands') and not import_box(box,False,'hands')
assert not import_box(box,True,'none')
assert import_box(dict(box,source='legacy_annotation'),False,'none')
assert import_box(dict(box,human_verified=True),False,'none')
assert import_box(dict(box,source='manual_box_edit'),False,'none')
assert not import_box(dict(box,source='loaded_annotation'),False,'hands',{'automatic':True})
assert state([box],None)=='model'
assert state([box],{'state':'verified','signature':signature([box])})=='viewed'
assert state([dict(box,human_verified=True)],None)=='viewed'
assert state([dict(box,source='manual_box_add')],None)=='manual'
assert state([box,box],None)=='conflict'
assert state([dict(box,x2=0)],None)=='conflict'
normalize=lambda label:label if label in ('Left_hand','Right_hand') else None
object_box=dict(box,label='part',id=5)
rows,omitted=unique_hand_candidates([box,dict(box,x2=10),object_box],normalize)
assert rows==[object_box] and omitted==2
rows,omitted=unique_hand_candidates([box,dict(box),object_box],normalize)
assert len(rows)==2 and omitted==1
manual=dict(box,source='manual_box_edit')
rows,omitted=unique_hand_candidates([box,manual,object_box],normalize)
assert manual in rows and object_box in rows and box not in rows and omitted==1
print('SELECTIVE_IMPORT_MANUAL_PRESERVATION_CONFLICT_GATE_MARKER_STATES_PASS')

from core.box_import import constrained_import_mode
assert constrained_import_mode('all',{'automatic_tracking_import_limit':'hands'})=='hands'
assert constrained_import_mode('none',{'automatic_tracking_import_limit':'hands'})=='none'
assert not import_box({'source':'','frame':0},False,'hands',{'automatic':True})
assert import_box({'source':'manual_sam_click','frame':0},False,'hands',{'automatic':True})
assert import_box({'source':'manual_box_edit','frame':0},False,'hands',{'automatic':True})
print('IMPORT_LIMIT_UNTAGGED_MACHINE_OBJECTS_HUMAN_SAM_ANCHORS_PASS')
