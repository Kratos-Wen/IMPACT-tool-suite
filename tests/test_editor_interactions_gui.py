"""Exercise interleaved editor actions through real Qt callbacks."""
import copy, json, os, sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication, QMessageBox
app = QApplication([])
for name in ('information', 'question', 'warning'):
    setattr(QMessageBox, name, lambda *a, **k: QMessageBox.No)
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
PROFILE.clear(); PROFILE.update(anomaly_labels=['attribute_a'], noun_classes=['part','tool'],
    verbs=['hold','place','insert'], noun_explanations={'part':'Target part.','tool':'Operating tool.'},
    verb_explanations={'hold':'Support.','place':'Placement.','insert':'Insertion.'})
refresh_aliases()
from ui.hoi_window import HOIWindow
from core.frame_review import valid_record
w = HOIWindow(); w._autosave_timer.stop(); w.player.frame_count=90; w.player.frame_rate=15
payload = dict(video_id='trial',frame_count=90,fps=15,
    object_library={'0':dict(label='part_1',category='part'),'5':dict(label='part_2',category='part'),'8':dict(label='tool_1',category='tool')},
    tracks={
        'T_OBJ_0':dict(object_id=0,category='part',boxes=[dict(frame=f,bbox=[10,10,30,30]) for f in (2,3,4,8,12)]),
        'T_OBJ_5':dict(object_id=5,category='part',boxes=[dict(frame=f,bbox=[35,10,55,30]) for f in (2,3,4,8,12)]),
        'T_OBJ_8':dict(object_id=8,category='tool',boxes=[dict(frame=f,bbox=[60,10,80,30]) for f in (2,3,4,8,12)])},
    hoi_events={'left_hand':[dict(event_id='left',start_frame=2,contact_onset_frame=2,end_frame=20,
        verb='hold',noun_object_id=0,instrument_object_id=8,anomaly_labels=[],anomaly_review_state='reviewed')],
        'right_hand':[dict(event_id='right',start_frame=2,contact_onset_frame=3,end_frame=20,
        verb='insert',noun_object_id=5,instrument_object_id=None,anomaly_labels=[],anomaly_review_state='reviewed')]})
w._load_annotations_v2(payload);w._apply_profile_libraries()
left = next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
right = next(e for e in w.events if e['hoi_data']['Right_hand'].get('verb'))
# Put both active hands in one event to catch the historical all-hand overwrite.
left['hoi_data']['Right_hand'] = copy.deepcopy(right['hoi_data']['Right_hand'])
w.events.remove(right); event_id=left['event_id'];w._set_selected_event(event_id,'Left_hand')
semantic_keys=('verb','noun_object_id','instrument_object_id','interaction_start','functional_contact_onset','interaction_end')
def semantics(hand): return {key:hand.get(key) for key in semantic_keys}
right_original=semantics(left['hoi_data']['Right_hand'])
def geometry(rows): return [(b['id'], b['orig_frame'], b['label'], b['x1'], b['y1'], b['x2'], b['y2']) for b in rows]
other_boxes=geometry([b for b in w.raw_boxes if b['id']!=0])
for cycle in range(40):
    w._set_selected_event(event_id,'Left_hand');w.player.current_frame=2
    left=w._find_event_by_id(event_id)
    assert semantics(left['hoi_data']['Right_hand'])==right_original
    w._sync_event_videomae_suggestions(event_id,left,[dict(label='place',score=.95)],target_hand='Left_hand')
    assert left['hoi_data']['Right_hand']['verb']=='insert'
    w._apply_verb_choice('place' if cycle%2 else 'hold')
    assert semantics(w._find_event_by_id(event_id)['hoi_data']['Right_hand'])==right_original
    start=2;end=20+(cycle%4);onset=start
    w._on_hoi_timeline_update(event_id,'Left_hand',start,end,onset)
    hand=w._find_event_by_id(event_id)['hoi_data']['Left_hand']
    assert hand['functional_contact_onset']==hand['interaction_start']==2
    assert hand['noun_object_id']==0 and hand['instrument_object_id']==8
    anchor=next(b for b in w.raw_boxes if b['id']==0 and b['orig_frame']==2)
    w._selected_edit_box=copy.deepcopy(anchor)
    w._on_box_edited(0,dict(anchor,frame=2,x1=10+(cycle%3)))
    w._verify_current_frame()
    record=w.frame_review['2']['O:0']
    assert valid_record(record,w._boxes_for_review('O:0',2))
    anchor=next(b for b in w.raw_boxes if b['id']==0 and b['orig_frame']==2)
    w._selected_edit_box=copy.deepcopy(anchor)
    w._on_box_edited(0,dict(anchor,frame=2,x1=anchor['x1']+1))
    assert not valid_record(record,w._boxes_for_review('O:0',2))
    w._hoi_undo();assert valid_record(w.frame_review['2']['O:0'],w._boxes_for_review('O:0',2))
    w._hoi_redo();assert not valid_record(w.frame_review['2']['O:0'],w._boxes_for_review('O:0',2))
    assert geometry([b for b in w.raw_boxes if b['id']!=0])==other_boxes
    # Current-frame deletion leaves the rest of the physical track intact.
    anchor=next(b for b in w.raw_boxes if b['id']==0 and b['orig_frame']==2)
    before=copy.deepcopy(w.raw_boxes);w._selected_edit_box=copy.deepcopy(anchor)
    w._on_box_edited(0,dict(anchor,frame=2,_action='delete'))
    assert not any(b['id']==0 and b['orig_frame']==2 for b in w.raw_boxes)
    assert sum(b['id']==0 for b in w.raw_boxes)==4
    w._hoi_undo();assert w.raw_boxes==before
    semantics_before=copy.deepcopy(w._find_event_by_id(event_id)['hoi_data'])
    for _ in range(3):
        w._refresh_events();w._update_status_label();w._refresh_label_explanations();app.processEvents()
    assert w._find_event_by_id(event_id)['hoi_data']==semantics_before
    out=w._build_payload_v2();w._load_annotations_v2(json.loads(json.dumps(out)))
    left=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
    right=next(e for e in w.events if e['hoi_data']['Right_hand'].get('verb'))
    assert semantics(right['hoi_data']['Right_hand'])==right_original
    if right is not left:
        left['hoi_data']['Right_hand']=copy.deepcopy(right['hoi_data']['Right_hand']);w.events.remove(right)
    event_id=left['event_id']
from core.assembly_timeline import put_state
base={'schema':'shared-assembly-1','states':[]}
w.shared_assembly=put_state(base,dict(frame=2,object_id=0,components=['part']))
w.shared_assembly=put_state(w.shared_assembly,dict(frame=6,object_id=0,components=['part','attachment']))
linked=dict(verb='hold',shared_assembly_ref=True,noun_object_id=0,instrument_object_id=None,interaction_start=2,interaction_end=10,anomaly_label='normal')
assert 'noun changes inside event; split at the change' in w._policy_missing(linked)
w.shared_assembly=put_state(base,dict(frame=2,object_id=0,components=['part','attachment']))
w.shared_assembly=put_state(w.shared_assembly,dict(frame=6,object_id=0,components=['part','attachment','handle']))
assert 'noun changes inside event; split at the change' not in w._policy_missing(linked)
w.shared_assembly=put_state(base,dict(frame=2,object_id=0,components=['part']))
w.shared_assembly=put_state(w.shared_assembly,dict(frame=6,object_id=5,components=['part']))
assert 'Object ID changes inside event; split at the change' in w._policy_missing(linked)
w._stop_autosave();w.close()
print('EDITOR_40_INTERLEAVED_CYCLES_HAND_ISOLATION_REVIEW_INVALIDATION_UNDO_DELETE_ROUNDTRIP_PASS')
