"""Reported pickup/hold ID and playhead regressions through real editor callbacks."""
import os,sys,copy,json
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QInputDialog
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
PROFILE.clear();PROFILE.update(noun_classes=['assembly','tool'],verbs=['pick_up','hold'],
 anomaly_labels=[],assembly_components=['shell','grip'],assembly_interfaces=[])
refresh_aliases()
app=QApplication([])
for name in ('information','warning','critical','question'):
    setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)
from ui.hoi_window import HOIWindow
from core.assembly_timeline import empty_timeline,put_state
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=90;w.player.frame_rate=15
payload=dict(schema='hoi-annotation',video_id='trial',frame_count=90,fps=15,
 object_library={'55':dict(label='assembly',category='assembly'),'8':dict(label='tool',category='tool')},
 tracks={f'T_OBJ_{uid}':dict(object_id=uid,category='assembly' if uid==55 else 'tool',
   boxes=[dict(frame=f,bbox=[10,10,30,30]) for f in [10,22]]) for uid in [55,8]},
 hoi_events={'left_hand':[
 dict(event_id='pickup',start_frame=10,contact_onset_frame=10,end_frame=22,verb='pick_up',noun_object_id=55),
 dict(event_id='hold',start_frame=25,contact_onset_frame=25,end_frame=40,verb='hold')], 'right_hand':[]})
w._load_annotations_v2(payload)
hold=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb')=='hold')
pickup=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb')=='pick_up')
w.player.current_frame=25;w._set_selected_event(hold['event_id'],'Left_hand')
def draw(label,frame=25,role='noun'):
    w.player.current_frame=frame;w._pending_draw_box_role=role
    w._on_box_edited(None,dict(label=label,frame=frame,_action='add',x1=12,y1=10,x2=33,y2=35))
draw('Left_hand',role='hand')
choices=[]
def reuse(parent,title,text,options,*args):
    choices.append(list(options));return options[0],True
QInputDialog.getItem=reuse
w.rad_draw_target.setChecked(True)
assert w._get_auto_draw_label() is None
draw('assembly')
assert choices[-1]==['Use existing [55] assembly','Create a new assembly instance']
assert w._hand_noun_object_id(w.event_draft['Left_hand'])==55
assert w.object_id_counter==56
box=next(b for b in w.raw_boxes if b['orig_frame']==25 and b['id']==55)
assert w._event_box_visible(box)
saved=copy.deepcopy(w.raw_boxes)
w._hoi_undo();assert not any(b['orig_frame']==25 and b['id']==55 for b in w.raw_boxes)
w._hoi_redo();assert w.raw_boxes==saved
assert w._hand_noun_object_id(w.event_draft['Left_hand'])==55
# Same selected instance on subsequent frames must never prompt or allocate another ID.
QInputDialog.getItem=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected instance prompt'))
draw('assembly',frame=26)
assert w.raw_boxes[-1]['id']==55 and w.object_id_counter==56
# Explicit registry labels with an ID also reuse the correct instance outside role shortcuts.
draw('[55] assembly',frame=27,role=None)
assert w.raw_boxes[-1]['id']==55
# Deliberately creating an independent assembly remains possible and reversible.
w.event_draft['Left_hand'].update(noun_object_id=None,target_object_id=None,shared_assembly_ref=False)
QInputDialog.getItem=lambda parent,title,text,options,*a:(options[-1],True)
draw('assembly',frame=28)
assert w.raw_boxes[-1]['id']==56 and w.object_id_counter==57
w._hoi_undo();assert 56 not in w.global_object_map.values()
w._hoi_redo();assert 56 in w.global_object_map.values()
# Canceling an instance choice leaves registry, geometry and event semantics untouched.
w.event_draft['Left_hand'].update(noun_object_id=None,target_object_id=None,shared_assembly_ref=False)
before=(copy.deepcopy(w.raw_boxes),copy.deepcopy(w.event_draft),copy.deepcopy(w.global_object_map),w.object_id_counter)
QInputDialog.getItem=lambda *a:('',False)
draw('assembly',frame=29)
assert before==(w.raw_boxes,w.event_draft,w.global_object_map,w.object_id_counter)
# Restore the known instance and capture actual player relation output.
w._apply_noun_choice(55)
lines=[]
w.player.set_overlay_relations=lambda value:lines.append(copy.deepcopy(value))
# Supply visible hand support on pickup frames as well.
draw('Left_hand',frame=10,role='hand');draw('Left_hand',frame=22,role='hand');draw('Left_hand',frame=26,role='hand')
w._set_selected_event(hold['event_id'],'Left_hand')
semantics=copy.deepcopy(w.event_draft)
for frame in [9,10,22,23,24,25,26,40,41]:
    w.player.current_frame=frame;w._update_overlay(frame)
    labels=[line['label'] for line in lines[-1]]
    assert 'hold' not in labels if frame<25 or frame>40 else True
    assert labels.count('pick_up')==1 if frame in [10,22] else 'pick_up' not in labels
    if frame in [25,26]:assert labels.count('hold')==1
    assert w.event_draft==semantics and w.selected_event_id==hold['event_id']
# Editing the focused verb produces a single preview only inside its own interval.
w.event_draft['Left_hand']['verb']='edited_hold';w.player.current_frame=25
w._update_overlay(25);assert [l['label'] for l in lines[-1]]==['edited_hold']
w._update_overlay(10);assert [l['label'] for l in lines[-1]]==['pick_up']
w.event_draft['Left_hand']['verb']='hold'
# Other-hand actions and independently composed instances are untouched by drawing/reuse.
w.shared_assembly=put_state(empty_timeline(),dict(frame=1,object_id=55,components=['shell','grip']))
w.shared_assembly=put_state(w.shared_assembly,dict(frame=1,object_id=56,components=['shell','grip']))
state=copy.deepcopy(w.shared_assembly)
w.player.current_frame=25;draw('assembly',frame=30)
assert w.raw_boxes[-1]['id']==55 and w.shared_assembly==state
out=w._build_payload_v2()
assert any(b['frame']==25 for b in out['tracks']['T_OBJ_55']['boxes'])
assert any(b['frame']==28 for b in out['tracks']['T_OBJ_56']['boxes'])
w._load_annotations_v2(json.loads(json.dumps(out)))
assert any(b['id']==55 and b['orig_frame']==25 for b in w.raw_boxes)
assert any(b['id']==56 and b['orig_frame']==28 for b in w.raw_boxes)
print('PICKUP_HOLD_REUSE_NEW_CANCEL_UNDO_ROUNDTRIP_AND_PLAYHEAD_OVERLAY_PASS',flush=True)
w._stop_autosave();w._mark_hoi_saved();w.close()
