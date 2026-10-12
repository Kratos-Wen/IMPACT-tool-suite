"""Repeated profile loads preserve physical IDs; numeric tool names stay distinct."""
import copy,os,sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.instance_category import instance_category
from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(noun_classes=['anti_vibration_handle','torx_screwdriver_15','torx_screwdriver_20','screw','assembly'],
    create_default_noun_instances=True,singleton_noun_classes=['anti_vibration_handle'],verbs=['hold'],anomaly_labels=['attribute_a'])
assert instance_category('torx_screwdriver_15',PROFILE)=='torx_screwdriver_15'
assert instance_category('torx_screwdriver_15_1_2',PROFILE)=='torx_screwdriver_15'
assert instance_category('torx_screwdriver_20_1',PROFILE)=='torx_screwdriver_20'
assert instance_category('anti_vibration_handle_1_1',PROFILE)=='anti_vibration_handle'
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QInputDialog
app=QApplication([])
for name in ('information','warning','critical','question'):setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)
from ui.hoi_window import HOIWindow
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=90;w.player.frame_rate=15
assert w.chk_edit_boxes.isChecked() and w.rad_draw_none.isEnabled() and not w.draw_mode_widget.isHidden()
payload=dict(schema='hoi-annotation',video_id='trial',frame_count=90,fps=15,
 object_library={'77':dict(label='anti_vibration_handle_1',category='anti_vibration_handle'),
 '78':dict(label='torx_screwdriver_15_1',category='torx_screwdriver_15'),
 '79':dict(label='torx_screwdriver_20_1',category='torx_screwdriver_20')},
 tracks={'T_OBJ_77':dict(object_id=77,category='anti_vibration_handle',boxes=[dict(frame=5,bbox=[1,2,10,20])])},
 hoi_events={'left_hand':[dict(event_id='first',start_frame=5,contact_onset_frame=5,end_frame=15,verb='hold',noun_object_id=77),
 dict(event_id='second',start_frame=20,contact_onset_frame=20,end_frame=30,verb='hold')],'right_hand':[]})
w._load_annotations_v2(payload);w._apply_profile_libraries()
baseline=(copy.deepcopy(w.global_object_map),w.object_id_counter,copy.deepcopy(w.raw_boxes),copy.deepcopy(w.events))
for _ in range(50):w._apply_profile_libraries()
assert baseline==(w.global_object_map,w.object_id_counter,w.raw_boxes,w.events)
assert [uid for name,uid in w.global_object_map.items() if instance_category(name,PROFILE)=='anti_vibration_handle']==[77]
options=[]
def reuse(parent,title,text,choices,*args):
 options.append(list(choices));return choices[0],True
QInputDialog.getItem=reuse
assert w._choose_drawn_instance_id('anti_vibration_handle')==(77,True)
assert options[-1]==['Use existing [77] anti_vibration_handle']
assert w._choose_drawn_instance_id('torx_screwdriver_15')==(78,True)
assert all('20' not in text for text in options[-1])
assert w._choose_drawn_instance_id('torx_screwdriver_20')==(79,True)
assert all('15' not in text for text in options[-1])
second=next(e for e in w.events if e['hoi_data']['Left_hand'].get('interaction_start')==20)
w.player.current_frame=20;w._set_selected_event(second['event_id'],'Left_hand')
w._pending_draw_box_role='noun';w._on_box_edited(None,dict(_action='add',label='anti_vibration_handle',frame=20,x1=3,y1=4,x2=12,y2=22))
assert w.raw_boxes[-1]['id']==77 and w._hand_noun_object_id(w.event_draft['Left_hand'])==77
assert w.object_id_counter==baseline[1]
saved=copy.deepcopy(w.raw_boxes);w._hoi_undo();w._hoi_redo();assert w.raw_boxes==saved
# A second screw is intentional; class-level uniqueness must not merge it.
w._choose_drawn_instance_id('screw');assert options[-1][-1]=='Create a new screw instance'
out=w._build_payload_v2();w._load_annotations_v2(out);w._apply_profile_libraries()
assert [uid for name,uid in w.global_object_map.items() if instance_category(name,PROFILE)=='anti_vibration_handle']==[77]
assert any(b['id']==77 and b['orig_frame']==20 for b in w.raw_boxes)
# A manual screw box creates a video-local instance and is explicitly bound to the event.
w.player.current_frame=20;second=next(e for e in w.events if e['hoi_data']['Left_hand'].get('interaction_start')==20)
w._set_selected_event(second['event_id'],'Left_hand')
w.rad_draw_none.setChecked(True)
def object_without_hand(parent,title,text,choices,*args):
    assert title=='New Box';return choices[1],True
QInputDialog.getItem=object_without_hand
assert w._get_auto_draw_label() is None and w._pending_draw_box_role is None
# Manual mode must not inherit the current noun even when this hand has no box.
old_screws={uid for name,uid in w.global_object_map.items() if instance_category(name,PROFILE)=='screw'}
def create(parent,title,text,choices,*args):
    assert choices[-1]=='Create a new screw instance';return choices[-1],True
QInputDialog.getItem=create
w._on_box_edited(None,dict(_action='add',label='screw',frame=20,x1=40,y1=40,x2=50,y2=50))
new_screw=next(b['id'] for b in w.raw_boxes if b['orig_frame']==20 and b['label']=='screw')
assert new_screw not in old_screws and w._object_name_for_id(new_screw)=='screw'
assert w._hand_noun_object_id(w.event_draft['Left_hand'])==77
w.combo_inline_noun.setCurrentIndex(w.combo_inline_noun.findData(new_screw))
assert w._hand_noun_object_id(w.event_draft['Left_hand'])==new_screw
assert any(b['id']==new_screw for b in w.player.overlay_boxes)
assert next(e for e in w.events if e['hoi_data']['Left_hand'].get('interaction_start')==5)['hoi_data']['Left_hand']['noun_object_id']==77
w._pending_draw_box_role='noun';w._on_box_edited(None,dict(_action='add',label='screw',frame=25,x1=41,y1=41,x2=51,y2=51))
assert next(b['id'] for b in w.raw_boxes if b['orig_frame']==25)==new_screw
out=w._build_payload_v2();w._load_annotations_v2(out)
assert str(new_screw) in w._build_payload_v2()['object_library']
assert next(e for e in w.events if e['hoi_data']['Left_hand'].get('interaction_start')==20)['hoi_data']['Left_hand']['noun_object_id']==new_screw
w._load_annotations_v2(dict(payload,video_id='another_trial'));assert new_screw not in w.global_object_map.values()
w._stop_autosave();w._mark_hoi_saved();w.close()
print('PROFILE_50_REFRESHES_SINGLETON_REUSE_MANUAL_NEW_SCREW_BIND_REOPEN_VIDEO_SCOPE_PASS')
