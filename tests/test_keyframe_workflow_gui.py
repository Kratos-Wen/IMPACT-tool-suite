"""SOE workflow, implicit box visits, six-type anomaly UI and shared boundaries."""
import copy,os,sys,time
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime,_configure_app_font
_bootstrap_qt_runtime()
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication,QMessageBox,QPushButton,QComboBox,QDialogButtonBox
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
PROFILE.clear();PROFILE.update(default_anomaly_label='unreviewed',human_review_rate=2,
    anomaly_labels=['attribute_'+x for x in 'abcdef'],verbs=['insert','hold','place'],
    noun_classes=['part','tool','assembly'],assembly_components=['part','attachment'],assembly_interfaces=[],
    create_default_noun_instances=True,singleton_noun_classes=['tool'],singleton_object_ids={'tool':1002})
refresh_aliases()
from ui.hoi_window import HOIWindow
from core.frame_review import valid_record
from core.event_intervals import interiors_overlap
app=QApplication([]);_configure_app_font(app)
for method in ('information','warning','question','critical'):setattr(QMessageBox,method,lambda *a,**k:QMessageBox.No)

def wait_until(condition,timeout=4000):
    start=time.monotonic()
    while not condition() and (time.monotonic()-start)*1000<timeout:QTest.qWait(10)
    assert condition(),'Timed out waiting for an editor transition'

def drain_events(milliseconds=500):
    done=[];QTimer.singleShot(milliseconds,lambda:done.append(True));wait_until(lambda:done)

class Capture:
    def __init__(self):self.position=0
    def set(self,key,value):self.position=int(value);return True
    def grab(self):self.position+=1;return True
    def read(self):
        value=self.position;self.position+=1
        return True,np.full((32,32,3),value,np.uint8)
    def release(self):pass

payload=dict(schema='hoi-annotation',video_id='trial',fps=15,frame_count=180,
    object_library={'1002':dict(label='tool_1',category='tool'),'2000':dict(label='part_1',category='part')},
    tracks={
        'T_OBJ_1002':dict(object_id=1002,category='tool',boxes=[dict(frame=f,bbox=[2,2,10,10],source='automatic') for f in (0,5,7,14)]),
        'T_OBJ_2000':dict(object_id=2000,category='part',boxes=[dict(frame=f,bbox=[12,12,22,22],source='automatic') for f in (0,5,7,14)]),
        'T_LEFT_HAND':dict(category='Left_hand',boxes=[dict(frame=f,bbox=[5,5,20,20],source='automatic') for f in (0,5,7,14)])},
    hoi_events={'left_hand':[dict(event_id='left',start_frame=0,contact_onset_frame=7,end_frame=14,
        verb='insert',noun_object_id=2000,instrument_object_id=1002)],'right_hand':[]})
w=HOIWindow();w._autosave_timer.stop();w._show_quick_start_on_startup=False
w.player.frame_count=180;w.player.frame_rate=15
w._load_annotations_v2(payload);w._apply_profile_libraries()
event=w.events[0];w._set_selected_event(event['event_id'],'Left_hand')
for hand in (event['hoi_data']['Left_hand'],w.event_draft['Left_hand']):
    for field in ('interaction_start','functional_contact_onset','interaction_end','verb','noun_object_id'):
        w._set_hand_field_state(hand,field,source='machine_candidate',status='suggested')
for combo in (w.combo_anomaly,w.combo_inline_anomaly):
    assert combo.count()==6 and combo.currentText()=='normal'
    assert all(combo.findData(name)==-1 for name in ('normal','unknown','unreviewed','confirm'))
assert not w._policy_missing(w.event_draft['Left_hand'])
assert {i['frame'] for i in w._frame_review_issues()}=={0,7,14},'A fixed-rate grid remained active'
assert not w._hand_completion_state(w.event_draft['Left_hand'],'Left_hand')['complete']
w.player.cap=Capture();w.player._media_path='synthetic';w.player.crop_start=0;w.player.crop_end=179
w.show();w.activateWindow();drain_events(150)
for frame in (0,7,14):
    w.player.seek(frame);w._refresh_boxes_for_frame(frame,lightweight=True);w._set_frame_controls(frame)
    w._schedule_box_observation()
    wait_until(lambda:w._box_evidence_complete('H:Left_hand',frame))
assert not w._frame_review_issues()
assert w._hand_completion_state(w.event_draft['Left_hand'],'Left_hand')['complete']
assert w.event_draft['Left_hand']['_event_extra']['source_field_states']['verb']['status']=='suggested'
assert not w._hand_completion_state(w.events[0]['hoi_data']['Left_hand'],'Left_hand')['suggested_fields']
segments=w._hoi_segments_for_hand('Left_hand')
assert segments[0]['complete'] and segments[0]['color'].name()==w._hoi_color_for_verb('insert').name()
# An explicit no-contact decision needs only the two observable endpoint frames.
hand=w.event_draft['Left_hand']
for field in ('interaction_start','interaction_end','verb','noun_object_id'):
    w._set_hand_field_state(hand,field,source='machine_candidate',status='suggested')
with patch('ui.annotation_policy.QInputDialog.getItem',return_value=('No contact occurred',True)),patch('ui.annotation_policy.QInputDialog.getText',return_value=('The attempt ends before contact',True)):
    w._record_contact_exception()
w._schedule_box_observation()
wait_until(lambda:not w._hand_completion_state(w.event_draft['Left_hand'],'Left_hand')['suggested_fields'])
assert w.event_draft['Left_hand']['functional_contact_onset'] is None
assert w._hand_completion_state(w.event_draft['Left_hand'],'Left_hand')['complete']
w._clear_contact_exception()
w._set_hand_field_state(w.event_draft['Left_hand'],'functional_contact_onset',value=7,source='manual',status='confirmed')
w._apply_draft_to_selected_event()
# Multiple anomaly types share a keyframe without separate reviewer switches.
w.combo_anomaly.toggle('attribute_a');w.combo_anomaly.toggle('attribute_b')
w.player.seek(5);w._refresh_boxes_for_frame(5,lightweight=True);w.btn_anomaly_keyframe.click()
hand=w.event_draft['Left_hand']
assert hand['required_review_frames']==[5]
assert set(hand['_event_extra']['anomaly_evidence'])=={'attribute_a','attribute_b'}
assert {i['frame'] for i in w._frame_review_issues()}=={5}
wait_until(lambda:not w._frame_review_issues())
assert w._hand_completion_state(w.event_draft['Left_hand'],'Left_hand')['complete']
w.combo_anomaly.toggle('attribute_a');w.combo_anomaly.toggle('attribute_b');assert w.combo_anomaly.currentText()=='normal'
# Playback and hover previews never certify the geometry of a paused keyframe.
w.frame_review={}
for box in w.raw_boxes:box['human_verified']=False
w._bump_bbox_revision();w._bump_query_state_revision();w.player.seek(0)
w.player.play();w.player.timer.stop();w._refresh_boxes_for_frame(0,lightweight=True)
drain_events();assert not w.frame_review
w.player.pause();w.player.seek(7,preview_only=True);w._schedule_box_observation()
drain_events();assert not w.frame_review and w.player.current_frame==0
# A manual geometry correction counts immediately, with no review checkbox.
w.player.seek(0);w._refresh_boxes_for_frame(0,lightweight=True)
for frame in (0,5,7,14):
    for entity in w._review_entities(w.event_draft['Left_hand'],'Left_hand',frame):
        for box in list(w._boxes_for_review(entity,frame)):
            w._selected_edit_box=copy.deepcopy(box)
            w._on_box_edited(box['id'],dict(box,frame=frame,x1=box['x1']+1))
state=w._hand_completion_state(w.event_draft['Left_hand'],'Left_hand')
assert not w._frame_review_issues() and state['complete'], (w._frame_review_issues(),state,w.event_draft['Left_hand'])
saved=w._build_payload_v2();geometry=copy.deepcopy(saved['tracks']);mapping=copy.deepcopy(saved['provenance']['video_local_instances'])
for _ in range(10):w._load_annotations_v2(w._build_payload_v2())
assert w._build_payload_v2()['provenance']['video_local_instances']==mapping
assert set(w._build_payload_v2()['object_library'])==set(saved['object_library'])
assert all('_1' not in row['label'] for row in w._build_payload_v2()['object_library'].values())
# Event intervals and timeline hit testing agree on a shared boundary frame.
assert not interiors_overlap(0,65,65,100) and interiors_overlap(0,65,64,100)
row=w.hoi_timeline.actor_rows['Left_hand']
original=row._get_segments
row._get_segments=lambda:[dict(event_id=1,start=0,end=65,onset=0,verb='hold'),dict(event_id=2,start=65,end=100,onset=65,verb='place')]
assert not row._overlaps(65,100,2) and row._overlaps(64,100,2)
row.set_selected(2);assert row._segment_at(65)['event_id']==2
assert row._hit_segment(row.frame_to_x(65),30)[0]['event_id']==2
row._get_segments=original
second=w._on_hoi_timeline_create('Left_hand',14,30,14);assert second is not None
assert w._on_hoi_timeline_create('Left_hand',13,25,13) is None
w._on_hoi_timeline_update(second,'Left_hand',14,31,14)
assert w._find_event_by_id(second)['hoi_data']['Left_hand']['interaction_start']==14
# Directly accessible assembly creation is cancellable and has no ID side effects.
before=(copy.deepcopy(w.shared_assembly),copy.deepcopy(w.global_object_map),w.object_id_counter)
def inspect_assembly():
    dialog=app.activeModalWidget();button=dialog.findChild(QPushButton,'newAssemblyInstance');assert button
    button.click();assert dialog.findChild(QComboBox,'assemblyInstance').currentData() is None
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel).click()
QTimer.singleShot(0,inspect_assembly);w.btn_inline_assembly.click()
assert before==(w.shared_assembly,w.global_object_map,w.object_id_counter)
# macOS uses readable point sizes and scaling also affects fonts.
with patch('app.sys.platform','darwin'):
    _configure_app_font(app);assert app.font().pointSizeF()>=11
    w._set_ui_scale(.85);assert w._preferred_base_font_pt()>=10.5
    w._set_ui_scale(1.2);assert w._preferred_base_font_pt()>=13.2-1e-6
w.hide();w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media();w.deleteLater();app.processEvents()
print('SIX_ANOMALIES_NORMAL_SOE_IMPLICIT_VISITS_MANUAL_BOXES_SHARED_BOUNDARIES_LOCAL_IDS_ASSEMBLY_ENTRY_MAC_FONT_PASS')
