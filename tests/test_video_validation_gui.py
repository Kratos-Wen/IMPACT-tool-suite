"""Clickable bbox ticks, provenance changes, exact seeks and large-track stress."""
import copy,os,sys,time
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
import numpy as np
from PyQt5.QtCore import Qt,QPoint
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtTest import QTest
from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(verbs=['hold'],noun_classes=['part','tool','assembly'],anomaly_labels=['attribute_a'],singleton_noun_classes=['tool'])
from ui.hoi_window import HOIWindow
from core.frame_review import signature

app=QApplication([])
for name in ('information','warning','question','critical'):setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)

class Capture:
    def __init__(self):self.position=0
    def set(self,key,value):self.position=int(value);return True
    def read(self):
        value=self.position;self.position+=1
        return True,np.full((32,32,3),value%255,np.uint8)
    def release(self):pass

class CountedRows(list):
    scans=0
    def __iter__(self):self.scans+=1;return super().__iter__()

payload=dict(schema='hoi-annotation',video_id='fixture',fps=15,frame_count=20000,
    object_library={'1001':dict(label='part_1',category='part'),'1002':dict(label='tool_1',category='tool')},
    tracks={'T_OBJ_1001':dict(object_id=1001,category='part',boxes=[dict(frame=f,bbox=[1,2,8,9],source='automatic' if f==20 else 'manual_box_add') for f in (10,20,30)]),
            'T_OBJ_1002':dict(object_id=1002,category='tool',boxes=[dict(frame=20,bbox=[12,2,18,9],source='manual_box_add')]),
            'T_LEFT_HAND':dict(category='Left_hand',boxes=[dict(frame=f,bbox=[5,12,12,20],source='automatic') for f in (10,20,30)])},
    hoi_events={'left_hand':[dict(event_id='fixture',start_frame=0,contact_onset_frame=0,end_frame=100,verb='hold',noun_object_id=1001,instrument_object_id=1002)],'right_hand':[]})
w=HOIWindow();w._autosave_timer.stop();w._show_quick_start_on_startup=False
w.player.frame_count=20000;w.player.frame_rate=15;w.player.crop_start=0;w.player.crop_end=19999;w.player.cap=Capture();w.player._media_path='fixture'
w._load_annotations_v2(payload);w._set_selected_event(w.events[0]['event_id'],'Left_hand')

from core.video_validation import annotation_signature,checked_record
w.player._frame_w=32;w.player._frame_h=32
original=w._build_payload_v2()
assert not original['video_validation']
w.validator_name='Fixture reviewer'
w.btn_validation.setChecked(True)
assert w.video_validation['status']=='in_progress'
w.resize(1000,800);w.show()
for _ in range(4):app.processEvents()
assert w.btn_mark_validated.isVisible()
assert w.width()<=1000,(w.width(),w.minimumSizeHint().width())
assert w.btn_mark_validated.width()>=w.btn_mark_validated.fontMetrics().horizontalAdvance(w.btn_mark_validated.text())+8
w._mark_hoi_saved();w._mark_video_validated()
assert w.video_validation['status']=='validated'
assert w._hoi_has_unsaved_changes(),'Validation alone must be saved'
import tempfile,json
from PyQt5.QtWidgets import QFileDialog
with tempfile.TemporaryDirectory() as directory:
    target=Path(directory)/'reviewed.json'
    with patch.object(QFileDialog,'getSaveFileName',return_value=(str(target),'JSON Files (*.json)')), patch.object(w,'_require_participant_code_for_study_save',return_value=True), patch.object(w,'_check_incomplete_hoi',return_value=(True,True)):
        w._save_annotations_json()
    assert target.is_file(),'Formal save writes the annotation'
    payload=json.loads(target.read_text())
    assert payload['video_validation']['status']=='validated','Ctrl+S must retain validation'
assert payload['video_validation']['reviewer']=='Fixture reviewer'
assert payload['video_validation']['validated_at']
assert annotation_signature(original)==annotation_signature(payload),'Marking does not change annotation content'
snapshot=w._autosave_snapshot();assert snapshot['video_validation']['status']=='validated'
w._load_annotations_v2(payload)
loaded=w._build_payload_v2()
if loaded['video_validation']['status']!='validated':
    from core.video_validation import annotation_content
    a,b=annotation_content(payload),annotation_content(loaded)
    print('ROUNDTRIP_DIFF',[(k,a[k],b[k]) for k in a if a[k]!=b[k]],flush=True)
assert loaded['video_validation']['status']=='validated','Save/reload must preserve validation'
w.player.current_frame=25;w._set_frame_controls(25)
assert w.video_validation['status']=='validated','Navigation does not invalidate review'
w.btn_validation.setChecked(False)
w._push_undo()
w.event_draft['Left_hand']['verb']='pick_up'
w._apply_draft_to_selected_event()
assert w.video_validation['status']=='needs_review'
w._hoi_undo()
assert w.video_validation['status']=='validated','Undo restores the reviewed content and status'
w._hoi_redo()
assert w.video_validation['status']=='needs_review','Redo restores the edited status'
w._mark_validation_change()
assert w.video_validation['status']=='needs_review','Edits outside validation mode must invalidate'
w._restore_state(snapshot)
assert w.video_validation['status']=='validated','Recovery preserves validation metadata'
changed=copy.deepcopy(payload);changed['hoi_events']['left_hand'][0]['verb']='pick_up'
assert checked_record(payload['video_validation'],changed)['status']=='needs_review'
changed=copy.deepcopy(payload);next(iter(changed['tracks'].values()))['boxes'][0]['bbox'][0]+=1
assert checked_record(payload['video_validation'],changed)['status']=='needs_review'
changed=copy.deepcopy(payload);changed['editor_view']={'start_frame':5,'end_frame':50}
assert checked_record(payload['video_validation'],changed)['status']=='validated'
w._load_annotations_v2(changed)
assert w.video_validation['status']=='validated'
changed=copy.deepcopy(payload);changed['hoi_events']['left_hand'][0]['end_frame']=99
w._load_annotations_v2(changed)
assert w.video_validation['status']=='needs_review','Stale externally edited JSON must not claim validated'
w._load_annotations_v2(original)
assert not w.video_validation,'Validation never leaks to a new/unvalidated file'
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media();w.deleteLater();app.processEvents()
print('VIDEO_VALIDATION_MARK_SAVE_RELOAD_RECOVERY_EDIT_INVALIDATION_NAVIGATION_AND_SOURCE_GUARD_PASS')
