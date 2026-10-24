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

import tempfile,json
with tempfile.TemporaryDirectory() as directory:
    folder=Path(directory)/'fixture';folder.mkdir();w.video_path=str(folder/'video.mp4')
    flags={'schema':'impact-review-flags','trial':'fixture','frame_count':20000,'notes':[
        dict(id='one',frame=20,hand='left_hand',event_id='fixture',message='Check the visible screw',kind='bbox'),
        dict(id='two',frame=30,hand='left_hand',event_id='fixture',message='Check event timing',kind='semantic')]}
    (folder/'review_flags.json').write_text(json.dumps(flags))
    w._refresh_reviewer_notes();assert len(w._reviewer_notes())==2
    before=copy.deepcopy(w.events);boxes=copy.deepcopy(w.raw_boxes)
    w._jump_reviewer_note();assert w.player.current_frame==20
    assert 'visible screw' in w.lbl_reviewer_notes.text()
    assert w.events==before and w.raw_boxes==boxes
    ticks=w._bbox_frame_markers();assert any(m['entity']=='review:one' for m in ticks)
    assert w._bbox_frame_markers() is ticks
    w._resolve_reviewer_note();assert len(w._reviewer_notes())==1
    assert w.events==before and w.raw_boxes==boxes
    del w._reviewer_notes_key
    assert len(w._reviewer_notes())==1,'Dismissal persists without touching annotations'
    w._jump_reviewer_note();assert w.player.current_frame==30
    flags['notes'][0]['message']='Changed review note'
    (folder/'review_flags.json').write_text(json.dumps(flags));del w._reviewer_notes_key
    assert len(w._reviewer_notes())==2,'Old dismissals cannot hide a newer note batch'
    w.video_path=str(Path(directory)/'other'/'video.mp4');w._refresh_reviewer_notes()
    assert not w._reviewer_notes(),'Notes must not leak across tasks'
    w.video_path=str(folder/'video.mp4');flags['frame_count']=30
    (folder/'review_flags.json').write_text(json.dumps(flags));w._refresh_reviewer_notes()
    assert not w._reviewer_notes() and w._reviewer_notes_error,'Reject mismatched source versions'
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media();w.deleteLater();app.processEvents()
print('REVIEWER_NOTES_NAVIGATION_MARKERS_PERSISTENCE_SOURCE_GUARD_NO_ANNOTATION_MUTATION_PASS')
