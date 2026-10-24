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

w.player._frame_w=32;w.player._frame_h=32;w.video_path='frame-copy-fixture.mp4'
w._jump_bbox_frame(20)
source=copy.deepcopy(w.raw_boxes);lib=copy.deepcopy(w.global_object_map);counter=w.object_id_counter
noun_id=w._hand_noun_object_id(w.event_draft['Left_hand'])
w._copy_frame_boxes();assert w._frame_box_clipboard['frame']==20
assert w.raw_boxes==source
w._jump_bbox_frame(40);w._paste_frame_boxes()
new=[b for b in w.raw_boxes if b['orig_frame']==40]
assert len(new)==3
assert all(b['source']=='manual_box_copy' and not b['human_verified'] and not b['locked'] and b['copied_from_frame']==20 for b in new)
assert w.object_id_counter==counter and w.global_object_map==lib
assert [b for b in w.raw_boxes if b['orig_frame']!=40]==source
assert any(m['frame']==40 for m in w._bbox_frame_markers())
length=len(w.raw_boxes);w._paste_frame_boxes();assert len(w.raw_boxes)==length,'Repeated paste cannot create duplicate boxes'
w._hoi_undo();assert w.raw_boxes==source,'Undo removes only the paste'
w._hoi_redo();assert len([b for b in w.raw_boxes if b['orig_frame']==40])==3
occupied=dict(new[1],orig_frame=50,x1=2,y1=3,x2=9,y2=10,source='manual_box_add');w.raw_boxes.append(occupied)
w._jump_bbox_frame(50);w._paste_frame_boxes()
assert len([b for b in w.raw_boxes if b['orig_frame']==50])==3
assert occupied in w.raw_boxes,'Existing target box is never replaced'
w._jump_bbox_frame(60);w.event_draft['Left_hand']['noun_object_id']=9999;w.event_draft['Left_hand']['target_object_id']=9999
w._paste_frame_boxes()
assert len([b for b in w.raw_boxes if b['orig_frame']==60])==2,'Changed Object ID must not inherit another object geometry'
assert not any(b['orig_frame']==60 and b['id']==noun_id for b in w.raw_boxes)
w.event_draft['Left_hand']['noun_object_id']=noun_id;w.event_draft['Left_hand']['target_object_id']=noun_id
w.video_path='different-video.mp4';before=copy.deepcopy(w.raw_boxes);w._paste_frame_boxes();assert w.raw_boxes==before
w.video_path='frame-copy-fixture.mp4';w._apply_draft_to_selected_event()
exported=w._build_payload_v2()
assert sum(b.get('copied_from_frame')==20 for t in exported['tracks'].values() for b in t['boxes'])==8
w._load_annotations_v2(exported)
assert sum(b.get('copied_from_frame')==20 for b in w.raw_boxes)==8,'Source frame metadata survives reload'
from core.frame_copy import paste_plan
sample=dict(x1=1,y1=1,x2=4,y2=4,id=1,orig_frame=20,_copy_frame=20)
added,skips=paste_plan({'O:1':[sample,sample]},30,{'O:1'},set());assert not added and skips
added,skips=paste_plan({'O:1':[dict(sample,x1=float('nan'))]},30,{'O:1'},set());assert not added
# Frame offsets remain source-video frame coordinates.
w._set_selected_event(w.events[0]['event_id'],'Left_hand')
w.start_offset=5;w._jump_bbox_frame(45);w._copy_frame_boxes();w._jump_bbox_frame(70);w._paste_frame_boxes()
rows=[b for b in w.raw_boxes if b.get('copied_from_frame')==45]
assert len(rows)==3 and all(b['orig_frame']==65 for b in rows),(len(rows),getattr(w,'_last_frame_copy_message',''),getattr(w,'_frame_box_clipboard',None))
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media();w.deleteLater();app.processEvents()
print('FRAME_COPY_IDS_SOURCE_UNCHANGED_NO_OVERWRITE_UNDO_REDO_SCOPE_SAVE_RELOAD_OFFSETS_PASS')
