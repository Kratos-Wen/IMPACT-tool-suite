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

PROFILE['noun_classes'].append('new_part')
reviewed=dict(schema='hoi-annotation',video_id='human-fixture',fps=15,frame_count=20000,frame_size=[32,32],
 object_library={'0':dict(label='part',category='part'),'12':dict(label='tool',category='tool'),'40':dict(label='tool',category='tool')},
 tracks={'T_OBJ_0':dict(object_id=0,category='part',boxes=[dict(frame=10,bbox=[1,2,8,9],source='manual')]),
         'T_OBJ_12':dict(object_id=12,category='tool',boxes=[]),'T_OBJ_40':dict(object_id=40,category='tool',boxes=[])},
 hoi_events={'left_hand':[dict(event_id='reviewed',start_frame=0,contact_onset_frame=0,end_frame=20,verb='hold',noun_object_id=0,target_object_id=0,instrument_object_id=None,tool_object_id=12,
       links={'target_track_id':'T_OBJ_0','tool_track_id':'T_OBJ_12'},interaction={'target':'part','tool':'tool'})],'right_hand':[]})
w.player._frame_w=32;w.player._frame_h=32
w._load_annotations_v2(reviewed,'reviewed.json')
assert set(w.global_object_map.values())=={0,12,40}
w._apply_profile_libraries()
assert set(w.global_object_map.values())=={0,12,40},'A later profile refresh must not invent unused IDs'
result=w._build_payload_v2()
event=result['hoi_events']['left_hand'][0]
assert event['noun_object_id']==0 and event['instrument_object_id'] is None
assert set(result['object_library'])=={'0','12','40'}
assert w._snapshot_state()['loaded_reviewed_registry'] is True
w._load_annotations_v2(result,'reviewed.json')
assert set(w.global_object_map.values())=={0,12,40}
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media();w.deleteLater();app.processEvents()
print('REVIEWED_IMPORT_PRESERVES_ZERO_SPARSE_IDS_EXPLICIT_EMPTY_ROLE_NO_PHANTOM_OBJECT_PASS')
