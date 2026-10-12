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
    tracks={'T_OBJ_1001':dict(object_id=1001,category='part',boxes=[dict(frame=f,bbox=[1,2,8,9],source='automatic') for f in (10,20,30)]),
            'T_OBJ_1002':dict(object_id=1002,category='tool',boxes=[dict(frame=20,bbox=[12,2,18,9],source='manual_box_add')]),
            'T_LEFT_HAND':dict(category='Left_hand',boxes=[dict(frame=f,bbox=[5,12,12,20],source='automatic') for f in (10,20,30)])},
    hoi_events={'left_hand':[dict(event_id='fixture',start_frame=0,contact_onset_frame=0,end_frame=100,verb='hold',noun_object_id=1001,instrument_object_id=1002)],'right_hand':[]})
w=HOIWindow();w._autosave_timer.stop();w._show_quick_start_on_startup=False
w.player.frame_count=20000;w.player.frame_rate=15;w.player.crop_start=0;w.player.crop_end=19999;w.player.cap=Capture();w.player._media_path='fixture'
w._load_annotations_v2(payload);w._set_selected_event(w.events[0]['event_id'],'Left_hand')
markers=w._bbox_frame_markers();assert {m['frame'] for m in markers}=={10,20,30}
assert any(m['role']=='I' and m['state']=='manual' for m in markers)
hand=w.event_draft['Left_hand'];uid=w._hand_noun_object_id(hand);tool=w._hand_instrument_object_id(hand)
rows=w._boxes_for_review('O:'+str(uid),20)
w.frame_review={'20':{'O:'+str(uid):dict(state='verified',signature=signature(rows))}};w._bump_query_state_revision()
assert any(m['role']=='O' and m['frame']==20 and m['state']=='viewed' for m in w._bbox_frame_markers())
strip=w.hoi_timeline.box_strip;strip.resize(900,strip.height());w.hoi_timeline._set_view(0,100)
# Slider changes must repaint the strip along with the event rows.
with patch.object(strip,'update') as repaint:
    w.hoi_timeline._on_view_start_changed(5);assert repaint.called
    repaint.reset_mock();w.hoi_timeline._on_view_span_changed(0);assert repaint.called
    repaint.reset_mock();w.hoi_timeline._refresh_view_slider();assert repaint.called
w.hoi_timeline._set_view(0,100)
QTest.mouseClick(strip,Qt.LeftButton,pos=QPoint(strip.frame_to_x(20),strip.lane_height+7))
assert w.player.current_frame==20 and w.player._displayed_frame==20
assert int(w.player._last_frame_bgr[0,0,0])==20
# Paused observation must immediately repaint and recolor a machine hand tick.
with patch.object(w,'isVisible',return_value=True),patch.object(w,'isActiveWindow',return_value=True),patch.object(strip,'update') as repaint:
    w._observed_frame_context=w._observation_context();w._observe_displayed_boxes()
    assert repaint.called
    assert any(m['role']=='H' and m['frame']==20 and m['state']=='viewed' for m in w._bbox_frame_markers())
strip.grab().save(str(Path(os.environ.get('IMPACT_HOI_SETTINGS_DIR','/tmp')).parent/'bbox_strip_preview.png'))
w._jump_bbox_direction(1);assert w.player.current_frame==30
w._jump_bbox_direction(-1);assert w.player.current_frame==20
before=copy.deepcopy(w._build_payload_v2())
for mode in ('hands','none','all'):
    w._box_import_override=mode;w._load_annotations_v2(payload);w._set_selected_event(w.events[0]['event_id'],'Left_hand')
    assert w._hand_noun_object_id(w.event_draft['Left_hand'])==uid
    assert w._hand_instrument_object_id(w.event_draft['Left_hand'])==tool
    assert any(b['source']=='manual_box_add' for b in w.raw_boxes)
    assert any(w._normalize_hand_label(b['label']) for b in w.raw_boxes)==(mode!='none')
    assert any(b['id']==uid and not w._normalize_hand_label(b['label']) for b in w.raw_boxes)==(mode=='all')
assert payload['object_library']['1001']['label']=='part_1', 'Import changed the source draft'
# Dense background tracks must not leak into event ticks or cause rescans per paint.
w.raw_boxes=CountedRows([dict(id=999,orig_frame=f,label='background',source='automatic',x1=1,y1=1,x2=5,y2=5) for f in range(20000)]+copy.deepcopy(w.raw_boxes))
w._rebuild_bboxes_from_raw();w._bump_bbox_revision();w._bump_query_state_revision()
w._bbox_frame_markers();scans=w.raw_boxes.scans;stable_ids=copy.deepcopy(w.global_object_map)
for _ in range(200):
    assert all(m['entity']!='O:999' for m in w._bbox_frame_markers())
    strip._layout();strip.hit(strip.frame_to_x(20),strip.lane_height+7)
assert w.raw_boxes.scans==scans,'Painting repeatedly rescanned the full track list'
for i in range(100):
    w._jump_bbox_frame((10,20,30)[i%3]);assert w.global_object_map==stable_ids
    assert w.player._displayed_frame==w.player.current_frame
# A geometry edit and undo update the ticks without changing the object's identity.
box=w._boxes_for_review('O:'+str(uid),20)[0];w._selected_edit_box=copy.deepcopy(box)
w._on_box_edited(uid,dict(box,frame=20,x1=2))
assert any(m['frame']==20 and m['role']=='O' and m['state']=='manual' for m in w._bbox_frame_markers())
w._hoi_undo();assert w.global_object_map==stable_ids
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media();w.deleteLater();app.processEvents()
print('BBOX_CLICK_EXACT_FRAME_ROLE_COLORS_IMPORT_CHOICES_20000_BOX_STRESS_100_SEEKS_ID_PRESERVATION_PASS')
