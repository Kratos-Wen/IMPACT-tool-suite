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
timeline=w.hoi_timeline;row=timeline.actor_rows['Left_hand'];row.resize(900,48);timeline._set_view(0,100)
assert not hasattr(timeline,'box_strip')
assert timeline.layout().count()==2,'Only existing event rows and view controls may occupy space'
assert len(timeline.actor_rows)==2 and row.sizeHint().height()==48
assert timeline.actor_rows['Right_hand']._hit_box_marker(row.frame_to_x(20),30) is None
# Slider changes must repaint markers inside the existing event row.
with patch.object(row,'update') as repaint:
    w.hoi_timeline._on_view_start_changed(5);assert repaint.called
    repaint.reset_mock();w.hoi_timeline._on_view_span_changed(0);assert repaint.called
    repaint.reset_mock();w.hoi_timeline._refresh_view_slider();assert repaint.called
w.hoi_timeline._set_view(0,100)
saved_intervals=copy.deepcopy(w.events)
QTest.mouseClick(row,Qt.LeftButton,pos=QPoint(row.frame_to_x(20),30))
assert w.events==saved_intervals and not row._dragging and row._pending_box_jump is None
assert w.player.current_frame==20 and w.player._displayed_frame==20
assert int(w.player._last_frame_bgr[0,0,0])==20
assert w.combo_inline_verb.currentData() == 'hold'
assert w.combo_inline_noun.currentData() == uid
assert w.combo_inline_instrument.currentData() == tool
# Paused observation must immediately repaint and recolor a machine hand tick.
with patch.object(w,'isVisible',return_value=True),patch.object(w,'isActiveWindow',return_value=True),patch.object(row,'update') as repaint:
    w._observed_frame_context=w._observation_context();w._observe_displayed_boxes()
    assert repaint.called
    assert any(m['role']=='H' and m['frame']==20 and m['state']=='viewed' for m in w._bbox_frame_markers())
row.grab().save(str(Path(os.environ.get('IMPACT_HOI_SETTINGS_DIR','/tmp')).parent/'bbox_event_preview.png'))
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
row=timeline.actor_rows['Left_hand'];row.resize(900,48)
# Dense background tracks must not leak into event ticks or cause rescans per paint.
w.raw_boxes=CountedRows([dict(id=999,orig_frame=f,label='background',source='automatic',x1=1,y1=1,x2=5,y2=5) for f in range(20000)]+copy.deepcopy(w.raw_boxes))
w._rebuild_bboxes_from_raw();w._bump_bbox_revision();w._bump_query_state_revision()
w._bbox_frame_markers();row._box_markers.layout(row._selected_segment())
scans=w.raw_boxes.scans;stable_ids=copy.deepcopy(w.global_object_map)
for _ in range(200):
    assert all(m['entity']!='O:999' for m in w._bbox_frame_markers())
    row._box_markers.layout(row._selected_segment());row._hit_box_marker(row.frame_to_x(20),30)
assert w.raw_boxes.scans==scans,'Painting repeatedly rescanned the full track list'
for i in range(100):
    w._jump_bbox_frame((10,20,30)[i%3]);assert w.global_object_map==stable_ids
    assert w.player._displayed_frame==w.player.current_frame
# A geometry edit and undo update the ticks without changing the object's identity.
box=w._boxes_for_review('O:'+str(uid),20)[0];w._selected_edit_box=copy.deepcopy(box)
w._on_box_edited(uid,dict(box,frame=20,x1=2))
assert any(m['frame']==20 and m['role']=='O' and m['state']=='manual' for m in w._bbox_frame_markers())
w._hoi_undo();assert w.global_object_map==stable_ids
row=timeline.actor_rows['Left_hand'];row.resize(900,48)
# A click on a bbox frame must not become an event move; dragging from the same
# line is also navigation-only. The upper SOE handle area remains editable.
local_segments=[dict(event_id=17,start=10,end=80,onset=40,verb='hold')]
local_markers=[dict(frame=f,role='O',entity='O:1',state='manual') for f in (10,20,40,80)]
row._get_segments=lambda:local_segments
row._box_markers.get_markers=lambda:local_markers
row.set_selected(17)
row._on_select=lambda *args:None
jumps=[];updates=[];activations=[]
row._on_box_marker=jumps.append
row._on_update=lambda *args:updates.append(args)
row._on_activate=lambda *args:activations.append(args)
for frame in (10,20,40,80):
    QTest.mouseClick(row,Qt.LeftButton,pos=QPoint(row.frame_to_x(frame),30))
    assert jumps[-1]==frame and not updates
assert row._hit_box_marker(row.frame_to_x(40),9) is None
QTest.mousePress(row,Qt.LeftButton,pos=QPoint(row.frame_to_x(40),9))
assert row._mode=='move_onset' and row._dragging
QTest.mouseRelease(row,Qt.LeftButton,pos=QPoint(row.frame_to_x(40),9))
assert updates[-1]==(17,'Left_hand',10,80,40)
updates.clear()
QTest.mousePress(row,Qt.LeftButton,pos=QPoint(row.frame_to_x(20),30))
assert row._pending_box_jump and not row._dragging
point=QPoint(row.frame_to_x(25),30)
event=QMouseEvent(QMouseEvent.MouseMove,point,Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
app.sendEvent(row,event)
QTest.mouseRelease(row,Qt.LeftButton,pos=point)
assert not updates and jumps[-1]==20 and not row._pending_box_jump
# The event body away from ticks still supports intentional dragging.
QTest.mousePress(row,Qt.LeftButton,pos=QPoint(row.frame_to_x(60),30))
point=QPoint(row.frame_to_x(65),30)
app.sendEvent(row,QMouseEvent(QMouseEvent.MouseMove,point,Qt.NoButton,Qt.LeftButton,Qt.NoModifier))
QTest.mouseRelease(row,Qt.LeftButton,pos=point)
assert updates[-1]==(17,'Left_hand',15,85,45)
QTest.mouseDClick(row,Qt.LeftButton,pos=QPoint(row.frame_to_x(20),30))
assert jumps[-1]==20 and not activations
# Qt's trailing release after a double-click must not commit a stale drag.
updates.clear()
QTest.mouseRelease(row,Qt.LeftButton,pos=QPoint(row.frame_to_x(20),30))
assert not updates and not row._dragging and row._pending_box_jump is None
QTest.mouseDClick(row,Qt.LeftButton,pos=QPoint(row.frame_to_x(60),30))
assert activations[-1]==(17,'Left_hand')
# No selected event means no marker hit; event-only filtering remains local.
row.set_selected(None);assert row._hit_box_marker(row.frame_to_x(20),30) is None
row.set_selected(17)
local_markers=local_markers+[dict(frame=95,role='O',entity='O:1',state='manual')]
assert row._hit_box_marker(row.frame_to_x(95),30) is None
# Dense event tracks coalesce to the available pixels and retain exact frame
# access. Repeated paints and hit tests reuse their layout, not all track rows.
local_segments[0].update(start=0,end=19999,onset=0)
local_markers=[dict(frame=f,role='H',entity='H:Left_hand',state='model') for f in range(20000)]
timeline._set_view(0,20000)
layout=row._box_markers.layout(row._selected_segment())
assert len(layout)<=row.width()-row.get_gutter()
assert sum(len(group['frames']) for group in layout.values())==20000
builds=row._box_markers.rebuilds
row.grab()  # Qt may polish/resize the hidden widget on its first dense render.
row._box_markers.layout(row._selected_segment());builds=row._box_markers.rebuilds
start=time.perf_counter()
for _ in range(100):
    row.grab();assert row._hit_box_marker(row.frame_to_x(8000),30)['frame'] in range(7970,8031)
assert row._box_markers.rebuilds==builds
elapsed=time.perf_counter()-start
assert elapsed<8,'Repeated timeline rendering is too slow: '+str(elapsed)
# Invalid/duplicate frames use the orange warning line. Geometry/provenance
# changes provide a new cached source; switching hand/event clears old ticks.
local_markers=[dict(frame=8000,role='O',entity='O:1',state='conflict')]
groups=row._box_markers.layout(row._selected_segment())
assert len(groups)==1 and next(iter(groups.values()))['conflict']
assert 'duplicate / invalid' in row._hit_box_marker(row.frame_to_x(8000),30)['tooltip']
print('DENSE_TIMELINE_100_PAINTS_SECONDS',round(elapsed,3))
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media();w.deleteLater();app.processEvents()
print('BBOX_INLINE_TICKS_NO_EXTRA_ROWS_EXACT_SEEK_DRAG_SOE_IMPORT_20000_BOX_STRESS_ID_PRESERVATION_PASS')
