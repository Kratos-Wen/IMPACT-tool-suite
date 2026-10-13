"""Ten independent blank-video annotation journeys using live Qt and real files."""
import copy,json,os,sys,tempfile,time
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog,QInputDialog
from PyQt5.QtCore import QEvent,Qt,QPoint
from PyQt5.QtTest import QTest
from core.project_profile import PROFILE
from core.safe_storage import atomic_json
from core.hand_identity import hand_box_id
from ui.hoi_window import HOIWindow
app=QApplication([]);errors=[]
original_excepthook=sys.excepthook
def record_exception(typ,value,tb):
 errors.append((typ.__name__,str(value)));original_excepthook(typ,value,tb)
sys.excepthook=record_exception
for name in ('information','warning','critical'):
 setattr(QMessageBox,name,lambda *a,**k:QMessageBox.Ok)
QMessageBox.question=lambda *a,**k:QMessageBox.Yes
profile=dict(noun_classes=['part','tool','fastener','assembly'],verbs=['hold','pick_up','insert'],
 anomaly_labels=['attribute_'+c for c in 'abcdef'],singleton_noun_classes=['part','tool'],
 create_default_noun_instances=True,assembly_components=['part','fastener'],assembly_interfaces=[])
def dispose(w):
 w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True
 w.close();w.player.release_media();w.deleteLater();app.sendPostedEvents(None,QEvent.DeferredDelete)
def geometry(w):
 return sorted((b['id'],b['orig_frame'],w._noun_display_name(b['label']),b['x1'],b['y1'],b['x2'],b['y2']) for b in w.raw_boxes)
started=time.monotonic()
with tempfile.TemporaryDirectory(prefix='First annotation 中文 ') as directory:
 root=Path(directory);base=root/'source.mp4'
 writer=cv2.VideoWriter(str(base),cv2.VideoWriter_fourcc(*'mp4v'),15,(96,64));assert writer.isOpened()
 for frame in range(160):writer.write(np.full((64,96,3),frame%255,np.uint8))
 writer.release()
 for reviewer in range(10):
  folder=root/str(reviewer);folder.mkdir();video=folder/'video.mp4'
  video.write_bytes(base.read_bytes());atomic_json(folder/'project_profile.json',profile,backup=False)
  os.environ['IMPACT_HOI_SETTINGS_DIR']=str(folder/'settings');os.environ['IMPACT_RESTORE_LAST_PROJECT']='0'
  PROFILE.clear();PROFILE.update(profile)
  w=HOIWindow();w._show_quick_start_on_startup=False
  assert w._load_video(str(video));w._autosave_timer.stop();assert not w.events and not w.raw_boxes
  w.show();app.processEvents();ids={};expected=[]
  for index in range(8):
   side=('Left_hand','Right_hand')[index%2];start=(index//2)*30;end=start+20;onset=start+7
   event_id=w._on_hoi_timeline_create(side,start,end,onset);assert event_id is not None
   assert w.combo_inline_noun.isEnabled() and w.combo_inline_noun.count()>1
   verb=('pick_up','insert','hold')[index%3]
   w.combo_inline_verb.setCurrentIndex(w.combo_inline_verb.findData(verb))
   category='part' if index<2 else 'fastener'
   # The first two fasteners are explicitly different physical instances.
   if category=='fastener' and index in (2,3):
    w.event_draft[side].update(noun_object_id=None,target_object_id=None)
    w._pending_draw_box_role=None
    with patch.object(QInputDialog,'getItem',side_effect=lambda parent,title,text,options,*a:(options[-1],True)):
     w._on_box_edited(None,dict(_action='add',label=category,frame=start,x1=40,y1=15,x2=55,y2=30))
    uid=w.raw_boxes[-1]['id'];ids[index]=uid
   else:uid=w.global_object_map['part'] if category=='part' else ids[2+(index%2)]
   w.combo_inline_noun.setCurrentIndex(w.combo_inline_noun.findData(uid))
   tool=w.global_object_map['tool'];w.combo_inline_instrument.setCurrentIndex(w.combo_inline_instrument.findData(tool))
   hand=w._selected_hand_data();assert hand['verb']==verb and w._hand_noun_object_id(hand)==uid
   assert w._hand_instrument_object_id(hand)==tool
   frames=sorted(set((hand['interaction_start'],hand['functional_contact_onset'],hand['interaction_end'])))
   if verb=='hold':assert hand['functional_contact_onset']==hand['interaction_start']
   for frame in frames:
    w.player.seek(frame)
    for role,label,x in [('hand',side,2),('noun',category,40),('instrument','tool',70)]:
     existing=next((b for b in w.raw_boxes if b['orig_frame']==frame and b['id']==({'hand':hand_box_id(side),'noun':uid,'instrument':tool}[role])),None)
     if existing:continue
     w._pending_draw_box_role=role
     w._on_box_edited(None,dict(_action='add',label=label,frame=frame,x1=x,y1=15,x2=x+14,y2=35))
     assert any(b['id']=={'hand':hand_box_id(side),'noun':uid,'instrument':tool}[role] for b in w.player.overlay_boxes)
   # Redrawing an existing entity replaces its one current-frame box.
   # A novice may draw again instead of finding a resize handle.
   for role,label,entity in [('hand',side,hand_box_id(side)),('noun',category,uid)]:
    w.player.seek(start);previous=geometry(w);w._pending_draw_box_role=role
    w._on_box_edited(None,dict(_action='add',label=label,frame=start,x1=10,y1=10,x2=25,y2=25))
    rows=[b for b in w.raw_boxes if b['id']==entity and b['orig_frame']==start]
    assert len(rows)==1 and rows[0]['x1']==10,('Redraw left overlapping boxes',rows)
    w._hoi_undo();assert geometry(w)==previous
    w._hoi_redo();assert len([b for b in w.raw_boxes if b['id']==entity and b['orig_frame']==start])==1
   if index%3==1:
    w.combo_inline_anomaly.toggle('attribute_a');w.combo_inline_anomaly.toggle('attribute_b')
    w.player.seek(start+10);w.btn_anomaly_keyframe.click()
    assert start+10 in w._selected_hand_data()['required_review_frames']
   # Edit/delete/undo/redo one frame without altering any other physical track.
   box=next(b for b in w.raw_boxes if b['id']==uid and b['orig_frame']==start)
   original=geometry(w);w._selected_edit_box=copy.deepcopy(box)
   w._on_box_edited(uid,dict(box,frame=start,x1=box['x1']+1));edited=geometry(w)
   w._hoi_undo();assert geometry(w)==original
   w._hoi_redo();assert geometry(w)==edited
   box=next(b for b in w.raw_boxes if b['id']==uid and b['orig_frame']==start)
   w._selected_edit_box=copy.deepcopy(box);w._on_box_edited(uid,dict(box,frame=start,_action='delete'))
   w._hoi_undo();assert geometry(w)==edited
   # Click manual bbox lines through the actual timeline mouse handler.
   row=w.hoi_timeline.actor_rows[side];w.hoi_timeline._set_view(start,25)
   before=copy.deepcopy(w.events)
   QTest.mouseClick(row,Qt.LeftButton,pos=QPoint(row.frame_to_x(start),30))
   assert w.events==before and w.player.current_frame==start
   assert w.combo_inline_verb.currentData()==verb and w.combo_inline_noun.currentData()==uid
   expected.append((side,start,end,verb,uid,tool))
   w._checkpoint_tracking();app.processEvents()
  assert ids[2]!=ids[3] and len(w.global_object_map.values())==len(set(w.global_object_map.values()))
  for side in ('Left_hand','Right_hand'):
   assert {b['id'] for b in w.raw_boxes if b['label']==side}=={hand_box_id(side)}
  target=folder/'reviewed.json';w._require_participant_code_for_study_save=lambda:True
  # Incomplete anomaly keyframes may be saved as drafts by explicitly accepting the save dialog.
  with patch.object(QFileDialog,'getSaveFileName',return_value=(str(target),'')):
   w._save_annotations_json()
  assert target.is_file(),errors
  before=geometry(w);saved=json.loads(target.read_text());registry=copy.deepcopy(saved['object_library'])
  dispose(w)
  restored=HOIWindow();restored._show_quick_start_on_startup=False
  assert restored._load_video(str(video));restored._autosave_timer.stop()
  # Standalone packages do not assume a reviewed.json without task metadata.
  restored._load_annotations_v2(json.loads(target.read_text()))
  assert geometry(restored)==before, ('Geometry changed',set(before)-set(geometry(restored)),set(geometry(restored))-set(before))
  assert restored._build_payload_v2()['object_library']==registry, ('Registry changed',registry,restored._build_payload_v2()['object_library'])
  for side,start,end,verb,uid,tool in expected:
   event=next(e for e in restored.events if e['hoi_data'][side].get('interaction_start')==start)
   restored._set_selected_event(event['event_id'],side);hand=restored._selected_hand_data()
   assert (hand['interaction_end'],hand['verb'],restored._hand_noun_object_id(hand),restored._hand_instrument_object_id(hand))==(end,verb,uid,tool)
  dispose(restored);assert not errors,errors
  print('FIRST_ANNOTATION_REVIEWER',reviewer+1,'EIGHT_EVENTS_SAVE_REOPEN_PASS',flush=True)
print('TEN_BLANK_VIDEO_JOURNEYS_80_EVENTS_QT_EXCEPTIONS_ZERO_SECONDS',round(time.monotonic()-started,2))
