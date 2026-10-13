"""One identity per anatomical hand through imports, edits, undo and JSON/XML export."""
import os,sys,copy,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog
from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(noun_classes=['part'],verbs=['hold'],anomaly_labels=[])
from core.hand_identity import hand_box_id
from ui.hoi_window import HOIWindow
app=QApplication([])
for name in ('information','warning','critical','question'):setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=100;w.player.frame_rate=15
payload=dict(schema='hoi-annotation',video_id='fixture',frame_count=100,fps=15,
 object_library={'0':dict(label='part',category='part')},
 tracks={f'T_{side.upper()}':dict(category=side,object_id=None,
 boxes=[dict(frame=f,bbox=[10+index*30,10,25+index*30,30],source='manual') for f in range(60)])
 for index,side in enumerate(('left_hand','right_hand'))},
 hoi_events={'left_hand':[dict(start_frame=0,contact_onset_frame=0,end_frame=20,verb='hold',noun_object_id=0),
 dict(start_frame=21,contact_onset_frame=21,end_frame=40,verb='hold',noun_object_id=0)],'right_hand':[]})
w._load_annotations_v2(payload)
def assert_ids():
 for label in ('Left_hand','Right_hand'):
  rows=[b for b in w.raw_boxes if b['label']==label]
  assert rows and {b['id'] for b in rows}=={hand_box_id(label)},(label,{b['id'] for b in rows})
 assert hand_box_id('Left_hand')!=hand_box_id('Right_hand')
assert_ids()
# A manual frame reuses the hand identity, and an edit affects only its own frame.
w._set_selected_event(w.events[0]['event_id'],'Left_hand');w.player.current_frame=90
w._on_box_edited(None,dict(_action='add',label='Left_hand',frame=90,x1=1,y1=2,x2=15,y2=20))
assert_ids();before=copy.deepcopy(w.raw_boxes)
box=next(b for b in w.raw_boxes if b['label']=='Left_hand' and b['orig_frame']==10)
w._on_box_edited(box['id'],dict(box,frame=10,x1=12))
assert next(b for b in w.raw_boxes if b['label']=='Left_hand' and b['orig_frame']==10)['x1']==12
assert [b for b in w.raw_boxes if b['orig_frame']!=10]==[b for b in before if b['orig_frame']!=10]
w._hoi_undo();assert_ids();w._hoi_redo();assert_ids()
# Deleting one hand at one frame must not delete the other side or other frames.
box=next(b for b in w.raw_boxes if b['label']=='Left_hand' and b['orig_frame']==10)
w._on_box_edited(box['id'],dict(box,frame=10,_action='delete'))
assert not any(b['label']=='Left_hand' and b['orig_frame']==10 for b in w.raw_boxes)
assert any(b['label']=='Right_hand' and b['orig_frame']==10 for b in w.raw_boxes)
assert_ids()
for _ in range(5):
 saved=w._build_payload_v2()
 assert saved['tracks']['T_LEFT_HAND']['hand_id']=='H_LEFT'
 assert saved['tracks']['T_RIGHT_HAND']['hand_id']=='H_RIGHT'
 w._load_annotations_v2(copy.deepcopy(saved));assert_ids()
# XML has exactly two stable tracks; exporting twice must not advance a counter.
with tempfile.TemporaryDirectory() as d:
 path=Path(d)/'hands.xml';QFileDialog.getSaveFileName=lambda *a,**k:(str(path),'')
 import xml.etree.ElementTree as ET
 for _ in range(2):
  counter=w.box_id_counter;w._save_hands_xml();assert w.box_id_counter==counter
  assert {(t.get('id'),t.get('label')) for t in ET.parse(path).getroot().findall('track')}=={('0','Left_hand'),('1','Right_hand')}
 parsed=w._parse_cvat_xml(str(path))
 assert {b['id'] for b in parsed}=={-1,-2}
# Existing recovery/dense tracking records with per-frame IDs normalize on rebuild.
for i,b in enumerate(w.raw_boxes):b['id']=i+3000
w._rebuild_bboxes_from_raw();assert_ids()
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close()
print('TWO_STABLE_HAND_IDS_MANUAL_EDIT_DELETE_UNDO_JSON_XML_RECOVERY_PASS')
