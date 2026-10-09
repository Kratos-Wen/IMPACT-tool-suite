import os,sys,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
app=QApplication([])
for method in ('information','warning','critical'):
 setattr(QMessageBox,method,lambda *a,**k:QMessageBox.Ok)
QMessageBox.question=lambda *a,**k:QMessageBox.No
from ui.hoi_window import HOIWindow
from core.frame_review import signature,valid_record
w=HOIWindow();w.player.frame_count=60;w.player.frame_rate=15
w._register_object_entry(4,'part');w.start_offset=5;w.end_frame=40
w.raw_boxes=[dict(id=4,orig_frame=2,label='part',source='manual',human_verified=True,x1=1,y1=2,x2=10,y2=20),dict(id=0,orig_frame=2,label='Left_hand',source='manual',human_verified=True,x1=2,y1=3,x2=12,y2=22)]
w.frame_review={'7':{'O:4':{'state':'verified','signature':signature([w.raw_boxes[0]])},'H:Left_hand':{'state':'verified','signature':signature([w.raw_boxes[1]])}}}
for iteration in range(10):
 payload=w._build_payload_v2()
 assert payload['tracks']['T_OBJ_4']['boxes'][0]['frame']==7
 assert payload['tracks']['T_LEFT_HAND']['boxes'][0]['frame']==7
 w._load_annotations_v2(payload)
 assert w.start_offset==5 and w.end_frame==40 and w.spin_start_offset.value()==5
 assert all(b['orig_frame']==2 for b in w.raw_boxes)
 assert valid_record(w.frame_review['7']['O:4'],[b for b in w.raw_boxes if b['label']=='part'])
 assert valid_record(w.frame_review['7']['H:Left_hand'],[b for b in w.raw_boxes if b['label']=='Left_hand'])
 w._rebuild_bboxes_from_raw();assert len(w.bboxes[7])==2
w._stop_autosave();w.deleteLater()
print('OFFSET_10_ROUNDTRIPS_ABSOLUTE_FRAMES_AND_VERIFICATION_PRESERVED_PASS')
