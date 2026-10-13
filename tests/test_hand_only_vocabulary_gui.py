"""Hand-only sample import must keep usable vocabulary and compact object allocation."""
import os,sys,json,tempfile,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QComboBox,QListWidget,QDialogButtonBox
from PyQt5.QtCore import QTimer,Qt
from core.project_profile import PROFILE
PROFILE.clear()
app=QApplication([])
for name in ('information','warning','critical','question'):
 setattr(QMessageBox,name,lambda *a,**k:QMessageBox.Yes)
from ui.hoi_window import HOIWindow
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=1710;w.player.frame_rate=15
payload=dict(schema='hoi-annotation',video_id='fixture',frame_count=1710,fps=15,object_library={},
 tracks={'T_LEFT_HAND':dict(category='Left_hand',boxes=[dict(frame=f,bbox=[1,2,10,20]) for f in range(1533)]),
 'T_RIGHT_HAND':dict(category='Right_hand',boxes=[dict(frame=f,bbox=[21,2,30,20]) for f in range(1533)])},
 hoi_events={'left_hand':[dict(start_frame=0,contact_onset_frame=0,end_frame=40,verb='')],'right_hand':[]})
w._load_annotations_v2(payload)
assert not w.global_object_map
before=copy.deepcopy(w.raw_boxes)
with tempfile.TemporaryDirectory() as d:
 root=Path(d);(root/'sample').mkdir();profile=root/'project_profile.json'
 # Older profiles omit create_default_noun_instances.
 profile.write_text(json.dumps(dict(noun_classes=['part','attachment','tool','assembly'],verbs=['hold','pick_up'],
 assembly_components=['part','attachment'],anomaly_labels=[])))
 w._auto_load_local_assets_for_video(str(root/'sample'/'video.mp4'))
 assert w.raw_boxes==before,'Loading vocabulary replaced imported hand geometry'
 assert {'part','attachment','tool'} <= set(w.global_object_map)
 assert max(w.global_object_map.values())<10,w.global_object_map
 event=w.events[0];w._set_selected_event(event['event_id'],'Left_hand')
 assert w.combo_inline_noun.isEnabled(),'Noun incorrectly requires a verb first'
 assert w.combo_inline_noun.findData(w.global_object_map['part'])>=0
 assert w.combo_inline_instrument.findData(w.global_object_map['tool'])>=0
 w.player.cap=object();w.player.pause=lambda:None
 def create():
  dialog=app.activeModalWidget();dialog.findChild(QComboBox,'assemblyInstance').setCurrentIndex(0)
  parts=dialog.findChild(QListWidget,'assemblyComponents')
  for i in range(parts.count()):parts.item(i).setCheckState(Qt.Checked)
  dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
 QTimer.singleShot(0,create);w._open_assembly_editor()
 ids=[uid for name,uid in w.global_object_map.items() if name=='assembly']
 assert len(ids)==1 and ids[0]<10,ids
 assert w.combo_inline_noun.findData(ids[0])>=0
 w.player.cap=None
 saved=copy.deepcopy(w._build_payload_v2());w._load_annotations_v2(saved)
 assert ids[0] in w.global_object_map.values()
 # Explicitly loading the same profile keeps all IDs and manually created entities.
 old=copy.deepcopy(w.global_object_map);assert w._load_project_profile(str(profile))
 assert w.global_object_map==old
 w.player.cap=None
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close()
print('HAND_ONLY_PROFILE_CHOICES_NO_VERB_GATE_COMPACT_ASSEMBLY_ID_ROUNDTRIP_PASS')
