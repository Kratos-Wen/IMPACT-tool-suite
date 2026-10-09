import os,sys,json,tempfile,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog
from unittest.mock import patch
app=QApplication([])
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.warning=lambda *a,**k:QMessageBox.Ok
QMessageBox.critical=lambda *a,**k:QMessageBox.Ok
QMessageBox.question=lambda *a,**k:QMessageBox.Yes
from ui.hoi_window import HOIWindow
from core.safe_storage import read_recovery
with tempfile.TemporaryDirectory(prefix='impact 中文 ') as d:
 root=Path(d);video=root/'video.mp4';video.touch()
 w=HOIWindow();w.video_path=str(video);w.player.frame_count=60;w.player.frame_rate=15;w.player.current_frame=7
 w._open_autosave();w._autosave_timer.stop()
 competitor=HOIWindow()
 try:competitor._reserve_autosave(video);raise AssertionError('second writer allowed')
 except ValueError:pass
 competitor._stop_autosave();competitor.deleteLater()
 w.raw_boxes=[dict(id=5,orig_frame=7,label='part',source='manual',human_verified=True,x1=1,y1=2,x2=10,y2=20)]
 w.global_object_map={'part':5};w.id_to_category={'part':'part'};w.object_id_counter=6
 w.event_draft['Left_hand'].update(verb='hold',noun_object_id=5,interaction_start=7,interaction_end=None)
 w.frame_review={'7':{'O:5':{'status':'reviewed'}}}
 # Stress coalescing: at most one writer, edits made while it runs survive the next tick.
 for i in range(100):
  w.raw_boxes[0]['x1']=i;w._checkpoint_tracking()
 w._finish_autosave();w._autosave_tick();w._finish_autosave()
 stored,_=read_recovery(w._autosave_path,w._autosave_identity())
 assert stored['state']['raw_boxes'][0]['x1']==99 and stored['state']['event_draft']['Left_hand']['interaction_end'] is None
 w._stop_autosave();w.deleteLater()
 restored=HOIWindow();restored.video_path=str(video);restored.player.frame_count=60;restored.player.frame_rate=15
 restored._open_autosave();restored._autosave_timer.stop()
 assert restored.raw_boxes[0]['x1']==99 and restored.event_draft['Left_hand']['verb']=='hold'
 assert restored.frame_review=={'7':{'O:5':{'status':'reviewed'}}}
 assert restored.global_object_map=={'part':5} and restored.id_to_category=={'part':'part'}
 restored._stop_autosave();restored.deleteLater()
print('AUTOSAVE_DRAFT_TRACK_REVIEW_RECOVERY_SINGLE_WRITER_STRESS_PASS')

# Actual abrupt process exit: no closeEvent, unlock, or final save runs.
import subprocess
with tempfile.TemporaryDirectory() as d:
 video=Path(d)/'video.mp4';video.touch()
 child_code="""
import os,sys
from PyQt5.QtWidgets import QApplication,QMessageBox
from ui.hoi_window import HOIWindow
app=QApplication([]);QMessageBox.question=lambda *a,**k:QMessageBox.Yes
w=HOIWindow();w.video_path=sys.argv[1];w.player.frame_count=60;w.player.frame_rate=15
w._open_autosave();w._autosave_timer.stop()
w.raw_boxes=[dict(id=8,orig_frame=3,label='part',source='sam_correction',human_verified=False,x1=2,y1=3,x2=9,y2=12)]
w._autosave_tick();w._finish_autosave()
os._exit(23)
"""
 result=subprocess.run([sys.executable,'-c',child_code,str(video)],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=40)
 assert result.returncode==23,result.stderr
 recovered=HOIWindow();recovered.video_path=str(video);recovered.player.frame_count=60;recovered.player.frame_rate=15
 recovered._open_autosave();recovered._autosave_timer.stop()
 assert recovered.raw_boxes[0]['id']==8 and recovered.raw_boxes[0]['source']=='sam_correction'
 recovered._stop_autosave();recovered.deleteLater()
print('ABRUPT_PROCESS_EXIT_STALE_LOCK_AND_TRACK_RECOVERY_PASS')
