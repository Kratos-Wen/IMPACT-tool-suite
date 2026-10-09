import os,sys,json,tempfile
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog
app=QApplication([]);errors=[]
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.warning=lambda *a,**k:errors.append(str(a[2])) or QMessageBox.Ok
QMessageBox.critical=lambda *a,**k:errors.append(str(a[2])) or QMessageBox.Ok
from ui.hoi_window import HOIWindow
from core.safe_storage import atomic_json
with tempfile.TemporaryDirectory() as d:
 folder=Path(d);video=folder/'video.mp4';video.touch();target=folder/'reviewed.json'
 w=HOIWindow();w.video_path=str(video);w.player.frame_count=30;w.player.frame_rate=15;w._task_resume_path=str(target)
 w._open_autosave();w._autosave_timer.stop()
 w._register_object_entry(5,'part');w.raw_boxes=[dict(id=5,orig_frame=0,label='part',source='manual',x1=1,y1=2,x2=3,y2=4)]
 w._require_participant_code_for_study_save=lambda:True;w._check_incomplete_hoi=lambda **k:(True,[])
 w._autosave_tick();w._finish_autosave()
 with patch.object(QFileDialog,'getSaveFileName',return_value=(str(folder/'annotations.json'),'')):
  w._save_annotations_json()
 assert not (folder/'annotations.json').exists() and w._hoi_has_unsaved_changes()
 atomic_json(target,{'previous':True})
 with patch.object(QFileDialog,'getSaveFileName',return_value=(str(target),'')),patch('ui.hoi_window.atomic_json',side_effect=OSError('disk full')):
  w._save_annotations_json()
 assert json.loads(target.read_text())=={'previous':True} and w._hoi_has_unsaved_changes()
 assert w._autosave_path.is_file(),'Recovery lost after failed manual save'
 with patch.object(QFileDialog,'getSaveFileName',return_value=(str(target),'')):
  w._save_annotations_json()
 assert json.loads(target.read_text())['schema']=='hoi-annotation' and not w._hoi_has_unsaved_changes()
 assert not w._autosave_path.exists(),'Successful manual save left stale recovery'
 w._stop_autosave();w.deleteLater()
print('SOURCE_OVERWRITE_BLOCK_DISK_FULL_OLD_SAVE_AND_RECOVERY_PRESERVED_PASS')
