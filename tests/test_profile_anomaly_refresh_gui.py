"""Old UI options never survive activation of a new task or restored profile."""
import os,sys,tempfile,json
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox
from core.safe_storage import atomic_json
from ui.hoi_window import HOIWindow
app=QApplication([]);warnings=[]
for name in ('information','question','critical'):setattr(QMessageBox,name,lambda *a,**k:QMessageBox.Yes)
QMessageBox.warning=lambda *a,**k:(warnings.append(str(a[2])),QMessageBox.Ok)[1]
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);os.environ['IMPACT_RESTORE_LAST_PROJECT']='0'
 old=root/'old.json';profile=root/'current.json'
 atomic_json(old,dict(noun_classes=['part'],verbs=['hold'],anomaly_labels=['old_kind','old_other']),backup=False)
 labels=['attribute_'+c for c in 'abcdef']
 atomic_json(profile,dict(noun_classes=['part'],verbs=['hold'],anomaly_labels=labels,anomaly_aliases={'old_kind':'attribute_a'}),backup=False)
 def video(folder):
  folder.mkdir();p=folder/'video.mp4';v=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),15,(64,64));assert v.isOpened()
  for _ in range(30):v.write(np.full((64,64,3),100,np.uint8))
  v.release();return p
 def close(w):
  w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media()
 def assert_current(w):
  for combo in (w.combo_anomaly,w.combo_inline_anomaly):
   assert [combo.itemData(i) for i in range(combo.count())]==labels
   assert str(profile.resolve()) in combo.toolTip()
   combo.setCurrentText('attribute_a');assert str(profile.resolve()) in combo.toolTip()
  assert w._normalize_anomaly_label('old_kind')=='attribute_a'
 for status in ('pending','ready'):
  folder=root/status;p=video(folder);os.environ['IMPACT_HOI_SETTINGS_DIR']=str(root/(status+'-settings'))
  atomic_json(folder/'task.json',dict(schema='IMPACT-TASK-1',trial_id='trial',video='video.mp4',status=status,bundle_root='..',
    project_profile='../current.json',annotations='annotations.json',resume_annotations='reviewed.json',frame_count=30,fps=15),backup=False)
  atomic_json(folder/'annotations.json',dict(schema='hoi-annotation',video_id='trial',frame_count=30,fps=15,object_library={},tracks={},hoi_events={'left_hand':[],'right_hand':[]}),backup=False)
  w=HOIWindow();w._show_quick_start_on_startup=False;assert w._load_project_profile(str(old),checkpoint=False)
  assert w.combo_anomaly.count()==2;assert w._load_video(str(p));w._autosave_timer.stop();assert_current(w);close(w)
 # A standalone project with an external profile restores the UI as well as data.
 folder=root/'standalone';p=video(folder);os.environ['IMPACT_HOI_SETTINGS_DIR']=str(root/'standalone-settings')
 w=HOIWindow();w._show_quick_start_on_startup=False;assert w._load_project_profile(str(profile),checkpoint=False)
 assert w._load_video(str(p));w._autosave_timer.stop();w._remember_project_session(force=True);close(w)
 restored=HOIWindow();restored._show_quick_start_on_startup=False;assert restored._load_project_profile(str(old),checkpoint=False)
 os.environ['IMPACT_RESTORE_LAST_PROJECT']='1'
 assert restored._restore_last_project_session();restored._autosave_timer.stop();assert_current(restored);close(restored)
 assert len(warnings)==1 and 'No event annotations imported' in warnings[0],warnings
print('PENDING_READY_STANDALONE_RESTORE_NEW_PROFILE_OPTIONS_AND_SOURCE_TOOLTIP_PASS')
