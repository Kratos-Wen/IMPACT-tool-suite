"""Open a task with optional hand XML; preserve manual edits and deleted hands."""
import os,sys,tempfile,json,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox
from core.safe_storage import atomic_json
from core.hand_assets import read_bound_hands,merge_hand_candidates
from ui.hoi_window import HOIWindow
app=QApplication([]);warnings=[]
for name in ('information','question','critical'):setattr(QMessageBox,name,lambda *a,**k:QMessageBox.Yes)
QMessageBox.warning=lambda *a,**k:(warnings.append(str(a[2])),QMessageBox.Ok)[1]
with tempfile.TemporaryDirectory() as directory:
 root=Path(directory);os.environ['IMPACT_HOI_SETTINGS_DIR']=str(root/'settings');os.environ['IMPACT_RESTORE_LAST_PROJECT']='0'
 video=root/'video.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(64,64));assert writer.isOpened()
 for _ in range(30):writer.write(np.full((64,64,3),100,np.uint8))
 writer.release()
 atomic_json(root/'project_profile.json',dict(noun_classes=['part'],verbs=['hold'],anomaly_labels=[],automatic_tracking_import='hands'),backup=False)
 atomic_json(root/'annotations.json',dict(schema='hoi-annotation',video_id='trial',frame_count=30,fps=15,
  object_library={},tracks={'T_LEFT_HAND':dict(category='left_hand',boxes=[dict(frame=0,bbox=[1,2,8,9],source='manual_box_add')])},hoi_events={'left_hand':[],'right_hand':[]}),backup=False)
 xml='<annotations schema="impact-hand-candidates-1" video_id="trial" frame_count="30" fps="15"><track id="0" label="Left_hand"><box frame="0" xtl="10" ytl="10" xbr="20" ybr="20"/><box frame="10" xtl="10" ytl="10" xbr="20" ybr="20"/></track></annotations>'
 hands=root/'hands.xml';hands.write_text(xml)
 atomic_json(root/'task.json',dict(schema='IMPACT-TASK-1',trial_id='trial',video='video.mp4',status='ready',project_profile='project_profile.json',annotations='annotations.json',resume_annotations='reviewed.json',hand_tracks='hands.xml',frame_count=30,fps=15),backup=False)
 w=HOIWindow();w._show_quick_start_on_startup=False;assert w._load_video(str(video));w._autosave_timer.stop()
 assert next(b for b in w.raw_boxes if b['orig_frame']==0)['x1']==1
 assert any(b['orig_frame']==10 and b['source']=='hand_sidecar_candidate' for b in w.raw_boxes)
 assert all(b['id']==-1 for b in w.raw_boxes)
 rows=copy.deepcopy(w.raw_boxes);candidates=read_bound_hands(hands,'trial',30,15)
 assert merge_hand_candidates(rows,candidates)==rows,'Reopening duplicated hands'
 original=[b for b in rows if b['orig_frame']==0]
 assert merge_hand_candidates(original,candidates,suppressed=['Left_hand:10'])==original
 assert merge_hand_candidates(original,candidates,reviews={'10':{'H:Left_hand':{'state':'not_visible'}}})==original
 for bad in (xml.replace('video_id="trial"','video_id="another"'),xml.replace('frame_count="30"','frame_count="31"'),xml.replace('fps="15"','fps="nan"'),xml.replace('frame="10"','frame="40"'),'<broken>'):
  hands.write_text(bad)
  try:read_bound_hands(hands,'trial',30,15);raise AssertionError('Invalid hand file accepted')
  except ValueError:pass
 assert not warnings,warnings
 w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media()
 hands.write_text(xml)
 task=json.loads((root/'task.json').read_text());task['status']='pending';task.pop('annotations')
 atomic_json(root/'task.json',task,backup=False)
 w=HOIWindow();w._show_quick_start_on_startup=False;assert w._load_video(str(video));w._autosave_timer.stop()
 assert any(b['orig_frame']==10 and b['source']=='hand_sidecar_candidate' for b in w.raw_boxes),'Pending task did not load bound hand XML'
 assert all(b['id']==-1 for b in w.raw_boxes)
 w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media()
print('TASK_HAND_XML_AUTOMATIC_LOAD_BINDING_HUMAN_PRESERVATION_SUPPRESSION_IDEMPOTENCE_PASS')
