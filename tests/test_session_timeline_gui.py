"""Real video reopen, recovery precedence and timeline-only navigation."""
import os,sys,tempfile,json,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtCore import Qt,QPoint,QPointF
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import QApplication,QMessageBox
import cv2,numpy as np
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
PROFILE.clear();PROFILE.update(noun_classes=['part'],verbs=['hold'],anomaly_labels=[])
refresh_aliases();app=QApplication([]);warnings=[]
QMessageBox.warning=lambda *a,**k:(warnings.append(str(a[2])),QMessageBox.Ok)[1]
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.question=lambda *a,**k:QMessageBox.Yes
from ui.hoi_window import HOIWindow
from core.safe_storage import atomic_json
with tempfile.TemporaryDirectory(prefix='IMPACT session space ') as d:
 root=Path(d).resolve();os.environ['IMPACT_HOI_SETTINGS_DIR']=str(root/'settings')
 os.environ.pop('IMPACT_RESTORE_LAST_PROJECT',None)
 profile=root/'project_profile.json';atomic_json(profile,dict(noun_classes=['part'],verbs=['hold'],anomaly_labels=[]),backup=False)
 video=root/'video.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(64,64))
 assert writer.isOpened()
 for i in range(60):writer.write(np.full((64,64,3),i,dtype=np.uint8))
 writer.release()
 payload=dict(video_id='trial',frame_count=60,fps=15,object_library={'0':dict(label='part',category='part')},
 tracks={'T_OBJ_0':dict(object_id=0,category='part',boxes=[dict(frame=10,bbox=[10,10,30,30])])},
 hoi_events={'left_hand':[dict(start_frame=5,contact_onset_frame=10,end_frame=30,verb='hold',noun_object_id=0)],'right_hand':[]})
 atomic_json(root/'annotations.json',payload,backup=False)
 atomic_json(root/'task.json',dict(schema='IMPACT-TASK-1',trial_id='trial',video='video.mp4',
  status='ready',project_profile='project_profile.json',annotations='annotations.json',resume_annotations='reviewed.json',frame_count=60,fps=15),backup=False)
 def window():
  w=HOIWindow();w._autosave_timer.stop();w._start_handtrack_precompute=lambda *a,**k:False
  w._maybe_warn_full_assist_semantic_unavailable=lambda *a,**k:None
  return w
 w=window();assert w._load_video(str(video))
 event=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
 w._set_selected_event(event['event_id'],'Left_hand');w.player.seek(12)
 w.hoi_timeline._set_view(4,20);w._remember_project_session(force=True)
 record=json.loads(w._project_session_path().read_text());assert record['frame']==12
 w._stop_autosave();w._mark_hoi_saved();w.close();w.player.clear()
 # Reopen the task and restore view with no video picker or model inference.
 reopened=window();assert reopened._restore_last_project_session()
 assert reopened.player.current_frame==12 and reopened.hoi_timeline.get_view_start()==4,(reopened.player.current_frame,reopened.hoi_timeline.get_view_start(),reopened._autosave_recovered)
 assert reopened._hand_noun_object_id(reopened._selected_hand_data())==0
 assert reopened.current_annotation_path==str(root/'reviewed.json')
 # Actual wheel events on the timeline never alter events or the video crop.
 timeline=reopened.hoi_timeline;timeline.resize(1000,160);timeline.show();app.processEvents()
 row=timeline.actor_rows['Left_hand'];row.resize(980,48)
 before=copy.deepcopy(reopened.events);crop=(reopened.start_offset,reopened.end_frame)
 timeline._set_view(4,40)
 x=(row.width()+timeline.get_gutter())//2
 anchor_before=timeline.get_view_start()+((x-timeline.get_gutter())/(row.width()-timeline.get_gutter()))*timeline.get_view_span()
 def wheel(modifiers,delta):
  event=QWheelEvent(QPointF(x,20),QPointF(row.mapToGlobal(QPoint(x,20))),QPoint(),QPoint(0,delta),Qt.NoButton,modifiers,Qt.NoScrollPhase,False)
  QApplication.sendEvent(row,event)
 wheel(Qt.ControlModifier,120);assert timeline.get_view_span()==32
 anchor_after=timeline.get_view_start()+((x-timeline.get_gutter())/(row.width()-timeline.get_gutter()))*timeline.get_view_span()
 assert abs(anchor_before-anchor_after)<=1
 old_start=timeline.get_view_start();wheel(Qt.ShiftModifier,-120);assert timeline.get_view_start()>old_start
 for _ in range(80):wheel(Qt.MetaModifier,120);wheel(Qt.ShiftModifier,-120)
 assert 0<=timeline.get_view_start()<=60-timeline.get_view_span()
 wheel(Qt.ControlModifier,-120);assert timeline.get_view_span()>=15
 old_view=(timeline.get_view_start(),timeline.get_view_span());row._dragging=True
 wheel(Qt.ControlModifier,120);assert (timeline.get_view_start(),timeline.get_view_span())==old_view
 row._dragging=False
 assert reopened.events==before and (reopened.start_offset,reopened.end_frame)==crop
 # Keyframes are SOE and explicit required frames; no event boundaries change.
 reopened.event_draft['Left_hand']['required_review_frames']=[17]
 reopened.player.seek(12);reopened._jump_event_keyframe(1);assert reopened.player.current_frame==17
 reopened._jump_event_keyframe(-1);assert reopened.player.current_frame==10
 for sid,frame in [('hoi.jump_start',5),('hoi.jump_onset',10),('hoi.jump_end',30)]:
  reopened._frame_review_actions[sid][0].trigger();assert reopened.player.current_frame==frame
 assert reopened.events==before
 # A newer recovery snapshot is authoritative over the older remembered crop/view.
 reopened.player.seek(10)
 anchor=next(b for b in reopened.raw_boxes if b['id']==0 and b['orig_frame']==10)
 reopened._selected_edit_box=copy.deepcopy(anchor)
 reopened._on_box_edited(0,dict(anchor,frame=10,x1=11))
 reopened._checkpoint_tracking();reopened._finish_autosave()
 reopened._remember_project_session(force=True)
 remembered=json.loads(reopened._project_session_path().read_text())
 remembered.update(start=0,end=59,frame=2)
 atomic_json(reopened._project_session_path(),remembered,backup=False)
 reopened._stop_autosave();reopened._mark_hoi_saved();reopened.close();reopened.player.clear()
 recovered=window();assert recovered._restore_last_project_session()
 assert recovered._autosave_recovered
 assert next(b for b in recovered.raw_boxes if b['id']==0 and b['orig_frame']==10)['x1']==11
 assert recovered.player.current_frame==10
 recovered._stop_autosave();recovered._mark_hoi_saved();recovered.close();recovered.player.clear()
 # Missing files leave a blank editor and retain the saved session for relocation.
 video.rename(root/'moved.mp4');missing=window();saved_session=missing._project_session_path().read_bytes()
 assert not missing._restore_last_project_session() and not missing.video_path
 assert missing._project_session_path().read_bytes()==saved_session
 missing._stop_autosave();missing._mark_hoi_saved();missing.close()
 assert len(warnings)==1 and 'previous video' in warnings[0].lower(),warnings
 # The actual application host schedules restoration and flushes recovery on close.
 (root/'moved.mp4').rename(video)
 record=json.loads((root/'settings/last_project.json').read_text());record['mode']='manual'
 atomic_json(root/'settings/last_project.json',record,backup=False)
 from ui.main_window import MainWindow
 host=MainWindow();app.processEvents()
 assert host.hoi_window.video_path==str(video) and host.hoi_window._manual_mode_enabled()
 # A failed recovery flush rejects close and permits a successful retry.
 original_stop=host.hoi_window._stop_autosave
 def failed_stop():raise OSError('Simulated recovery write failure')
 host.hoi_window._stop_autosave=failed_stop
 host.hoi_window._mark_hoi_saved();host.close()
 assert not host.hoi_window._close_request_approved and not getattr(host.hoi_window,'_autosave_stopped',False)
 assert warnings[-1]=='Simulated recovery write failure'
 host.hoi_window._stop_autosave=original_stop
 host.hoi_window._mark_hoi_saved();host.close();app.processEvents()
 assert host.hoi_window._autosave_stopped
 host.hoi_window.player.clear()
print('SESSION_TASK_RESUME_RECOVERY_MISSING_FILE_TIMELINE_WHEEL_KEYFRAME_PASS')
