"""Exercise interrupted project transitions and recovery with live Qt widgets."""
import copy,json,os,sys,tempfile,shutil
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
import cv2,numpy as np
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
from core.safe_storage import atomic_json,read_recovery
PROFILE.clear();PROFILE.update(noun_classes=['part'],verbs=['hold'],anomaly_labels=[])
refresh_aliases();app=QApplication([]);warnings=[];failures=[]
QMessageBox.warning=lambda *a,**k:(warnings.append(str(a[2])),QMessageBox.Ok)[1]
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.question=lambda *a,**k:QMessageBox.Discard if a[1]=='Save Before Loading Video' else QMessageBox.Yes
from ui.hoi_window import HOIWindow

def check(name,fn):
    try:fn();print(name,'PASS',flush=True)
    except Exception as exc:
        failures.append((name,type(exc).__name__,str(exc)))
        print(name,'FAIL',type(exc).__name__,str(exc),flush=True)

def dispose(w):
    w._stop_autosave();w._mark_hoi_saved();w.close();w.player.release_media()

with tempfile.TemporaryDirectory(prefix='impact failure paths ') as d:
    root=Path(d).resolve();os.environ['IMPACT_HOI_SETTINGS_DIR']=str(root/'settings')
    os.environ.pop('IMPACT_RESTORE_LAST_PROJECT',None)
    video=root/'video.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(64,64))
    assert writer.isOpened()
    for i in range(60):writer.write(np.full((64,64,3),i,dtype=np.uint8))
    writer.release()
    atomic_json(root/'project_profile.json',dict(noun_classes=['part'],verbs=['hold'],anomaly_labels=[]),backup=False)
    atomic_json(root/'annotations.json',dict(video_id='trial',frame_count=60,fps=15,
        object_library={'0':dict(label='part',category='part')},
        tracks={'T_OBJ_0':dict(object_id=0,category='part',boxes=[dict(frame=10,bbox=[10,10,30,30])])},
        hoi_events={'left_hand':[dict(start_frame=5,contact_onset_frame=10,end_frame=30,verb='hold',noun_object_id=0)],'right_hand':[]}),backup=False)
    atomic_json(root/'task.json',dict(schema='IMPACT-TASK-1',trial_id='trial',video='video.mp4',status='ready',
        project_profile='project_profile.json',annotations='annotations.json',resume_annotations='reviewed.json',frame_count=60,fps=15),backup=False)
    def window():
        w=HOIWindow();w._autosave_timer.stop();w._start_handtrack_precompute=lambda *a,**k:False
        w._maybe_warn_full_assist_semantic_unavailable=lambda *a,**k:None
        return w
    w=window();assert w._load_video(str(video))
    event=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
    w._set_selected_event(event['event_id'],'Left_hand');w.player.seek(10)
    w.event_draft['Left_hand']['interaction_end']=29
    def draft_navigation():
        w._jump_to_selected_end()
        assert w.player.current_frame==29,w.player.current_frame
        assert event['hoi_data']['Left_hand']['interaction_end']==30
    check('DRAFT_SOE_NAVIGATION',draft_navigation)
    w.player.seek(10);w._checkpoint_tracking();w._finish_autosave();w._remember_project_session(force=True)
    record=json.loads(w._project_session_path().read_text());record['selected_hand']='Right_hand'
    atomic_json(w._project_session_path(),record,backup=False)
    dispose(w)
    recovered=window()
    def recovery_selection():
        assert recovered._restore_last_project_session()
        assert recovered._autosave_recovered
        assert recovered.selected_hand_label=='Left_hand',recovered.selected_hand_label
        assert recovered.event_draft['Left_hand']['interaction_end']==29
        restored_event=recovered._find_event_by_id(event['event_id'])
        assert restored_event['hoi_data']['Left_hand']['interaction_end']==30, 'Recovery must not commit the saved draft'
    check('RECOVERY_SELECTION_DRAFT_PRECEDENCE',recovery_selection)
    dispose(recovered)
    # A damaged session record must be handled by the same startup warning as missing files.
    malformed=window();atomic_json(malformed._project_session_path(),[],backup=False)
    def malformed_record():
        assert malformed._restore_last_project_session() is False
        assert not malformed.video_path
    check('MALFORMED_SESSION_ROOT',malformed_record);dispose(malformed)
    # Player and editor must both retain the old media on failed opening.
    w=window();assert w._load_video(str(video));w._set_selected_event(event['event_id'],'Left_hand')
    w.player.seek(12);w.raw_boxes[0]['x1']=12;w._push_undo()
    corrupt=root/'broken.mp4';corrupt.write_bytes(b'not a video')
    before=w._snapshot_state();cap=w.player.cap;reserved=w._autosave_reserved;annotation=w.current_annotation_path
    def player_failure():
        assert w.player.load(str(corrupt)) is False
        assert w.player.cap is cap and cap.isOpened()
        assert w.player.current_frame==12
    check('PLAYER_REJECTS_CORRUPT_WITHOUT_RELEASING_CURRENT',player_failure)
    # Restore media only to isolate the editor transition test on an unfixed build.
    if w.player.cap is not cap:
        w.player.load(str(video));w.player.seek(12)
    cap=w.player.cap
    def editor_failure():
        assert not w._load_video(str(corrupt))
        assert w._snapshot_state()==before,'Failed video open changed the current workspace'
        assert w.player.cap is cap and cap.isOpened()
        assert w._autosave_reserved==reserved and w.current_annotation_path==annotation
        assert len(w._hoi_undo_stack)==1
    check('EDITOR_FAILED_VIDEO_RETAINS_WORK_AND_UNDO',editor_failure)
    dispose(w)
    # A failed session-bookmark read is independent of the recovery writer.
    w=window();assert w._load_video(str(video));w.raw_boxes[0]['x1']=13
    original_stat=Path.stat;calls=[0]
    def flaky_stat(path,*args,**kwargs):
        if str(path)==str(video):
            calls[0]+=1
            if calls[0]==2:raise OSError('Simulated transient session stat failure')
        return original_stat(path,*args,**kwargs)
    w._session_saved_at=0
    def independent_autosave():
        with patch.object(Path,'stat',flaky_stat):w._autosave_tick()
        w._finish_autosave()
        document,_=read_recovery(w._autosave_path,w._autosave_identity())
        assert document['state']['raw_boxes'][0]['x1']==13
    check('SESSION_FAILURE_DOES_NOT_BLOCK_RECOVERY_WRITE',independent_autosave)
    # Rejected and accepted switches both use a real second decoder. Cancellation
    # and lock errors keep the old workspace; acceptance flushes the latest draft.
    destination=root/'next';destination.mkdir();next_video=destination/'video.mp4'
    shutil.copyfile(video,next_video)
    before=w._snapshot_state();cap=w.player.cap;reserved=w._autosave_reserved
    def canceled_switch():
        with patch.object(QMessageBox,'question',return_value=QMessageBox.Cancel):
            assert not w._load_video(str(next_video))
        assert w._snapshot_state()==before and w.player.cap is cap and w._autosave_reserved==reserved
    check('CANCELED_VIDEO_SWITCH_RETAINS_CURRENT',canceled_switch)
    original_mkdir=Path.mkdir
    def denied_directory(path,*args,**kwargs):
        if path==destination/'.impact-recovery':raise PermissionError('Simulated read-only destination')
        return original_mkdir(path,*args,**kwargs)
    def denied_switch():
        with patch.object(Path,'mkdir',denied_directory):assert not w._load_video(str(next_video))
        assert w._snapshot_state()==before and w.player.cap is cap and w._autosave_reserved==reserved
    check('DESTINATION_LOCK_FAILURE_RETAINS_CURRENT',denied_switch)
    w._set_selected_event(event['event_id'],'Left_hand')
    w.event_draft['Left_hand']['interaction_end']=28;w.raw_boxes[0]['x1']=17
    recovery_path=w._autosave_path;identity=w._autosave_identity()
    def successful_switch():
        assert w._load_video(str(next_video))
        document,_=read_recovery(recovery_path,identity)
        assert document['state']['event_draft']['Left_hand']['interaction_end']==28
        assert document['state']['raw_boxes'][0]['x1']==17
        assert w.video_path==str(next_video) and w.player.cap is not cap and w.player.cap.isOpened()
        assert not w.raw_boxes and w.current_annotation_path==''
    check('SUCCESSFUL_SWITCH_FLUSHES_FINAL_OLD_DRAFT',successful_switch)
    dispose(w)
if failures:raise AssertionError(failures)
print('EDITOR_FAILURE_PATHS_PASS')
