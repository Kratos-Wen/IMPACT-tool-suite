"""One manual workflow, legacy session migration, imported candidates and editing."""
import copy,json,os,sys,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
import cv2,numpy as np
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
from core.safe_storage import atomic_json
PROFILE.clear();PROFILE.update(noun_classes=['part','tool','assembly'],verbs=['hold','pick_up'],anomaly_labels=[])
refresh_aliases();app=QApplication([])
for name in ('information','warning','critical','question'):
    setattr(QMessageBox,name,lambda *a,**k:QMessageBox.Yes)
from ui.hoi_window import HOIWindow

def unexpected(*a,**k):
    raise AssertionError('Opening a manual project started local automatic inference')

with tempfile.TemporaryDirectory(prefix='Manual workflow ') as directory:
    root=Path(directory).resolve();os.environ['IMPACT_HOI_SETTINGS_DIR']=str(root/'settings')
    os.environ['IMPACT_RESTORE_LAST_PROJECT']='0'
    video=root/'video.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(64,64))
    assert writer.isOpened()
    for i in range(40):writer.write(np.full((64,64,3),i,dtype=np.uint8))
    writer.release()
    profile=dict(noun_classes=['part','tool','assembly'],verbs=['hold','pick_up'],anomaly_labels=[])
    atomic_json(root/'project_profile.json',profile,backup=False)
    payload=dict(schema='hoi-annotation',video_id='fixture',frame_count=40,fps=15,
        object_library={'7':dict(label='part',category='part')},
        tracks={'T_OBJ_7':dict(object_id=7,category='part',boxes=[dict(frame=5,bbox=[10,10,30,30],source='model')])},
        hoi_events={'left_hand':[dict(start_frame=5,contact_onset_frame=5,end_frame=15,verb='hold',noun_object_id=7)],'right_hand':[]})
    atomic_json(root/'annotations.json',payload,backup=False)
    atomic_json(root/'task.json',dict(schema='IMPACT-TASK-1',trial_id='fixture',video='video.mp4',status='ready',
        project_profile='project_profile.json',annotations='annotations.json',resume_annotations='reviewed.json',frame_count=40,fps=15),backup=False)
    w=HOIWindow();w._autosave_timer.stop()
    assert w._experiment_mode_key()=='manual' and w.combo_experiment_mode.count()==1
    assert w.combo_experiment_mode.isHidden()
    w._set_experiment_mode_key('full_assist');w._apply_experiment_mode_ui();w._apply_user_study_ui_profile()
    assert w._experiment_mode_key()=='manual'
    for name in ('act_precompute_hand_tracks','act_load_yolo_model','act_load_semantic_adapter','act_load_videomae_model','act_batch_apply_action_labels'):
        action=getattr(w,name);assert not action.isEnabled() and not action.isVisible(),name
    assert not w.sc_detect.isEnabled()
    assert not w.detect_menu.menuAction().isVisible() and not w.assist_menu.menuAction().isVisible()
    assert not w._start_handtrack_precompute(str(video),force=True)
    assert not w._semantic_assist_enabled() and not w._detection_assist_enabled()
    w._semantic_feedback_pending=100;w._semantic_feedback_feature_dim=4
    w._maybe_schedule_semantic_training()
    assert w._semantic_adapter_train_worker is None
    w._start_handtrack_precompute=unexpected
    w._load_yolo_model_from_path=unexpected
    # Loading/restoring a task must retain candidates without starting a model.
    assert w._load_video(str(video))
    state=w._current_annotation_ready_state()
    assert state['mode_ready'] and 'ontology' not in state['required_missing'],state
    event=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
    w._set_selected_event(event['event_id'],'Left_hand');w.player.current_frame=5;w._refresh_boxes_for_frame(5)
    box=next(b for b in w.raw_boxes if b['id']==7)
    assert any(b['id']==7 for b in w.player.edit_boxes)
    w._on_box_edited(7,dict(box,x1=12,frame=5,_action='edit'))
    assert next(b for b in w.raw_boxes if b['id']==7)['x1']==12
    saved=copy.deepcopy(w._build_payload_v2());w._load_annotations_v2(saved)
    assert next(b for b in w.raw_boxes if b['id']==7)['x1']==12
    assert w._timeline_onset_state_for_mode()['status']=='confirmed'
    w._remember_project_session(force=True)
    record=json.loads(w._project_session_path().read_text());record['mode']='full_assist'
    atomic_json(w._project_session_path(),record,backup=False)
    w._stop_autosave();w._mark_hoi_saved();w.close();w.player.release_media()
    os.environ['IMPACT_RESTORE_LAST_PROJECT']='1'
    other=HOIWindow();other._autosave_timer.stop();other._start_handtrack_precompute=unexpected
    assert other._restore_last_project_session()
    assert other._experiment_mode_key()=='manual' and other.events
    other._stop_autosave();other._mark_hoi_saved();other.close();other.player.release_media()
print('MANUAL_ONLY_LEGACY_SESSION_CANDIDATE_EDIT_SAVE_NO_MODEL_PASS')
