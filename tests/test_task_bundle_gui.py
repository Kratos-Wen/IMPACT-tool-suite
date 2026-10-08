import os,sys,json,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
QMessageBox.information=lambda *a,**k: QMessageBox.Ok
QMessageBox.question=lambda *a,**k: QMessageBox.No
QMessageBox.warning=lambda *a,**k: QMessageBox.Ok
app=QApplication([])
from ui.hoi_window import HOIWindow
w=HOIWindow();w.player.frame_count=100;w.player.frame_rate=15
with tempfile.TemporaryDirectory() as d:
 r=Path(d).resolve();(r/'video.mp4').touch();(r/'weights.pt').touch()
 (r/'profile.json').write_text(json.dumps({'noun_aliases':{'old':'new'},'anomaly_labels':['normal']}))
 payload=dict(video_id='trial',frame_count=100,fps=15,tracks={},hoi_events={'left_hand':[],'right_hand':[]})
 (r/'annotations.json').write_text(json.dumps(payload))
 manifest=dict(schema='IMPACT-TASK-1',trial_id='trial',video='video.mp4',status='ready',project_profile='profile.json',annotations='annotations.json',resume_annotations='reviewed.json',sam_checkpoint='weights.pt',frame_count=100,fps=15)
 (r/'task.json').write_text(json.dumps(manifest));assert w._auto_load_task_bundle(str(r/'video.mp4'))
 assert w.current_annotation_path==str(r/'reviewed.json') and os.environ['IMPACT_SAM2_CHECKPOINT']==str(r/'weights.pt')
 (r/'reviewed.json').write_text(json.dumps(dict(payload,resume_marker=True)))
 w._load_annotations_v2=lambda data,annotation_path='': setattr(w,'_test_loaded_payload',data)
 assert w._auto_load_task_bundle(str(r/'video.mp4'))
 assert w._test_loaded_payload['resume_marker'] is True
 manifest['status']='pending';(r/'task.json').write_text(json.dumps(manifest))
 assert w._auto_load_task_bundle(str(r/'video.mp4')) and w._test_loaded_payload['resume_marker'] is True
 (r/'reviewed.json').unlink()
 previous=w._test_loaded_payload
 assert w._auto_load_task_bundle(str(r/'video.mp4')) and w._test_loaded_payload is previous
 w.video_path=str(r/'video.mp4')
 assert w._build_payload_v2()['video_id']=='trial'
 manifest['status']='ready'
 manifest['frame_count']=99;(r/'task.json').write_text(json.dumps(manifest))
 try:w._auto_load_task_bundle(str(r/'video.mp4'));raise AssertionError('mismatch accepted')
 except ValueError:pass
 print('TASK_GUI_AUTOLOAD_AND_TIMELINE_GUARD_PASS')
