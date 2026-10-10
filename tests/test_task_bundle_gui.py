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
 w.video_path=str(r/'video.mp4')
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
 w.player.current_frame=0;w.start_offset=0
 w.raw_boxes=[dict(id=0,orig_frame=0,label='Left_hand',x1=1,y1=2,x2=10,y2=12),dict(id=0,orig_frame=0,label='part',x1=15,y1=15,x2=40,y2=45)]
 w._selected_edit_box=w.raw_boxes[1]
 w._selected_hand_data=lambda:dict(interaction_start=0,interaction_end=2,noun_object_id=0)
 w.selected_event_id=1;w.selected_hand_label='Left_hand'
 w.event_draft={'Left_hand':dict(verb='hold',interaction_start=0,interaction_end=2,noun_object_id=0)}
 w._choose_correction_batch=lambda targets:([t for t in targets if t['entity']=='O:0'],0,1)
 w._start_next_tracking_job=lambda:None
 w._start_correction_propagation()
 assert len(w._tracking_jobs)==1
 first=next(iter(w._tracking_jobs.values()));request=first['request']
 assert request['requests'][0]['id']==0
 assert request['requests'][0]['bbox']==[15,15,40,45]
 assert json.loads((Path(first['directory'])/'request.json').read_text())==request
 first['status']='failed'
 old_boxes=[dict(b) for b in w.raw_boxes]
 w._start_correction_propagation()
 second=next(j for j in w._tracking_jobs.values() if j['id']!=first['id'])
 branch=second['request']['requests'][0]
 (Path(second['directory'])/'result.json').write_text(json.dumps(dict(complete=True,request_key=second['request'],results=[dict(job_index=0,end=branch['end'],boxes=[dict(frame=branch['end'],x1=16,y1=15,x2=41,y2=45)],empty_frames=[])])))
 w.events.append(dict(event_id=99,frames=[0,2],hoi_data={}))
 w._merge_tracking_job(second)
 import time
 until=time.monotonic()+5
 while second['status']=='saving':
  assert time.monotonic()<until
  app.processEvents();time.sleep(.002)
 assert second['status']=='applied','An unrelated event must not invalidate tracking'
 assert next(b for b in w.raw_boxes if b['label']=='Left_hand')==old_boxes[0]
 assert any(b['id']==0 and b['label']=='part' and b['orig_frame']==1 for b in w.raw_boxes)
 w._mark_hoi_saved();w.close()
 print('TASK_GUI_AUTOLOAD_TIMELINE_IDENTITY_AND_CORRECTION_NAMESPACE_PASS')
