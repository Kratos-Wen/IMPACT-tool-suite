"""Real Qt processes: editable while running, scoped merge, cancel and recovery."""
import copy,json,os,sys,tempfile,time,threading
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtCore import QTimer,QProcess
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
PROFILE.clear();PROFILE.update(noun_classes=['part','tool'],verbs=['hold','align'],anomaly_labels=[],assembly_interfaces=[])
refresh_aliases();app=QApplication([]);warnings=[]
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.warning=lambda *a,**k:(warnings.append(str(a[2])),QMessageBox.Ok)[1]
QMessageBox.question=lambda *a,**k:QMessageBox.Yes
from ui.hoi_window import HOIWindow
from ui.tracking_tasks import write_recovery as real_write_recovery

FAKE='''import json,sys,time,os
from pathlib import Path
def save(path,data):
 tmp=Path(path+'.tmp');tmp.write_text(json.dumps(data));os.replace(tmp,path)
print(json.dumps({'type':'ready'}),flush=True)
for line in sys.stdin:
 m=json.loads(line)
 if m.get('shutdown'):break
 q=json.loads(Path(m['request']).read_text(encoding='utf-8'));results=[];done=0
 total=sum(abs(r['end']-r['start']) for r in q['requests'])
 for i,r in enumerate(q['requests']):results.append(dict(job_index=i,end=r['start'],boxes=[],empty_frames=[]))
 for i,r in enumerate(q['requests']):
  direction=r['direction']
  for f in range(r['start']+direction,r['end']+direction,direction):
   time.sleep(.025);b=r['bbox'];results[i]['end']=f
   results[i]['boxes'].append(dict(frame=f,x1=b[0]+1,y1=b[1],x2=b[2]+1,y2=b[3]));done+=1
   save(m['output'],dict(results=results,complete=False,request_key=q))
   print(json.dumps(dict(type='progress',job_id=m['job_id'],phase='tracking',completed=done,total=total)),flush=True)
 save(m['output'],dict(results=results,complete=True,request_key=q))
 print(json.dumps(dict(type='completed',job_id=m['job_id'])),flush=True)
'''

def wait(predicate,limit=10):
    end=time.monotonic()+limit
    while not predicate():
        assert time.monotonic()<end,'Timed out waiting for Qt tracking state'
        app.processEvents();time.sleep(.002)

with tempfile.TemporaryDirectory(prefix='Tracking space 中文 ') as d:
    root=Path(d);backend=root/'backend 中文.py';backend.write_text(FAKE,encoding='utf-8')
    weights=root/'small.pt';weights.touch();video=root/'video.mp4';video.touch()
    os.environ['IMPACT_SAM2_CHECKPOINT']=str(weights);os.environ['IMPACT_SAM2_PYTHON']=sys.executable
    os.environ['IMPACT_HOI_SETTINGS_DIR']=str(root/'settings')
    payload=dict(video_id='trial',frame_count=40,fps=15,
        object_library={str(i):dict(label='part' if i==0 else 'tool',category='part' if i==0 else 'tool') for i in [0,8]},
        tracks={f'T_OBJ_{i}':dict(object_id=i,category='part' if i==0 else 'tool',boxes=[dict(frame=f,bbox=[10+i,10,30+i,30],source='model') for f in range(5,16)]) for i in [0,8]},
        hoi_events={'left_hand':[dict(start_frame=5,contact_onset_frame=5,end_frame=15,verb='hold',noun_object_id=0,instrument_object_id=8)],'right_hand':[]})
    def window():
        w=HOIWindow();w._autosave_timer.stop();w._load_annotations_v2(copy.deepcopy(payload));w.video_path=str(video);w.player.frame_count=40;w.player.frame_rate=15
        event=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'));w._set_selected_event(event['event_id'],'Left_hand');w.player.current_frame=10
        for b in w.raw_boxes:
            if b['orig_frame']==10:b['source']='manual_box_edit'
        w._rebuild_bboxes_from_raw();w._tracking_worker_path=lambda:str(backend)
        w._choose_correction_batch=lambda targets:([t for t in targets if t['entity']=='O:0'],5,15)
        w._open_autosave('')
        return w
    w=window();heartbeats=[];timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:heartbeats.append(time.monotonic()));timer.start()
    baseline=copy.deepcopy(w.raw_boxes);w._start_correction_propagation();job=w._tracking_jobs[w._tracking_active]
    assert not app.activeModalWidget(),'Tracking must not open a blocking progress dialog'
    tool_requests,notes=w._plan_correction_batch([t for t in w._correction_targets() if t['entity']=='O:8'],5,15)
    queued=w._enqueue_tracking(dict(job['request'],requests=tool_requests))
    assert queued['status']=='queued' and w._tracking_active==job['id']
    w._cancel_tracking_job(queued['id']);assert queued['status']=='cancelled' and w._tracking_active==job['id']
    wait(lambda:job['completed']>=2)
    # Continue annotating another entity and editing the event verb during inference.
    other=next(b for b in w.raw_boxes if b['id']==8 and b['orig_frame']==11);other.update(x1=25,source='manual_box_edit')
    w.event_draft['Left_hand']['verb']='align'
    gate=threading.Event()
    def delayed_save(*args,**kwargs):
        assert gate.wait(5),'Slow disk test did not release save worker'
        return real_write_recovery(*args,**kwargs)
    with patch('ui.tracking_tasks.write_recovery',new=delayed_save):
        wait(lambda:job['status']=='saving');count=len(heartbeats)
        wait(lambda:len(heartbeats)>=count+3)
        gate.set();wait(lambda:job['status']=='applied')
    assert len(heartbeats)>=3
    assert other in w.raw_boxes and w.event_draft['Left_hand']['verb']=='align'
    assert any(b['id']==0 and b['orig_frame']==11 and b['source']=='sam2_correction_proposal' for b in w.raw_boxes)
    applied=copy.deepcopy(w.raw_boxes);w._hoi_undo()
    assert any(b['id']==0 and b['orig_frame']==11 and b['source']=='model' for b in w.raw_boxes)
    assert any(b['id']==8 and b['orig_frame']==11 and b['x1']==25 for b in w.raw_boxes)
    w._hoi_redo();assert w.raw_boxes==applied
    recovery=json.loads(w._autosave_path.read_text(encoding='utf-8'))
    assert any(b['source']=='sam2_correction_proposal' for b in recovery['state']['raw_boxes'])
    service=w._tracking_service;assert service is not None
    # A second task uses the same process. An intermediate manual edit becomes
    # a new anchor; only its continuation reruns, with the same physical ID.
    w._start_correction_propagation();second=w._tracking_jobs[w._tracking_active]
    wait(lambda:second['completed']>=1)
    manual=next(b for b in w.raw_boxes if b['id']==0 and b['orig_frame']==13);manual.update(x1=14,source='manual_box_edit')
    w._bump_bbox_revision()
    wait(lambda:second['status']=='applied_partial' and any(j.get('retry')==1 for j in w._tracking_jobs.values()))
    retry=next(j for j in w._tracking_jobs.values() if j.get('retry')==1);wait(lambda:retry['status']=='applied')
    assert w._tracking_service is service and manual in w.raw_boxes
    assert retry['request']['requests'][0]['start']==13
    assert all(not b.get('human_verified',False) for b in w.raw_boxes if b.get('source')=='sam2_correction_proposal')
    # A seed edit invalidates its old branch and automatically uses the new seed.
    w._start_correction_propagation();third=w._tracking_jobs[w._tracking_active]
    wait(lambda:third['completed']>=1)
    seed=next(b for b in w.raw_boxes if b['id']==0 and b['orig_frame']==10);seed.update(x1=15,source='manual_box_edit');w._bump_bbox_revision()
    wait(lambda:third['status']=='needs_new_anchor')
    wait(lambda:w._tracking_active is None)
    assert seed in w.raw_boxes
    # Cancellation retains on-disk partial candidates and never merges them.
    before=copy.deepcopy(w.raw_boxes);w._start_correction_propagation();cancelled=w._tracking_jobs[w._tracking_active]
    wait(lambda:cancelled['completed']>=1);w._cancel_tracking_job();wait(lambda:w._tracking_active is None)
    assert cancelled['status']=='cancelled' and w.raw_boxes==before
    partial=json.loads((Path(cancelled['directory'])/'result.json').read_text());assert any(r['boxes'] for r in partial['results'])
    # A result stays with its source video even if the user switches while it runs.
    w._start_correction_propagation();switched=w._tracking_jobs[w._tracking_active]
    wait(lambda:switched['completed']>=1);other_video=root/'other video.mp4';other_video.touch()
    w.video_path=str(other_video);untouched=copy.deepcopy(w.raw_boxes)
    wait(lambda:switched['status']=='ready');assert w.raw_boxes==untouched
    w.video_path=str(video);w._tracking_attach_video();wait(lambda:switched['status']=='applied')
    # Closing during propagation is allowed; the durable queue resumes on reopen.
    w._start_correction_propagation();interrupted=w._tracking_jobs[w._tracking_active]
    wait(lambda:interrupted['completed']>=1)
    w._mark_hoi_saved();w.close();assert interrupted['status']=='interrupted';assert w._tracking_service is None
    reopened=window();reopened._tracking_attach_video();wait(lambda:reopened._tracking_active is None and reopened._tracking_jobs[interrupted['id']]['status']!='saving')
    assert reopened._tracking_jobs[interrupted['id']]['status'] in ('applied','applied_partial','needs_new_anchor')
    # A missing interpreter is a visible task failure, not a modal interruption.
    reopened._release_tracking_service();wait(lambda:reopened._tracking_service is None)
    os.environ['IMPACT_SAM2_PYTHON']=str(root/'missing python')
    reopened._start_correction_propagation();failed=reopened._tracking_jobs[reopened._tracking_active]
    wait(lambda:failed['status']=='failed');assert failed['error']
    reopened._mark_hoi_saved();reopened.close();timer.stop()
assert not warnings,warnings
print('REAL_BACKGROUND_QUEUE_PROGRESS_EDIT_MERGE_RESEED_CANCEL_RECOVERY_FAILURE_PASS')
