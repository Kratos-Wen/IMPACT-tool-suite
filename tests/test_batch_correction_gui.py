"""Batch direction planning, live dialog, stale results and one-step undo."""
import os,sys,tempfile,json,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtCore import Qt,QTimer,QProcess
from PyQt5.QtWidgets import QApplication,QMessageBox,QListWidget,QSpinBox,QDialogButtonBox
from unittest.mock import patch
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
PROFILE.clear();PROFILE.update(noun_classes=['assembly','tool'],verbs=['hold'],anomaly_labels=[],
 assembly_components=['part','grip'],assembly_interfaces=[])
refresh_aliases();app=QApplication([])
warnings=[]
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.warning=lambda *a,**k:(warnings.append(str(a[2])),QMessageBox.Ok)[1]
QMessageBox.question=lambda *a,**k:QMessageBox.Yes
from ui.hoi_window import HOIWindow
from core.assembly_timeline import empty_timeline,put_state
from core.correction_propagation import apply_batch
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=40;w.player.frame_rate=15
payload=dict(video_id='trial',frame_count=40,fps=15,
 object_library={str(i):dict(label=('assembly' if i!=8 else 'tool')+'_'+str(i),category='assembly' if i!=8 else 'tool') for i in [0,5,8]},
 tracks={f'T_OBJ_{i}':dict(object_id=i,category='assembly' if i!=8 else 'tool',boxes=[dict(frame=f,bbox=[10+i,10,30+i,30],source='model') for f in range(5,16)]) for i in [0,5,8]},
 hoi_events={'left_hand':[dict(start_frame=5,contact_onset_frame=10,end_frame=15,verb='hold',noun_object_id=0,instrument_object_id=8)],
 'right_hand':[dict(start_frame=8,contact_onset_frame=10,end_frame=12,verb='hold',noun_object_id=5)]})
w._load_annotations_v2(payload)
left=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
right=next(e for e in w.events if e['hoi_data']['Right_hand'].get('verb'))
left['hoi_data']['Right_hand']=copy.deepcopy(right['hoi_data']['Right_hand']);w.events.remove(right)
w._set_selected_event(left['event_id'],'Left_hand');w.player.current_frame=10
for hand,uid in [('Left_hand',0),('Right_hand',100)]:
 for frame in range(5,16):w.raw_boxes.append(dict(id=uid,label=hand,orig_frame=frame,x1=1,y1=1,x2=10,y2=10,source='model'))
w.shared_assembly=put_state(empty_timeline(),dict(frame=0,object_id=0,components=['part','grip']))
w.shared_assembly=put_state(w.shared_assembly,dict(frame=12,object_id=0,components=['part','grip','attachment']))
for box in w.raw_boxes:
 if box['orig_frame']==10 or (box['id']==8 and box['orig_frame']==13):box['source']='manual_box_edit'
w._rebuild_bboxes_from_raw()
w._selected_edit_box=next(b for b in w.raw_boxes if b['id']==0 and b['label'].startswith('assembly') and b['orig_frame']==10)
targets=w._correction_targets();assert {t['entity'] for t in targets}=={'H:Left_hand','H:Right_hand','O:0','O:5','O:8'}
def dialog_actions():
 dialog=app.activeModalWidget();assert dialog
 listing=dialog.findChild(QListWidget,'correctionTargets')
 for i in range(listing.count()):
  item=listing.item(i);item.setCheckState(Qt.Checked if item.data(Qt.UserRole)['entity'] in ['H:Left_hand','O:0','O:8'] else Qt.Unchecked)
 dialog.findChild(QSpinBox,'correctionStart').setValue(5)
 dialog.findChild(QSpinBox,'correctionEnd').setValue(15)
 dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok).click()
QTimer.singleShot(0,dialog_actions)
selection=w._choose_correction_batch(targets);chosen,begin,end=selection
requests,notes=w._plan_correction_batch(chosen,begin,end)
assert len(requests)==6
assert next(r for r in requests if r['entity_kind']=='object' and r['id']==0 and r['direction']==1)['end']==11
assert next(r for r in requests if r['entity_kind']=='object' and r['id']==8 and r['direction']==1)['end']==12
assert any(r['entity_kind']=='hand' and r['id']==0 for r in requests)
results=[dict(job_index=i,end=r['end'],boxes=[dict(frame=r['end'],x1=2,y1=2,x2=12,y2=12)],empty_frames=[]) for i,r in enumerate(requests)]
old=copy.deepcopy(w.raw_boxes);updated,completed=apply_batch(old,requests,results)
assert w.raw_boxes==old and len(completed)==6
for box in old:
 if box.get('source')=='manual_box_edit' or box['id'] in [5,100]:assert box in updated
for bad in [results[:-1],[dict(r,job_index=0) for r in results],
 [dict(results[0],end=100)]+results[1:],[dict(results[0],empty_frames=[100])]+results[1:]]:
 try:apply_batch(old,requests,bad)
 except ValueError:pass
 else:raise AssertionError('Invalid batch accepted')
 assert w.raw_boxes==old
with tempfile.TemporaryDirectory() as d:
 checkpoint=Path(d)/'weights.pt';checkpoint.touch();os.environ['IMPACT_SAM2_CHECKPOINT']=str(checkpoint)
 w.video_path=str(Path(d)/'video.mp4');w._choose_correction_batch=lambda targets:selection
 def capture(process,executable,args):
  request=json.loads(Path(args[1]).read_text());assert len(request['requests'])==6
  Path(args[2]).write_text(json.dumps(dict(results=results)))
 with patch.object(QProcess,'start',new=capture):w._start_correction_propagation()
 w._correction_process.finished.emit(0,QProcess.NormalExit)
 assert w.raw_boxes==updated
 w._hoi_undo();assert w.raw_boxes==old
 w._hoi_redo();assert w.raw_boxes==updated
 with patch.object(QProcess,'start',new=capture):w._start_correction_propagation()
 w.event_draft['Left_hand']['verb']='edited'
 w._correction_process.finished.emit(0,QProcess.NormalExit)
 assert w.raw_boxes==updated
 assert not warnings,warnings
w._stop_autosave();w._mark_hoi_saved();w.close()
print('BATCH_DIALOG_BIDIRECTIONAL_NAMESPACE_HUMAN_ANCHOR_ASSEMBLY_UNDO_STALE_PASS')
