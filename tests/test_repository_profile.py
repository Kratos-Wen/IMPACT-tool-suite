"""Installed defaults win over stale Shared paths, including missing assets."""
import os,sys,json,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['IMPACT_USE_BUNDLED_PROFILE']='1'
os.environ['IMPACT_RESTORE_LAST_PROJECT']='0'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.project_profile import PROFILE,REPOSITORY_PROFILE,resolve_profile_path,activate_project_profile
expected=json.loads(REPOSITORY_PROFILE.read_text(encoding='utf-8-sig'))
assert PROFILE==expected
assert resolve_profile_path('/missing/Shared/project_profile.json')==REPOSITORY_PROFILE
assert len(PROFILE['anomaly_labels'])==6
specs={s['id']:s for s in PROFILE['assembly_interfaces']}
assert specs['bearing_plate--gearbox_housing']['fastener_by_model']=={'A':'screw_short','B':'screw_black'}
assert specs['adapter_plate--gearbox_housing']['fastener_class']=='screw_long'
assert specs['adapter_plate--gearbox_housing']['slot_components']==['nut']
assert 'wire' in PROFILE['counted_object_states']
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
app=QApplication([])
for method in ('information','warning','critical'):setattr(QMessageBox,method,lambda *a,**k:QMessageBox.Ok)
from ui.hoi_window import HOIWindow
import cv2,numpy as np
w=HOIWindow();w._autosave_timer.stop()
assert w.act_load_project_profile.text()=='Reload project configuration'
with tempfile.TemporaryDirectory(prefix='repo profile 中文 ') as temp:
 root=Path(temp);video=root/'video.mp4'
 writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(64,48));assert writer.isOpened()
 for _ in range(10):writer.write(np.full((48,64,3),100,np.uint8))
 writer.release();w.player.cap=cv2.VideoCapture(str(video));assert w.player.cap.isOpened()
 w.player.frame_count=10;w.player.frame_rate=15;w.video_path=str(video)
 annotation={'schema':'hoi-annotation','video_id':'1234567_r0_a_a','video_path':'video.mp4','frame_count':10,'fps':15,'object_library':{},'tracks':{},'hoi_events':{'left_hand':[],'right_hand':[]}}
 (root/'annotations.json').write_text(json.dumps(annotation))
 (root/'task.json').write_text(json.dumps(dict(schema='IMPACT-TASK-1',trial_id=annotation['video_id'],video='video.mp4',bundle_root='.',project_profile='Shared/project_profile.json',annotations='annotations.json',status='ready',frame_count=10,fps=15)))
 assert w._auto_load_task_bundle(str(video)) # Missing Shared must not block loading.
 assert PROFILE==expected
 shared=root/'Shared';shared.mkdir();stale=shared/'project_profile.json';stale.write_text('not even valid JSON')
 assert w._auto_load_task_bundle(str(video))
 assert PROFILE==expected
 assert w._load_project_profile(str(stale)) and PROFILE==expected
 assert os.environ['IMPACT_PROJECT_PROFILE']==str(REPOSITORY_PROFILE)
 # Old launch scripts with missing profile paths also use the repository copy.
 import launch
 captured=[];oldrun=launch.runpy.run_path;oldargv=sys.argv[:];oldcwd=os.getcwd()
 launch.runpy.run_path=lambda *a,**k:captured.append(os.environ['IMPACT_PROJECT_PROFILE'])
 try:
  sys.argv=['launch.py','--project-profile',str(root/'missing.json')];launch.main()
 finally:launch.runpy.run_path=oldrun;sys.argv=oldargv;os.chdir(oldcwd)
 assert captured==[str(REPOSITORY_PROFILE)]
 w.player.release_media();w._stop_autosave()
 # External fixtures/custom projects remain an explicit opt-out, not a fallback.
 external=root/'custom.json';external.write_text(json.dumps({'noun_aliases':{'old':'new'}}))
 os.environ['IMPACT_USE_BUNDLED_PROFILE']='0';before=id(PROFILE);activate_project_profile(external)
 assert id(PROFILE)==before and PROFILE['noun_aliases']=={'old':'new'}
 os.environ['IMPACT_USE_BUNDLED_PROFILE']='1';activate_project_profile(external)
 assert PROFILE==expected
w.deleteLater()
print('REPOSITORY_PROFILE_DEFAULT_STALE_MISSING_TASK_LAUNCH_RELOAD_AND_EXTERNAL_OPT_OUT_PASS')
