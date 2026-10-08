import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.task_assets import resolve_task
from core.project_profile import PROFILE,activate_project_profile
from core.noun_aliases import normalize_noun_aliases
with tempfile.TemporaryDirectory(prefix='bundle 中文 ') as d:
 root=Path(d).resolve(); folder=root/'tasks'/'install'/'trial';folder.mkdir(parents=True); video=folder/'video.mp4';video.touch()
 assert resolve_task(video) is None
 manifest=dict(schema='IMPACT-TASK-1',video='video.mp4',bundle_root='../../..',project_profile='../../../project_profile.json',annotations='annotations.json')
 p=folder/'task.json';p.write_text(json.dumps(manifest));task=resolve_task(video)
 assert task['paths']['project_profile']==root/'project_profile.json'
 manifest['video']='other.mp4';p.write_text(json.dumps(manifest))
 try:resolve_task(video);raise AssertionError('wrong identity accepted')
 except ValueError:pass
 manifest['video']='video.mp4';manifest['annotations']='../../../../outside.json';p.write_text(json.dumps(manifest))
 try:resolve_task(video);raise AssertionError('escape accepted')
 except ValueError:pass
 profile=root/'project_profile.json';profile.write_text(json.dumps({'noun_aliases':{'old':'new'}}))
 before=id(PROFILE);activate_project_profile(profile)
 assert id(PROFILE)==before and normalize_noun_aliases({'noun':'old'})['noun']=='new'
print('TASK_PATH_IDENTITY_AND_DYNAMIC_PROFILE_PASS')
