import json,tempfile,importlib.util,ast
from pathlib import Path
root=Path(__file__).resolve().parent.parent
import os,sys,tempfile,json
sys.path.insert(0,str(root))
_profile_tmp=tempfile.TemporaryDirectory()
_profile_path=Path(_profile_tmp.name)/'profile.json'
_profile_path.write_text(json.dumps({'noun_aliases':{'tool_alias':'tool_a','tool_a':'tool_a'},'orientation_required_prefixes':['sample_sideways_']}))
os.environ['IMPACT_PROJECT_PROFILE']=str(_profile_path)

for v in ('',):
 spec=importlib.util.spec_from_file_location('guard',root/v/'utils/ego_orientation_guard.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 with tempfile.TemporaryDirectory() as tmp:
  path=Path(tmp)/'sample_sideways_rgb.mp4'
  assert not m.check_ego_orientation(path)[0]
  Path(str(path)+'.orientation.json').write_text(json.dumps({'pixel_orientation':'upright_ccw90','rotation_ccw_degrees':90}))
  assert m.check_ego_orientation(path)[0]
  assert m.check_ego_orientation(Path(tmp)/'sample_upright_rgb.mp4')[0]
 ast.parse((root/v/'ui/video_player.py').read_text(encoding="utf-8"));print(v,'orientation guard PASS')
