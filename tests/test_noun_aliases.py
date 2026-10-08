import importlib.util,copy,ast
from pathlib import Path
root=Path(__file__).resolve().parent.parent
import os,sys,tempfile,json
sys.path.insert(0,str(root))
_profile_tmp=tempfile.TemporaryDirectory()
_profile_path=Path(_profile_tmp.name)/'profile.json'
_profile_path.write_text(json.dumps({'noun_aliases':{'tool_alias':'tool_a','tool_a':'tool_a'},'orientation_required_prefixes':['sample_sideways_']}))
os.environ['IMPACT_PROJECT_PROFILE']=str(_profile_path)

for v in ('',):
 spec=importlib.util.spec_from_file_location('aliases',root/v/'core/noun_aliases.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 old={'class_map':{'4':'tool_alias','8':'tool_a'},'boxes':[{'id':0,'label':'tool_alias','x1':1,'y1':2,'x2':3,'y2':4}], 'interaction':{'instrument':'tool_alias'},'text':'tool_alias','machine_snapshot':{'noun':'tool_alias'},'object_library':[{'id':81,'label':'tool_alias_2','category':'tool_alias'}],'classes':['tool_alias','tool_a']}
 snapshot=copy.deepcopy(old);new=module.normalize_noun_aliases(old)
 assert old==snapshot and new['boxes'][0]['id']==0 and new['boxes'][0]['x1']==1
 assert new['class_map']=={'4':'tool_a','8':'tool_a'} and new['interaction']['instrument']=='tool_a'
 assert new['text']=='tool_alias' and new['machine_snapshot']==old['machine_snapshot']
 assert new['object_library'][0]['id']==81 and new['object_library'][0]['label']=='tool_a_2'
 assert new['classes']==['tool_a','tool_a'];assert module.normalize_noun_aliases(new)==new
 ast.parse((root/v/'ui/hoi_window.py').read_text());print(v,'aliases PASS')
