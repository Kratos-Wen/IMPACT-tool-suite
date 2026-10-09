import json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.safe_storage import atomic_json,content_digest,write_recovery,read_recovery
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'reviewed.json';atomic_json(p,{'revision':1})
 real=os.replace
 def fail(src,dst):
  if Path(dst)==p:raise OSError('simulated interrupted replacement')
  return real(src,dst)
 with patch('core.safe_storage.os.replace',side_effect=fail):
  try:atomic_json(p,{'revision':2});raise AssertionError('failure ignored')
  except OSError:pass
 assert json.loads(p.read_text())['revision']==1
 atomic_json(p,{'revision':2});assert json.loads(Path(str(p)+'.bak').read_text())['revision']==1
 for bad in (float('nan'),object()):
  try:atomic_json(p,{'invalid':bad});raise AssertionError('invalid written')
  except (ValueError,TypeError):pass
  assert json.loads(p.read_text())['revision']==2
 recovery=Path(d)/'recovery.json';identity={'trial':'test','frames':15}
 baseline={'events':[],'raw_boxes':[]};state={'events':[{'draft':True}],'raw_boxes':[{'id':3,'frame':1}],'class_map':{0:'part','1':'tool'}}
 assert write_recovery(recovery,{'state':state,'identity':identity},baseline)
 assert not write_recovery(recovery,{'state':dict(state,current_frame=12),'identity':identity},baseline)
 state2=dict(state,raw_boxes=[{'id':3,'frame':2}]);write_recovery(recovery,{'state':state2,'identity':identity},baseline)
 recovery.write_text('{broken')
 document,source=read_recovery(recovery,identity);assert source.name.endswith('.bak') and document['state']['raw_boxes'][0]['frame']==1
 try:read_recovery(recovery,{'trial':'other','frames':15});raise AssertionError('wrong identity restored')
 except ValueError:pass
 assert content_digest(state)==content_digest(json.loads(json.dumps(state)))
print('ATOMIC_FAILURE_BACKUP_CORRUPTION_IDENTITY_PASS')
