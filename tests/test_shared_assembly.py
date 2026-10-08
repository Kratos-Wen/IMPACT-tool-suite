import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.assembly_timeline import validate_timeline,put_state,state_at,noun_at,resolve_object,next_change
base={'schema':'shared-assembly-1','states':[]}
a=put_state(base,dict(frame=10,object_id=0,components=['gearbox_housing'],interfaces={}))
b=put_state(a,dict(frame=20,object_id=0,components=['gearbox_housing','bearing_plate'],interfaces={'bearing--housing':{'1':True,'2':True}}))
assert base['states']==[] and noun_at(b,9) is None and noun_at(b,10)=='gearbox_housing' and noun_at(b,20)=='assembly'
left={'shared_assembly_ref':True,'verb':'hold','instrument_object_id':7}
right={'shared_assembly_ref':True,'verb':'tighten','instrument_object_id':8}
before=copy.deepcopy([left,right]);assert resolve_object(left,b,20)==resolve_object(right,b,20)==0
assert [left,right]==before
assert resolve_object({'noun_object_id':9},b,20)==9
assert next_change(b,10)==20 and next_change(b,20) is None
c=put_state(b,dict(frame=20,object_id=0,components=['gearbox_housing'],interfaces={'bearing--housing':{'1':False,'2':True}}))
assert noun_at(c,20)=='gearbox_housing' and len(c['states'])==2 and noun_at(b,20)=='assembly'
assert validate_timeline(json.loads(json.dumps(b)))==b
for state in (dict(frame=-1,object_id=0,components=['gearbox_housing']),dict(frame=1,object_id=0,components=[]),dict(frame=1,object_id=0,components=['assembly']),dict(frame=1,object_id=0,components=['a','a'])):
 try:put_state(base,state)
 except ValueError:pass
 else:raise AssertionError(state)
print('SHARED_TIMELINE_CORE_PASS')
