import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.object_states import *
profile={'counted_object_states':{'sample':{'classes':['raw','prepared'],'maximum':2,'noun_by_count':{'0':'raw','1':'prepared','2':'prepared'}}}}
data=empty_states()
for f,n in [(10,2),(20,1),(30,0)]:data=put_state(data,dict(frame=f,object_id=4,kind='sample',count=n),profile)
assert state_at(data,4,9) is None
assert [state_at(data,4,f)['count'] for f in [10,19,20,29,30]]==[2,2,1,1,0]
assert noun_at(data,4,30,'prepared',profile)=='raw'
assert noun_at(data,4,20,'raw',profile)=='prepared'
assert noun_at(data,5,20,'prepared',profile)=='prepared'
assert next_instance_id({'raw':4},[{'id':7}],data,2)==8
new=put_state(data,dict(frame=31,object_id=8,parent_object_id=4,kind='sample',count=0),profile)
assert state_at(new,4,31)['count']==0 and state_at(data,8,31) is None
assert validate_states(new,profile,40,{4,8})==new
for changes in [dict(count=True),dict(count=3),dict(frame=-1),dict(object_id=True),dict(parent_object_id=8)]:
    bad=dict(frame=31,object_id=8,kind='sample',count=0);bad.update(changes)
    try:put_state(data,bad,profile)
    except ValueError:pass
    else:raise AssertionError(changes)
from core.local_instances import localize_instances
source={'object_library':{'104':{'label':'raw'},'108':{'label':'raw'}},'tracks':{},'hoi_events':{},
        'object_states':{'schema':SCHEMA,'states':[dict(frame=1,object_id=108,parent_object_id=104,kind='sample',count=0)]}}
localized=localize_instances(source,{})
assert localized['object_states']['states'][0]['object_id']==2
assert localized['object_states']['states'][0]['parent_object_id']==1
assert localize_instances(localized,{})['object_states']==localized['object_states']
print('INSTANCE_COUNT_TIMELINE_ID_STABILITY_LOCALIZATION_VALIDATION_PASS')
