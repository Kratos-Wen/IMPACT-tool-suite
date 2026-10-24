from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.annotation_refs import instance_reference
tracks={'T_OBJ_9':9}
for role,field,alias,link in [('object','noun_object_id','target_object_id','target_track_id'),('instrument','instrument_object_id','tool_object_id','tool_track_id')]:
    stale={'links':{link:'T_OBJ_9'},'interaction':{'target':'part','tool':'part'}}
    assert instance_reference({**stale,field:None,alias:9},role,tracks,lambda _:8) is None
    assert instance_reference({**stale,field:0},role,tracks,lambda _:8)==0
    assert instance_reference(stale,role,tracks,lambda _:8)==9
    assert instance_reference({alias:7},role,tracks)==7
print('EXPLICIT_NULL_ZERO_AND_LEGACY_FALLBACK_INSTANCE_REFERENCE_PASS')
