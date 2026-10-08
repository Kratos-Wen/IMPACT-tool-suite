import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.frame_review import signature,valid_record,missing_frames,snap_edge
box=dict(x1=1,y1=2,x2=10,y2=12)
record={'state':'verified','signature':signature([box])}
assert valid_record(record,[box])
assert not valid_record(record,[dict(box,x1=2)]) and not valid_record(record,[])
assert not valid_record(record,[box,box])
assert valid_record({'state':'not_visible'},[])
assert not valid_record({'state':'not_visible'},[box])
assert not valid_record(record,[dict(box,x1=float('nan'))])
assert not valid_record(record,[dict(box,x2=0)])
assert missing_frames(0,0,0,[],15)==[0]
assert not missing_frames(0,14,0,[0,7,14],15)
assert 14 in missing_frames(0,14,0,[0,7],15)
assert 3 in missing_frames(0,14,0,[0,7,14],15,extra=[3])
assert snap_edge(1,0,1,1,0,1)==1
assert snap_edge(8,0,100,1,0,10)==8
print('FRAME_REVIEW_INVALIDATION_COVERAGE_AND_FINE_SNAP_PASS')
