"""Recovery/undo copies retain nested metadata without sharing live editor rows."""
import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.editor_snapshot import clone_boxes

shared = {'points': [[1, 2]], 'visibility': {'valid': True}}
rows = [dict(id=0, orig_frame=1, label='part', x1=1.0, meta=shared),
        dict(id=2, orig_frame=2, label='Left_hand', meta=shared)]
rows.append(rows[0])
snapshot = clone_boxes(rows)
assert snapshot == copy.deepcopy(rows)
assert snapshot[0] is snapshot[2] and snapshot[0] is not rows[0]
assert snapshot[0]['meta'] is snapshot[1]['meta']
rows[0]['x1'] = 5
shared['points'][0][0] = 99
assert snapshot[0]['x1'] == 1.0 and snapshot[0]['meta']['points'] == [[1, 2]]
cycle = {'id': 0}
cycle['self'] = cycle
cyclic = [cycle]
cycle['rows'] = cyclic
copied = clone_boxes(cyclic)
assert copied[0]['self'] is copied[0] and copied[0]['rows'] is copied
print('EDITOR_SNAPSHOT_NESTED_METADATA_ALIAS_AND_CYCLE_ISOLATION_PASS')
