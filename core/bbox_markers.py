"""Revision-cached frame index and provenance colors for bbox navigation."""
from core.box_import import automatic
from core.frame_review import signature,valid_record

COLORS={'reviewer':'#d97706','manual':'#2563eb','viewed':'#15803d','model':'#94a3b8','conflict':'#d97706'}
PRIORITY={'model':0,'viewed':1,'manual':2,'conflict':3}
CAPTIONS={'reviewer':'Reviewer note: inspect this frame','manual':'Drawn / edited','viewed':'Viewed','model':'Model proposal','conflict':'Resolve duplicate / invalid boxes'}

def manually_annotated(box):
    """Geometry added/edited by a person, rather than viewed machine output."""
    source = str(box.get('source') or '').strip().lower()
    return source.startswith(('manual', 'human', 'hands_xml')) or source == 'loaded_annotation'

def state(boxes,record):
    try:
        if len(boxes)!=1: return 'conflict'
        signature(boxes)
    except (TypeError,ValueError,KeyError):return 'conflict'
    box=boxes[0];source=str(box.get('source','')).lower()
    if source.startswith(('manual','human')) or not automatic(dict(box,human_verified=False,locked=False)):return 'manual'
    if box.get('human_verified') or box.get('locked') or valid_record(record,boxes):return 'viewed'
    return 'model'

class BoxIndex:
    def __init__(self,boxes,normalize_hand,include=None):
        self.by_frame={}
        for box in boxes:
            if include is not None and not include(box):continue
            frame=int(box.get('orig_frame',-1));hand=normalize_hand(box.get('label'))
            entity='H:'+hand if hand else 'O:'+str(box.get('id'))
            self.by_frame.setdefault(frame,{}).setdefault(entity,[]).append(box)
        self.frames=sorted(self.by_frame)
