"""Scoped three-way merge of durable tracking proposals into a live editor."""
import copy
from core.correction_propagation import apply, matches, protected, apply_batch

def scoped_boxes(boxes, request):
    lo,hi=sorted((request['start'],request['end']))
    return [copy.deepcopy(b) for b in boxes if matches(b,request) and lo<=int(b['orig_frame'])+request['offset']<=hi]

def frame_rows(boxes, request, frame):
    return [b for b in boxes if matches(b,request) and int(b['orig_frame'])+request['offset']==frame]

def geometry(rows):
    return sorted((str(b.get('id')),str(b.get('label')),*(float(b[k]) for k in ('x1','y1','x2','y2'))) for b in rows)

def merge_branch(current, baseline, request, result, allowed_end, changed_review_frames=()):
    """Cut at the first edited frame. Seed changes invalidate only this branch."""
    start=request['start'];direction=request['direction'];end=result.get('end')
    # Validate the worker contract against the original snapshot first.
    apply_batch(baseline,[request],[dict(result,job_index=0)])
    expected=set(range(start+direction,end+direction,direction)) if end!=start else set()
    supplied={r['frame'] for r in result.get('boxes',[])}|set(result.get('empty_frames',[]))
    if supplied!=expected:raise ValueError('Tracking result has gaps; retain it without applying')
    original_anchor=frame_rows(baseline,request,start);live_anchor=frame_rows(current,request,start)
    if len(live_anchor)!=1 or geometry(original_anchor)!=geometry(live_anchor):
        return copy.deepcopy(current),None,dict(reason='anchor_changed',frame=start)
    if allowed_end is None or (allowed_end-start)*direction<0:
        return copy.deepcopy(current),None,dict(reason='event_or_identity_changed')
    end=start+direction*min((end-start)*direction,(allowed_end-start)*direction)
    conflict=None
    for frame in range(start+direction,end+direction,direction):
        old=frame_rows(baseline,request,frame);live=frame_rows(current,request,frame)
        if old!=live or frame in changed_review_frames or any(protected(b) for b in live):
            conflict=dict(reason='intermediate_frame_changed',frame=frame)
            end=frame-direction;break
    if end==start:return copy.deepcopy(current),None,conflict
    bounded=dict(request,end=end)
    accepted=dict(result,end=end,boxes=[r for r in result.get('boxes',[]) if 0<(r['frame']-start)*direction<=(end-start)*direction],
        empty_frames=[f for f in result.get('empty_frames',[]) if 0<(f-start)*direction<=(end-start)*direction])
    return apply(current,bounded,accepted['boxes']),(bounded,accepted),conflict

def bindings_valid(old,current):
    """A verb or Onset edit does not alter the physical tracking target."""
    return bool(current and old['event_id']==current['event_id'] and old['actor']==current['actor']
        and old['role']==current['role'] and old['identity']==current['identity']
        and old.get('object_name')==current.get('object_name'))
