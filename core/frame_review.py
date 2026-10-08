"""Geometry-bound human review: model output never counts as human evidence."""
import hashlib,json,math

def signature(boxes):
    rows=sorted(tuple(float(b[k]) for k in ('x1','y1','x2','y2')) for b in boxes)
    if any(not all(math.isfinite(v) for v in r) or r[2]<=r[0] or r[3]<=r[1] for r in rows):raise ValueError('Invalid box geometry')
    return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest()

def valid_record(record,boxes):
    try:
        return bool(record and ((record.get('state')=='verified' and len(boxes)==1 and record.get('signature')==signature(boxes)) or
                    (record.get('state')=='not_visible' and not boxes)))
    except (ValueError,TypeError,KeyError):return False

def missing_frames(start,end,onset,reviewed,fps,rate=2,extra=()):
    if start>end or fps<=0 or rate<=0:raise ValueError('Invalid interval or rate')
    step=max(1,int(math.floor(fps/rate)))
    valid=sorted({f for f in reviewed if start<=f<=end})
    required={start,end,*[f for f in extra if start<=f<=end]}
    if onset is not None and start<=onset<=end:required.add(onset)
    missing=required-set(valid)
    anchors=sorted({start,end,*valid})
    for a,b in zip(anchors,anchors[1:]):
        for f in range(a+step,b,step):missing.add(f)
    return sorted(missing)

def snap_edge(frame,start,end,x,start_x,end_x,pixels=3,frames=1):
    frame=max(start,min(frame,end))
    eligible=[edge for edge,pos in ((start,start_x),(end,end_x)) if abs(frame-edge)<=frames and abs(x-pos)<=pixels]
    return min(eligible,key=lambda edge:abs(frame-edge)) if eligible else frame
