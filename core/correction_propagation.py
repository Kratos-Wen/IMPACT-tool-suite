"""Bounded, single-instance correction propagation. No implicit acceptance."""
import copy,math

def object_instance(box, uid):
    label=str(box.get("label", "")).strip().lower().replace("-", "_").replace(" ", "_")
    return box.get("id")==uid and label not in {"left_hand", "right_hand", "hand_left", "hand_right", "lefthand", "righthand", "left", "right", "hand"}

def protected(box):
    return bool(box.get("locked") or box.get("human_verified") or str(box.get("source", "")).startswith(("manual", "human")) or box.get("source") == "materialized_handtrack")

def matches(box, request):
    if request.get('entity_kind') == 'hand':
        return str(box.get('label','')).casefold()==str(request.get('label','')).casefold()
    return object_instance(box, request['id'])

def plan(raw_boxes, anchor, start, end, offset=0, entity_kind='object'):
    uid=anchor.get("id")
    if uid is None: raise ValueError("Select a box with an instance ID.")
    if end == start: raise ValueError("Choose an interval beyond the corrected frame.")
    direction=1 if end>start else -1
    request={"id":uid,"label":anchor.get("label", ""),"entity_kind":entity_kind}
    stop=end
    for box in raw_boxes:
        frame=int(box.get("orig_frame", -1))+offset
        if matches(box, request) and 0 < (frame-start)*direction <= (stop-start)*direction and protected(box): stop=frame-direction
    if (stop-start)*direction <= 0: raise ValueError("The next frame is already a human anchor.")
    return {"id":uid,"label":anchor.get("label", ""),"start":start,"end":stop,"direction":direction,"offset":offset,"entity_kind":entity_kind,
            "bbox":[float(anchor[k]) for k in ("x1","y1","x2","y2")], "class_id":anchor.get("class_id")}

def apply(raw_boxes, request, results):
    uid=request["id"]; start=request["start"]; end=request["end"]; offset=request["offset"]
    direction=1 if end>start else -1
    by_frame={}
    for result in results:
        frame=int(result["frame"])
        if not 0 < (frame-start)*direction <= (end-start)*direction: raise ValueError("Prediction outside requested interval")
        coords=[float(result[k]) for k in ("x1","y1","x2","y2")]
        if not all(math.isfinite(c) for c in coords) or not (coords[2]>coords[0] and coords[3]>coords[1]): raise ValueError("Invalid predicted box")
        if frame in by_frame: raise ValueError("Duplicate predicted frame")
        by_frame[frame]=dict(result)
    # Empty masks have no box: remove old automatic boxes rather than carry them through occlusion.
    kept=[]
    for box in raw_boxes:
        frame=int(box.get("orig_frame", -1))+offset
        if matches(box, request) and 0 < (frame-start)*direction <= (end-start)*direction:
            if protected(box): raise ValueError("A human anchor changed while tracking; rerun.")
            continue
        kept.append(copy.deepcopy(box))
    for frame,row in sorted(by_frame.items()):
        new={"id":uid,"label":request["label"],"orig_frame":frame-offset,"source":"sam2_correction_proposal","human_verified":False,"locked":False}
        if request.get("class_id") is not None: new["class_id"]=request["class_id"]
        new.update({k:float(row[k]) for k in ("x1","y1","x2","y2")})
        kept.append(new)
    return kept

def apply_batch(raw_boxes, requests, results):
    """Validate every branch before applying one undoable batch transaction."""
    if len(results)!=len(requests):raise ValueError('Incomplete tracking batch')
    indexed={}
    for result in results:
        index=result.get('job_index')
        if type(index) is not int or not 0<=index<len(requests) or index in indexed:
            raise ValueError('Invalid tracking branch identity')
        indexed[index]=result
    updated=copy.deepcopy(raw_boxes);completed=[]
    for index,original in enumerate(requests):
        result=indexed[index];request=dict(original)
        end=result.get('end');direction=request['direction']
        if type(end) is not int or not 0<=(end-request['start'])*direction<=(request['end']-request['start'])*direction:
            raise ValueError('Invalid branch end frame')
        predicted={row['frame'] for row in result.get('boxes',[])}
        empty=result.get('empty_frames',[])
        if len(empty)!=len(set(empty)) or predicted.intersection(empty):raise ValueError('Conflicting frame predictions')
        if any(type(f) is not int or not 0<(f-request['start'])*direction<=(end-request['start'])*direction for f in empty):
            raise ValueError('Empty-mask frame outside completed branch')
        if end==request['start']:
            if result.get('boxes') or empty:raise ValueError('Predictions beyond unchanged anchor')
            continue
        request['end']=end
        updated=apply(updated,request,result.get('boxes',[]))
        completed.append((request,result))
    return updated,completed
