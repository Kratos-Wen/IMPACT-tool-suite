"""Bounded, single-instance correction propagation. No implicit acceptance."""
import copy

def object_instance(box, uid):
    label=str(box.get("label", "")).strip().lower().replace("-", "_").replace(" ", "_")
    return box.get("id")==uid and label not in {"left_hand", "right_hand", "hand_left", "hand_right", "lefthand", "righthand", "left", "right", "hand"}

def protected(box):
    return bool(box.get("locked") or box.get("human_verified") or str(box.get("source", "")).startswith(("manual", "human")) or box.get("source") == "materialized_handtrack")

def plan(raw_boxes, anchor, start, end, offset=0):
    uid=anchor.get("id")
    if uid is None: raise ValueError("Select a box with an instance ID.")
    if end == start: raise ValueError("Choose an interval beyond the corrected frame.")
    direction=1 if end>start else -1
    stop=end
    for box in raw_boxes:
        frame=int(box.get("orig_frame", -1))+offset
        if object_instance(box, uid) and 0 < (frame-start)*direction <= (stop-start)*direction and protected(box): stop=frame-direction
    if (stop-start)*direction <= 0: raise ValueError("The next frame is already a human anchor.")
    return {"id":uid,"label":anchor.get("label", ""),"start":start,"end":stop,"direction":direction,"offset":offset,
            "bbox":[float(anchor[k]) for k in ("x1","y1","x2","y2")], "class_id":anchor.get("class_id")}

def apply(raw_boxes, request, results):
    uid=request["id"]; start=request["start"]; end=request["end"]; offset=request["offset"]
    direction=1 if end>start else -1
    by_frame={}
    for result in results:
        frame=int(result["frame"])
        if not 0 < (frame-start)*direction <= (end-start)*direction: raise ValueError("Prediction outside requested interval")
        coords=[float(result[k]) for k in ("x1","y1","x2","y2")]
        if not (coords[2]>coords[0] and coords[3]>coords[1]): raise ValueError("Invalid predicted box")
        if frame in by_frame: raise ValueError("Duplicate predicted frame")
        by_frame[frame]=dict(result)
    # Empty masks have no box: remove old automatic boxes rather than carry them through occlusion.
    kept=[]
    for box in raw_boxes:
        frame=int(box.get("orig_frame", -1))+offset
        if object_instance(box, uid) and 0 < (frame-start)*direction <= (end-start)*direction:
            if protected(box): raise ValueError("A human anchor changed while tracking; rerun.")
            continue
        kept.append(copy.deepcopy(box))
    for frame,row in sorted(by_frame.items()):
        new={"id":uid,"label":request["label"],"orig_frame":frame-offset,"source":"sam2_correction_proposal","human_verified":False,"locked":False}
        if request.get("class_id") is not None: new["class_id"]=request["class_id"]
        new.update({k:float(row[k]) for k in ("x1","y1","x2","y2")})
        kept.append(new)
    return kept
