"""Plan a bounded frame paste. Never allocate IDs or replace existing evidence."""
import copy
import math

def valid_box(box):
    try:
        values=[float(box[k]) for k in ('x1','y1','x2','y2')]
        return all(math.isfinite(v) for v in values) and values[2]>values[0] and values[3]>values[1]
    except (KeyError,TypeError,ValueError):
        return False

def paste_plan(copied, target_frame, allowed_entities, occupied_entities):
    additions=[];skipped=[]
    for entity,rows in copied.items():
        if entity not in allowed_entities:
            skipped.append((entity,'not_in_current_event'));continue
        if entity in occupied_entities:
            skipped.append((entity,'target_already_has_box'));continue
        if len(rows)!=1 or not valid_box(rows[0]):
            skipped.append((entity,'ambiguous_or_invalid_source'));continue
        source=rows[0];box=copy.deepcopy(source)
        for key in ('human_verified','locked','confidence','automatic','synthetic'):
            box.pop(key,None)
        box.update(orig_frame=target_frame,source='manual_box_copy',locked=False,human_verified=False,
                   copied_from_frame=source.get('_copy_frame'),copied_from_source=source.get('source',''))
        box.pop('_copy_frame',None)
        additions.append((entity,box))
    return additions,skipped
