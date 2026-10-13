"""Exactly bound optional hand candidates; existing editor evidence takes precedence."""
import math
import xml.etree.ElementTree as ET
from core.hand_identity import hand_box_id,hand_entity_id

def read_bound_hands(path,trial,frame_count,fps):
    try:root=ET.parse(path).getroot()
    except ET.ParseError as ex:raise ValueError('Invalid hand XML') from ex
    if root.get('schema')!='impact-hand-candidates-1' or root.get('video_id')!=str(trial):
        raise ValueError('Hand file belongs to another video or lacks exact binding metadata')
    asset_fps=float(root.get('fps','-1'))
    if int(root.get('frame_count','-1'))!=frame_count or not math.isfinite(asset_fps) or abs(asset_fps-fps)>.01:
        raise ValueError('Hand file timing differs from the loaded video')
    rows=[];seen=set()
    for track in root.findall('track'):
        label=track.get('label')
        if hand_box_id(label) is None:raise ValueError('Hand asset contains an unsupported entity')
        label='Left_hand' if hand_box_id(label)==-1 else 'Right_hand'
        for node in track.findall('box'):
            if node.get('outside','0')=='1':continue
            frame=int(node.get('frame','-1'))
            coords=[float(node.get(key,'nan')) for key in ('xtl','ytl','xbr','ybr')]
            if not 0<=frame<frame_count or not all(math.isfinite(v) for v in coords) or not (0<=coords[0]<coords[2] and 0<=coords[1]<coords[3]):
                raise ValueError('Hand asset has invalid frames or geometry')
            if (label,frame) in seen:raise ValueError('Hand asset has duplicate anatomical boxes')
            seen.add((label,frame))
            rows.append(dict(id=hand_box_id(label),entity_id=hand_entity_id(label),label=label,orig_frame=frame,
                             source='hand_sidecar_candidate',human_verified=False,locked=False,
                             **dict(zip(('x1','y1','x2','y2'),coords))))
    return rows

def merge_hand_candidates(existing,candidates,*,suppressed=(),reviews=None,offset=0):
    keys={(hand_box_id(row.get('label')),row.get('orig_frame')) for row in existing if hand_box_id(row.get('label')) is not None}
    deleted=set(suppressed);reviews=reviews or {};result=list(existing)
    for row in candidates:
        label=row['label'];frame=row['orig_frame'];display=frame+offset
        if ((row['id'],frame) in keys or f'{label}:{display}' in deleted or
                reviews.get(str(display),{}).get('H:'+label,{}).get('state')=='not_visible'):
            continue
        result.append(dict(row));keys.add((row['id'],frame))
    return result
