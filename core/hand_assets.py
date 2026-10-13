"""Exactly bound optional hand candidates; existing editor evidence takes precedence."""
import math
import json
import xml.etree.ElementTree as ET
from core.hand_identity import hand_box_id,hand_entity_id

def read_json_hands(path,frame_count,fps,width,height,trial=None):
    """Read only anatomical hand candidates; never import an Object registry or events."""
    with open(path,encoding='utf-8-sig') as stream:data=json.load(stream)
    if not isinstance(data,dict) or not isinstance(data.get('tracks'),dict):
        raise ValueError('Hand JSON needs a tracks dictionary')
    if int(data.get('frame_count',-1))!=frame_count or not math.isfinite(float(data.get('fps',-1))) or abs(float(data['fps'])-fps)>.01:
        raise ValueError('Hand JSON timing differs from the loaded video')
    if trial and str(data.get('video_id',''))!=str(trial):
        raise ValueError('Hand JSON belongs to another task')
    mode=data.get('bbox_mode','xyxy')
    if mode not in ('xyxy','xywh'):raise ValueError('Unsupported hand JSON bbox mode')
    groups={}
    for track in data['tracks'].values():
        if not isinstance(track,dict):raise ValueError('Invalid JSON hand track')
        label=str(track.get('category') or '').strip().lower().replace(' ','_')
        if hand_box_id(label) is None:continue
        label='Left_hand' if hand_box_id(label)==-1 else 'Right_hand'
        for box in track.get('boxes',[]):
            if not isinstance(box,dict):raise ValueError('Invalid JSON hand box')
            if box.get('outside') or box.get('visible') is False:continue
            frame=box.get('frame');coords=box.get('bbox')
            if type(frame) is not int or not 0<=frame<frame_count or not isinstance(coords,(list,tuple)) or len(coords)!=4:
                raise ValueError('Hand JSON has invalid frames or geometry')
            coords=[float(v) for v in coords]
            if mode=='xywh':coords=[coords[0],coords[1],coords[0]+coords[2],coords[1]+coords[3]]
            if data.get('bbox_normalized'):coords=[coords[0]*width,coords[1]*height,coords[2]*width,coords[3]*height]
            if not all(math.isfinite(v) for v in coords) or not (0<=coords[0]<coords[2] and 0<=coords[1]<coords[3]):
                raise ValueError('Hand JSON has invalid frames or geometry')
            row=dict(id=hand_box_id(label),entity_id=hand_entity_id(label),label=label,orig_frame=frame,
                     source_video_id=str(data.get('video_id') or ''),
                     source='hand_sidecar_candidate',human_verified=False,locked=False,
                     **dict(zip(('x1','y1','x2','y2'),coords)))
            groups.setdefault((label,frame),{})[tuple(coords)]=row
    # Conflicting fragments do not establish which box is the actual hand.
    return [next(iter(rows.values())) for rows in groups.values() if len(rows)==1],sum(len(rows)>1 for rows in groups.values())

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
