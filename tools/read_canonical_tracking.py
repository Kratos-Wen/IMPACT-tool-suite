"""Read one prediction chunk through a video-specific canonical identity registry."""
import argparse,gzip,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.singleton_instances import normalize_tracking_frame

def read_chunk(path,registry):
    with gzip.open(path,'rt',encoding='utf-8') as stream:chunk=json.load(stream)
    if chunk.get('trial_id',chunk.get('trial')) not in (None,registry['trial']):
        raise ValueError('Registry belongs to another video')
    chunk['frames']=[normalize_tracking_frame(frame,registry) for frame in chunk['frames']]
    chunk['identity_registry_version']=registry['version']
    chunk['canonical_global_namespace']=registry['trial']+'::'
    instances={};mask_aliases={}
    for source in chunk.get('instances',[]):
        alias=registry['aliases'].get(source.get('track_uid'))
        if alias:
            mask_aliases[source['track_id']]=alias
            item=dict(source,track_uid=alias['track_uid'],track_id=alias['object_id'],global_instance_uid=alias['global_instance_uid'])
            target=instances.setdefault(alias['track_uid'],item)
            target.setdefault('source_track_uids',[]).append(source['track_uid'])
        else:instances[source['track_uid']]=source
    chunk['instances']=list(instances.values())
    # Masks must use the same canonical identity as the frame boxes.
    active={frame['frame']:{track['track_uid'] for track in frame.get('tracks',[])} for frame in chunk['frames']}
    masks=[];groups=defaultdict(list);withheld=[]
    for source in chunk.get('masks',[]):
        alias=mask_aliases.get(source.get('track_id'))
        if alias is None:masks.append(source);continue
        item=dict(source,source_track_id=source['track_id'],track_id=alias['object_id'],
                  track_uid=alias['track_uid'],global_instance_uid=alias['global_instance_uid'])
        if alias['track_uid'] not in active.get(source['frame'],set()):withheld.append(item);continue
        groups[(source['frame'],alias['track_uid'])].append(item)
    for rows in groups.values():
        shapes={json.dumps({key:row.get(key) for key in ('shape','bbox_xyxy','encoding','data')},sort_keys=True) for row in rows}
        if len(shapes)==1:masks.append(rows[0])
        else:withheld.extend(rows)
    chunk['masks']=masks;chunk['withheld_identity_masks']=withheld
    return chunk

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--chunk',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();registry=json.loads(args.registry.read_text())
    if args.output.resolve()==args.chunk.resolve():raise ValueError('Keep source predictions unchanged')
    result=read_chunk(args.chunk,registry)
    with gzip.open(args.output,'wt',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,allow_nan=False)
