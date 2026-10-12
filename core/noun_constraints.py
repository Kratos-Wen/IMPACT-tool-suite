"""Apply project-configured retired noun policy without inventing a category."""
from copy import deepcopy
import re


def constrain_nouns(data, profile):
    rejected=set(profile.get('rejected_noun_classes',[]))
    if not rejected:return data
    library=data.get('object_library',{});tracks=data.get('tracks',{})
    def category(info):
        return re.sub(r'(?:_\d+)+$','',str(info.get('category') or info.get('label') or '').lower())
    ids={int(uid) for uid,info in library.items() if category(info) in rejected}
    tids={tid for tid,track in tracks.items() if track.get('object_id') in ids or category(track) in rejected}
    ids.update(track['object_id'] for tid,track in tracks.items() if tid in tids and type(track.get('object_id')) is int)
    if ids or tids:
        archive=data.setdefault('provenance',{}).setdefault('rejected_noun_candidates',{})
        archive.setdefault('object_library',{}).update({uid:deepcopy(info) for uid,info in library.items() if int(uid) in ids})
        archive.setdefault('tracks',{}).update({tid:deepcopy(track) for tid,track in tracks.items() if tid in tids})
        for uid in list(library):
            if int(uid) in ids:del library[uid]
        for tid in tids:tracks.pop(tid,None)
        roles={'noun_object_id','target_object_id','instrument_object_id','tool_object_id','shared_assembly_id'}
        def clear(value):
            if isinstance(value,list):return [clear(x) for x in value]
            if not isinstance(value,dict):return value
            out={}
            for k,v in value.items():
                if k in ('provenance','migration_review','source_record'):out[k]=v
                elif k in roles and type(v) is int and v in ids:out[k]=None
                elif k.endswith('_track_id') and isinstance(v,str) and v in tids:out[k]=None
                elif k in ('target','noun','instrument','tool') and isinstance(v,str) and re.sub(r'(?:_\d+)+$','',v.lower()) in rejected:out[k]=''
                else:out[k]=clear(v)
            return out
        data=clear(data)
    for field in ('noun_library','object_classes'):
        if isinstance(data.get(field),list):data[field]=[x for x in data[field] if x not in rejected]
    return data
