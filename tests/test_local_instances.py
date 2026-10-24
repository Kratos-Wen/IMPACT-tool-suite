"""Import aliases preserve geometry, distinct screws, hand IDs and assembly merges."""
import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.local_instances import localize_instances
from core.assembly_timeline import validate_timeline,resolve_object
from core.frame_review import signature

profile=dict(noun_classes=['part','attachment','grip','screw','torx_screwdriver_15','torx_screwdriver_20','assembly'],singleton_noun_classes=['grip'])
library={str(uid):dict(label=name,category=category) for uid,name,category in (
    (1002,'grip_1','grip'),(99999,'grip_2','grip'),(2000,'screw_1','screw'),(2001,'screw_2','screw'),
    (1019,'torx_screwdriver_15_1','torx_screwdriver_15'),(1020,'torx_screwdriver_20_1','torx_screwdriver_20'),
    (3000,'assembly_1','assembly'),(3001,'assembly_2','assembly'))}
tracks={f'T_OBJ_{uid}':dict(object_id=int(uid),category=entry['category'],boxes=[dict(frame=2,bbox=[int(uid)%5,2,10,20],source='manual')]) for uid,entry in library.items()}
tracks['T_LEFT_HAND']=dict(object_id=1002,category='Left_hand',boxes=[dict(frame=2,bbox=[5,5,15,25])])
event=dict(noun_object_id=99999,instrument_object_id=1019,start_frame=2,end_frame=20,links={'target_track_id':'T_OBJ_99999'},
    annotation_state={'field_suggestions':{'noun_object_id':{'value':99999}}},migration_review={'source_object_id':99999})
source=dict(object_library=library,tracks=tracks,hoi_events={'left_hand':[event]},frame_review={'2':{'O:2000':{'state':'not_visible'},'H:Left_hand':{'state':'not_visible'}}},
    assembly_default_reference=3000,shared_assembly={'schema':'shared-assembly-2','states':[
        dict(frame=0,object_id=3000,components=['part','attachment']),dict(frame=0,object_id=3001,components=['grip','spring']),
        dict(frame=30,object_id=3000,components=['part','attachment','grip','spring'])],
        'merges':[dict(frame=30,source_object_id=3001,target_object_id=3000)]},
    provenance={'raw_response':{'noun_object_id':99999,'track_id':'T_OBJ_99999'}})
original=copy.deepcopy(source);out=localize_instances(source,profile);assert source==original
aliases=out['provenance']['video_local_instances']['original_to_local']
assert aliases['1002']==aliases['99999'] and aliases['2000']!=aliases['2001']
assert aliases['3000']!=aliases['3001'] and max(map(int,out['object_library']))<100
assert out['object_library'][str(aliases['2000'])]['label']==out['object_library'][str(aliases['2001'])]['label']=='screw'
assert out['object_library'][str(aliases['1019'])]['label']=='torx_screwdriver_15'
for old,track in source['tracks'].items():
    key=out['provenance']['video_local_instances']['source_track_aliases'].get(old,old)
    for box in track['boxes']:assert box in out['tracks'][key]['boxes']
assert out['tracks']['T_LEFT_HAND']==tracks['T_LEFT_HAND']
assert out['provenance']['raw_response']==source['provenance']['raw_response']
row=out['hoi_events']['left_hand'][0]
assert row['noun_object_id']==row['annotation_state']['field_suggestions']['noun_object_id']['value']==aliases['1002']
assert row['links']['target_track_id']=='T_OBJ_'+str(aliases['1002'])
assert row['migration_review']['source_object_id']==99999
assert 'O:'+str(aliases['2000']) in out['frame_review']['2'] and 'H:Left_hand' in out['frame_review']['2']
timeline=validate_timeline(out['shared_assembly']);assert timeline['merges'][0]['source_object_id']==aliases['3001']
assert out['assembly_default_reference']==aliases['3000']
assert resolve_object(dict(shared_assembly_ref=True,shared_assembly_id=aliases['3001']),timeline,35)==aliases['3000']
for _ in range(20):assert localize_instances(out,profile)==out
# Deleting a different instance and reopening must retain every surviving ID.
del out['object_library'][str(aliases['2001'])]
assert localize_instances(out,profile)['object_library']==out['object_library']
# Physical singleton constraints must not merge independent composition histories.
assembly_source=copy.deepcopy(source)
for uid in ('3000','3001'):assembly_source['object_library'][uid]['category']='grip'
independent=localize_instances(assembly_source,profile)
mapping=independent['provenance']['video_local_instances']['original_to_local']
assert mapping['3000']!=mapping['3001']!=mapping['1002']
print('VIDEO_LOCAL_IDS_SINGLETON_ALIASES_DISTINCT_SCREWS_NUMERIC_TOOLS_HAND_NAMESPACES_MERGES_STABLE_REOPEN_PASS')

# Hand detector fragments are a separate namespace, not numbered Object instances.
hands_source=copy.deepcopy(source)
hands_source['object_library'].update({str(uid):dict(label='Left_hand',category='Left_hand') for uid in range(50,95)})
hands_out=localize_instances(hands_source,{**profile,'compact_import_ids':True})
assert len(hands_out['object_library'])==len(independent['object_library'])
assert not any(info['category'] in ('hand','left_hand','right_hand') for info in hands_out['object_library'].values())
assert set(hands_out['unassigned_hand_candidates']['object_library'])=={str(uid) for uid in range(50,95)}
assert hands_out['tracks']['T_LEFT_HAND']==source['tracks']['T_LEFT_HAND']
assert localize_instances(hands_out,{**profile,'compact_import_ids':True})==hands_out

# Human review packages keep their physical IDs, even when sparse or starting at zero.
reviewed=copy.deepcopy(source)
reviewed['object_library']['0']={'label':'part','category':'part'}
reviewed['hoi_events']['left_hand'][0]['noun_object_id']=0
preserved=localize_instances(reviewed,{**profile,'compact_import_ids':True},preserve_ids=True)
assert set(preserved['object_library'])==set(reviewed['object_library'])
assert preserved['hoi_events']['left_hand'][0]['noun_object_id']==0
assert preserved['provenance']['video_local_instances']['original_to_local']['99999']==99999
assert localize_instances(preserved,profile,preserve_ids=True)==preserved
print('REVIEWED_ZERO_SPARSE_LARGE_IDS_NEVER_RENUMBERED_PASS')
