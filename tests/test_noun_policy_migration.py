import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.migrate_noun_policy import migrate_document, updated_profile

source=dict(object_library={
    '1':dict(label='screw',category='screw'),
    '2':dict(label='unresolved_object',category='unresolved_object'),
    '3':dict(label='unresolved_object',category='unresolved_object'),
    '4':dict(label='unresolved_object',category='unresolved_object'),
    '5':dict(label='angle_grinder',category='angle_grinder')},
    tracks={'T_OBJ_1':dict(object_id=1,category='screw',boxes=[dict(frame=1,bbox=[1,2,3,4],source='manual_box_add')]),
            'T_OBJ_4':dict(object_id=4,category='unresolved_object',boxes=[dict(frame=2,bbox=[2,3,4,5])]),
            'T_OBJ_5':dict(object_id=5,category='angle_grinder',boxes=[dict(frame=3,bbox=[2,3,4,5])])},
    hoi_events={'left_hand':[dict(event_id='a',verb='tighten',noun_object_id=1,start_frame=1,
                                 contact_onset_frame=1,end_frame=2,interaction={'noun':'screw'}),
                             dict(event_id='b',verb='hold',noun_object_id=3,start_frame=3,end_frame=4)],'right_hand':[]},
    shared_assembly={'states':[dict(frame=1,object_id=5,components=['screw'])],'merges':[]},
    provenance={'old_label':'screw'})
before=copy.deepcopy(source)
result,report=migrate_document(source)
assert source==before
assert result['object_library']['1']['category']=='screw_long'
assert result['object_library']['5']['category']=='assembly'
assert report['removed_unused_ids']==[2]
assert {x['object_id'] for x in report['unresolved_objects']}=={3,4}
assert result['tracks']['T_OBJ_1']['boxes']==source['tracks']['T_OBJ_1']['boxes']
assert result['hoi_events']['left_hand'][0]['noun_object_id']==1
assert result['hoi_events']['left_hand'][0]['interaction']['noun']=='screw_long'
assert result['shared_assembly']['states'][0]['components']==['screw_long']
assert result['provenance']['old_label']=='screw'
assert migrate_document(result)[0]==result
profile=dict(noun_classes=['screw','screw_short','screw_black','angle_grinder','assembly'],
             assembly_components=['screw'],noun_aliases={'screw':'screw','angle_grinder':'angle_grinder'},
             model_component_rules={'B':dict(forbidden_components=['lever'])})
updated=updated_profile(profile)
assert profile['noun_aliases']['screw']=='screw'
assert 'screw' not in updated['noun_classes'] and 'angle_grinder' not in updated['noun_classes']
assert updated['noun_aliases']['screw']=='screw_long'
assert updated['model_component_rules']['A']['forbidden_components']==['screw_black']
assert updated['model_component_rules']['B']['forbidden_components']==['lever','screw_short']
assert updated_profile(updated)==updated
print('NOUN_POLICY_IDS_GEOMETRY_HISTORY_UNUSED_CLEANUP_AND_IDEMPOTENCE_PASS')
