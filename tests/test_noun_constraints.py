"""Retired candidates never become classes; uncertain geometry remains recoverable."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.annotation_migration import adapt_annotation
from core.local_instances import localize_instances
profile={'rejected_noun_classes':['retired_candidate'],'noun_classes':['assembly'],'noun_aliases':{'complete_device':'assembly'}}
source={'schema':'hoi-annotation','object_library':{'38':{'label':'complete_device','category':'complete_device'},'39':{'label':'retired_candidate','category':'retired_candidate'}},'tracks':{'T38':{'object_id':38,'category':'complete_device','boxes':[{'frame':1,'bbox':[1,2,3,4]}]},'T39':{'object_id':39,'category':'retired_candidate','boxes':[{'frame':2,'bbox':[2,3,4,5]}]}},'hoi_events':{'left_hand':[{'start_frame':0,'end_frame':4,'verb':'hold','noun_object_id':39,'links':{'target_track_id':'T39'},'interaction':{'target':'retired_candidate'},'anomaly_labels':[]}]}}
a=localize_instances(adapt_annotation(source,profile),profile)
assert a['object_library']['38']['category']=='assembly' and '39' not in a['object_library']
assert a['hoi_events']['left_hand'][0]['noun_object_id'] is None
assert a['hoi_events']['left_hand'][0]['links']['target_track_id'] is None
assert a['provenance']['rejected_noun_candidates']['tracks']['T39']['boxes']==source['tracks']['T39']['boxes']
b=localize_instances(adapt_annotation(a,profile),profile)
assert a['object_library']==b['object_library'] and a['tracks']==b['tracks']
assert source['object_library']['39']['category']=='retired_candidate'
print('RETIRED_NOUN_UNSET_GEOMETRY_ARCHIVED_ALIAS_ID_STABILITY_PASS')
