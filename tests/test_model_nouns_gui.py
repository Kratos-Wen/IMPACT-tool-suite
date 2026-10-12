"""Model compatibility prevents new mistakes without rewriting old evidence."""
import os,sys,copy,json
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
from core.noun_model_rules import noun_allowed
from ui.hoi_window import HOIWindow
PROFILE.clear();PROFILE.update(noun_classes=['fastener_long','fastener_dark','fastener_short'],noun_aliases={'fastener':'fastener_long'},model_component_rules={'A':{'forbidden_components':['fastener_dark']},'B':{'forbidden_components':['fastener_short']}},verbs=['hold'],anomaly_labels=[])
refresh_aliases();app=QApplication([])
messages=[]
for name in ('information','warning','critical','question'):setattr(QMessageBox,name,lambda *a,**k:messages.append(a[1:3]) or QMessageBox.No)
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=20;w.player.frame_rate=15
for model,bad in [('a',2),('b',3)]:
 source={'schema':'hoi-annotation','video_id':f'1234567_r2_{model}_a','frame_count':20,'fps':15,'object_library':{str(i):{'label':name,'category':name} for i,name in enumerate(['fastener','fastener_dark','fastener_short'],1)},'tracks':{'T1':{'object_id':1,'category':'fastener','boxes':[{'frame':0,'bbox':[10,10,30,30],'source':'manual_box_edit'}]}},'hoi_events':{'left_hand':[{'start_frame':0,'contact_onset_frame':0,'end_frame':10,'verb':'hold','noun_object_id':bad}],'right_hand':[]}}
 w._load_annotations_v2(source);w._apply_profile_libraries()
 assert w._object_name_for_id(1)=='fastener_long'
 assert w.combo_target.model().item(w.combo_target.findData(bad)).isEnabled() is False
 event=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
 w._set_selected_event(event['event_id'],'Left_hand');app.processEvents()
 assert w._hand_noun_object_id(w.event_draft['Left_hand'])==bad,'Imported incompatible reference must remain for correction'
 assert 'model-incompatible component' in w._policy_missing(w.event_draft['Left_hand'])
 w._apply_noun_choice(1)
 assert w._hand_noun_object_id(w.event_draft['Left_hand'])==1
 state=copy.deepcopy(w.event_draft);w._apply_noun_choice(bad)
 assert w.event_draft==state
 w.combo_target.setCurrentIndex(w.combo_target.findData(bad));assert w.event_draft==state
 assert w._choose_drawn_instance_id('fastener_dark' if bad==2 else 'fastener_short') is None
 w._refresh_model_noun_choices()
 assert not w.combo_inline_noun.model().item(w.combo_inline_noun.findData(bad)).isEnabled()
 out=w._build_payload_v2();assert out['tracks']['T_OBJ_1']['category']=='fastener_long'
 assert out['tracks']['T_OBJ_1']['boxes'][0]['bbox']==[10,10,30,30]
 w._load_annotations_v2(json.loads(json.dumps(out)))
 assert w._object_name_for_id(1)=='fastener_long'
 assert not any(x.get('category')=='fastener' for x in w._build_payload_v2()['object_library'].values())
assert noun_allowed('fastener_dark','1234567_r0_unknown_a',PROFILE),'Do not infer an unknown model'
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close()
print('MODEL_A_B_NOUN_CHOICES_OLD_CONFLICT_REVIEW_STABLE_IDS_AND_ALIAS_ROUNDTRIP_PASS')
