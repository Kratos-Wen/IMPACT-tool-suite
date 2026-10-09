"""Round-trip review attributes, unknown contact, migration and object identity."""
import copy,json,os,sys,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
app=QApplication([])
for method in ('information','question','warning'):setattr(QMessageBox,method,lambda *a,**k:QMessageBox.No)
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
from core.anomaly_attributes import export_review,display_value
from core.annotation_migration import adapt_annotation,canonical_verb
from ui.attribute_selector import AttributeSelector
from ui.hoi_window import HOIWindow,YoloInferenceWorker,_mediapipe_anatomical_label
from types import SimpleNamespace
assert canonical_verb('old-action',{'verb_aliases':{'old_action':'action'}})=='action'
PROFILE.clear();PROFILE.update(default_anomaly_label='unreviewed',anomaly_labels=['attribute_a','attribute_b'],noun_aliases={'old_part':'part','old_combo':'assembly'},legacy_assembly_mappings={'old_combo':['part','attachment']},legacy_anomaly_candidates={'old_error':['attribute_a']},assembly_components=['part','attachment','grip_handle'],assembly_interfaces=[dict(id='part--attachment',base_component='part',completed_component='attachment')]);refresh_aliases()
assert _mediapipe_anatomical_label('Left')=='right'
assert _mediapipe_anatomical_label('Right')=='left'
worker=YoloInferenceWorker(SimpleNamespace(names={0:'old_part'}),[],0.5,0.5,{0:'wrong_editor_class'})
assert worker._class_name_for_id(0)=='part'
selector=AttributeSelector();selector.setOptions(PROFILE['anomaly_labels'],'unreviewed')
selector.toggle('attribute_a');selector.toggle('attribute_b');assert export_review(selector.currentText(),PROFILE)==dict(anomaly_labels=['attribute_a','attribute_b'],anomaly_review_state='reviewed')
selector.toggle('unknown');assert export_review(selector.currentText(),PROFILE)['anomaly_review_state']=='partial'
selector.toggle('confirm');assert export_review(selector.currentText(),PROFILE)['anomaly_review_state']=='reviewed'
selector.toggle('normal');assert export_review(selector.currentText(),PROFILE)['anomaly_labels']==[]
source=dict(version='legacy',video_id='1234567_r1_a_a',fps=15,frame_count=60,object_library={'0':dict(label='old_combo_1',category='old_combo'),'5':dict(label='old_part_2',category='old_part')},tracks={'T_OBJ_0':dict(object_id=0,category='old_combo',boxes=[dict(frame=2,bbox=[1,2,10,20])]),'T_OBJ_5':dict(object_id=5,category='old_part',boxes=[dict(frame=3,bbox=[3,4,12,22])])},hoi_events={'left_hand':[dict(event_id='L_original',start_frame=2,contact_onset_frame=None,end_frame=10,verb='hold',noun_object_id=0,links=dict(target_track_id='T_OBJ_0'),anomaly_label='old_error')],'right_hand':[]})
source['hoi_events']['left_hand'][0]['annotation_state']={'field_suggestions':{'noun_object_id':{'value':0}}}
original=copy.deepcopy(source);converted=adapt_annotation(source,PROFILE);assert source==original
assert converted['schema']=='hoi-annotation' and 'version' not in converted
assert converted['shared_assembly']['states'][0]['interfaces']['part--attachment']=={'1':None,'2':None}
assert converted['tracks']['T_OBJ_5']['boxes']==source['tracks']['T_OBJ_5']['boxes']
assert converted['hoi_events']['left_hand'][0]['anomaly_review_state']=='unreviewed'
assert adapt_annotation(converted,PROFILE)==converted
alias_source=copy.deepcopy(source)
alias_source['object_library']['7']={'label':'old_combo_2','category':'old_combo'}
alias_source['hoi_events']['left_hand'][0]['annotation_state']['field_suggestions']['noun_object_id']['value']=7
alias_converted=adapt_annotation(alias_source,PROFILE)
assert alias_converted['hoi_events']['left_hand'][0]['annotation_state']['field_suggestions']['noun_object_id']['value']==0
# A newly registered physical instance never inherits a retired alias mapping.
alias_converted['object_library']['7']={'label':'new_part','category':'new_part'}
alias_converted['hoi_events']['left_hand'][0]['annotation_state']['field_suggestions']['noun_object_id']['value']=7
assert adapt_annotation(alias_converted,PROFILE)['hoi_events']['left_hand'][0]['annotation_state']['field_suggestions']['noun_object_id']['value']==7
payload=copy.deepcopy(converted);event=payload['hoi_events']['left_hand'][0]
event.update(anomaly_labels=['attribute_a','attribute_b'],anomaly_review_state='reviewed',onset_review_state='unknown',onset_reason='contact hidden',anomaly_evidence={'attribute_a':{'view':'ego','intervals':[[3,5]]},'attribute_b':{'view':'ego','intervals':[[6,8]]}})
payload['hoi_events']['right_hand']=[dict(event_id='R_original',start_frame=3,contact_onset_frame=3,end_frame=9,verb='insert',noun_object_id=5,anomaly_labels=[],anomaly_review_state='reviewed',links={'target_track_id':'T_OBJ_5'})]
w=HOIWindow();w.player.frame_count=60;w.player.frame_rate=15
w._load_annotations_v2(payload)
for iteration in range(25):
 out=w._build_payload_v2();left=out['hoi_events']['left_hand'][0];right=out['hoi_events']['right_hand'][0]
 assert left['anomaly_labels']==['attribute_a','attribute_b'],left
 assert left['contact_onset_frame'] is None and left['onset_reason']=='contact hidden'
 assert left['event_id']=='L_original' and right['event_id']=='R_original'
 assert right['verb']=='insert' and right['noun_object_id']==5
 assert left['anomaly_evidence']==event['anomaly_evidence']
 assert out['schema']=='hoi-annotation' and 'version' not in out
 w._load_annotations_v2(json.loads(json.dumps(out)))
assert w._normalize_anomaly_label('old_temporal_category')=='unknown'
# Auxiliary event graph exports the same multilabel state as the main editor.
from core.structured_event_graph import build_hoi_event_graph
current_graph=build_hoi_event_graph(w.events)
assert current_graph['schema']=='hoi-event-graph'
assert any(row['anomaly_labels']==['attribute_a','attribute_b'] for row in current_graph['events'])
assert not any('anomaly_label' in row for row in current_graph['events'])
w._assembly_default_reference=True;w._reset_event_draft();w.events=[dict(event_id=99,frames=[0,20],hoi_data=copy.deepcopy(w.event_draft))]
assert not any(w._build_payload_v2()['hoi_events'].values()),'Idle hand generated event'
assert not build_hoi_event_graph(w.events)['events'],'Idle shared hand generated graph event'
# Real modal composition editor: unknown screw state and independent hand survive.
from PyQt5.QtCore import QTimer,Qt
from PyQt5.QtWidgets import QComboBox,QListWidget,QTableWidget,QDialogButtonBox
w.events=[];w._load_annotations_v2(payload)
w.selected_event_id=w.events[0]['event_id'];w.selected_hand_label='Left_hand'
w.event_draft=copy.deepcopy(w.events[0]['hoi_data'])
w.event_draft['Right_hand']=dict(verb='insert',noun_object_id=5,target_object_id=5,instrument_object_id=None,interaction_start=3,interaction_end=9)
w.events[0]['hoi_data']=copy.deepcopy(w.event_draft)
app.processEvents()
w.player.cap=True;w.player.current_frame=2
w.player.seek=lambda frame: setattr(w.player,'current_frame',frame)
w._jump_first_incomplete=lambda:None
w._refresh_boxes_for_frame=lambda *a:None
w._load_hand_draft_to_ui=lambda *a:None
before_right=copy.deepcopy(w.event_draft['Right_hand'])
def edit_composition():
 dlg=app.activeModalWidget();parts=dlg.findChild(QListWidget);table=dlg.findChild(QTableWidget)
 assert table.cellWidget(0,1).checkState()==Qt.PartiallyChecked
 assert 'grip_handle' in [parts.item(i).text() for i in range(parts.count())]
 for i in range(parts.count()):parts.item(i).setCheckState(Qt.Checked if parts.item(i).text()=='part' else Qt.Unchecked)
 dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
QTimer.singleShot(0,edit_composition);w._open_assembly_editor()
assert w.event_draft['Right_hand']==before_right
assert w.shared_assembly['states'][0]['composition_review_state']=='reviewed'
assert w.shared_assembly['states'][0]['interfaces']['part--attachment']=={'1':None,'2':None}
w.player.cap=None
# Actual policy controls keep null contact through the normal UI-save path.
from unittest.mock import patch
from PyQt5.QtWidgets import QInputDialog
w._load_hand_draft_to_ui=HOIWindow._load_hand_draft_to_ui.__get__(w)
w._set_selected_event(w.events[0]['event_id'],'Left_hand')
with patch.object(QInputDialog,'getItem',return_value=('Not observable',True)),patch.object(QInputDialog,'getText',return_value=('contact hidden',True)):
 w._record_contact_exception()
w._save_ui_to_hand_draft('Left_hand');w._apply_draft_to_selected_event()
assert w.event_draft['Left_hand']['functional_contact_onset'] is None
out=w._build_payload_v2()['hoi_events']['left_hand'][0];assert out['contact_onset_frame'] is None and out['onset_reason']=='contact hidden'
# Byte-exact archiving is idempotent and does not replace an existing source.
from core.annotation_archive import archive_source
with tempfile.TemporaryDirectory() as directory:
    file=Path(directory)/'old.json';original_bytes=b'{"version":"old"}\n';file.write_bytes(original_bytes)
    from unittest.mock import patch
    cache=Path(directory)/'application_cache'
    with patch.dict(os.environ,{'IMPACT_SOURCE_ARCHIVE':str(cache)}):
        first=archive_source(file);second=archive_source(file)
    assert first==second and (cache/first['sha256']/first['filename']).read_bytes()==original_bytes
    assert not (file.parent/'archived').exists()
    assert file.read_bytes()==original_bytes
# Profile vocabularies are available for new work without creating a shared assembly.
PROFILE.update(verbs=['fresh_action'],noun_classes=['fresh_part','assembly'],create_default_noun_instances=True)
w.combo_verb.setCurrentIndex(-1);w._apply_profile_libraries()
assert any(label.name=='fresh_action' for label in w.verbs)
assert w.combo_verb.currentText()==''
assert any(name.startswith('fresh_part') for name in w.global_object_map)
generic=dict(verb='hold',noun_object_id=0,interaction_start=2,interaction_end=10,anomaly_label='normal')
w.global_object_map['assembly_1']=0
assert 'assembly composition' in w._policy_missing(generic)
w.close();print('CURRENT_ANNOTATION_25_ROUNDTRIPS_COMPOSITION_DIALOG_AND_ARCHIVE_PASS')
