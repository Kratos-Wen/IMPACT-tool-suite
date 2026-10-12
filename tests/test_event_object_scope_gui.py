"""Imported tracks stay editable and durable while unrelated objects stay hidden."""
import os,sys,json,copy,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
PROFILE.clear();PROFILE.update(noun_classes=['part','tool','background','other'],automatic_tracking_import='all',automatic_tracking_import_limit='all',verbs=['hold'],anomaly_labels=[])
refresh_aliases();app=QApplication([])
for name in ('information','warning','critical','question'):setattr(QMessageBox,name,lambda *a,**k:QMessageBox.Yes)
from ui.hoi_window import HOIWindow
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=30;w.player.frame_rate=15
source={'schema':'hoi-annotation','video_id':'trial','frame_count':30,'fps':15,'object_library':{str(i):{'label':name,'category':name} for i,name in enumerate(['part','tool','background','other'],1)},'tracks':{f'T{i}':{'object_id':i,'category':name,'boxes':[{'frame':f,'bbox':[10,10,30,30],'source':'model_candidate'} for f in (0,20)]} for i,name in enumerate(['part','tool','background','other'],1)},'hoi_events':{'left_hand':[{'event_id':'first','start_frame':0,'contact_onset_frame':0,'end_frame':10,'verb':'hold','noun_object_id':1,'instrument_object_id':2},{'event_id':'second','start_frame':20,'contact_onset_frame':20,'end_frame':29,'verb':'hold','noun_object_id':4}],'right_hand':[]},'provenance':{'machine_only':True}}
w._load_annotations_v2(source);assert len(w.raw_boxes)==8
assert not w._event_visible_object_ids(0)
event=next(e for e in w.events if e['hoi_data']['Left_hand'].get('interaction_start')==0)
w.player.current_frame=0;w._set_selected_event(event['event_id'],'Left_hand')
app.processEvents()
assert w.selected_event_id==event['event_id'],'Deferred load navigation must not override a selected event'
assert w._event_visible_object_ids(0)=={'1','2'}
assert not w._event_visible_object_ids(20),'Another event must not expose unrelated object tracks'
w.event_draft['Left_hand'].update(noun_object_id=None,target_object_id=None,instrument_object_id=None,tool_object_id=None)
assert not w._event_visible_object_ids(0)
w._apply_noun_choice(1);w.event_draft['Left_hand'].update(instrument_object_id=2,tool_object_id=2)
box=next(b for b in w.raw_boxes if b['id']==1 and b['orig_frame']==0)
w._selected_edit_box=copy.deepcopy(box);w._on_box_edited(1,dict(box,frame=0,x1=13.5))
assert next(b for b in w.raw_boxes if b['id']==1 and b['orig_frame']==0)['x1']==13.5
with tempfile.TemporaryDirectory() as folder:
 path=Path(folder)/'reviewed.json';QFileDialog.getSaveFileName=lambda *a,**k:(str(path),'JSON')
 w._check_incomplete_hoi=lambda **k:(True,[])
 w._require_participant_code_for_study_save=lambda:True
 w._ensure_semantic_adapter_loaded=lambda:False
 w._save_annotations_json();assert path.is_file()
 data=json.loads(path.read_text())
 assert data['tracks']['T_OBJ_1']['boxes'][0]['bbox'][0]==13.5
 assert len(data['tracks']['T_OBJ_3']['boxes'])==2,'Hidden background tracks must survive save'
 w._load_annotations_v2(data)
 assert next(b for b in w.raw_boxes if b['id']==1 and b['orig_frame']==0)['x1']==13.5
 assert len([b for b in w.raw_boxes if b['id']==3])==2
 w._load_annotations_v2(w._build_payload_v2())
 assert next(b for b in w.raw_boxes if b['id']==1 and b['orig_frame']==0)['x1']==13.5
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close()
print('EVENT_ONLY_OBJECT_TOOL_FILTER_EDIT_ATOMIC_SAVE_REOPEN_HIDDEN_TRACKS_PASS')
