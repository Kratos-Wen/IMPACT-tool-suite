"""Independent instances, merge geometry, real dialogs and recovery regression."""
import copy,json,os,sys,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.assembly_timeline import (empty_timeline,put_state,state_at,noun_at,resolve_object,
    reference_id,active_object_ids,merge_instances,track_segments,change_frames,validate_timeline)

base=put_state(empty_timeline(),dict(frame=1,object_id=0,components=['shell','grip']))
base=put_state(base,dict(frame=1,object_id=2,components=['lever','spring']))
assert state_at(base,1) is None
assert state_at(base,1,0)['components']==['shell','grip']
assert state_at(base,1,2)['components']==['lever','spring']
assert resolve_object(dict(shared_assembly_ref=True),base,1) is None
assert resolve_object(dict(shared_assembly_ref=True,noun_object_id=0),base,1)==0
assert resolve_object(dict(shared_assembly_ref=True,shared_assembly_id=2,noun_object_id=0),base,1)==2
saved=copy.deepcopy(base)
for f in range(2,202):
    updated=put_state(base,dict(frame=f,object_id=2,components=['lever','spring','part']))
    assert state_at(updated,f,0)==state_at(base,f,0)
assert base==saved
merged=merge_instances(base,2,0,30)
assert active_object_ids(merged,29)==[0,2] and active_object_ids(merged,30)==[0]
assert resolve_object(dict(shared_assembly_ref=True,shared_assembly_id=2),merged,29)==2
assert resolve_object(dict(shared_assembly_ref=True,shared_assembly_id=2),merged,30)==0
assert track_segments(merged,2,5,50)==[
    dict(object_id=2,track_id='T_OBJ_2',start_frame=5,end_frame=29),
    dict(object_id=0,track_id='T_OBJ_0',start_frame=30,end_frame=50)]
assert change_frames(merged,2,5,50)==[30]
assert validate_timeline(json.loads(json.dumps(merged)))==merged
chain=put_state(merged,dict(frame=1,object_id=7,components=['part','attachment']))
chain=merge_instances(chain,0,7,40)
assert [r['object_id'] for r in track_segments(chain,2,5,50)]==[2,0,7]
assert active_object_ids(chain,50)==[7]
for args in ((2,2,30),(0,2,35),(2,0,1)):
    try:merge_instances(merged,*args)
    except ValueError:pass
    else:raise AssertionError(args)
try:put_state(merged,dict(frame=31,object_id=2,components=['lever','spring']))
except ValueError:pass
else:raise AssertionError('Retired instance resurrected')
future=put_state(base,dict(frame=40,object_id=2,components=['lever','spring','part']))
assert merge_instances(future,2,0,30)['merges'][0]['superseded_states'][0]['frame']==40

from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import (QApplication,QMessageBox,QDialogButtonBox,QListWidget,QComboBox,QCheckBox)
from PyQt5.QtCore import QTimer,Qt
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
PROFILE.clear();PROFILE.update(noun_classes=['shell','grip','lever','spring','part','attachment','assembly','tool'],
    assembly_components=['shell','grip','lever','spring','part','attachment'],assembly_interfaces=[],
    verbs=['hold','insert'],anomaly_labels=['attribute_a'])
refresh_aliases()
app=QApplication([])
warnings=[]
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.warning=lambda *a,**k:(warnings.append(str(a[2])),QMessageBox.Ok)[1]
QMessageBox.question=lambda *a,**k:QMessageBox.Yes
from core.annotation_migration import adapt_annotation
profile=dict(noun_aliases={'old_main':'assembly','old_sub':'assembly'},
    legacy_assembly_mappings={'old_main':['shell','grip'],'old_sub':['lever','spring']})
legacy=dict(video_id='trial',tracks={
    'main':dict(object_id=0,category='old_main',boxes=[dict(frame=5,bbox=[1,2,10,20])]),
    'sub':dict(object_id=2,category='old_sub',boxes=[dict(frame=5,bbox=[21,2,30,20])])},
    object_library={'0':dict(label='old_main',category='old_main'),'2':dict(label='old_sub',category='old_sub')},
    hoi_events={'left_hand':[dict(start_frame=5,end_frame=20,verb='hold',noun_object_id=0)],
                'right_hand':[dict(start_frame=5,end_frame=20,verb='insert',noun_object_id=2)]})
original=copy.deepcopy(legacy);adapted=adapt_annotation(legacy,profile)
assert legacy==original and adapted['tracks']['main']['boxes']==original['tracks']['main']['boxes']
assert adapted['tracks']['sub']['boxes']==original['tracks']['sub']['boxes']
assert {s['object_id'] for s in adapted['shared_assembly']['states']}=={0,2}
assert adapted['hoi_events']['right_hand'][0]['shared_assembly_id']==2
assert adapt_annotation(adapted,profile)==adapted
previously_converted=copy.deepcopy(adapted)
previously_converted['object_library'].pop('2');previously_converted['tracks'].pop('sub')
previously_converted['shared_assembly']=put_state(empty_timeline(),dict(frame=1,object_id=0,components=['shell','grip']))
previously_converted['provenance']={'assembly_identity_map':{'2':0}}
previously_converted['hoi_events']['right_hand'][0].pop('shared_assembly_id')
assert adapt_annotation(previously_converted,profile)['hoi_events']['right_hand'][0]['shared_assembly_id']==0

from ui.hoi_window import HOIWindow
from core.frame_review import signature,valid_record
from core.structured_event_graph import build_hoi_event_graph
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=90;w.player.frame_rate=15
frames=(5,10,20,29,30,31,40,45,50)
payload=dict(schema='hoi-annotation',video_id='trial',frame_count=90,fps=15,shared_assembly=base,
    object_library={'0':dict(label='assembly_0',category='assembly'),
                    '2':dict(label='assembly_2',category='assembly'),'8':dict(label='tool',category='tool')},
    tracks={f'T_OBJ_{uid}':dict(object_id=uid,category='assembly' if uid!=8 else 'tool',
        boxes=[dict(frame=f,bbox=[10+uid*5,10,30+uid*5,30],source='manual',human_verified=True) for f in frames]) for uid in (0,2,8)},
    hoi_events={'left_hand':[dict(event_id='main',start_frame=5,contact_onset_frame=5,end_frame=50,verb='hold',
        noun_object_id=0,shared_assembly_ref=True,shared_assembly_id=0,anomaly_labels=[],anomaly_review_state='reviewed')],
        'right_hand':[dict(event_id='donor',start_frame=5,contact_onset_frame=20,end_frame=50,verb='insert',
        noun_object_id=2,shared_assembly_ref=True,shared_assembly_id=2,instrument_object_id=8,anomaly_labels=[],anomaly_review_state='reviewed')]})
w._load_annotations_v2(payload)
left=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
right=next(e for e in w.events if e['hoi_data']['Right_hand'].get('verb'))
left['hoi_data']['Right_hand']=copy.deepcopy(right['hoi_data']['Right_hand']);w.events.remove(right)
eid=left['event_id'];w._set_selected_event(eid,'Right_hand');w.player.current_frame=20
before_left=copy.deepcopy(w.event_draft['Left_hand'])
boxes={uid:[b for b in w.raw_boxes if b['id']==uid and b['orig_frame']==20] for uid in (0,2)}
w.frame_review={'20':{f'O:{uid}':dict(state='verified',signature=signature(rows)) for uid,rows in boxes.items()}}
left_review=copy.deepcopy(w.frame_review['20']['O:0'])
w._commit_assembly_update(put_state(base,dict(frame=20,object_id=2,components=['lever','spring','part'],composition_review_state='reviewed')),2,True,False)
assert w.event_draft['Left_hand']==before_left and w.frame_review['20']['O:0']==left_review
assert 'O:2' not in w.frame_review['20']
w._hoi_undo();assert w.shared_assembly==base and 'O:2' in w.frame_review['20']

# Real modal dialog: instance switch reloads its own checklist, and a new ID is unique.
w.player.cap=True;w.player.seek=lambda f:setattr(w.player,'current_frame',f)
def create_instance():
    dlg=app.activeModalWidget();combo=dlg.findChild(QComboBox,'assemblyInstance');parts=dlg.findChild(QListWidget,'assemblyComponents')
    combo.setCurrentIndex(combo.findData(0))
    assert {parts.item(i).text() for i in range(parts.count()) if parts.item(i).checkState()==Qt.Checked}=={'shell','grip'}
    combo.setCurrentIndex(combo.findData(2))
    assert {parts.item(i).text() for i in range(parts.count()) if parts.item(i).checkState()==Qt.Checked}=={'lever','spring'}
    combo.setCurrentIndex(0)
    for i in range(parts.count()):parts.item(i).setCheckState(Qt.Checked if parts.item(i).text() in ('part','attachment') else Qt.Unchecked)
    dlg.findChild(QCheckBox,'assemblyLink').setChecked(False)
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
QTimer.singleShot(0,create_instance);w._open_assembly_editor()
new_ids=set(active_object_ids(w.shared_assembly,20))-{0,2}
assert len(new_ids)==1 and w.event_draft['Left_hand']==before_left
w._hoi_undo();assert active_object_ids(w.shared_assembly,20)==[0,2]

# Real merge dialog: insert ID 2 into ID 0; do not overwrite either hand's action.
w.player.current_frame=30
def merge_dialog():
    dlg=app.activeModalWidget()
    source=dlg.findChild(QComboBox,'mergeSource');target=dlg.findChild(QComboBox,'mergeTarget')
    source.setCurrentIndex(source.findData(2));target.setCurrentIndex(target.findData(0))
    dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
QTimer.singleShot(0,merge_dialog);w._open_assembly_merge()
assert len(w.shared_assembly['merges'])==1 and w._event_visible_object_ids()=={'0','8'}
assert not w._event_box_visible(next(b for b in w.raw_boxes if b['id']==2 and b['orig_frame']==30))
assert w.event_draft['Right_hand']['verb']=='insert' and w.event_draft['Right_hand']['shared_assembly_id']==2
assert 'Object ID changes inside event; split at the change' not in w._policy_missing(w.event_draft['Right_hand'])
assert 'noun changes inside event; split at the change' not in w._policy_missing(w.event_draft['Right_hand'])
assert w._review_entities(w.event_draft['Right_hand'],'Right_hand',29)==['H:Right_hand','O:2','O:8']
assert w._review_entities(w.event_draft['Right_hand'],'Right_hand',30)==['H:Right_hand','O:0','O:8']
graph=build_hoi_event_graph(w.events,assembly_timeline=w.shared_assembly)
assert next(e for e in graph['events'] if e['verb']=='insert')['target_track_segments'][1]['object_id']==0
w._hoi_undo();assert active_object_ids(w.shared_assembly,30)==[0,2]
w._hoi_redo();assert active_object_ids(w.shared_assembly,30)==[0]
w.player.cap=None

for cycle in range(50):
    out=w._build_payload_v2()
    donor=out['hoi_events']['right_hand'][0]
    assert donor['verb']=='insert' and donor['shared_assembly_id']==2
    assert donor['contact_onset_frame']==20 and donor['start_frame']==5 and donor['end_frame']==50
    assert donor['links']['target_track_segments']==track_segments(w.shared_assembly,2,5,50)
    assert all(b['frame']<30 for b in out['tracks']['T_OBJ_2']['boxes'])
    assert [b['frame'] for b in out['tracks']['T_OBJ_2']['inactive_boxes']]==[30,31,40,45,50]
    w._load_annotations_v2(json.loads(json.dumps(out)))
    event=next(e for e in w.events if e['hoi_data']['Right_hand'].get('verb'))
    w._set_selected_event(event['event_id'],'Right_hand');w.player.current_frame=45
    assert w._hand_noun_object_id(w.event_draft['Right_hand'])==0
    w._save_ui_to_hand_draft('Right_hand');w._apply_draft_to_selected_event()
    assert w.event_draft['Right_hand']['shared_assembly_id']==2
    evidence=w._compute_sparse_evidence_state(w.event_draft['Right_hand'])
    assert evidence['noun_start']['object_id']==2 and evidence['noun_end']['object_id']==0
    assert evidence['noun_end']['status']=='confirmed'
    assert w._event_visible_object_ids()=={'0','8'}
    w._refresh_assembly_caption();app.processEvents()

# ID zero is a real default, not False; recovery retains timeline and both IDs.
w._assembly_default_reference=0;w._reset_event_draft()
assert w.event_draft['Left_hand']['shared_assembly_id']==0
with tempfile.TemporaryDirectory(prefix='assembly recovery ') as directory:
    video=Path(directory)/'video.mp4';video.touch();w.video_path=str(video)
    w._open_autosave();w._autosave_timer.stop()
    w._commit_assembly_update(put_state(w.shared_assembly,dict(frame=60,object_id=0,
        components=['shell','grip','lever','spring','part'],composition_review_state='reviewed')))
    w._finish_autosave();w._autosave_tick();w._finish_autosave()
    from core.safe_storage import read_recovery
    recovered,_=read_recovery(w._autosave_path,w._autosave_identity())
    assert recovered['state']['shared_assembly']==w.shared_assembly
    assert recovered['state']['assembly_default_reference']==0
    assert recovered['state']['event_draft']['Left_hand']['shared_assembly_id']==0
    w._stop_autosave()
w.deleteLater()
assert not warnings,warnings
print('MULTI_ASSEMBLY_200_ISOLATION_CHECKS_50_ROUNDTRIPS_REAL_DIALOG_MERGE_UNDO_RECOVERY_PASS')
