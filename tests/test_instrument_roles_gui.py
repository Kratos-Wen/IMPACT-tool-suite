"""Role filtering cannot change IDs, hide Object tools or erase invalid old fields."""
import os,sys,copy,json,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
PROFILE.clear();PROFILE.update(noun_classes=['shell','bolt','driver','cutter'],
    noun_aliases={'old_driver':'driver'},instrument_classes=['driver','cutter'],
    verbs=['pick_up','hold'],anomaly_labels=[],compact_import_ids=True,
    create_default_noun_instances=True)
refresh_aliases()
app=QApplication([]);messages=[]
for name in ('information','warning','critical','question'):
    setattr(QMessageBox,name,lambda *a,**k:messages.append(a[1:3]) or QMessageBox.No)
from ui.hoi_window import HOIWindow
from core.local_instances import localize_instances,SCHEMA
source=dict(schema='hoi-annotation',video_id='trial',frame_count=40,fps=15,
 object_library={'94':dict(label='shell',category='shell'),
                 '95':dict(label='old_driver_1',category='old_driver'),
                 '96':dict(label='bolt',category='bolt')},
 tracks={'old':dict(object_id=94,category='shell',boxes=[dict(frame=2,bbox=[1,2,12,20],source='manual_box_edit')])},
 hoi_events={'left_hand':[dict(event_id='one',start_frame=1,contact_onset_frame=2,end_frame=6,
   verb='hold',noun_object_id=94,instrument_object_id=95,anomaly_labels=[])],'right_hand':[]})
raw=copy.deepcopy(source);localized=localize_instances(source,PROFILE)
assert source==raw and list(localized['object_library'])==['1','2','3']
assert localized['hoi_events']['left_hand'][0]['instrument_object_id']==2
assert localized['tracks']['T_OBJ_1']['boxes'][0]['bbox']==[1,2,12,20]
# A previously localized/human-used ID must not be renumbered to please the display.
existing=copy.deepcopy(source);existing['provenance']={'video_local_instances':{'schema':SCHEMA}}
assert set(localize_instances(existing,PROFILE)['object_library'])=={'94','95','96'}
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=40;w.player.frame_rate=15
w._load_annotations_v2(source);w._apply_profile_libraries()
event=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
w._set_selected_event(event['event_id'],'Left_hand');app.processEvents()
def ids(combo):return {combo.itemData(i) for i in range(combo.count()) if combo.itemData(i) is not None}
def assert_roles():
    for combo in (w.combo_instrument,w.combo_inline_instrument):
        assert all(w._object_name_for_id(uid) in ('driver','cutter') for uid in ids(combo))
        assert 1 not in ids(combo) and 3 not in ids(combo)
    assert {1,2,3}.issubset(ids(w.combo_target)), 'A tool may still be Object in pickup/hold'
assert_roles();baseline=copy.deepcopy(w.events)
for _ in range(10):
    w._apply_profile_libraries();w._load_hand_draft_to_ui('Left_hand');w._update_inline_event_editor()
    assert_roles();assert w.events==baseline
# Typed labels and numeric IDs cannot bypass the filtered dropdown.
for combo,callback,text in [(w.combo_instrument,w._on_hoi_meta_changed,'shell'),
                           (w.combo_inline_instrument,w._on_inline_instrument_changed,'[3] bolt')]:
    old=combo.blockSignals(True);combo.setEditText(text);combo.blockSignals(old)
    callback();assert w._hand_instrument_object_id(w.event_draft['Left_hand'])==2
# Drawing a part in Instrument mode retains geometry without linking it as a tool.
w._assign_drawn_object_to_current_hand(1,'instrument')
assert w._hand_instrument_object_id(w.event_draft['Left_hand'])==2
saved=w._build_payload_v2();w._load_annotations_v2(json.loads(json.dumps(saved)))
w._apply_profile_libraries();assert_roles()
assert w._build_payload_v2()['object_library']==saved['object_library']
# An imported incorrect role remains visible as an invalid field, not a menu option.
bad=copy.deepcopy(source);bad['hoi_events']['left_hand'][0]['instrument_object_id']=94
w._load_annotations_v2(bad);w._apply_profile_libraries()
event=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
w._set_selected_event(event['event_id'],'Left_hand');app.processEvents()
assert_roles()
assert 'Invalid instrument [1]' in w.combo_instrument.currentText()
assert 'choose a valid instrument' in w._policy_missing(w.event_draft['Left_hand'])
w._on_inline_instrument_text_finished()
assert w._hand_instrument_object_id(w.event_draft['Left_hand'])==1
w.combo_verb.setCurrentText('pick_up')
assert w._hand_instrument_object_id(w.event_draft['Left_hand'])==1, 'Verb refresh silently cleared an old invalid reference'
assert w._build_payload_v2()['hoi_events']['left_hand'][0]['instrument_object_id']==1
# Correcting the field is explicit and survives undo/reopen.
w.combo_instrument.setCurrentIndex(w.combo_instrument.findData(2))
assert w._hand_instrument_object_id(w.event_draft['Left_hand'])==2
saved=w._build_payload_v2();w._load_annotations_v2(saved);w._apply_profile_libraries();assert_roles()
assert w._build_payload_v2()['hoi_events']['left_hand'][0]['instrument_object_id']==2
# Profile activation removes old role choices even when default instances are disabled.
PROFILE.update(instrument_classes=['driver'],create_default_noun_instances=False)
w._apply_profile_libraries()
assert ids(w.combo_instrument)=={2}
assert messages

# Hand-only import cannot carry the source's Object library, events or hand IDs.
import cv2,numpy as np
from core.hand_assets import read_json_hands
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);video=root/'video.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(64,48));assert writer.isOpened()
    for _ in range(40):writer.write(np.full((48,64,3),100,np.uint8))
    writer.release();w.player.cap=cv2.VideoCapture(str(video));assert w.player.cap.isOpened()
    path=root/'machine.json';fixture=copy.deepcopy(source)
    fixture['tracks'].update({
      'L1':dict(object_id=500,category='Left_hand',boxes=[dict(frame=2,bbox=[1,2,12,20]),dict(frame=3,bbox=[1,2,12,20]),dict(frame=5,bbox=[1,2,12,20])]),
      'L2':dict(object_id=999,category='Left_hand',boxes=[dict(frame=2,bbox=[1,2,12,20]),dict(frame=3,bbox=[2,3,14,21])]),
      'R':dict(object_id=998,category='Right_hand',boxes=[dict(frame=2,bbox=[21,2,30,20]),dict(frame=4,bbox=[21,2,30,20])])})
    path.write_text(json.dumps(fixture))
    candidates,omitted=read_json_hands(path,40,15,64,48,'trial')
    assert omitted==1 and {r['id'] for r in candidates}=={-1,-2}
    w.raw_boxes.append(dict(id=-1,label='Left_hand',orig_frame=2,x1=8,y1=8,x2=20,y2=22,source='manual_box_edit'))
    w._suppressed_hand_boxes=['Right_hand:4']
    before=(copy.deepcopy(w.events),copy.deepcopy(w.global_object_map),w.object_id_counter)
    QFileDialog.getOpenFileName=lambda *a,**k:(str(path),'')
    w._load_hands_json()
    assert (w.events,w.global_object_map,w.object_id_counter)==before
    assert any(r['id']==-1 and r['orig_frame']==2 and r['x1']==8 for r in w.raw_boxes)
    assert not any(r['id']==-1 and r['orig_frame']==3 for r in w.raw_boxes)
    assert not any(r['id']==-2 and r['orig_frame']==4 for r in w.raw_boxes)
    baseline=copy.deepcopy(w.raw_boxes);w._load_hands_json();assert w.raw_boxes==baseline
    fixture['fps']=30;path.write_text(json.dumps(fixture));w._load_hands_json()
    assert w.raw_boxes==baseline
    w.player.release_media()
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close()
print('INSTRUMENT_ROLE_FILTER_TYPED_IDS_DRAWING_INVALID_IMPORT_SAVE_REOPEN_AND_SMALL_IMPORT_IDS_PASS')
