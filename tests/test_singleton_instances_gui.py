"""Canonical machine instances survive real editor loading, editing and saving."""
import copy,os,sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_singleton_instances import out,profile
from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(profile,verbs=['hold'],anomaly_labels=['attribute_a'],create_default_noun_instances=True)
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QInputDialog
app=QApplication([])
for name in ('information','warning','critical','question'):setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)
from ui.hoi_window import HOIWindow
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=30;w.player.frame_rate=15
w._load_annotations_v2(copy.deepcopy(out));w._apply_profile_libraries()
left=next(e for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
w._set_selected_event(left['event_id'],'Left_hand');w.player.current_frame=3
canonical_uid=w._annotation_provenance['video_local_instances']['original_to_local']['1002']
assert 0<canonical_uid<100
w._pending_draw_box_role='noun'
QInputDialog.getItem=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Reasked known identity'))
w._on_box_edited(None,dict(_action='add',label='anti_vibration_handle',frame=3,x1=2,y1=3,x2=12,y2=22))
assert w.raw_boxes[-1]['id']==canonical_uid and w._hand_noun_object_id(w.event_draft['Left_hand'])==canonical_uid
saved=w._build_payload_v2();assert saved['object_library'][str(canonical_uid)]['label']=='anti_vibration_handle'
assert saved['provenance']['singleton_identity']==out['provenance']['singleton_identity']
w._load_annotations_v2(saved)
assert any(b['id']==canonical_uid and b['orig_frame']==3 for b in w.raw_boxes)
w._stop_autosave();w._mark_hoi_saved();w.close()
# An empty project uses small video-local IDs, independently of old reserved IDs.
w=HOIWindow();w._autosave_timer.stop();w.global_object_map={};w.id_to_category={};w.raw_boxes=[];w.object_id_counter=0
w._apply_profile_libraries();baseline=(copy.deepcopy(w.global_object_map),w.object_id_counter)
for _ in range(30):w._apply_profile_libraries()
assert (w.global_object_map,w.object_id_counter)==baseline
assert 0<w.global_object_map['anti_vibration_handle']<100
assert w.global_object_map['torx_screwdriver_15']!=w.global_object_map['torx_screwdriver_20']
assert max(w.global_object_map.values())<100
w._stop_autosave();w._mark_hoi_saved();w.close()
print('CANONICAL_MACHINE_EDITOR_ADD_SAVE_REOPEN_AND_EMPTY_PROJECT_FIXED_IDS_PASS')
