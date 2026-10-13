"""Unchanged paints reuse checks; semantic, geometry and review changes invalidate them."""
import copy,os,sys
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
from core.project_profile import PROFILE
from core.frame_review import signature
PROFILE.clear();PROFILE.update(noun_classes=['part'],verbs=['hold','insert'],anomaly_labels=[])
from ui.hoi_window import HOIWindow
app=QApplication([])
for method in ('warning','critical','information','question'):setattr(QMessageBox,method,lambda *a,**k:QMessageBox.No)
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=20000;w.player.frame_rate=15
w._load_annotations_v2(dict(schema='hoi-annotation',frame_count=20000,fps=15,
 object_library={'1':dict(label='part',category='part')},
 tracks={'T_OBJ_1':dict(object_id=1,category='part',boxes=[dict(frame=f,bbox=[1,2,8,9],source='manual_box_add') for f in (0,10)]),
 'T_LEFT_HAND':dict(category='left_hand',boxes=[dict(frame=f,bbox=[10,12,20,24],source='manual_box_add') for f in (0,10)])},
 hoi_events={'left_hand':[dict(start_frame=0,contact_onset_frame=0,end_frame=10,verb='hold',noun_object_id=1)],'right_hand':[]}))
event=w.events[0];w._set_selected_event(event['event_id'],'Left_hand');hand=event['hoi_data']['Left_hand']
assert w._hoi_segments_for_hand('Left_hand')[0]['complete']
w._collect_incomplete_hoi();w._refresh_sparse_evidence_snapshots()
# Canonicalization can modify fields on the first pass, so warm all caches.
w._hoi_segments_for_hand('Left_hand');w._collect_incomplete_hoi()
with patch.object(w,'_hand_completion_state',side_effect=AssertionError('Paint recomputed completeness')),patch.object(w,'_compute_sparse_evidence_state',side_effect=AssertionError('Refresh recomputed sparse evidence')):
 for _ in range(50):
  assert w._hoi_segments_for_hand('Left_hand')[0]['complete']
  w._collect_incomplete_hoi();w._refresh_sparse_evidence_snapshots()
result=w._hoi_segments_for_hand('Left_hand');result[0]['complete']=False;result[0]['color'].setNamedColor('black')
assert w._hoi_segments_for_hand('Left_hand')[0]['complete']
hand['verb']='insert'
assert w._hoi_segments_for_hand('Left_hand')[0]['verb']=='insert'
# Missing geometry must turn the border incomplete, including after undo/recovery.
before=copy.deepcopy(w.raw_boxes);w.raw_boxes=[b for b in w.raw_boxes if b['orig_frame']!=10]
w._rebuild_bboxes_from_raw();assert not w._hoi_segments_for_hand('Left_hand')[0]['complete']
issues=w._collect_incomplete_hoi();assert issues
issues[0]['missing'].append('caller mutation');issues[0]['frame']=-100
assert all(i['frame']>=0 and 'caller mutation' not in i['missing'] for i in w._collect_incomplete_hoi())
w.raw_boxes=before;w._rebuild_bboxes_from_raw();assert w._hoi_segments_for_hand('Left_hand')[0]['complete']
# Review transitions may occur without replacing their nested dictionary.
for box in w.raw_boxes:box.update(source='automatic',human_verified=False)
w._bump_bbox_revision();w._bump_query_state_revision()
assert not w._hoi_segments_for_hand('Left_hand')[0]['complete']
w.frame_review={}
for frame in (0,10):
 w.frame_review[str(frame)]={entity:dict(state='verified',signature=signature(w._boxes_for_review(entity,frame))) for entity in ('H:Left_hand','O:1')}
assert w._hoi_segments_for_hand('Left_hand')[0]['complete']
w.frame_review['10']['O:1']['signature']='invalid'
assert not w._hoi_segments_for_hand('Left_hand')[0]['complete']
w.frame_review['10']['O:1']['signature']=signature(w._boxes_for_review('O:1',10))
assert w._hoi_segments_for_hand('Left_hand')[0]['complete']
PROFILE['deprecated_verbs']=['insert']
assert not w._hoi_segments_for_hand('Left_hand')[0]['complete']
PROFILE['deprecated_verbs']=[];assert w._hoi_segments_for_hand('Left_hand')[0]['complete']
# Grid work scales with screen width, rather than the length of the video.
row=w.hoi_timeline.actor_rows['Left_hand'];row.resize(900,48)
for span in (100,20000,2000000):
 w.hoi_timeline._set_view(0,span)
 step=row._major_grid_step(15)
 assert step>=15 and step%15==0 and row.get_span()/step<30
# Qt must not rebuild identical styles while the playhead moves.
chip=w.lbl_incomplete_chip;w._set_status_chip(chip,'Stable','ok')
with patch.object(chip,'setStyleSheet',wraps=chip.setStyleSheet) as style:
 for _ in range(50):w._set_status_chip(chip,'Stable','ok')
 assert style.call_count==0
 w._set_status_chip(chip,'Missing','warn');assert style.call_count==1
assert chip.text()=='Missing'
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close()
print('RENDER_CACHE_50_CYCLES_SEMANTICS_GEOMETRY_INPLACE_REVIEW_PROFILE_GRID_INVALIDATION_PASS')
