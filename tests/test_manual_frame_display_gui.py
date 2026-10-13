"""Dense candidates never create manual ticks or fill unselected hand frames."""
import copy, os, sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(verbs=['hold'],noun_classes=['part'],anomaly_labels=[])
app=QApplication([])
for name in ('information','warning','question','critical'):
    setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)
from ui.hoi_window import HOIWindow
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=20000;w.player.frame_rate=15
payload=dict(schema='hoi-annotation',video_id='fixture',frame_count=20000,fps=15,
 object_library={'8':dict(label='part',category='part')},
 tracks={'T_LEFT_HAND':dict(category='Left_hand',automatic=True,
    boxes=[dict(frame=f,bbox=[1,2,8,9],source='machine_candidate') for f in range(20000)]),
    'T_OBJ_8':dict(object_id=8,category='part',boxes=[dict(frame=12,bbox=[15,2,25,9],source='manual_box_add')])},
 hoi_events={'left_hand':[dict(start_frame=0,contact_onset_frame=5,end_frame=19999,verb='hold',noun_object_id=8)],'right_hand':[]})
w._load_annotations_v2(payload);w._set_selected_event(w.events[0]['event_id'],'Left_hand')
assert {m['frame'] for m in w._bbox_frame_markers()}=={12}
assert all(m['role']=='O' for m in w._bbox_frame_markers())
def handbox(frame):return next(b for b in w.raw_boxes if b['orig_frame']==frame and b['label']=='Left_hand')
for f in (0,5,19999,12):assert w._event_box_visible(handbox(f))
for f in (1,6,13,10000):assert not w._event_box_visible(handbox(f))
# Viewing/locking a candidate is not a manual geometry edit.
box=handbox(6);box['human_verified']=True;box['locked']=True
w._bump_bbox_revision();w._bump_query_state_revision()
assert not w._event_box_visible(box)
assert {m['frame'] for m in w._bbox_frame_markers()}=={12}
w.event_draft['Left_hand']['required_review_frames']=[6]
assert w._event_box_visible(box)
assert {m['frame'] for m in w._bbox_frame_markers()}=={12}
# Editing one candidate creates one tick and remains visible between SOE frames.
box=handbox(13);box['source']='manual_box_edit';box['x1']=2
w._bump_bbox_revision();w._bump_query_state_revision()
assert {m['frame'] for m in w._bbox_frame_markers()}=={12,13}
assert w._event_box_visible(box)
w._refresh_boxes_for_frame(14)
assert not any(w._normalize_hand_label(b.get('label')) for b in w.player.overlay_boxes)
assert not w._frame_has_selected_hand_box(14, 'Left_hand')
assert not w.player.overlay_relations
w._refresh_boxes_for_frame(13)
assert any(w._normalize_hand_label(b.get('label')) for b in w.player.overlay_boxes)
# Export/reload keeps dense candidates, manual provenance and sparse display.
out=copy.deepcopy(w._build_payload_v2());w._load_annotations_v2(out)
w._set_selected_event(w.events[0]['event_id'],'Left_hand')
assert len([b for b in w.raw_boxes if b['label']=='Left_hand'])==20000
assert {m['frame'] for m in w._bbox_frame_markers()}=={12,13}
assert not w._event_box_visible(handbox(14))
# Review changes must not rescan dense geometry or rebuild all event ticks.
class CountedRows(list):
    scans = 0
    def __iter__(self):
        self.scans += 1
        return super().__iter__()
w.raw_boxes = CountedRows(w.raw_boxes)
w._manual_bbox_index()
w._frame_review_issues()
scans = w.raw_boxes.scans
from unittest.mock import patch
with patch.object(w, '_bbox_frame_markers', side_effect=AssertionError('Hand display rebuilt all event markers')):
    for _ in range(50):
        w._bump_query_state_revision()
        w._frame_review_issues()
        assert not w._event_box_visible(handbox(14))
# handbox() intentionally iterates; the indexed checks must add no scans.
assert w.raw_boxes.scans == scans + 50
w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.deleteLater();app.processEvents()
print('DENSE_CANDIDATES_MANUAL_ONLY_TICKS_SELECTED_HAND_FRAMES_ROUNDTRIP_PASS')
