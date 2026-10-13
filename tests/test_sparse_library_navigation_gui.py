"""Sparse imports keep vocabulary; navigation does not validate unchanged boundaries."""
import copy
import os
import sys
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication, QMessageBox
from core.project_profile import PROFILE
PROFILE.clear()
PROFILE.update(verbs=['pick_up', 'hold', 'receive'], noun_classes=['part', 'tool', 'assembly'],
               create_default_noun_instances=True, anomaly_labels=[])
app = QApplication([])
for name in ('information', 'warning', 'critical', 'question'):
    setattr(QMessageBox, name, lambda *a, **k: QMessageBox.No)
from ui.hoi_window import HOIWindow
w = HOIWindow()
w._autosave_timer.stop()
w.player.frame_count = 100
w.player.frame_rate = 15
payload = dict(schema='hoi-annotation', video_id='fixture', frame_count=100, fps=15,
               object_library={'8': dict(label='part', category='part')}, tracks={},
               hoi_events={'left_hand': [
                   dict(start_frame=0, contact_onset_frame=0, end_frame=30, verb='hold', noun_object_id=8),
                   dict(start_frame=30, contact_onset_frame=30, end_frame=60, verb='hold', noun_object_id=8)],
                   'right_hand': []})
for library in (None, {}, {'7': 'hold'}):
    data = copy.deepcopy(payload)
    if library is not None:
        data['verb_library'] = library
    w._load_annotations_v2(data)
    ev = w.events[0]
    w._set_selected_event(ev['event_id'], 'Left_hand')
    assert {'pick_up', 'hold', 'receive'} <= {v.name for v in w.verbs}
    assert w.combo_inline_verb.findData('receive') >= 0
    assert w.combo_inline_verb.currentData() == 'hold'
    assert w.combo_inline_noun.findData(8) >= 0
    assert w.combo_inline_noun.currentData() == 8
    assert len([uid for name, uid in w.global_object_map.items() if name == 'tool']) == 1
    events = copy.deepcopy(w.events)
    undo_depth = len(w._hoi_undo_stack)
    with patch.object(QMessageBox, 'warning') as warning:
        w._on_hoi_timeline_update(ev['event_id'], 'Left_hand', 0, 30, 0)
        assert not warning.called, 'Clicking an unchanged imported event warned about overlap'
        assert w.events == events and len(w._hoi_undo_stack) == undo_depth
        w._on_hoi_timeline_update(ev['event_id'], 'Left_hand', 0, 31, 0)
        assert warning.call_count == 1, 'Actual same-hand overlap must still be rejected'
        assert w.events == events and len(w._hoi_undo_stack) == undo_depth
    stable_ids = copy.deepcopy(w.global_object_map)
    w._apply_profile_libraries()
    assert w.global_object_map == stable_ids
# A hands-only file with no events or object library retains usable project choices.
w._load_annotations_v2(dict(schema='hoi-annotation', tracks={}, hoi_events={}, verb_library={}))
assert {'pick_up', 'hold', 'receive'} <= {v.name for v in w.verbs}
assert {'part', 'tool'} <= set(w.global_object_map)
w._stop_autosave()
w._mark_hoi_saved()
w._close_request_approved = True
w.close()
w.deleteLater()
app.processEvents()
print('SPARSE_VOCABULARY_NOOP_BOUNDARY_GUARD_PASS')
