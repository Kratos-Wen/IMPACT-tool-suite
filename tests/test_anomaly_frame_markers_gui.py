"""Anomaly markers appear without boxes and navigate without editing events."""
import copy
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMessageBox
from core.project_profile import PROFILE

PROFILE.clear()
PROFILE.update(verbs=['hold'], noun_classes=['part'], anomaly_labels=[])
from ui.hoi_window import HOIWindow

app = QApplication([])
for name in ('information', 'warning', 'critical', 'question'):
    setattr(QMessageBox, name, lambda *a, **k: QMessageBox.No)

with tempfile.TemporaryDirectory() as directory:
    os.environ['IMPACT_HOI_SETTINGS_DIR'] = directory
    w = HOIWindow()
    w._autosave_timer.stop()
    w.player.frame_count = 200
    w.player.frame_rate = 15
    payload = dict(schema='hoi-annotation', video_id='markers', frame_count=200, fps=15,
        object_library={'1': dict(label='part', category='part')}, tracks={},
        hoi_events={'left_hand': [dict(event_id='left', verb='hold', start_frame=10,
            contact_onset_frame=20, end_frame=100, noun_object_id=1)], 'right_hand': []})
    w._load_annotations_v2(payload)
    event_id = w.events[0]['event_id']
    w._set_selected_event(event_id, 'Left_hand')
    row = w.hoi_timeline.actor_rows['Left_hand']
    row.resize(900, 48)
    w.hoi_timeline._set_view(0, 120)
    assert not row._segments()[0]['anomaly_frames']

    w.player.current_frame = 40
    w._require_current_frame()
    assert row._segments()[0]['anomaly_frames'] == (40,)
    assert not w.raw_boxes, 'Adding a marker must not fabricate boxes'
    assert w.hoi_timeline.actor_rows['Right_hand']._hit_box_marker(row.frame_to_x(40), 37) is None

    # Even SOE-coincident anomaly frames have their own bottom marker.
    w.player.current_frame = 20
    w._require_current_frame()
    assert row._segments()[0]['anomaly_frames'] == (20, 40)
    assert row._hit_box_marker(row.frame_to_x(20), 9) is None
    assert row._hit_box_marker(row.frame_to_x(20), 37)['frame'] == 20
    w._hoi_undo()
    assert row._segments()[0]['anomaly_frames'] == (40,)

    saved = copy.deepcopy(w._build_payload_v2())
    assert saved['hoi_events']['left_hand'][0]['required_review_frames'] == [40]
    w._load_annotations_v2(saved)
    event_id = w.events[0]['event_id']
    row.set_selected(None)
    assert row._segments()[0]['anomaly_frames'] == (40,)
    marker = row._hit_box_marker(row.frame_to_x(40), 37)
    assert marker['frame'] == 40 and 'Anomaly frame 40' == marker['tooltip']
    before = copy.deepcopy(w.events)
    jumps = []
    def seek(frame):
        jumps.append(frame)
        w.player.current_frame = frame
    w.player.seek = seek
    QTest.mouseClick(row, Qt.LeftButton, pos=QPoint(row.frame_to_x(40), 37))
    assert jumps[-1] == 40 and w.selected_event_id == event_id
    assert w.events == before and not row._dragging
    QTest.mouseDClick(row, Qt.LeftButton, pos=QPoint(row.frame_to_x(40), 37))
    QTest.mouseRelease(row, Qt.LeftButton, pos=QPoint(row.frame_to_x(40), 37))
    assert w.events == before and not row._dragging

    # Markers are visible when unselected and don't spill into the gutter on pan.
    row.set_selected(None)
    preview = row.grab().toImage()
    x = row.frame_to_x(40)
    scale = preview.devicePixelRatio()
    assert preview.pixelColor(round((x + 2) * scale), round(37 * scale)).name() == '#7e22ce'
    preview.save('/tmp/impact-anomaly-frame-marker.png')
    w.hoi_timeline._set_view(50, 100)
    assert not row._anomaly_marker_positions(row._segments()[0])
    assert row._hit_box_marker(row.frame_to_x(40), 37) is None
    w._stop_autosave()
    w.deleteLater()

print('ANOMALY_FRAME_MARKERS_ADD_UNDO_SAVE_LOAD_UNSELECTED_CLICK_SOE_CLIP_PASS')
