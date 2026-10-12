"""Exercise frame caching, preview coalescing and review-index invalidation."""
import copy
import os
import sys
import tempfile
import time
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest
from ui.hoi_window import HOIWindow
from ui.video_player import VideoPlayer

app = QApplication([])
for name in ('information', 'warning', 'critical', 'question'):
    setattr(QMessageBox, name, lambda *a, **k: QMessageBox.No)


def wait_until(condition, milliseconds=3000):
    start = time.monotonic()
    while not condition() and (time.monotonic() - start) * 1000 < milliseconds:
        QTest.qWait(10)
    assert condition(), 'Timed out waiting for Qt preview processing'


def drain_events():
    # Confirm a later Qt timer actually ran before checking cancelled work.
    done = []
    QTimer.singleShot(100, lambda: done.append(True))
    wait_until(lambda: done)


class Capture:
    def __init__(self):
        self.position = 0
        self.reads = 0
        self.seeks = 0

    def set(self, key, frame):
        assert key == cv2.CAP_PROP_POS_FRAMES
        self.position = int(frame)
        self.seeks += 1
        return True

    def read(self):
        self.reads += 1
        image = np.full((32, 32, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, image

    def release(self):
        pass


class CountedRows(list):
    def __init__(self, rows):
        super().__init__(rows)
        self.scans = 0

    def __iter__(self):
        self.scans += 1
        return super().__iter__()


w = HOIWindow()
w._autosave_timer.stop()
w.raw_boxes = CountedRows([dict(id=0, orig_frame=i, label='part', x1=1, y1=2, x2=3, y2=4)
                          for i in range(20000)])
raw = w.raw_boxes[10]
assert w._boxes_for_review('O:0', 10)[0] is raw
scans = w.raw_boxes.scans
for frame in range(100):
    assert w._boxes_for_review('O:0', frame)[0] is w.raw_boxes[frame]
assert w.raw_boxes.scans == scans, 'Playback rescanned full raw geometry'
w._boxes_for_review('O:0', 10)[0]['human_verified'] = True
assert raw['human_verified']
raw.update(orig_frame=30000, label='Left_hand')
w._rebuild_bboxes_from_raw()
assert not w._boxes_for_review('O:0', 10)
assert w._boxes_for_review('H:Left_hand', 30000) == [raw]
w.start_offset = 7
assert w._boxes_for_review('H:Left_hand', 30007) == [raw]
assert not w._boxes_for_review('H:Left_hand', 30000)
w.start_offset = 0
w._label_suggestions()
scans = w.raw_boxes.scans
for _ in range(100):
    assert 'part' in w._label_suggestions()
assert w.raw_boxes.scans == scans, 'Playback rebuilt all label suggestions'
w.raw_boxes.append(dict(id=2, orig_frame=1, label='new_part', x1=1, y1=2, x2=3, y2=4))
assert 'new_part' in w._label_suggestions()
w.raw_boxes[-1]['label'] = 'renamed_part'
w._bbox_revision += 1
assert 'renamed_part' in w._label_suggestions() and 'new_part' not in w._label_suggestions()
w.global_object_map = {'first_name': 2}
w._label_suggestions()
w.global_object_map = {'new_name': 2}
assert 'new_name' in w._label_suggestions() and 'first_name' not in w._label_suggestions()
# Replacement with the same length must never reuse another task's index.
w.raw_boxes = [dict(b, id=9) for b in w.raw_boxes]
assert not w._boxes_for_review('O:0', 5)
assert w._boxes_for_review('O:9', 5)
w.raw_boxes = []
w._rebuild_bboxes_from_raw()

player = w.player
capture = Capture()
player.cap = capture
player._media_path = 'synthetic.mp4'
preview_calls = []
def preview_decode(path, frame):
    preview_calls.append(frame)
    return np.full((32, 32, 3), frame, dtype=np.uint8)
player._preview_decoder.decode = preview_decode
player.frame_count = 120
player.crop_start, player.crop_end = 0, 119
player.seek(20)
reads, seeks = capture.reads, capture.seeks
player.seek(20)
assert (capture.reads, capture.seeks) == (reads, seeks)
player.seek(21)
assert capture.seeks == seeks and int(player.get_current_frame_bgr()[0, 0, 0]) == 21
player.seek(70, preview_only=True)
assert player.current_frame == 21
player.play()
player.timer.stop()
player._on_timer()
assert player.current_frame == 22 and int(player.get_current_frame_bgr()[0, 0, 0]) == 22
player.pause()
copy_frame = player.get_current_frame_bgr()
copy_frame[:] = 255
assert int(player.get_current_frame_bgr()[0, 0, 0]) == 22
player._frame_cache_budget = 2 * 32 * 32 * 3
for frame in range(30, 50):
    player.seek(frame)
    assert player._decoded_frame_bytes <= player._frame_cache_budget
assert len(player._decoded_frames) <= 2

# Mouse movement coalesces to one preview and never refreshes all event panels.
w._refresh_events = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Full event refresh during hover'))
player.seek(0)
reads = capture.reads
for frame in range(1, 71):
    w._on_hoi_timeline_hover(frame)
assert capture.reads == reads
wait_until(lambda: player._displayed_frame == 70)
assert capture.reads == reads and preview_calls == [70] and player._displayed_frame == 70 and player.current_frame == 0
w._on_hoi_timeline_hover(40)
w._on_hoi_timeline_hover(-1)
drain_events()
assert player._displayed_frame == 0 and player.current_frame == 0
w._on_hoi_timeline_hover(30)
player.seek(90)
drain_events()
assert player._displayed_frame == 90 and player.current_frame == 90
# Release commits the precise slider frame and clears any delayed preview.
w._refresh_events = lambda *a, **k: None
w.slider.setRange(0, 119)
player.play()
w._on_slider_pressed()
assert not player.is_playing
for frame in range(1, 80):
    w._on_slider_changed(frame)
w.slider.blockSignals(True)
w.slider.setValue(80)
w.slider.blockSignals(False)
w._on_slider_released()
drain_events()
assert player.current_frame == player._displayed_frame == 80
player.release_media()
assert not player._decoded_frames and player._decoded_frame_bytes == 0
w._stop_autosave()
w.deleteLater()

# Validate actual decoder pixels through random seek, preview, cache and replay.
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'video.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 15, (64, 64))
    assert writer.isOpened()
    for frame in range(40):
        writer.write(np.full((64, 64, 3), frame * 5, np.uint8))
    writer.release()
    reference = cv2.VideoCapture(str(path))
    p = VideoPlayer()
    assert p.load(str(path))
    for frame in (1, 2, 20, 19, 20, 4, 5, 4, 35):
        p.seek(frame)
        reference.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, expected = reference.read()
        assert ok and np.array_equal(p.get_current_frame_bgr(), expected)
    p.seek(20, preview_only=True)
    p.play()
    p.timer.stop()
    p._on_timer()
    reference.set(cv2.CAP_PROP_POS_FRAMES, 36)
    ok, expected = reference.read()
    assert p.current_frame == 36 and ok and np.array_equal(p.get_current_frame_bgr(), expected)
    p.release_media()
    assert p.load(str(path)) and p.current_frame == p._displayed_frame == 0
    p.release_media()
    reference.release()
    p.deleteLater()
print('EDITOR_FRAME_INDEX_LABEL_CACHE_PREVIEW_COALESCING_NATIVE_PIXEL_REPLAY_PASS')
