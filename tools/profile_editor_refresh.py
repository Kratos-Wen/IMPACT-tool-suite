"""Profile editor callbacks on a recovery copy without touching project files."""
import argparse
import cProfile
import json
import os
import pstats
import sys
import tempfile
from pathlib import Path
from time import perf_counter

parser = argparse.ArgumentParser()
parser.add_argument('--recovery', required=True)
parser.add_argument('--profile', required=True)
args = parser.parse_args()
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['IMPACT_PROJECT_PROFILE'] = str(Path(args.profile).resolve())
os.environ['IMPACT_RESTORE_LAST_PROJECT'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
with tempfile.TemporaryDirectory(prefix='impact-profile-') as directory:
    os.environ['IMPACT_HOI_SETTINGS_DIR'] = directory
    from app import _bootstrap_qt_runtime
    _bootstrap_qt_runtime()
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from PyQt5.QtCore import QTimer
    from ui.hoi_window import HOIWindow
    app = QApplication([])
    for name in ('information', 'warning', 'critical', 'question'):
        setattr(QMessageBox, name, lambda *a, **k: QMessageBox.No)
    window = HOIWindow()
    for timer in window.findChildren(QTimer):
        timer.stop()
    window._set_experiment_mode_key('manual')
    document = json.loads(Path(args.recovery).read_text())
    state = document['state']
    window.player.frame_count = document['identity']['frames']
    window.player.frame_rate = document['identity']['fps']
    window.start_offset = state.get('start_offset', 0)
    window.end_frame = state.get('end_frame')
    profiler = cProfile.Profile()
    profiler.enable()
    for name, operation in (
        ('restore', lambda: window._restore_state(state)),
        ('undo snapshot', window._push_undo),
        ('refresh events', window._refresh_events),
        ('autosave snapshot', window._autosave_snapshot),
        ('frame refresh', lambda: window._refresh_boxes_for_frame(state['current_frame'])),
    ):
        begin = perf_counter()
        operation()
        print(f'{name}: {perf_counter()-begin:.3f}s', flush=True)
    profiler.disable()
    pstats.Stats(profiler).strip_dirs().sort_stats('cumtime').print_stats(25)
    candidates = [box for box in window.raw_boxes
                  if box.get('orig_frame') == state['current_frame'] - window.start_offset]
    if candidates:
        box = dict(candidates[0])
        window.player.current_frame = state['current_frame']
        window._selected_edit_box = dict(box)
        edited = dict(box, frame=state['current_frame'], _action='update', x1=box['x1']+1)
        begin = perf_counter()
        window._on_box_edited(box['id'], edited)
        print(f'actual box edit (no profiling overhead): {perf_counter()-begin:.3f}s', flush=True)
    window._stop_autosave()
    window.deleteLater()
