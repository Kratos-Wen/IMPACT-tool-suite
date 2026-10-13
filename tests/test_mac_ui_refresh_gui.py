"""Light controls remain readable on dark desktops; repaint caches invalidate."""
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime, _configure_app_font
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication, QMessageBox, QDialog, QComboBox
from PyQt5.QtGui import QColor, QPalette
from ui.hoi_window import HOIWindow

app = QApplication([])
for name in ('information', 'warning', 'critical', 'question'):
    setattr(QMessageBox, name, lambda *a, **k: QMessageBox.No)
_configure_app_font(app)
dark = app.palette()
for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
    dark.setColor(role, QColor('white'))
dark.setColor(QPalette.Window, QColor('#202020'))
app.setPalette(dark)
w = HOIWindow()
w._autosave_timer.stop()
w.resize(1400, 900)
w.show()
app.processEvents()
for widget in (w.combo_target, w.combo_instrument, w.spin_jump, w.btn_inline_edit_boxes):
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        assert widget.palette().color(role).lightness() < 160, (widget, role)
w.grab().save('/tmp/impact-mac-ui.png')
dialog = QDialog(w)
field = QComboBox(dialog)
dialog.ensurePolished()
field.ensurePolished()
assert dialog.palette().color(QPalette.Window).lightness() > 200
assert field.palette().color(QPalette.Text).lightness() < 160
w._frame_review_issues()
review_key = w._frame_review_cache_key
w._frame_review_issues()
assert review_key == w._frame_review_cache_key

w.events = [dict(event_id=i, hoi_data={'Left_hand': dict(
    interaction_start=i*10, interaction_end=i*10+9,
    functional_contact_onset=i*10+2, verb='hold')}) for i in range(200)]
original = w._hand_completion_state
calls = []
def completion(hand, hand_key=None):
    calls.append(hand)
    return original(hand, hand_key=hand_key)
w._hand_completion_state = completion
begin = perf_counter()
first = w._hoi_segments_for_hand('Left_hand')
cold = perf_counter()-begin
assert len(calls) == 200
begin = perf_counter()
for _ in range(20):
    assert w._hoi_segments_for_hand('Left_hand') == first
warm = (perf_counter()-begin)/20
assert len(calls) == 200
w.events[0]['hoi_data']['Left_hand']['verb'] = 'pick_up'
w._hoi_segments_for_hand('Left_hand')
assert len(calls) == 400
w.frame_review = dict(getattr(w, 'frame_review', {}))
w._hoi_segments_for_hand('Left_hand')
assert len(calls) == 600
w._bbox_revision += 1
w._hoi_segments_for_hand('Left_hand')
assert len(calls) == 800
w._stop_autosave()
w._mark_hoi_saved()
w.deleteLater()
print(f'DARK_PALETTE_READABILITY_AND_CACHE_PASS cold={cold:.4f}s warm={warm:.4f}s')
