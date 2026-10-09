"""Read-only guide, profile switching and nested signal blocking regressions."""
import copy, os, sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox, QLineEdit, QComboBox, QTableWidget, QDialogButtonBox, QTabWidget
app = QApplication([])
for name in ('information', 'question', 'warning'):
    setattr(QMessageBox, name, lambda *a, **k: QMessageBox.No)
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
from core.label_glossary import explanation, glossary_rows
PROFILE.clear(); PROFILE.update(
    noun_classes=['part', 'tool'], verbs=['hold', 'place', 'insert'],
    noun_aliases={'old_part': 'part'}, anomaly_labels=['attribute_a'],
    noun_explanations={'part': {'en': 'Individual target part.', 'zh': '独立零件。'}, 'tool': {'en': 'Operating tool.', 'zh': '操作工具。'}},
    verb_explanations={'hold': {'en': 'Independent support.', 'zh': '独立支撑。'}, 'place': {'en': 'Controlled placement.', 'zh': '受控放置。'}, 'insert': {'en': 'Feed into a hole.', 'zh': '送入孔中。'}})
refresh_aliases()
assert explanation('[0] old_part_1', 'noun') == 'Individual target part.'
assert explanation('[0] OLD_PART_1', 'noun') == 'Individual target part.'
assert explanation('part', 'noun', 'zh') == '独立零件。'
assert dict(glossary_rows('verb', ['custom_action']))['custom_action'].startswith('Added label:')
from ui.hoi_window import HOIWindow
w = HOIWindow(); w._autosave_timer.stop(); w.player.frame_count = 60; w.player.frame_rate = 15
payload = dict(video_id='trial', frame_count=60, fps=15,
    object_library={'0': dict(label='part_1', category='part'), '8': dict(label='tool_1', category='tool')},
    tracks={}, hoi_events={'left_hand': [dict(event_id='left', start_frame=2, contact_onset_frame=2, end_frame=10,
        verb='hold', noun_object_id=0, instrument_object_id=8, anomaly_labels=[], anomaly_review_state='reviewed')],
        'right_hand': [dict(event_id='right', start_frame=3, contact_onset_frame=4, end_frame=12,
        verb='insert', noun_object_id=0, instrument_object_id=None, anomaly_labels=[], anomaly_review_state='reviewed')]})
w._load_annotations_v2(payload); w._apply_profile_libraries()
left_id = next(e['event_id'] for e in w.events if e['hoi_data']['Left_hand'].get('verb'))
w._set_selected_event(left_id, 'Left_hand'); app.processEvents()
baseline = copy.deepcopy(w.events)
for cycle in range(30):
    w._update_verb_combo(); w._load_hand_draft_to_ui('Left_hand')
    w._refresh_label_explanations(); w._update_inline_event_editor(); w._refresh_events(); app.processEvents()
    assert w.events == baseline, 'Read-only refresh changed event data'
assert w.combo_target.itemData(w.combo_target.findData(0), Qt.ToolTipRole) == 'Individual target part.'
def inspect_dialog():
    dialog = app.activeModalWidget(); tabs = dialog.findChild(QTabWidget); tables = [tabs.widget(i) for i in range(tabs.count())]
    assert len(tables) == 2 and all(table.editTriggers() == QTableWidget.NoEditTriggers for table in tables)
    search = dialog.findChild(QLineEdit); search.setText('controlled placement')
    assert sum(not tables[1].isRowHidden(r) for r in range(tables[1].rowCount())) == 1
    search.clear(); dialog.findChild(QComboBox).setCurrentIndex(1)
    assert any(tables[0].item(r, 1).text() == '独立零件。' for r in range(tables[0].rowCount()))
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Close).click()
QTimer.singleShot(0, inspect_dialog); w._open_label_guide()
assert w.events == baseline
controls = [w.combo_verb, w.combo_target, w.combo_instrument, w.combo_anomaly, w.combo_inline_verb, w.combo_inline_noun, w.combo_inline_instrument, w.combo_inline_anomaly]
for combo in controls: combo.blockSignals(True)
w._update_verb_combo(); w._load_hand_draft_to_ui('Left_hand')
assert all(combo.signalsBlocked() for combo in controls), 'Nested refresh released the outer signal guard'
for combo in controls: combo.blockSignals(False)
# ID 0 is a valid confirmed noun, including when the verb is edited.
w._set_hand_field_state(w.event_draft['Left_hand'], 'noun_object_id', source='manual', status='confirmed')
w._apply_draft_to_selected_event(); w.combo_verb.setCurrentText('place')
assert w.event_draft['Left_hand']['noun_object_id'] == 0
assert w.event_draft['Left_hand']['_field_state']['noun_object_id']['status'] == 'confirmed'
right = next(e['hoi_data']['Right_hand'] for e in w.events if e['hoi_data']['Right_hand'].get('verb'))
assert right['verb'] == 'insert' and right['instrument_object_id'] is None
PROFILE['noun_explanations']['part']['en'] = 'Updated project definition.'
w._refresh_label_explanations()
assert w.combo_target.itemData(w.combo_target.findData(0), Qt.ToolTipRole) == 'Updated project definition.'
w._register_object_entry(50, 'Left_hand_1')
w._register_object_entry(51, 'hand')
assert w.combo_target.findData(50) == w.combo_instrument.findData(50) == -1
assert w.combo_target.findData(51) == -1
w._rebuild_object_combos(0, 8)
assert w.combo_target.findData(50) == -1 and w.combo_instrument.findData(51) == -1
assert w.global_object_map['Left_hand_1'] == 50, 'Filtering erased the source registry'
PROFILE.update(deprecated_verbs=['old_action'], ambiguous_nouns=['unresolved_object'])
w._register_object_entry(52, 'unresolved_object_1')
assert 'identify Object category' in w._policy_missing(dict(verb='hold',noun_object_id=52,anomaly_label='normal'))
assert 'replace legacy verb' in w._policy_missing(dict(verb='old_action',noun_object_id=0,anomaly_label='normal'))
assert 'hand used as Object/instrument' in w._policy_missing(dict(verb='hold',noun_object_id=50,anomaly_label='normal'))
w._stop_autosave(); w.close()
print('LABEL_GUIDE_PROFILE_REFRESH_NESTED_SIGNALS_AND_ZERO_ID_PASS')
