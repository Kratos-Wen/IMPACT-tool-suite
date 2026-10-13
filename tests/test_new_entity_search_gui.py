"""New manual entities are immediately editable; fuzzy search never overwrites exact IDs."""
import os,sys,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QInputDialog
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(noun_classes=['screw_long','screw_short'],verbs=['hold'],anomaly_labels=[])
from core.label_search import ranked_labels
assert ranked_labels('scre_logn',['screw_short','screw_long'])[0]=='screw_long'
assert ranked_labels('short',['screw_long','screw_short'])==['screw_short']
assert ranked_labels('screw short',['screw_long','screw_short'])[0]=='screw_short'
assert ranked_labels('12',['screw_long'])==[]
app=QApplication([])
for name in ('information','warning','critical','question'):setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)
from ui.hoi_window import HOIWindow
w=HOIWindow();w._autosave_timer.stop();w.player.frame_count=100;w.player.frame_rate=15
w._load_annotations_v2(dict(schema='hoi-annotation',video_id='fixture',frame_count=100,fps=15,
 object_library={'5':dict(label='screw_long',category='screw_long'),'9':dict(label='screw_short',category='screw_short')},tracks={},
 hoi_events={'left_hand':[dict(start_frame=0,contact_onset_frame=0,end_frame=20,verb='hold',noun_object_id=5)],'right_hand':[]}))
w._set_selected_event(w.events[0]['event_id'],'Left_hand');w.rad_draw_none.setChecked(True)
QInputDialog.getItem=lambda parent,title,text,options,*args:(options[-1],True)
w._pending_draw_box_role=None
w._on_box_edited(None,dict(_action='add',label='screw_long',frame=0,x1=1,y1=2,x2=8,y2=9))
uid=w.raw_boxes[-1]['id']
assert uid !=5 and w._hand_noun_object_id(w.event_draft['Left_hand'])==5
assert any(b['id']==uid for b in w.player.overlay_boxes),'Unbound manual new entity disappeared'
assert any(b['id']==uid for b in w.player.edit_boxes),'New box is not editable'
# Binding the new instance makes it visible as an event entity, including after reload.
w.combo_inline_noun.setCurrentIndex(w.combo_inline_noun.findData(uid))
assert w._hand_noun_object_id(w.event_draft['Left_hand'])==uid
assert any(b['id']==uid for b in w.player.overlay_boxes)
saved=copy.deepcopy(w._build_payload_v2());w._load_annotations_v2(saved)
w._set_selected_event(w.events[0]['event_id'],'Left_hand');w._refresh_boxes_for_frame(0)
assert any(b['id']==uid for b in w.player.overlay_boxes)
# Drawing with a noun role updates binding AND redraws immediately.
w.event_draft['Left_hand'].update(noun_object_id=None,target_object_id=None)
w._pending_draw_box_role='noun'
w._on_box_edited(None,dict(_action='add',label='screw_short',frame=0,x1=12,y1=2,x2=19,y2=9))
new_id=w.raw_boxes[-1]['id']
assert w._hand_noun_object_id(w.event_draft['Left_hand'])==new_id
assert any(b['id']==new_id for b in w.player.overlay_boxes)
# Instance dropdown suggestions retain the ID even when names are identical.
combo=w.combo_inline_noun
combo.lineEdit().setText('scre_logn');combo.lineEdit().textEdited.emit('scre_logn')
model=combo.completer().completionModel()
choices=[model.index(i,0).data() for i in range(model.rowCount())]
assert f'[{uid}] screw_long' in choices
combo.completer().activated[str].emit(f'[{uid}] screw_long')
assert w._hand_noun_object_id(w.event_draft['Left_hand'])==uid
# Real label-entry dialog: partial names, typos, exact labels and numeric IDs.
w.player.edit_label_suggestions=['screw_long','screw_short']
dlg=w.player._rename_dialog('',title='New box');dlg.show();app.processEvents()
for text,expected in [('short','screw_short'),('scre_logn','screw_long'),('screw short','screw_short')]:
 dlg.line.setText(text);dlg.line.textEdited.emit(text)
 assert dlg.line.completer().completionModel().index(0,0).data()==expected
 dlg.line.apply_completion();assert dlg.line.text() in ({text,expected} if text=='screw short' else {text}), ('Focus/OK silently chose a fuzzy candidate',text,dlg.line.text())
 dlg.line.apply_completion(accept_suggestion=True);assert dlg.line.text()==expected
# A stale popup must never turn an exact short label into a long one.
dlg.line.setText('long');dlg.line.textEdited.emit('long');dlg.line.setText('screw_short')
assert w.player._resolve_label_input(dlg)[0]=='screw_short'
for text in (str(uid),f'[{uid}] screw_long'):
 dlg.line.setText(text)
 assert w.player._resolve_label_input(dlg)[0]==text
dlg.line.setText('custom_screw_long')
assert w.player._resolve_label_input(dlg)[0]=='custom_screw_long'
# Tab accepts the current ranked suggestion through the actual key event.
dlg.line.setText('scre_logn');dlg.line.textEdited.emit('scre_logn');dlg.line.setFocus()
QTest.keyClick(dlg.line,Qt.Key_Tab);assert dlg.line.text()=='screw_long'
dlg.reject();w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close()
print('NEW_ENTITY_VISIBLE_EDITABLE_BIND_RELOAD_FUZZY_TAB_EXACT_LABEL_INSTANCE_ID_PASS')
