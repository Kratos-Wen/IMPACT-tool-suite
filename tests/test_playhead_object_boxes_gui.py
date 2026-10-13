"""Object/tool boxes follow fallback relations outside the selected event."""
import copy
import os
import sys
from pathlib import Path

os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication, QMessageBox
from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(noun_classes=['part','tool'],verbs=['hold'],anomaly_labels=[])
from ui.hoi_window import HOIWindow

app=QApplication([])
for name in ('information','warning','critical','question'):
    setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)
w=HOIWindow();w._autosave_timer.stop()
w.player.frame_count=40;w.player.frame_rate=15
w.global_object_map={'part':0,'tool':8,'part_2':2,'tool_2':9,'unrelated':99}
def hand(start,end,noun,tool):
    return dict(interaction_start=start,functional_contact_onset=start,interaction_end=end,
                verb='hold',noun_object_id=noun,instrument_object_id=tool)
w.events=[dict(event_id=1,hoi_data={'Right_hand':hand(10,20,0,8)}),
          dict(event_id=2,hoi_data={'Right_hand':hand(21,30,2,9)})]
w.selected_event_id=1;w.selected_hand_label='Right_hand'
w.event_draft=copy.deepcopy(w.events[0]['hoi_data'])
w.raw_boxes=[dict(id=uid,orig_frame=frame,label=label,x1=1,y1=2,x2=10,y2=20)
             for frame in (10,21,31)
             for uid,label in ((100,'Right_hand'),(0,'part'),(8,'tool'),(2,'part_2'),(9,'tool_2'),(99,'unrelated'))]
w._rebuild_bboxes_from_raw()
before=copy.deepcopy(w.event_draft)
for frame,expected in ((10,{'0','8'}),(21,{'2','9'}),(31,set())):
    w.player.current_frame=frame
    assert w._event_visible_object_ids(frame)==expected
    w._refresh_boxes_for_frame(frame,skip_events=True,lightweight=True)
    visible={str(box['id']) for box in w.player.overlay_boxes if box['id']!=100}
    assert visible==expected,(frame,visible)
    editable={str(box['id']) for box in w.player.edit_boxes if box['id']!=100}
    assert editable==expected,(frame,editable)
    assert w.event_draft==before and w.selected_event_id==1
w._stop_autosave();w.deleteLater()
print('PLAYHEAD_NOUN_INSTRUMENT_BOXES_AND_EDIT_CONTEXT_PASS')
