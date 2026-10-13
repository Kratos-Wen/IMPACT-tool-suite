"""Objects at the playhead remain visible after leaving the selected event."""
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
PROFILE.clear();PROFILE.update(noun_classes=['part','assembly','tool'],verbs=['hold'],anomaly_labels=[])
from ui.hoi_window import HOIWindow

app=QApplication([])
for name in ('information','warning','critical','question'):
    setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)
w=HOIWindow();w._autosave_timer.stop()
w.player.frame_count=1600;w.player.frame_rate=15
w.global_object_map={'part':0,'assembly':101,'tool':8,'unrelated':99}
def hand(start,end,noun,tool=None):
    return dict(interaction_start=start,functional_contact_onset=start,interaction_end=end,
                verb='hold',noun_object_id=noun,target_object_id=noun,instrument_object_id=tool)
w.events=[dict(event_id=104,hoi_data={'Right_hand':hand(630,1280,0)}),
          dict(event_id=23,hoi_data={'Left_hand':hand(1277,1532,101)}),
          dict(event_id=105,hoi_data={'Right_hand':hand(1297,1322,101,8)})]
w.selected_event_id=104;w.selected_hand_label='Right_hand'
w.event_draft=copy.deepcopy(w.events[0]['hoi_data'])
w.raw_boxes=[dict(id=uid,orig_frame=frame,label=label,source='manual_box_add',x1=1,y1=2,x2=10,y2=20)
             for frame in (1270,1280,1298,1350,1540)
             for uid,label in ((-2,'Right_hand'),(0,'part'),(101,'assembly'),(8,'tool'),(99,'unrelated'))]
w._rebuild_bboxes_from_raw()
before=copy.deepcopy(w.event_draft)
for frame,expected in ((1270,{'0'}),(1280,{'0'}),(1298,{'101','8'}),(1350,{'101'}),(1540,set())):
    w.player.current_frame=frame
    assert w._event_visible_object_ids(frame)==expected
    w._refresh_boxes_for_frame(frame,skip_events=True,lightweight=True)
    visible={str(box['id']) for box in w.player.overlay_boxes if box['id']!=-2}
    assert visible==expected,(frame,visible)
    editable={str(box['id']) for box in w.player.edit_boxes if box['id']!=-2}
    assert editable==expected,(frame,editable)
    assert w.event_draft==before and w.selected_event_id==104
w._stop_autosave();w.deleteLater()
print('FRAME_1298_UNSELECTED_ASSEMBLY_INSTRUMENT_DISPLAY_AND_EDIT_PASS')
