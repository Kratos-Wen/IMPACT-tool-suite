"""Discover local hand XML and draw missing hand/noun/instrument in order."""
import copy
import os
import sys
import tempfile
from pathlib import Path

os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication, QMessageBox, QInputDialog
from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(noun_classes=['screw_long','tool'],verbs=['align'],anomaly_labels=[])
from ui.hoi_window import HOIWindow

app=QApplication([])
warnings=[]
QMessageBox.warning=lambda *a,**k:(warnings.append(str(a[2])),QMessageBox.Ok)[1]
for name in ('information','critical','question'):
    setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)
QInputDialog.getItem=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Automatic sequence prompted for hand/instance'))
w=HOIWindow();w._autosave_timer.stop()
w.player.frame_count=30;w.player.frame_rate=15
w._set_experiment_mode_key('manual')
w.global_object_map={'screw_long':0,'tool':8}
w.id_to_category={'screw_long':'screw_long','tool':'tool'}
w._rebuild_object_combos()
w.events=[dict(event_id=1,hoi_data={'Left_hand':dict(verb='align',interaction_start=2,
    functional_contact_onset=3,interaction_end=5,noun_object_id=0,target_object_id=0,
    instrument_object_id=8,tool_object_id=8,anomaly_label='normal')})]
w.player.current_frame=2
w._set_selected_event(1,'Left_hand')
with tempfile.TemporaryDirectory() as directory:
    root=Path(directory);video=root/'sample.mp4'
    xml=root/'hand.xml'
    xml.write_text('<annotations><track label="left"><box frame="2" xtl="1" ytl="2" xbr="10" ybr="20"/></track></annotations>')
    assert w._auto_load_video_hands(video)==1
    assert w._auto_load_video_hands(video)==0
    assert w.raw_boxes[0]['orig_frame']==2,'Sparse XML frame was incorrectly aligned to zero'
    def draw(expected):
        resolved=w._get_auto_draw_label()
        assert resolved and w._pending_draw_box_role==expected,(resolved,w._pending_draw_box_role)
        label,cls=resolved
        w._on_box_edited(None,dict(_action='add',label=label,class_id=cls,frame=w.player.current_frame,
                                  x1=20,y1=20,x2=30,y2=30))
    assert w.rad_draw_target.isChecked()
    draw('noun')
    assert w.rad_draw_inst.isChecked()
    draw('instrument')
    assert w._hand_noun_object_id(w.event_draft['Left_hand'])==0
    assert w._hand_instrument_object_id(w.event_draft['Left_hand'])==8
    # A tracking box exists even when the SOE-only overlay hides this middle frame.
    w.raw_boxes.append(dict(w.raw_boxes[0],orig_frame=4))
    w._rebuild_bboxes_from_raw();w.player.current_frame=4
    assert w._next_event_draw_role()=='noun'
    # Moving to a frame without tracking starts with hand, then noun and tool.
    w.player.current_frame=3
    w._sync_event_draw_mode()
    assert w.btn_inline_edit_boxes.text()=='Edit Boxes [ON: Hand]'
    draw('hand');draw('noun');draw('instrument')
    # Empty instrument never creates an unassigned tool step.
    w.event_draft['Left_hand'].update(instrument_object_id=None,tool_object_id=None)
    w.player.current_frame=5
    w._sync_event_draw_mode()
    draw('hand');draw('noun')
    assert w.rad_draw_target.isChecked() and w._next_event_draw_role()=='noun'
    w.rad_draw_none.setChecked(True)
    assert w._get_auto_draw_label() is None
    assert w._pending_draw_box_role is None
    # Imported XML fills gaps, never restores intentionally deleted hands.
    w.raw_boxes=[];w._rebuild_bboxes_from_raw()
    w._suppressed_hand_boxes=['Left_hand:2']
    assert w._auto_load_video_hands(video)==0
    w._suppressed_hand_boxes=[]
    w.frame_review={'2':{'H:Left_hand':{'state':'not_visible'}}}
    assert w._auto_load_video_hands(video)==0
    w.frame_review={}
    w.raw_boxes=[dict(id=-1,orig_frame=2,label='Left_hand',source='manual_box_edit',x1=3,y1=4,x2=12,y2=22)]
    w._rebuild_bboxes_from_raw()
    before=copy.deepcopy(w.raw_boxes)
    assert w._auto_load_video_hands(video)==0 and w.raw_boxes==before
    # Native XML frames retain their video position with a crop offset.
    w.raw_boxes=[];w.start_offset=2;w._rebuild_bboxes_from_raw()
    assert w._auto_load_video_hands(video)==1 and w.raw_boxes[0]['orig_frame']==0
    assert len(w._boxes_for_review('H:Left_hand',2))==1
    assert not warnings,warnings
w._stop_autosave();w.deleteLater()
print('LOCAL_XML_MANUAL_MODE_NATIVE_FRAMES_AND_HAND_NOUN_OPTIONAL_TOOL_SEQUENCE_PASS')
