"""Measure video space and check readable compact controls at desktop sizes."""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--baseline',action='store_true')
parser.add_argument('--video')
args=parser.parse_args()
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime, _configure_app_font
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox,QPushButton
from PyQt5.QtCore import QTimer
import cv2
import numpy as np
from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(noun_classes=['screw_long','tool'],verbs=['tighten'],anomaly_labels=[])
from ui.hoi_window import HOIWindow

app=QApplication([]);_configure_app_font(app)
for name in ('information','warning','critical','question'):
    setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)
with tempfile.TemporaryDirectory() as directory:
    os.environ['IMPACT_HOI_SETTINGS_DIR']=directory
    w=HOIWindow();w._autosave_timer.stop()
    for timer in w.findChildren(QTimer):timer.stop()
    w._set_experiment_mode_key('manual')
    w.onboarding_banner.hide()
    w.player.frame_count=300;w.player.frame_rate=15
    payload=dict(schema='hoi-annotation',video_id='layout',frame_count=300,fps=15,
        object_library={'1':dict(label='screw_long',category='screw_long'),'2':dict(label='tool',category='tool')},
        tracks={},hoi_events={'left_hand':[],'right_hand':[dict(event_id='R_001',verb='tighten',
            start_frame=10,contact_onset_frame=20,end_frame=200,noun_object_id=1,instrument_object_id=2)]})
    w._load_annotations_v2(payload)
    w.player.current_frame=20;w._set_selected_event(w.events[0]['event_id'],'Right_hand')
    image=np.full((720,720,3),(90,130,100),np.uint8)
    if args.video:
        capture=cv2.VideoCapture(args.video)
        capture.set(cv2.CAP_PROP_POS_FRAMES,1633)
        ok,image=capture.read();capture.release();assert ok
    w.player._show_native_frame(image,20)
    results=[]
    tag='before' if args.baseline else 'after'
    for width,height in ((1440,900),(1280,800),(1920,1080),(1000,800),(1280,800)):
        w.resize(width,height);w.show()
        for _ in range(3):app.processEvents()
        w._tracking_row.show()
        w._tracking_progress.hide()
        w._tracking_label.setText('Tracking: ready')
        for _ in range(3):app.processEvents()
        controls=[w.combo_inline_verb,w.combo_inline_noun,w.combo_inline_instrument,w.combo_inline_anomaly,
                  w.btn_inline_edit_boxes,w.btn_correct_track,w.btn_anomaly_keyframe]
        for control in controls:
            assert control.isVisible()
            assert control.height()>=control.fontMetrics().height()+4,(control.text() if hasattr(control,'text') else 'combo',control.height())
        if not args.baseline:
            assert w.width()<=width
            assert w._inline_fields_wide==(w.inline_event_card.width()>=1100)
            assert w._inline_commands_compact==w._inline_fields_wide
            assert w.combo_inline_verb.geometry().top()==w.combo_inline_anomaly.geometry().top() if w._inline_fields_wide else w.combo_inline_verb.geometry().top()<w.combo_inline_anomaly.geometry().top()
            if w._inline_commands_compact:
                assert w.inline_event_card.height()<100
                assert abs(w.lbl_inline_event_title.geometry().center().y()-w.btn_inline_edit_boxes.geometry().center().y())<=1
            for index, control in enumerate(controls):
                for other in controls[index+1:]:
                    assert not control.geometry().intersects(other.geometry())
            assert w.player.height()>=w.height()*.40
        result=dict(size=[w.width(),w.height()],video_height=w.player.height(),editor_height=w.inline_event_card.height())
        results.append(result)
        w.grab().save(f'/tmp/impact-video-{tag}-{width}.png')
    w._set_advanced_inspector_visible(True)
    for _ in range(3):app.processEvents()
    w.grab().save(f'/tmp/impact-video-{tag}-advanced.png')
    for width in (1280,1440):
        w.resize(width,900)
        for _ in range(4):app.processEvents()
        for control in w.inline_event_card.findChildren(QPushButton):
            if control.isVisible():
                assert control.width()>=control.fontMetrics().horizontalAdvance(control.text())+8,control.text()
        w.grab().save(f'/tmp/impact-video-{tag}-advanced-{width}.png')
    print(json.dumps(results))
    w._stop_autosave();w.deleteLater()
