"""Exercise real canvas gestures rather than calling the box callback directly."""
import os,sys,tempfile,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
import cv2,numpy as np
from PyQt5.QtWidgets import QApplication,QMessageBox,QInputDialog
from PyQt5.QtCore import Qt,QPoint,QEvent,QTimer
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(noun_classes=['part'],verbs=['hold'],anomaly_labels=[])
from ui.hoi_window import HOIWindow
app=QApplication([])
for name in ('information','warning','critical','question'):
 setattr(QMessageBox,name,lambda *a,**k:QMessageBox.Yes)
QInputDialog.getItem=lambda parent,title,text,options,*a:(options[0],True)
with tempfile.TemporaryDirectory() as directory:
 os.environ['IMPACT_HOI_SETTINGS_DIR']=directory
 path=Path(directory)/'video.mp4'
 writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),15,(320,240));assert writer.isOpened()
 for _ in range(15):writer.write(np.full((240,320,3),100,np.uint8))
 writer.release()
 w=HOIWindow();w._show_quick_start_on_startup=False;assert w._load_video(str(path));w._autosave_timer.stop()
 w.show();app.processEvents();w._on_hoi_timeline_create('Left_hand',0,10,0)
 w._apply_verb_choice('hold');p=w.player;p.seek(0);app.processEvents();p.grab()
 assert p.edit_enabled and p.edit_add_enabled
 original_dialog=p._rename_dialog
 def answer_label(*a,**k):
  dialog=original_dialog(*a,**k)
  def accept():
   dialog.line.setText('Left_hand');dialog.accept()
  QTimer.singleShot(0,accept)
  return dialog
 p._rename_dialog=answer_label
 w.rad_draw_none.setChecked(True)
 def screen(ix,iy):
  x,y,width,height=p._last_draw_rect
  return QPoint(round(x+ix*width/p._frame_w),round(y+iy*height/p._frame_h))
 def drag(start,end,modifiers=Qt.NoModifier):
  QTest.mousePress(p,Qt.LeftButton,modifiers,pos=start)
  app.sendEvent(p,QMouseEvent(QEvent.MouseMove,end,Qt.NoButton,Qt.LeftButton,modifiers))
  QTest.mouseRelease(p,Qt.LeftButton,modifiers,pos=end);app.processEvents();p.grab()
 for cycle in range(20):
  print('CANVAS_CYCLE',cycle,flush=True)
  before=copy.deepcopy(w.raw_boxes)
  drag(screen(70,90),screen(140,150),Qt.ControlModifier)
  assert len(w.raw_boxes)==1 and w.raw_boxes[0]['label']=='Left_hand'
  assert p.overlay_boxes and p.edit_boxes
  w._hoi_undo();assert w.raw_boxes==before
  w._hoi_redo();assert len(w.raw_boxes)==1
  p.grab();box=copy.deepcopy(w.raw_boxes[0])
  rect=p._box_screen_rect(p.edit_boxes[0]);center=rect.center()
  drag(center,center+QPoint(12,10))
  assert w.raw_boxes[0]['x1']!=box['x1'],'Canvas drag did not edit geometry'
  assert w.raw_boxes[0]['id']==box['id']
  center=p._box_screen_rect(p.edit_boxes[0]).center()
  QTest.mouseClick(p,Qt.RightButton,pos=center)
  assert not w.raw_boxes and not p.edit_boxes
  w._hoi_undo();assert len(w.raw_boxes)==1
 w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True
 w.close();p.release_media();w.deleteLater();app.sendPostedEvents(None,QEvent.DeferredDelete)
print('CANVAS_20_CTRL_DRAW_REDRAW_MOVE_RIGHT_DELETE_UNDO_REDO_CYCLES_PASS')
