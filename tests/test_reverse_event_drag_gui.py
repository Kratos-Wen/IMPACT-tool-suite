"""Real pointer events create in either direction and retain closed interval rules."""
import os,sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt,QPointF,QEvent
from PyQt5.QtGui import QMouseEvent,QColor
from ui.hoi_timeline import HOITimelineRow
app=QApplication([]);segments=[];created=[];updated=[]
row=HOITimelineRow('Left_hand','Left',lambda:segments,lambda _:QColor('blue'),
 lambda *a:None,lambda *a:updated.append(a),lambda *a:(created.append(a),len(created))[1],
 lambda *a:None,None,lambda *a:None,lambda:120,lambda:0,lambda:100,lambda:15,lambda:100)
row.resize(1100,48);row.show();app.processEvents()
def pointer(kind,frame,y=30):
 p=QPointF(row.frame_to_x(frame),y)
 button=Qt.NoButton if kind==QEvent.MouseMove else Qt.LeftButton
 buttons=Qt.NoButton if kind==QEvent.MouseButtonRelease else Qt.LeftButton
 QApplication.sendEvent(row,QMouseEvent(kind,p,button,buttons,Qt.NoModifier))
def drag(start,end,intermediate=None,y=30):
 pointer(QEvent.MouseButtonPress,start,y)
 if intermediate is not None:pointer(QEvent.MouseMove,intermediate,y)
 pointer(QEvent.MouseButtonRelease,end,y)
for a,b,move in [(20,60,55),(60,20,25),(30,15,70),(30,30,None)]:
 drag(a,b,move);assert created[-1][1:3]==tuple(sorted((a,b))),created[-1]
 assert created[-1][1]<=created[-1][3]<=created[-1][2]
segments[:]=[dict(event_id=10,start=40,end=80,onset=55,verb='hold')]
drag(40,20,25);assert updated[-1][2:]==(20,80,55),updated[-1]
drag(55,45,50,y=8);assert updated[-1][2:]==(40,80,45),updated[-1]
drag(65,45,50);assert updated[-1][2:]==(20,60,35),updated[-1]
segments[:]=[dict(event_id=20,start=0,end=19,onset=0,verb='hold')]
drag(50,20,25);assert created[-1][1:3]==(20,50)
drag(60,19,19);assert created[-1][1]>19,'Reverse drag bypassed same-hand overlap protection'
row.close();print('REVERSE_FORWARD_CROSS_ANCHOR_RELEASE_RESIZE_MOVE_AND_ADJACENCY_PASS')
