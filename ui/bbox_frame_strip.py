"""Three thin role lanes, sharing the event timeline's exact coordinate transform."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter,QColor,QPen
from PyQt5.QtWidgets import QToolTip
from ui.timeline import BaseTimelineRow
from core.bbox_markers import COLORS,PRIORITY,CAPTIONS

class BBoxFrameStrip(BaseTimelineRow):
    def __init__(self,timeline,get_markers,on_jump):
        super().__init__(timeline._get_fc,timeline.get_view_start,timeline.get_view_span,timeline._get_fps,timeline.get_gutter,timeline)
        self.timeline=timeline;self.get_markers=get_markers;self.on_jump=on_jump
        self.lane_height=18;self.setFixedHeight(3*self.lane_height);self.setMouseTracking(True);self._layout_key=None;self._groups={}
        self.setToolTip('H: hand · O: object · I: instrument. Click a tick to jump to its exact frame. Blue: drawn/edited · green: viewed · gray: model · amber: conflict. Ctrl/Command+wheel zooms; Shift+wheel pans.')

    def _layout(self):
        markers=self.get_markers() or [];start=self.get_vs();end=start+self.get_span()
        key=(id(markers),start,end,self.width(),self.get_gutter())
        if key!=self._layout_key:
            groups={}
            for row in markers:
                if not start<=row['frame']<=end:continue
                x=self.frame_to_x(row['frame']);lane=('H','O','I').index(row['role'])
                k=(x,lane);previous=groups.get(k)
                if previous is None or PRIORITY[row['state']]>PRIORITY[previous['state']]:groups[k]=row
            self._groups=groups;self._layout_key=key;self._markers_source=markers
        return self._groups

    def paintEvent(self,event):
        painter=QPainter(self);painter.fillRect(self.rect(),QColor('#f8fafc'))
        painter.setFont(self.font())
        for lane,role in enumerate(('H','O','I')):
            y=lane*self.lane_height
            painter.setPen(QColor('#64748b'));painter.drawText(5,y+14,role)
            painter.setPen(QColor('#e2e8f0'));painter.drawLine(self.get_gutter(),y+14,self.width(),y+14)
        for (x,lane),row in self._layout().items():
            painter.setPen(QPen(QColor(COLORS[row['state']]),2 if row['state']!='model' else 1))
            painter.drawLine(x,lane*self.lane_height+4,x,lane*self.lane_height+12)
        if self.current_frame is not None:
            painter.setPen(QPen(QColor('#111827'),1));x=self.frame_to_x(self.current_frame)
            painter.drawLine(x,0,x,self.height()-1)

    def hit(self,x,y):
        if x<self.get_gutter():return None
        lane=max(0,min(2,int(y)//self.lane_height));groups=self._layout()
        hits=[(abs(px-x),-PRIORITY[row['state']],row['frame'],row) for (px,l),row in groups.items() if l==lane and abs(px-x)<=3]
        return min(hits,key=lambda item:item[:3])[-1] if hits else None

    def mousePressEvent(self,event):
        if event.button()==Qt.LeftButton:
            row=self.hit(event.x(),event.y())
            if row:self.on_jump(row['frame']);event.accept();return
        event.ignore()

    def mouseMoveEvent(self,event):
        row=self.hit(event.x(),event.y());self.setCursor(Qt.PointingHandCursor if row else Qt.ArrowCursor)
        if row:QToolTip.showText(event.globalPos(),f"Frame {row['frame']} · {row['role']} {row['entity']} · {CAPTIONS[row['state']]}",self)
        else:QToolTip.hideText()

    def wheelEvent(self,event):
        if self.timeline._wheel_navigation(event,self):self.update()
        else:super().wheelEvent(event)
