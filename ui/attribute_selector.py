"""Compact checked dropdown; codes stay separate from the visible captions."""
from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtWidgets import QComboBox

class AttributeSelector(QComboBox):
    selectionChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._values = []
        self._captions = {}
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText('Normal')
        self.view().viewport().installEventFilter(self)
        self.setMinimumContentsLength(20)

    def setOptions(self, options, selected, captions=None):
        old = self.blockSignals(True)
        self.clear()
        self._captions = dict(captions or {})
        for code in options:
            if code in ('normal','unknown','unreviewed','confirm'):
                continue
            if self.findData(code)>=0:
                continue
            caption=self._captions.get(code,code.replace('_',' ').capitalize())
            self.addItem(caption,code)
            item=self.model().item(self.count()-1)
            item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
        self.setCurrentText(selected)
        self.blockSignals(old)

    def currentText(self):
        return '; '.join(self._values) if self._values else 'normal'

    def setCurrentText(self, text):
        values=[x.strip() for x in str(text or '').split(';') if x.strip()]
        available={self.itemData(i) for i in range(self.count())}
        self._values=[x for x in dict.fromkeys(values) if x in available]
        self._refresh_summary()

    def _refresh_summary(self):
        captions=[]
        for i in range(self.count()):
            code=self.itemData(i)
            self.model().item(i).setData(Qt.Checked if code in self._values else Qt.Unchecked,Qt.CheckStateRole)
            if code in self._values:captions.append(self.itemText(i))
        self.lineEdit().setText(', '.join(captions) if captions else 'Normal')
        self.setToolTip('\n'.join(captions) if captions else 'Normal. Select every anomaly type visible in this event.')

    def toggle(self, code):
        if code == 'normal':
            self._values=[]
        elif self.findData(code)<0:
            return
        else:
            if code in self._values:self._values.remove(code)
            else:self._values.append(code)
        self._refresh_summary()
        self.selectionChanged.emit()
        self.activated[int].emit(max(0,self.findData(code)))

    def eventFilter(self, obj, event):
        if obj is self.view().viewport() and event.type()==QEvent.MouseButtonRelease:
            index=self.view().indexAt(event.pos())
            if index.isValid():self.toggle(self.itemData(index.row()))
            return True
        return super().eventFilter(obj,event)

    def keyPressEvent(self,event):
        if self.view().isVisible() and event.key()==Qt.Key_Space:
            index=self.view().currentIndex()
            if index.isValid():self.toggle(self.itemData(index.row()))
            event.accept();return
        super().keyPressEvent(event)
