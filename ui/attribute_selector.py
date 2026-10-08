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
        self.lineEdit().setPlaceholderText('Review attributes...')
        self.view().viewport().installEventFilter(self)
        self.setMinimumContentsLength(20)

    def setOptions(self, options, selected, captions=None):
        old = self.blockSignals(True)
        self.clear()
        self._captions = dict(captions or {})
        for code in list(options)+['normal','unknown','unreviewed','confirm']:
            if self.findData(code)>=0:
                continue
            caption=self._captions.get(code, {'normal':'Reviewed normal','unknown':'Other attributes unknown','unreviewed':'Unreviewed','confirm':'Confirm selected attributes'}.get(code,code.replace('_',' ').capitalize()))
            self.addItem(caption,code)
            item=self.model().item(self.count()-1)
            item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
        self.setCurrentText(selected)
        self.blockSignals(old)

    def currentText(self):
        return '; '.join(self._values) if self._values else 'unreviewed'

    def setCurrentText(self, text):
        values=[x.strip() for x in str(text or 'unreviewed').split(';') if x.strip()]
        available={self.itemData(i) for i in range(self.count())}
        self._values=[x for x in dict.fromkeys(values) if x in available] or ['unreviewed']
        self._refresh_summary()

    def _refresh_summary(self):
        captions=[]
        for i in range(self.count()):
            code=self.itemData(i)
            self.model().item(i).setData(Qt.Checked if code in self._values else Qt.Unchecked,Qt.CheckStateRole)
            if code in self._values:captions.append(self.itemText(i))
        self.lineEdit().setText(', '.join(captions))
        self.setToolTip('\n'.join(captions))

    def toggle(self, code):
        if self.findData(code)<0:return
        if code == 'confirm':
            self._values=[v for v in self._values if v not in ('normal','unknown','unreviewed')]
            if not self._values:self._values=['normal']
        elif code in ('normal','unreviewed'):
            self._values=[code]
        else:
            self._values=[v for v in self._values if v not in ('normal','unreviewed')]
            if code in self._values:self._values.remove(code)
            else:self._values.append(code)
            if not self._values:self._values=['unreviewed']
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
