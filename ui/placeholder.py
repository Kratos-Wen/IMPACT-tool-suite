from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
)
from PyQt5.QtCore import Qt


class PlaceholderPane(QWidget):
    def __init__(
        self,
        title: str,
        message: str,
        parent=None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        lbl_msg = QLabel(message)
        lbl_msg.setWordWrap(True)
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_msg)
        layout.addStretch(1)
