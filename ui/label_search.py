"""Ranked completion for instance selectors; selection preserves the instance ID."""
from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtWidgets import QCompleter
from core.label_search import ranked_labels


def install_instance_search(combo):
    model = QStringListModel(combo)
    completer = QCompleter(model, combo)
    completer.setCaseSensitivity(Qt.CaseInsensitive)
    completer.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
    combo.setCompleter(completer)
    def update(text):
        labels = []
        for index in range(combo.count()):
            item = combo.model().item(index) if hasattr(combo.model(), 'item') else None
            if combo.itemData(index) is not None and (item is None or item.isEnabled()):
                labels.append(combo.itemText(index))
        choices = ranked_labels(text, labels)
        model.setStringList(choices)
        completer.setCompletionPrefix('')
        if choices:
            completer.complete()
        else:
            completer.popup().hide()
    def select(text):
        index = combo.findText(text, Qt.MatchExactly)
        if index >= 0:
            combo.setCurrentIndex(index)
    combo.lineEdit().textEdited.connect(update)
    completer.activated[str].connect(select)
    return completer
