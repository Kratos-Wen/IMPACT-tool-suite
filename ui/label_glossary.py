"""Read-only label guide and selector tooltips; never changes event fields."""
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLineEdit, QComboBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, QDialogButtonBox)
from core.label_glossary import glossary_rows, label_tooltip


def decorate_combo(combo, kind, objects=None):
    if combo is None:
        return
    objects = objects or {}
    by_id = {uid: name for name, uid in objects.items()}
    for row in range(combo.count()):
        label = by_id.get(combo.itemData(row), combo.itemText(row)) if kind == 'noun' else combo.itemText(row)
        if label and str(label).lower() not in ('none', 'choose noun...', 'choose instrument...'):
            combo.setItemData(row, label_tooltip(label, kind), Qt.ToolTipRole)


class LabelGlossaryMixin:
    def _open_label_guide(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('Noun and verb guide')
        layout = QVBoxLayout(dialog)
        search = QLineEdit(); search.setPlaceholderText('Search label or explanation')
        language = QComboBox(); language.addItem('English', 'en'); language.addItem('中文', 'zh')
        layout.addWidget(search); layout.addWidget(language)
        tabs = QTabWidget(); layout.addWidget(tabs)
        tables = {}
        for kind, title in (('noun', 'Nouns'), ('verb', 'Verbs')):
            table = QTableWidget(0, 2); table.setHorizontalHeaderLabels(['Label', 'Meaning / use'])
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            table.setColumnWidth(0, 190); table.setWordWrap(True)
            tabs.addTab(table, title); tables[kind] = table

        def populate():
            extras = {'noun': list(self.global_object_map), 'verb': [v.name for v in self.verbs]}
            for kind, table in tables.items():
                rows = glossary_rows(kind, extras[kind], language.currentData())
                table.setRowCount(len(rows))
                for row, (name, meaning) in enumerate(rows):
                    table.setItem(row, 0, QTableWidgetItem(name))
                    table.setItem(row, 1, QTableWidgetItem(meaning))
                table.resizeRowsToContents()
            filter_rows()

        def filter_rows():
            query = search.text().casefold().strip()
            for table in tables.values():
                for row in range(table.rowCount()):
                    text = ' '.join(table.item(row, col).text() for col in range(2)).casefold()
                    table.setRowHidden(row, bool(query and query not in text))

        search.textChanged.connect(filter_rows); language.currentIndexChanged.connect(populate)
        buttons = QDialogButtonBox(QDialogButtonBox.Close); buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons); populate(); dialog.resize(810, 570); dialog.exec_()

    def _refresh_label_explanations(self):
        for name, kind in (('combo_verb', 'verb'), ('combo_inline_verb', 'verb'),
                           ('combo_target', 'noun'), ('combo_instrument', 'noun'),
                           ('combo_inline_noun', 'noun'), ('combo_inline_instrument', 'noun')):
            decorate_combo(getattr(self, name, None), kind, self.global_object_map)
        panel = getattr(self, 'label_panel', None)
        if panel:
            for row in range(panel.verb_list.count()):
                item = panel.verb_list.item(row)
                item.setToolTip(label_tooltip(item.data(Qt.UserRole) or item.text(), 'verb'))
