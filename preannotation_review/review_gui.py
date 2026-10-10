"""Explicit human decisions over immutable proposals, with native-frame video."""
import copy
import gzip
import json
import re
import uuid
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QFont, QFontDatabase, QFontMetrics
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QPlainTextEdit, QListWidget,
    QFormLayout, QScrollArea, QSlider, QFileDialog, QMessageBox, QShortcut, QCheckBox,
    QTabWidget, QDialog, QDialogButtonBox, QInputDialog)
from video_player import VideoPlayer
from review_core import ReviewDocument, VERBS, STATES, BOUNDARIES, now

STATUS_NAMES = ['待审核', '已审核', '已审核但无法确定', '拒绝候选', '重复事件']


def button(text, action):
    b = QPushButton(text); b.clicked.connect(action); return b


def form_spin(maximum):
    w = QSpinBox(); w.setRange(-1, maximum); w.setSpecialValueText('未知'); return w


class ReviewPlayer(VideoPlayer):
    def mouseDoubleClickEvent(self, event):
        # Instance labels are edited through a scoped dialog, never by renaming
        # a rendered label that also contains the stable track ID.
        if self.edit_enabled and event.button() == Qt.LeftButton:
            return
        super().mouseDoubleClickEvent(event)


from frame_checks import ReviewFrameMixin

class ReviewWindow(ReviewFrameMixin,QMainWindow):
    def edit_shared_assembly(self):
        if not self.doc or not self.require_reviewer() or not self.apply_current():return
        from assembly_bridge import ReviewAssemblyBridge
        bridge=ReviewAssemblyBridge(self)
        bridge._open_assembly_editor()
        bridge.deleteLater()

    def merge_shared_assembly(self):
        if not self.doc or not self.require_reviewer() or not self.apply_current():return
        from assembly_bridge import ReviewAssemblyBridge
        bridge=ReviewAssemblyBridge(self);bridge._open_assembly_merge();bridge.deleteLater()

    def undo_shared_assembly(self):
        snapshot=getattr(self,'_assembly_undo',None)
        if not snapshot or not self.doc:return
        if snapshot.get('base_snapshot_sha256')!=self.doc.data.get('base_snapshot_sha256'):
            self._assembly_undo=None;return
        # Restore assembly-only fields; never undo subsequent verb/box edits.
        for key in ('shared_assembly','assembly_default_reference'):
            if key in snapshot:self.doc.data[key]=copy.deepcopy(snapshot[key])
            else:self.doc.data.pop(key,None)
        refs={r['event_uid']:{k:copy.deepcopy(r['value'].get(k)) for k in ('shared_assembly_ref','shared_assembly_id','object_instance_id')} for r in snapshot['events']}
        for row in self.doc.events:
            if row['event_uid'] in refs:row['value'].update(refs[row['event_uid']])
        self._assembly_undo=None;self.dirty=True;self.frame_changed(self.player.current_frame)
    def __init__(self):
        super().__init__()
        # Some cluster Qt installations do not discover system fallback fonts.
        fallback = Path(__file__).resolve().parent / 'fonts/IMPACTReviewCJK.ttf'
        if fallback.exists(): QFontDatabase.addApplicationFont(str(fallback))
        families = set(QFontDatabase().families())
        for family in ('Microsoft YaHei UI', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', 'IMPACT Review CJK'):
            if family in families and QFontMetrics(QFont(family, 9)).inFont('中'):
                self.setFont(QFont(family, 9)); break
        self.doc = None; self.frames = {}; self.current_index = -1
        self.loading = False; self.dirty = False; self.selected_track = None
        self.setWindowTitle('IMPACT 预标注审核 — 机器建议需人工确认')
        self.resize(1550, 980)
        menu=self.menuBar().addMenu("Frame review")
        for label,key,fn in [("Verify visible boxes","Ctrl+Return",self.verify_review_frame),("Next frame needing review","Alt+N",lambda:self.next_review_frame(1)),("Previous frame needing review","Alt+P",lambda:self.next_review_frame(-1))]:
            action=menu.addAction(label,fn);action.setShortcut(QKeySequence(key))
            from utils.shortcut_settings import load_shortcut_bindings,default_shortcut_bindings,shortcut_value
            sid={"Ctrl+Return":"hoi.verify_frame","Alt+N":"hoi.next_review_frame","Alt+P":"hoi.prev_review_frame"}[key]
            action.setShortcut(QKeySequence(shortcut_value(load_shortcut_bindings(),default_shortcut_bindings(),sid,key)))
        menu.addAction("Mark entity not visible...",self.mark_review_not_visible)
        menu.addAction("Require review at current frame",self.require_review_current_frame)
        menu.addAction("Clear automatic track in event...",self.clear_review_event_track)
        menu.addAction("Undo track clear",self.undo_review_track_clear)
        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central)
        top = QHBoxLayout(); layout.addLayout(top)
        top.addWidget(button('打开 review / reviewed JSON', self.choose_file))
        top.addWidget(QLabel('标注人编号'))
        self.reviewer = QLineEdit(); self.reviewer.setMaximumWidth(170); top.addWidget(self.reviewer)
        top.addWidget(button('保存回传文件 Ctrl+S', self.save))
        top.addWidget(button('新增漏标事件', self.add_event))
        top.addWidget(button('Shared assembly...', self.edit_shared_assembly))
        top.addWidget(button('Merge assemblies...', self.merge_shared_assembly))
        top.addWidget(button('Undo assembly change', self.undo_shared_assembly))
        top.addWidget(button('下一处待审核', self.next_pending))
        self.summary = QLabel('打开批次中某段视频的 review.json'); top.addWidget(self.summary, 1)
        self.banner = QLabel('左右始终指操作者本人的左右；帧号从 0 开始，End 为包含的最后一帧。')
        self.banner.setWordWrap(True); layout.addWidget(self.banner)
        split = QSplitter(Qt.Horizontal); layout.addWidget(split, 1)
        self.tabs = QTabWidget(); split.addWidget(self.tabs)
        self.event_list = QListWidget(); self.tabs.addTab(self.event_list, '事件候选')
        wp = QWidget(); wl = QVBoxLayout(wp); self.window_list = QListWidget(); wl.addWidget(self.window_list)
        self.window_note = QLineEdit(); self.window_note.setPlaceholderText('该窗口的漏标／异常情况或审核备注'); wl.addWidget(self.window_note)
        wl.addWidget(button('本窗口已完整回放并检查漏标', self.mark_window))
        self.tabs.addTab(wp, '全片回放检查')
        self.event_list.currentRowChanged.connect(self.select_event)
        self.window_list.currentRowChanged.connect(self.select_window)
        mid = QWidget(); ml = QVBoxLayout(mid); split.addWidget(mid)
        self.player = ReviewPlayer(status_cb=self.message)
        self.player.on_frame_advanced = self.frame_changed
        self.player.setMinimumSize(480, 330); ml.addWidget(self.player, 1)
        self.slider = QSlider(Qt.Horizontal); self.slider.valueChanged.connect(self.seek); ml.addWidget(self.slider)
        controls = QHBoxLayout(); ml.addLayout(controls)
        controls.addWidget(button('◀ 一帧', lambda: self.seek(self.player.current_frame - 1)))
        controls.addWidget(button('播放／暂停', self.play_pause))
        controls.addWidget(button('一帧 ▶', lambda: self.seek(self.player.current_frame + 1)))
        self.frame_spin = QSpinBox(); self.frame_spin.valueChanged.connect(self.seek)
        controls.addWidget(self.frame_spin)
        self.frame_label = QLabel(''); controls.addWidget(self.frame_label, 1)
        marks = QHBoxLayout(); ml.addLayout(marks)
        for key, text in zip(BOUNDARIES, ('当前帧 → Start', '当前帧 → Onset', '当前帧 → End')):
            marks.addWidget(button(text, lambda _, k=key: self.mark_boundary(k)))
        marks.addWidget(button('Onset 无法确定', lambda: self.fields['onset_frame'].setValue(-1)))
        geo = QHBoxLayout(); ml.addLayout(geo)
        self.show_all = QCheckBox('显示所有框'); self.show_all.setChecked(False); self.show_all.hide()
        self.show_all.toggled.connect(lambda: self.frame_changed(self.player.current_frame)); geo.addWidget(self.show_all)
        self.edit_boxes = QCheckBox('编辑当前帧框（暂停后）')
        self.edit_boxes.setChecked(True)
        self.edit_boxes.toggled.connect(self.toggle_edit); geo.addWidget(self.edit_boxes)
        geo.addWidget(button('修改所选实例／名词／左右手', self.edit_instance))
        geo.addWidget(button('删除所选框（仅本帧）', self.hide_box))
        ml.addWidget(QLabel('选框后可拖动边框；Ctrl+拖拽补框；Ctrl+滚轮缩放。框修改只作用于当前帧。'))
        self.box_label = QLabel('未选中实例'); self.box_label.setWordWrap(True); ml.addWidget(self.box_label)
        right = QScrollArea(); right.setWidgetResizable(True); split.addWidget(right)
        form_widget = QWidget(); right.setWidget(form_widget); fl = QVBoxLayout(form_widget)
        form = QFormLayout(); fl.addLayout(form)
        self.uid_label = QLabel(''); self.uid_label.setWordWrap(True); form.addRow('事件 UID', self.uid_label)
        self.origin_label = QLabel(''); self.origin_label.setWordWrap(True); form.addRow('候选来源', self.origin_label)
        self.hand = QComboBox(); self.hand.addItems(['left', 'right', 'unknown']); form.addRow('本人的左／右手', self.hand)
        self.verb = QComboBox(); self.verb.setEditable(True); self.verb.addItems([''] + VERBS); form.addRow('动词（可直接输入）', self.verb)
        self.definition = QLineEdit(); form.addRow('新增动词的定义', self.definition)
        self.ids = {}
        for key, label in [('instrument_instance_id', 'I 工具实例（无工具可留空）'), ('object_instance_id', 'O 交互对象实例')]:
            combo = QComboBox(); combo.setEditable(True); combo.setMinimumContentsLength(15); self.ids[key] = combo; form.addRow(label, combo)
        self.fields = {}
        for key, label in zip(BOUNDARIES, ['Start 起始帧', 'Onset 接触帧', 'End 结束帧（含）']):
            self.fields[key] = form_spin(10000000); form.addRow(label, self.fields[key])
        self.intervals = QPlainTextEdit(); self.intervals.setMaximumHeight(85)
        form.addRow('边界不确定区间（JSON）', self.intervals)
        self.truncated_start = QCheckBox('窗口开始前事件已开始')
        self.truncated_end = QCheckBox('窗口结束后事件仍继续')
        form.addRow(self.truncated_start); form.addRow(self.truncated_end)
        self.status = QComboBox(); self.status.addItems(STATUS_NAMES); form.addRow('人工决定', self.status)
        self.unknown = QLineEdit(); form.addRow('无法确定的原因', self.unknown)
        self.duplicate = QLineEdit(); form.addRow('重复指向的完整事件 UID', self.duplicate)
        self.notes = QPlainTextEdit(); self.notes.setMaximumHeight(85); form.addRow('人工备注／拒绝原因', self.notes)
        fl.addWidget(button('应用本事件的修改', self.apply_current))
        self.evidence = QPlainTextEdit(); self.evidence.setReadOnly(True); self.evidence.setMinimumHeight(120)
        fl.addWidget(QLabel('原始模型依据与问题（保留，不随修改消失）')); fl.addWidget(self.evidence)
        split.setSizes([280, 810, 460])
        for w in [self.hand, self.verb, self.status] + list(self.ids.values()):
            w.currentTextChanged.connect(self.mark_dirty)
        for w in [self.definition, self.unknown, self.duplicate, self.reviewer]:
            w.textChanged.connect(self.mark_dirty)
        for w in self.fields.values(): w.valueChanged.connect(self.mark_dirty)
        for w in (self.notes, self.intervals): w.textChanged.connect(self.mark_dirty)
        self.truncated_start.toggled.connect(self.mark_dirty); self.truncated_end.toggled.connect(self.mark_dirty)
        QShortcut(QKeySequence('Ctrl+S'), self, activated=self.save)

    def message(self, text):
        self.statusBar().showMessage(str(text), 12000)

    def mark_dirty(self, *_):
        if not self.loading and self.doc:
            self.dirty = True
            self.frame_changed(self.player.current_frame)

    def choose_file(self):
        p, _ = QFileDialog.getOpenFileName(self, '打开审核文件', '', 'JSON (*.json)')
        if p: self.open_path(p)

    def confirm_leave(self):
        if not self.dirty: return True
        answer = QMessageBox.question(self, '尚未保存', '保存本视频的修改后继续？', QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if answer == QMessageBox.Cancel: return False
        if answer == QMessageBox.Save: return bool(self.save())
        return True

    def open_path(self, path):
        if not self.confirm_leave(): return False
        try:
            doc = ReviewDocument(path)
            if not doc.video_path.is_file() or not doc.tracks_path.is_file():
                raise ValueError('缺少 video.mp4 或 tracks.json.gz；请完整解压批次并保留目录结构')
            if doc.video_path.stat().st_size != doc.data['source']['source_fingerprint']['bytes']:
                raise ValueError('视频文件大小与发放版本不同，请使用包内原视频')
            with gzip.open(doc.tracks_path, 'rt', encoding='utf-8') as f: tracks = json.load(f)
            frames = {r['frame']: dict(r,tracks=[]) if __import__('os').environ.get('IMPACT_SKIP_AUTOMATIC_TRACKS')=='1' else r for r in tracks['frames']}
            if len(frames) != doc.frame_count: raise ValueError('轨迹帧数不一致')
            self.player.pause()
            self.loading = True
            if not self.player.load(str(doc.video_path)): raise ValueError('无法解码视频')
            if self.player.frame_count != doc.frame_count: raise ValueError('视频帧数与预标注不一致，禁止错位审核')
            self.doc = doc; self.frames = frames; self.current_index = -1; self.selected_track = None
            self.reviewer.setText(doc.data.get('reviewer', self.reviewer.text()))
            self.slider.setRange(0, doc.frame_count - 1); self.frame_spin.setRange(0, doc.frame_count - 1)
            for w in self.fields.values(): w.setMaximum(doc.frame_count - 1)
            self.refresh_ids()
            self.event_list.blockSignals(True); self.event_list.clear()
            for row in doc.events: self.event_list.addItem(self.event_label(row))
            self.event_list.blockSignals(False)
            self.window_list.blockSignals(True); self.window_list.clear()
            for w in doc.data['windows']: self.window_list.addItem(self.window_label(w))
            self.window_list.blockSignals(False)
            self.banner.setText(doc.data['trial_id'] + ' | 固定批次 ' + doc.data['snapshot_id'] + ' | 左右为本人左右；帧号 0 起，End 包含。机器框和语义均待验证。')
            self.loading = False; self.dirty = False
            if doc.events: self.event_list.setCurrentRow(0)
            else: self.seek(0)
            self.refresh_summary(); return True
        except Exception as exc:
            self.loading = False
            # A failed decoder/source check must never leave another video's
            # image visible next to the preceding document's editable labels.
            self.doc = None; self.current_index = -1; self.frames = {}
            self.player.release_media()
            QMessageBox.critical(self, '打开失败', str(exc)); return False

    def refresh_ids(self):
        if not self.doc: return
        for combo in self.ids.values():
            text = combo.currentText(); combo.blockSignals(True); combo.clear(); combo.addItem('')
            for uid in sorted(self.doc.instance_ids()):
                meta = self.doc.instance(uid, self.player.current_frame)
                combo.addItem(uid + ' | ' + meta.get('label', ''))
            combo.setCurrentText(text); combo.blockSignals(False)

    @staticmethod
    def event_label(row):
        e = row['value']; r = row['review']
        return f"{STATUS_NAMES[STATES.index(r['status'])]} | {e.get('hand')} {e.get('verb')}\nS {e.get('start_frame')} C {e.get('onset_frame')} E {e.get('end_frame')}"

    @staticmethod
    def window_label(w):
        return f"{'✓' if w['review']['status'] == 'replayed' else '待回放'} {w['start_frame']}–{w['end_frame']}\n{w['candidate_count']} 候选 / {w['rejected_candidate_count']} 结构疑问"

    def refresh_summary(self):
        if self.doc:
            c = self.doc.completion()
            self.summary.setText(f"事件决定 {c['events_resolved']}/{c['events_total']}；回放 {c['windows_replayed']}/{c['windows_total']}")

    def select_event(self, index):
        if self.loading or not self.doc or index < 0: return
        if not self.apply_current():
            self.event_list.blockSignals(True); self.event_list.setCurrentRow(self.current_index); self.event_list.blockSignals(False); return
        self.current_index = index; row = self.doc.events[index]; e, r = row['value'], row['review']
        self.loading = True
        self.uid_label.setText(row['event_uid'])
        self.origin_label.setText(row['origin'] + ('；细化阶段撤回，需要人工决定' if (row.get('original') or {}).get('refinement_withdrew_candidate') else ''))
        self.hand.setCurrentText(e.get('hand', 'unknown')); self.verb.setCurrentText(e.get('verb') or '')
        self.definition.setText(e.get('novel_verb_definition') or '')
        for key, combo in self.ids.items(): combo.setCurrentText(e.get(key) or '')
        self.ids['object_instance_id'].setEnabled(not e.get('shared_assembly_ref',False))
        for key, field in self.fields.items(): field.setValue(e[key] if type(e.get(key)) is int else -1)
        self.intervals.setPlainText(json.dumps(e.get('boundary_intervals', []), ensure_ascii=False))
        self.truncated_start.setChecked(bool(e.get('truncated_at_window_start')))
        self.truncated_end.setChecked(bool(e.get('truncated_at_window_end')))
        self.status.setCurrentIndex(STATES.index(r['status'])); self.unknown.setText(r.get('unknown_reason', ''))
        self.duplicate.setText(r.get('duplicate_of', '')); self.notes.setPlainText(r.get('notes', ''))
        self.evidence.setPlainText(json.dumps(row['original'], ensure_ascii=False, indent=2))
        self.loading = False
        self.seek(e.get('start_frame') if type(e.get('start_frame')) is int else 0)

    def apply_current(self):
        if self.loading or not self.doc or self.current_index < 0: return True
        row = self.doc.events[self.current_index]; before = copy.deepcopy(row)
        try:
            e = copy.deepcopy(row['value']); r = copy.deepcopy(row['review'])
            e.update(hand=self.hand.currentText(), verb=self.verb.currentText().strip(),
                     novel_verb_definition=self.definition.text().strip() or None,
                     boundary_intervals=json.loads(self.intervals.toPlainText() or '[]'),
                     truncated_at_window_start=self.truncated_start.isChecked(), truncated_at_window_end=self.truncated_end.isChecked())
            if e['verb'].casefold() in VERBS: e['verb'] = e['verb'].casefold()
            for key, combo in self.ids.items(): e[key] = combo.currentText().split('|')[0].strip() or None
            if e.get('shared_assembly_ref'):
                e['object_instance_id']=row['value'].get('object_instance_id')
            for key, field in self.fields.items(): e[key] = None if field.value() < 0 else field.value()
            if e.get('verb')=='hold' and not row['value'].get('verb') and e.get('onset_frame') is None and e.get('start_frame') is not None:
                e['onset_frame']=e['start_frame']
                field=self.fields['onset_frame'];field.blockSignals(True);field.setValue(e['start_frame']);field.blockSignals(False)
            r.update(status=STATES[self.status.currentIndex()], notes=self.notes.toPlainText(),
                     unknown_reason=self.unknown.text(), duplicate_of=self.duplicate.text().strip())
            changed = e != row['value'] or any(r[k] != row['review'].get(k) for k in ('status', 'notes', 'unknown_reason', 'duplicate_of'))
            if not changed: return True
            r.update(reviewer=self.reviewer.text().strip(), updated_at_utc=now())
            candidate = dict(row, value=e, review=r)
            issues = self.doc.validate_row(candidate)
            if issues: raise ValueError('\n'.join(issues))
            row.update(value=e, review=r)
            self.doc.data['audit_log'].append(dict(action='edit_event', event_uid=row['event_uid'],
                at_utc=now(), reviewer=r['reviewer'], previous_value=before['value'], previous_review=before['review']))
            self.event_list.item(self.current_index).setText(self.event_label(row))
            self.dirty = True; self.refresh_summary(); self.frame_changed(self.player.current_frame); return True
        except Exception as exc:
            QMessageBox.warning(self, '请修正或保留为待审核', str(exc)); return False

    def seek(self, frame):
        if not self.doc: return
        self.player.pause(); self.player.seek(max(0, min(int(frame), self.doc.frame_count - 1)))
        self.frame_changed(self.player.current_frame)

    def play_pause(self):
        if not self.doc: return
        if self.player.is_playing: self.player.pause()
        else:
            self.edit_boxes.setChecked(False); self.player.play()

    def frame_changed(self, frame):
        if not self.doc or self.loading: return
        for w in (self.slider, self.frame_spin):
            w.blockSignals(True); w.setValue(frame); w.blockSignals(False)
        record = self.frames.get(frame, {})
        self.frame_label.setText(f"/ {self.doc.frame_count - 1}  PTS {record.get('pts_seconds', 0):.3f}s")
        overrides = self.doc.data['box_overrides'].get(str(frame), {})
        boxes = {}
        for t in record.get('tracks', []):
            boxes[f"EGO_T{t['track_id']:06d}"] = copy.deepcopy(t)
        for uid, box in overrides.items(): boxes.setdefault(uid, {}).update(box)
        selected_ids = set()
        if self.current_index >= 0:
            event = self.doc.events[self.current_index]['value']
            selected_ids = {w.currentText().split(' | ', 1)[0].strip() for w in self.ids.values()}
            if event.get('shared_assembly_ref'):
                from core.assembly_timeline import resolve_object
                uid=resolve_object(event,self.doc.data.get('shared_assembly',{}),frame)
                old=event.get('object_instance_id')
                selected_ids.discard(old)
                if uid is not None:selected_ids.add(f'EGO_T{uid:06d}')
                self.ids['object_instance_id'].blockSignals(True)
                self.ids['object_instance_id'].setCurrentText(f'EGO_T{uid:06d}' if uid is not None else '')
                self.ids['object_instance_id'].setEnabled(False)
                self.ids['object_instance_id'].blockSignals(False)
        displayed = []
        for uid, t in boxes.items():
            from core.assembly_timeline import active_id
            if uid.startswith('EGO_T') and uid[5:].isdigit() and active_id(self.doc.data.get('shared_assembly',{}),int(uid[5:]),frame)!=int(uid[5:]):continue
            b = t.get('bbox_xyxy'); meta = self.doc.instance(uid, frame)
            if not b or t.get('visible') is not True: continue
            if uid not in selected_ids and meta.get('anatomical_hand') not in ('left', 'right'): continue
            hand = meta.get('anatomical_hand', 'unknown')
            side = (' / 候选' + hand) if hand in ('left', 'right') else ''
            if meta.get('reviewer'): side = (' / 人工' + hand) if hand in ('left', 'right') else ''
            label = uid + ' ' + meta.get('label', '') + side
            displayed.append(dict(id=uid, frame=frame, x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=label,
                                  color='#00a878' if uid in overrides else '#e89f00',
                                  selected=uid == self.selected_track, thick=uid in selected_ids))
        from core.frame_review import valid_record
        if self.current_index>=0:
            roles=self._review_roles(self.doc.events[self.current_index]["value"],frame)
            row=self.doc.data.get("frame_review",{}).get(str(frame),{})
            done=sum(valid_record(row.get(r),self._role_boxes(r,frame)) for r in roles)
            self.frame_label.setText(self.frame_label.text()+f" | Human review {done}/{len(roles)}")
        self.player.set_overlay_boxes(displayed)
        self.player.set_edit_context(displayed, on_change=self.box_changed, on_select=self.box_selected,
            allow_add=self.edit_boxes.isChecked(), allow_edit=self.edit_boxes.isChecked(),
            selected_box=next((b for b in displayed if b['id'] == self.selected_track), None))

    def toggle_edit(self, enabled):
        if enabled: self.player.pause()
        self.frame_changed(self.player.current_frame)

    def box_selected(self, box):
        self.selected_track = box.get('id') if isinstance(box, dict) else None
        self.box_label.setText('选中 ' + self.selected_track if self.selected_track else '未选中实例')

    def require_reviewer(self):
        if self.reviewer.text().strip(): return True
        QMessageBox.warning(self, '需要标注人编号', '请在顶部填写标注人编号，以记录修改来源。'); return False

    def box_changed(self, uid, box):
        if not self.doc or not self.require_reviewer(): return
        frame = self.player.current_frame
        if box.get('_action') == 'delete':
            self.selected_track = uid
            self.hide_box()
            return
        if uid is None:
            choices = sorted(set(self.doc.data.get("instances", {})) | set(self.doc.data.get("human_instances", {})))
            choices.append('[Create new instance]')
            default = choices.index(self.selected_track) if self.selected_track in choices else len(choices) - 1
            chosen, ok = QInputDialog.getItem(self, 'Box instance ID', 'Attach this box to an existing ID, or create a new instance:', choices, default, False)
            if not ok:
                return
            if chosen != '[Create new instance]':
                uid = chosen
        if uid is None:
            uid = 'HUMAN_'  + uuid.uuid4().hex[:12]
            self.doc.data['human_instances'][uid] = dict(track_uid=uid, label=str(box.get('label') or ''),
                definition='人工框新增名称，需在实例编辑中补充定义', category_candidates=[str(box.get('label') or '')],
                anatomical_hand='unknown', identity_status='human_created', reviewer=self.reviewer.text().strip())
        b = [box[k] for k in ('x1', 'y1', 'x2', 'y2')]
        b = [max(0, min(float(v), float(self.player._frame_w if i % 2 == 0 else self.player._frame_h))) for i, v in enumerate(b)]
        if b[0] >= b[2] or b[1] >= b[3]: return
        self.doc.data['audit_log'].append(dict(action='edit_box', frame=frame, track_uid=uid,
            previous_override=copy.deepcopy(self.doc.data['box_overrides'].get(str(frame), {}).get(uid)),
            reviewer=self.reviewer.text().strip(), at_utc=now()))
        self.doc.data['box_overrides'].setdefault(str(frame), {})[uid] = dict(bbox_xyxy=b, visible=True,
            reviewer=self.reviewer.text().strip(), updated_at_utc=now(), geometry_source='human_current_frame')
        self.selected_track = uid; self.dirty = True; self.refresh_ids(); self.frame_changed(frame)

    def hide_box(self):
        if not self.doc or not self.selected_track or not self.require_reviewer(): return
        self.doc.data['audit_log'].append(dict(action='hide_box', frame=self.player.current_frame, track_uid=self.selected_track,
            previous_override=copy.deepcopy(self.doc.data['box_overrides'].get(str(self.player.current_frame), {}).get(self.selected_track)),
            reviewer=self.reviewer.text().strip(), at_utc=now()))
        self.doc.data['box_overrides'].setdefault(str(self.player.current_frame), {})[self.selected_track] = dict(
            bbox_xyxy=None, visible=False, reviewer=self.reviewer.text().strip(), updated_at_utc=now(), geometry_source='human_current_frame')
        self.dirty = True; self.frame_changed(self.player.current_frame)

    def edit_instance(self):
        if not self.doc or not self.require_reviewer(): return
        uid = self.selected_track
        if not uid:
            QMessageBox.information(self, '请选择实例', '勾选“编辑当前帧框”，在画面中单击要修改的框。'); return
        meta = self.doc.instance(uid, self.player.current_frame)
        dlg = QDialog(self); dlg.setWindowTitle('实例修改：' + uid); form = QFormLayout(dlg)
        noun = QLineEdit(meta.get('label', '')); form.addRow('名词（允许新词）', noun)
        definition = QLineEdit(meta.get('definition', '')); form.addRow('含义／修改依据（必填）', definition)
        side = QComboBox(); side.addItems(['unknown', 'left', 'right', 'not_hand']); side.setCurrentText(meta.get('anatomical_hand', 'unknown'))
        form.addRow('本人的左右手／非手', side)
        entity = QLineEdit(meta.get('physical_entity_id', '')); form.addRow('物理实体 ID（可空；跨片段关联）', entity)
        begin = QSpinBox(); end = QSpinBox()
        for s in (begin, end): s.setRange(0, self.doc.frame_count - 1); s.setValue(self.player.current_frame)
        form.addRow('修改生效起始帧', begin); form.addRow('修改生效结束帧（含）', end)
        form.addRow(QLabel('仅修改这段帧区间的语义；不会移动其他帧的框，也不会合并原始 track ID。'))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); form.addRow(buttons)
        buttons.accepted.connect(dlg.accept); buttons.rejected.connect(dlg.reject)
        if dlg.exec_() != QDialog.Accepted: return
        if not noun.text().strip() or not definition.text().strip() or begin.value() > end.value():
            QMessageBox.warning(self, '未应用', '需要名词、定义／依据以及有效的帧范围。'); return
        self.doc.data['instance_edits'].append(dict(track_uid=uid, label=noun.text().strip(), definition=definition.text().strip(),
            anatomical_hand=side.currentText(), physical_entity_id=entity.text().strip() or None,
            start_frame=begin.value(), end_frame=end.value(), reviewer=self.reviewer.text().strip(), updated_at_utc=now()))
        self.dirty = True; self.refresh_ids(); self.frame_changed(self.player.current_frame)

    def mark_boundary(self, key):
        if self.doc and self.current_index >= 0: self.fields[key].setValue(self.player.current_frame)

    def add_event(self):
        if not self.doc or not self.apply_current() or not self.require_reviewer(): return
        row = self.doc.add_event(self.player.current_frame, self.reviewer.text().strip())
        from core.assembly_timeline import active_id,active_object_ids,state_at
        default=self.doc.data.get('assembly_default_reference',False);data=self.doc.data.get('shared_assembly',{})
        ids=active_object_ids(data,self.player.current_frame)
        uid=active_id(data,default,self.player.current_frame) if type(default) is int and state_at(data,self.player.current_frame,default) else ids[0] if default is True and len(ids)==1 else None
        if uid is not None:row['value'].update(shared_assembly_ref=True,shared_assembly_id=uid)
        self.event_list.addItem(self.event_label(row)); self.dirty = True
        self.event_list.setCurrentRow(len(self.doc.events) - 1); self.tabs.setCurrentIndex(0); self.refresh_summary()

    def select_window(self, index):
        if not self.doc or index < 0 or self.loading: return
        if not self.apply_current(): return
        w = self.doc.data['windows'][index]
        self.window_note.setText(w['review'].get('notes', '')); self.seek(w['start_frame'])
        self.message('请完整播放至帧 ' + str(w['end_frame']) + '；零候选不代表无事件，发现漏标可新增。')

    def mark_window(self):
        if not self.doc or not self.require_reviewer() or not self.apply_current(): return
        i = self.window_list.currentRow()
        if i < 0: return
        w = self.doc.data['windows'][i]
        w['review'].update(status='replayed', reviewer=self.reviewer.text().strip(), notes=self.window_note.text(), updated_at_utc=now())
        self.window_list.item(i).setText(self.window_label(w)); self.dirty = True; self.refresh_summary()

    def next_pending(self):
        if not self.doc or not self.apply_current(): return
        n = len(self.doc.events)
        for offset in range(1, n + 1):
            i = (self.current_index + offset) % n
            if self.doc.events[i]['review']['status'] == 'needs_review':
                self.tabs.setCurrentIndex(0)
                if self.event_list.currentRow() == i: self.select_event(i)
                else: self.event_list.setCurrentRow(i)
                return
        for i, w in enumerate(self.doc.data['windows']):
            if w['review']['status'] != 'replayed':
                self.tabs.setCurrentIndex(1)
                if self.window_list.currentRow() == i: self.select_window(i)
                else: self.window_list.setCurrentRow(i)
                return
        self.message('事件决定与全片回放检查已记录；这不代表逐帧几何、异常标注或最终质检已完成。')

    def save(self):
        if not self.doc or not self.apply_current() or not self.require_reviewer(): return False
        code = re.sub(r'[^\w.-]', '_', self.reviewer.text().strip())
        suggested = self.doc.path if self.doc.path.name != 'review.json' else self.doc.path.with_name('reviewed_' + code + '.json')
        p, _ = QFileDialog.getSaveFileName(self, '保存回传 JSON（保留 review.json 原件）', str(suggested), 'JSON (*.json)')
        if not p: return False
        try:
            dest = self.doc.save(p, self.reviewer.text().strip()); self.dirty = False
            self.message('已保存：' + str(dest)); return True
        except Exception as exc:
            QMessageBox.warning(self, '保存失败', str(exc)); return False

    def closeEvent(self, event):
        if self.confirm_leave():
            self.player.release_media(); event.accept()
        else: event.ignore()
