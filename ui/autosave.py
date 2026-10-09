"""Background recovery writes; snapshots contain every box and unfinished draft."""
import copy
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PyQt5.QtCore import QTimer, QLockFile
from PyQt5.QtWidgets import QMessageBox, QLabel
from core.safe_storage import write_recovery, read_recovery, content_digest


class AutosaveMixin:
    def _init_autosave(self):
        self._autosave_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='impact-save')
        self._autosave_future = None
        self._autosave_lock = None
        self._autosave_reserved = None
        self._autosave_path = None
        self._autosave_baseline = None
        self._autosave_error_shown = False
        self.lbl_autosave = QLabel('Autosave: open a task', self)
        self.lbl_autosave.setToolTip('Recovery saves include event drafts, boxes, tracking, assembly and review records.')
        self.layout().addWidget(self.lbl_autosave)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(2000)
        self._autosave_timer.timeout.connect(self._autosave_tick)
        self._autosave_timer.start()

    def _autosave_snapshot(self):
        state = self._snapshot_state()
        state.update(start_offset=self.start_offset, end_frame=self.end_frame,
                     actors_config=copy.deepcopy(self.actors_config),
                     verbs=[dict(name=v.name, color_name=v.color_name, id=v.id) for v in self.verbs],
                     provenance=copy.deepcopy(getattr(self, '_annotation_provenance', {})))
        return state

    def _autosave_identity(self):
        path = Path(self.video_path).resolve()
        return dict(trial=str(getattr(self, '_task_trial_id', '') or path.stem),
                    video=path.name, bytes=path.stat().st_size if path.exists() else None,
                    frames=int(self.player.frame_count), fps=float(self.player.frame_rate))

    def _reserve_autosave(self, video_path):
        path = Path(video_path).resolve()
        self._finish_autosave()
        if self._autosave_reserved == path:
            return
        directory = path.parent / '.impact-recovery'
        directory.mkdir(exist_ok=True)
        # Separate standalone videos, even when they share a directory.
        recovery = directory / (path.name + '.json')
        lock = QLockFile(str(recovery) + '.lock')
        lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            raise ValueError('This video is already open in another editor. Close that editor before continuing.')
        if self._autosave_lock:
            self._autosave_lock.unlock()
        self._autosave_lock = lock
        self._autosave_reserved = path
        self._autosave_path = None
        self._autosave_baseline = None

    def _open_autosave(self, annotation_path=''):
        self._reserve_autosave(self.video_path)
        path = Path(self.video_path).resolve().parent / '.impact-recovery' / (Path(self.video_path).name + '.json')
        self._autosave_path = path
        self._autosave_baseline = self._autosave_snapshot()
        self._autosave_error_shown = False
        try:
            document, candidate = read_recovery(path, self._autosave_identity())
        except ValueError as exc:
            QMessageBox.warning(self, 'Recovery unavailable', str(exc))
            document = None
        if document:
            newer = not annotation_path or not Path(annotation_path).exists() or candidate.stat().st_mtime_ns > Path(annotation_path).stat().st_mtime_ns
            if newer and document['digest'] != content_digest(self._autosave_baseline):
                answer = QMessageBox.question(self, 'Recover unsaved work',
                    'An automatic recovery draft is available, including boxes and tracking.\nRestore it?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if answer == QMessageBox.Yes:
                    state = document['state']
                    self.start_offset = int(state.get('start_offset', 0))
                    self.end_frame = state.get('end_frame')
                    self.actors_config = state.get('actors_config', self.actors_config)
                    from core.models import LabelDef
                    self.verbs = [LabelDef(**v) for v in state.get('verbs', [])]
                    self._annotation_provenance = state.get('provenance', {})
                    state['class_map'] = {int(k) if str(k).isdigit() else k:v for k,v in state.get('class_map', {}).items()}
                    self._restore_state(state)
                    self.lbl_autosave.setText('Autosave: recovered draft; save reviewed.json when ready')
                    return
                # Keep declined work out of the active recovery slot.
                if path.exists():
                    path.rename(path.with_name(path.name + '.declined.' + str(time.time_ns())))
                backup = path.with_name(path.name + '.bak')
                if backup.exists():
                    backup.rename(backup.with_name(backup.name + '.declined.' + str(time.time_ns())))
        self.lbl_autosave.setText('Autosave: ready')

    def _autosave_tick(self):
        if not self._autosave_path or self._autosave_baseline is None:
            return
        if self._autosave_future:
            if not self._autosave_future.done():
                return  # No unbounded queue and no concurrent reads of live editor objects.
            try:
                changed = self._autosave_future.result()
                if changed:
                    self.lbl_autosave.setText('Autosave: saved ' + time.strftime('%H:%M:%S'))
                self._autosave_error_shown = False
            except Exception as exc:
                self.lbl_autosave.setText('Autosave failed — save manually')
                if not self._autosave_error_shown:
                    self._autosave_error_shown = True
                    QMessageBox.warning(self, 'Autosave failed', str(exc) + '\nSave your work to a writable local folder.')
            self._autosave_future = None
        # Incomplete event drafts are retained without export filters or validation dialogs.
        envelope = dict(identity=self._autosave_identity(), state=self._autosave_snapshot(), saved_at=time.time())
        self._autosave_future = self._autosave_executor.submit(write_recovery, self._autosave_path, envelope, self._autosave_baseline)

    def _finish_autosave(self):
        future = getattr(self, '_autosave_future', None)
        if future:
            future.result(timeout=10)
            self._autosave_future = None

    def _checkpoint_tracking(self):
        if not getattr(self, '_autosave_path', None):
            return
        self._autosave_tick()

    def _stop_autosave(self):
        self._finish_autosave()
        self._autosave_tick()
        self._finish_autosave()
        self._autosave_timer.stop()
        self._autosave_executor.shutdown(wait=True)
        if self._autosave_lock:
            self._autosave_lock.unlock()
