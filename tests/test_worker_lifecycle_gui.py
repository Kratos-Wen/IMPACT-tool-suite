"""Real Qt threads: closing never abandons a running training worker."""
import os,sys,tempfile,threading,time
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QMessageBox
from PyQt5.QtCore import QThread,QEvent
from core.project_profile import PROFILE
from core.noun_aliases import refresh_aliases
PROFILE.clear();PROFILE.update(noun_classes=['part'],verbs=['hold'],anomaly_labels=[])
refresh_aliases();app=QApplication([]);warnings=[]
QMessageBox.information=lambda *a,**k:QMessageBox.Ok
QMessageBox.warning=lambda *a,**k:(warnings.append(str(a[2])),QMessageBox.Ok)[1]
QMessageBox.question=lambda *a,**k:QMessageBox.Yes
from ui.hoi_window import HOIWindow

with tempfile.TemporaryDirectory() as d:
    root=Path(d);feedback=root/'feedback.jsonl';feedback.write_text('{}\n')
    w=HOIWindow();w._autosave_timer.stop();w._apply_profile_libraries()
    w._semantic_feedback_file=lambda:str(feedback)
    w._semantic_model_file=lambda:str(root/'model.pt')
    w._semantic_feedback_feature_dim=4
    w._semantic_adapter_train_config.update(train_every=1,min_samples=1)
    # Exercise completion, exceptional completion, and immediate replacement.
    for cycle in range(6):
        gate=threading.Event();entered=threading.Event()
        def training(**kwargs):
            entered.set();assert gate.wait(5), 'Test worker release timed out'
            if cycle%2:raise ValueError('Controlled training failure')
            return True,'Controlled training completion',None
        with patch('ui.hoi_window.train_adapter_from_feedback',new=training):
            w._close_request_approved=False;w._semantic_feedback_pending=1
            w._maybe_schedule_semantic_training();worker=w._semantic_adapter_train_worker
            assert worker is not None and entered.wait(2)
            assert worker.isRunning()
            # A previous close approval must not bypass the live-thread check.
            w._close_request_approved=True;w.close()
            assert not getattr(w,'_close_request_finalized',False)
            assert not getattr(w,'_autosave_stopped',False)
            w._maybe_schedule_semantic_training();assert w._semantic_adapter_train_worker is worker
            w._close_request_approved=False;gate.set()
            assert worker.wait(2000)
            # Native completion is queued until the UI processes events. Do not
            # replace the retained worker before that callback releases it.
            w._maybe_schedule_semantic_training();assert w._semantic_adapter_train_worker is worker
            deadline=time.monotonic()+5
            while w._semantic_adapter_train_worker is not None:
                assert time.monotonic()<deadline, 'Completion callback did not clear worker'
                app.processEvents();time.sleep(.001)
            assert not worker.isRunning(), 'Worker reference cleared before thread stopped'
    # The existing YOLO training lane must have the same close protection.
    gate=threading.Event();entered=threading.Event()
    class TrainingLane(QThread):
        def run(self):entered.set();assert gate.wait(5)
    worker=TrainingLane(w);w.train_worker=worker;worker.start();assert entered.wait(2)
    w._close_request_approved=True;assert not w._confirm_close_request()
    gate.set();assert worker.wait(2000)
    w._mark_hoi_saved();w.close();w.player.release_media()
    assert w._close_request_finalized and w._autosave_stopped
    w.deleteLater();app.sendPostedEvents(None,QEvent.DeferredDelete)
assert not warnings,warnings
print('TRAINING_THREAD_CLOSE_COMPLETION_FAILURE_REPLACEMENT_PASS')
