import os,sys,copy
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen';sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication,QComboBox,QPushButton,QDialogButtonBox,QMessageBox
from PyQt5.QtCore import QTimer
from core.project_profile import PROFILE
PROFILE.update(counted_object_states={'sample':{'classes':['raw','prepared'],'maximum':2,'noun_by_count':{'0':'raw','1':'prepared','2':'prepared'},'title':'Sample'}})
from ui.hoi_window import HOIWindow
from core.object_states import state_at
app=QApplication([])
for name in ('information','warning','critical'):setattr(QMessageBox,name,lambda *a,**k:QMessageBox.Ok)
w=HOIWindow();w.player.frame_count=120;w.player.frame_rate=15;w.player.current_frame=10
w._register_object_entry(1,'raw');w._register_object_entry(2,'prepared')
w.selected_hand_label='Left_hand';w.event_draft={'Left_hand':{'noun_object_id':1},'Right_hand':{'noun_object_id':1}}
w.events=[];w.selected_event_id=None
errors=[]
def modal(fn):
    def invoke():
        d=app.activeModalWidget()
        try:fn(d)
        except BaseException as exc:errors.append(exc);d.reject()
    QTimer.singleShot(0,invoke);w._open_instance_state_editor()
    assert not errors,errors
def save_count(d,n):
    c=d.findChild(QComboBox,'stateCount');c.setCurrentIndex(c.findData(n))
    d.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
original=copy.deepcopy(w.global_object_map)
modal(lambda d:save_count(d,2))
assert w.global_object_map==original and state_at(w.object_states,1,10)['count']==2
assert w.event_draft['Right_hand']['noun_object_id']==1
assert w._object_name_for_id(1)=='prepared'
snapshot=w._snapshot_state();assert snapshot['object_states']==w.object_states
w.player.current_frame=20;modal(lambda d:save_count(d,1))
w.player.current_frame=30;modal(lambda d:save_count(d,0))
assert w.global_object_map==original and w._object_name_for_id(1)=='raw'
w._restore_state(snapshot);assert state_at(w.object_states,1,30)['count']==2
payload=w._build_payload_v2();assert payload['object_states']==w.object_states
for _ in range(3):w._load_annotations_v2(payload);assert w.object_states==payload['object_states']
w.selected_hand_label='Left_hand';w.event_draft={'Left_hand':{'noun_object_id':1},'Right_hand':{'noun_object_id':1}}
w.player.current_frame=40;before=w._snapshot_state();counter=w.object_id_counter
def cancel_new(d):d.findChild(QPushButton,'newStateInstance').click();d.reject()
modal(cancel_new);assert w.object_id_counter==counter and w.global_object_map==before['global_object_map']
def cut(d):d.findChild(QPushButton,'newStateInstance').click();save_count(d,0)
modal(cut);fresh=w.event_draft['Left_hand']['noun_object_id']
assert fresh not in original.values() and w.event_draft['Right_hand']['noun_object_id']==1
assert state_at(w.object_states,fresh,40)['parent_object_id']==1
assert set(w.global_object_map.values())=={1,2,fresh}
assert state_at(w.object_states,1,40)['count']==2
# An old prepared category never fabricates an exact count.
w.event_draft['Left_hand']['noun_object_id']=2
def unknown(d):assert d.findChild(QComboBox,'stateCount').currentData() is None;d.reject()
modal(unknown)
before=w._snapshot_state();w._restore_state(before);assert w._snapshot_state()['object_states']==before['object_states']
# Explicit selection opens the editor; refreshes alone never do.
shown=[]
def dismiss_popup():
    d=app.activeModalWidget()
    if d is not None:shown.append(True);d.reject()
QTimer.singleShot(50,dismiss_popup)
w._queue_instance_state_editor();app.processEvents()
assert shown and not w._instance_state_dialog
bad=copy.deepcopy(w._build_payload_v2());bad['object_states']['states'][0]['count']=3
before=copy.deepcopy(w.object_states)
try:w._load_annotations_v2(bad)
except ValueError:pass
else:raise AssertionError('Invalid state accepted')
assert w.object_states==before
from core.safe_storage import write_recovery,read_recovery
import tempfile
with tempfile.TemporaryDirectory() as temp:
    path=Path(temp)/'recovery.json';snapshot=w._autosave_snapshot();identity={'trial':'test'}
    assert write_recovery(path,{'identity':identity,'state':snapshot}, {})
    restored,_=read_recovery(path,identity)
    assert restored['state']['object_states']==snapshot['object_states']
w._stop_autosave();w.deleteLater()
print('INSTANCE_STATE_REAL_WINDOW_COUNTS_CUT_ID_CANCEL_UNDO_SAVE_LOAD_PASS')
