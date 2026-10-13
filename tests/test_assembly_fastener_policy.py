"""Fasteners cannot reappear as parts through profiles, old states or promotions."""
import os, sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.assembly_policy import component_options, excluded_selections, slot_fastener
from core.project_profile import PROFILE
profile=dict(noun_classes=['shell','plate','bolt_long','bolt_short','bolt_dark','wrong_bolt','spring','retainer','assembly'],
    noun_aliases={'old_bolt':'bolt_long'},assembly_components=['shell','plate','spring','retainer','bolt_long','bolt_short','bolt_dark','wrong_bolt'],
    assembly_forbidden_components=['wrong_bolt'],assembly_interfaces=[
      dict(id='joint-one',base_component='shell',completed_component='plate',fastener_class='bolt_long',slot_components=['retainer']),
      dict(id='joint-two',fastener_by_model={'A':'bolt_short','B':'bolt_dark'})])
old=[dict(components=['shell','plate','old_bolt','bolt_dark_1','wrong_bolt','retainer_1'])]
assert component_options(profile,old)==['plate','shell','spring']
assert excluded_selections(old[0]['components'],profile)==['old_bolt','bolt_dark_1','wrong_bolt','retainer_1']
assert slot_fastener(profile['assembly_interfaces'][0],'B')=='bolt_long'
assert slot_fastener(profile['assembly_interfaces'][1],'A')=='bolt_short'
assert slot_fastener(profile['assembly_interfaces'][1],'B')=='bolt_dark'
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
from PyQt5.QtWidgets import QApplication, QWidget, QListWidget, QDialogButtonBox, QCheckBox
from PyQt5.QtCore import QTimer
from types import SimpleNamespace
from core.assembly_timeline import empty_timeline, put_state
from ui.assembly_editor import AssemblyEditorMixin
app=QApplication([]);PROFILE.clear();PROFILE.update(profile)
class Editor(QWidget, AssemblyEditorMixin):
    def _noun_model_trial(self):return '1234567_r0_a_a'
e=Editor();parent=QWidget();table=e._assembly_slot_table(parent)
assert 'bolt_long' in table.item(0,0).text()
assert 'retainer' in table.item(0,0).text()
assert table.rowCount()==2 and table.columnCount()==3
assert 'whole connection' in table.cellWidget(0,1).toolTip()
assert 'bolt_short' in table.item(1,0).text() and 'bolt_dark' not in table.item(1,0).text()
slots={'joint-one':{'1':True,'2':True},'joint-two':{'1':None,'2':False}}
e._assembly_load_slots(table,slots);assert e._assembly_read_slots(table)==slots
e._assembly_validate_components(['shell','plate'],slots)
for name in ('bolt_long','old_bolt','bolt_dark_1','wrong_bolt','retainer'):
    try:e._assembly_validate_components(['shell',name],slots)
    except ValueError:pass
    else:raise AssertionError('Excluded part accepted: '+name)
try:e._assembly_validate_components(['shell','plate'],{'joint-one':{'1':True,'2':False}})
except ValueError:pass
else:raise AssertionError('Incomplete connection accepted')
e.player=SimpleNamespace(cap=True,frame_count=30,current_frame=0,pause=lambda:None)
e.global_object_map={'assembly':5,'wrong_bolt':7};e.object_id_counter=8;e.raw_boxes=[]
e.selected_hand_label='Left_hand';e.event_draft={'Left_hand':{'noun_object_id':5}}
e.shared_assembly=put_state(empty_timeline(),dict(frame=0,object_id=5,
    components=['shell','plate','bolt_long','wrong_bolt','retainer'],interfaces=slots))
captured=[];e._commit_assembly_update=lambda data,*args:captured.append(data)
def inspect_and_save():
    dialog=app.activeModalWidget();parts=dialog.findChild(QListWidget,'assemblyComponents')
    assert {parts.item(i).text() for i in range(parts.count())}=={'shell','plate','spring'}
    dialog.findChild(QCheckBox,'assemblyLink').setChecked(False)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
QTimer.singleShot(0,inspect_and_save);e._open_assembly_editor()
assert captured[0]['states'][0]['components']==['plate','shell']
assert captured[0]['states'][0]['interfaces']==slots
assert e.shared_assembly['states'][0]['components']==['shell','plate','bolt_long','wrong_bolt','retainer']
e.event_draft={'Left_hand':{'noun_object_id':7}}
assert not any(uid==7 for uid,_,_ in e._assembly_instance_choices(0))
e.close()
parent.close()
print('ASSEMBLY_FASTENER_OPTIONS_LEGACY_ALIASES_MODEL_SLOTS_SAVE_GUARDS_PASS')
