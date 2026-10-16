"""Profile-driven instance state editor next to the selected Object."""
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QSpinBox,QPushButton,QDialogButtonBox,QMessageBox
from core.project_profile import PROFILE
from core.instance_category import instance_category
from core.object_states import empty_states,state_at,put_state,noun_at,next_instance_id

class InstanceStateMixin:
    def _instance_states(self):return getattr(self,'object_states',empty_states())

    def _instance_kind(self,uid):
        name=next((self._noun_display_name(n) for n,v in self.global_object_map.items() if v==uid),None)
        for kind,spec in PROFILE.get('counted_object_states',{}).items():
            if name in spec.get('classes',[]):return kind
        return None

    def _instance_state_noun(self,uid,frame,fallback):
        return noun_at(self._instance_states(),uid,frame,fallback,PROFILE)

    def _selected_state_instance(self):
        hand=getattr(self,'event_draft',{}).get(getattr(self,'selected_hand_label',None),{})
        return hand.get('noun_object_id',hand.get('target_object_id'))

    def _refresh_instance_state_ui(self):
        button=getattr(self,'btn_inline_instance_state',None)
        if button is None:return
        if not PROFILE.get('counted_object_states'):
            button.setVisible(False);return
        uid=self._selected_state_instance();kind=self._instance_kind(uid)
        button.setVisible(kind is not None)
        if kind is not None:
            spec=PROFILE['counted_object_states'][kind];s=state_at(self._instance_states(),uid,self._assembly_frame())
            count='?' if s is None or s['count'] is None else str(s['count'])
            button.setText(f"{spec.get('title','Object state')} [ID {uid}]: {count}/{spec['maximum']}…")
        # Refresh text only. Never emit selection changes or create instances on seek.
        kinds={n:k for k,spec in PROFILE.get('counted_object_states',{}).items() for n in spec.get('classes',[])}
        bases={obj:self._noun_display_name(name) for name,obj in self.global_object_map.items()}
        current={}
        for s in self._instance_states().get('states',[]):
            if s['frame']<=self._assembly_frame():current[s['object_id']]=s
        labels={}
        for obj,base in bases.items():
            if base not in kinds:continue
            s=current.get(obj)
            noun=base if s is None or s['count'] is None else PROFILE['counted_object_states'][s['kind']]['noun_by_count'][str(s['count'])]
            labels[obj]=f'[{obj}] {noun}'
        for name in ('combo_target','combo_inline_noun'):
            combo=getattr(self,name,None)
            if combo is None:continue
            old=combo.blockSignals(True)
            try:
                for i in range(combo.count()):
                    text=labels.get(combo.itemData(i))
                    if text is not None and combo.itemText(i)!=text:combo.setItemText(i,text)
            finally:combo.blockSignals(old)

    def _queue_instance_state_editor(self,*args):
        uid=self._selected_state_instance()
        if self._instance_kind(uid) is None:return
        self._instance_state_pending_uid=uid
        if getattr(self,'_instance_state_dialog',False) or getattr(self,'_instance_state_queued',False):return
        self._instance_state_queued=True
        video=getattr(self,'video_path',None)
        def show():
            self._instance_state_queued=False
            target=self._instance_state_pending_uid
            if video==getattr(self,'video_path',None) and self._selected_state_instance()==target:self._open_instance_state_editor()
        QTimer.singleShot(0,show)

    def _open_instance_state_editor(self):
        uid=self._selected_state_instance();kind=self._instance_kind(uid)
        if kind is None or getattr(self,'_instance_state_dialog',False):return
        if int(getattr(self.player,'frame_count',0))<=0:
            QMessageBox.information(self,'Object state','Load a video first.');return
        spec=PROFILE['counted_object_states'][kind];self.player.pause();self._instance_state_dialog=True
        dlg=QDialog(self);dlg.setWindowTitle(spec.get('title','Object state')+' by instance ID');layout=QVBoxLayout(dlg)
        layout.addWidget(QLabel('Same physical item: keep its ID. Both hands share its state.'))
        objects=QComboBox();objects.setObjectName('stateInstance')
        for obj in sorted(set(self.global_object_map.values())):
            if self._instance_kind(obj)==kind:objects.addItem(f'ID {obj}: {self._object_name_for_id(obj)}',obj)
        objects.setCurrentIndex(objects.findData(uid));layout.addWidget(objects)
        frame=QSpinBox();frame.setObjectName('stateFrame');frame.setRange(0,max(0,self.player.frame_count-1));frame.setValue(self._assembly_frame())
        layout.addWidget(QLabel('State takes effect at frame:'));layout.addWidget(frame)
        layout.addWidget(QLabel(spec.get('prompt','Completed count:')))
        count=QComboBox();count.setObjectName('stateCount');count.addItem('Not confirmed',None)
        for n in range(spec['maximum']+1):count.addItem(str(n),n)
        layout.addWidget(count)
        hint=QLabel(spec.get('help',''));hint.setWordWrap(True);layout.addWidget(hint)
        new=QPushButton(spec.get('new_label','New separated piece (new ID)'));new.setObjectName('newStateInstance');layout.addWidget(new)
        pending={'new':False,'parent':None}
        def load():
            if pending['new']:return
            s=state_at(self._instance_states(),objects.currentData(),frame.value())
            count.setCurrentIndex(count.findData(s['count'] if s else None))
        objects.currentIndexChanged.connect(load);frame.valueChanged.connect(load);load()
        def create():
            parent=objects.currentData();fresh=next_instance_id(self.global_object_map,self.raw_boxes,self._instance_states(),self.object_id_counter)
            pending.update(new=True,parent=parent)
            objects.blockSignals(True);objects.clear();objects.addItem(f'New piece: ID {fresh} (from ID {parent})',fresh);objects.blockSignals(False)
            new.setEnabled(False);count.setCurrentIndex(count.findData(0))
        new.clicked.connect(create)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons);buttons.rejected.connect(dlg.reject)
        def save():
            obj=objects.currentData()
            s=dict(object_id=obj,frame=frame.value(),kind=kind,count=count.currentData())
            if pending['new']:s['parent_object_id']=pending['parent'];s['origin']='separated_piece'
            try:
                updated=put_state(self._instance_states(),s,PROFILE)
                if pending['new']:
                    expected=next_instance_id(self.global_object_map,self.raw_boxes,self._instance_states(),self.object_id_counter)
                    if obj!=expected:raise ValueError('Object IDs changed; reopen the editor before creating this instance.')
            except ValueError as exc:QMessageBox.warning(dlg,'Invalid state',str(exc));return
            self._push_undo();self.object_states=updated
            if pending['new']:self._register_object_entry(obj,spec['noun_by_count']['0'])
            # Editing an instance state never rewrites event participants. A cut
            # event can still reference the parent before the new piece exists.
            self._bump_query_state_revision();self._load_hand_draft_to_ui(self.selected_hand_label)
            self._refresh_boxes_for_frame(self._assembly_frame());self._refresh_instance_state_ui();self._checkpoint_tracking();dlg.accept()
            if pending['new']:
                QMessageBox.information(self,spec.get('title','Object state'),
                    f'Created ID {obj}. Existing events keep their Object IDs. Select this new ID in Noun for events using the separated piece.')
        buttons.accepted.connect(save)
        try:dlg.exec_()
        finally:self._instance_state_dialog=False
