"""Per-instance composition shared by hands referencing the same ID."""
from copy import deepcopy
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog,QVBoxLayout,QLabel,QSpinBox,QComboBox,QListWidget,
    QListWidgetItem,QTableWidget,QTableWidgetItem,QCheckBox,QDialogButtonBox,QMessageBox)
from core.assembly_timeline import (state_at,put_state,noun_at,validate_timeline,empty_timeline,
    reference_id,resolve_object,active_id,active_object_ids,merge_instances)
from core.project_profile import PROFILE
from core.annotation_migration import model_from_trial
from core.label_glossary import label_tooltip

class AssemblyEditorMixin:
    def _assembly_data(self):
        return getattr(self,'shared_assembly',empty_timeline())

    def _assembly_frame(self):
        return int(getattr(getattr(self,'player',None),'current_frame',0))

    def _assembly_selected_id(self):
        hand=self.event_draft.get(self.selected_hand_label,{})
        uid=reference_id(hand,self._assembly_data(),self._assembly_frame())
        if uid is not None and state_at(self._assembly_data(),self._assembly_frame(),uid):
            return active_id(self._assembly_data(),uid,self._assembly_frame())
        ids=active_object_ids(self._assembly_data(),self._assembly_frame())
        return ids[0] if len(ids)==1 else None

    def _assembly_default_id(self,frame):
        default=getattr(self,'_assembly_default_reference',False)
        if type(default) is int:
            return active_id(self._assembly_data(),default,frame) if state_at(self._assembly_data(),frame,default) else None
        ids=active_object_ids(self._assembly_data(),frame)
        return ids[0] if default is True and len(ids)==1 else None

    def _set_assembly_reference(self,hand,uid):
        hand.update(shared_assembly_ref=True,shared_assembly_id=uid,noun_object_id=uid,target_object_id=uid)

    def _bind_assembly_object(self,hand,uid):
        if any(s['object_id']==uid for s in self._assembly_data().get('states',[])):
            self._set_assembly_reference(hand,uid)
        else:
            hand['shared_assembly_ref']=False;hand.pop('shared_assembly_id',None)

    def _unlink_shared_assembly(self):
        hand=self.event_draft.get(self.selected_hand_label)
        if hand is None:return
        self._push_undo();uid=resolve_object(hand,self._assembly_data(),self._assembly_frame())
        hand.update(noun_object_id=uid,target_object_id=uid,shared_assembly_ref=False)
        hand.pop('shared_assembly_id',None)
        for event in self.events:
            if event.get('event_id')==self.selected_event_id:
                other=event.get('hoi_data',{}).get(self.selected_hand_label)
                if other is not None:
                    other.update(noun_object_id=uid,target_object_id=uid,shared_assembly_ref=False)
                    other.pop('shared_assembly_id',None)
        self._bump_query_state_revision();self._load_hand_draft_to_ui(self.selected_hand_label)
        self._checkpoint_tracking()

    def _assembly_specs(self):return PROFILE.get('assembly_interfaces',[])

    def _assembly_slot_table(self,parent):
        table=QTableWidget(len(self._assembly_specs()),3,parent);table.setObjectName('assemblyInterfaces')
        table.setHorizontalHeaderLabels(['Interface','Slot 1 secured','Slot 2 secured'])
        for row,spec in enumerate(self._assembly_specs()):
            cell=QTableWidgetItem(spec['id']);cell.setFlags(cell.flags() & ~Qt.ItemIsEditable);table.setItem(row,0,cell)
            for col in (1,2):
                check=QCheckBox();check.setTristate(True)
                check.setToolTip('Checked: secured; unchecked: not secured; dash: unknown');table.setCellWidget(row,col,check)
        return table

    def _assembly_load_slots(self,table,interfaces):
        for row,spec in enumerate(self._assembly_specs()):
            for col,key in ((1,'1'),(2,'2')):
                value=interfaces.get(spec['id'],{}).get(key)
                table.cellWidget(row,col).setCheckState(Qt.PartiallyChecked if value is None else Qt.Checked if value else Qt.Unchecked)

    def _assembly_read_slots(self,table):
        return {spec['id']:{key:None if table.cellWidget(row,col).checkState()==Qt.PartiallyChecked else table.cellWidget(row,col).isChecked()
            for col,key in ((1,'1'),(2,'2'))} for row,spec in enumerate(self._assembly_specs())}

    def _assembly_validate_components(self,components,interfaces):
        model=model_from_trial(getattr(self,'_task_trial_id','')+' '+str(self.video_path))
        forbidden=set(PROFILE.get('model_component_rules',{}).get(model,{}).get('forbidden_components',[]))
        if forbidden.intersection(components):raise ValueError('Selected component is unavailable for this model.')
        for spec in self._assembly_specs():
            slots=interfaces.get(spec['id'],{})
            if spec.get('completed_component') in components and spec.get('base_component') in components and (set(slots)!={'1','2'} or not all(v is True for v in slots.values())):
                raise ValueError(f"{spec['id']}: both slots must be secured before including the completed component.")

    def _commit_assembly_update(self,updated,uid=None,link=False,both=False):
        """State, references, review and registry changes form one undo transaction."""
        updated=validate_timeline(updated);before=self._assembly_data();self._push_undo();self.shared_assembly=updated
        if uid is not None:
            state=state_at(updated,self._assembly_frame(),uid)
            label='assembly' if not state or len(state['components'])!=1 else state['components'][0]
            controls=[getattr(self,n,None) for n in ('combo_target','combo_instrument','combo_inline_noun')]
            blocked=[(c,c.blockSignals(True)) for c in controls if c is not None]
            try:self._register_object_entry(uid,label)
            finally:
                for c,old in blocked:c.blockSignals(old)
        if link and uid is not None:
            self._assembly_default_reference=uid
            def selected(label,hand):return label==self.selected_hand_label or (both and bool(hand.get('verb') or hand.get('interaction_start') is not None))
            for label,hand in self.event_draft.items():
                if selected(label,hand):self._set_assembly_reference(hand,uid)
            for event in self.events:
                if event.get('event_id')==self.selected_event_id:
                    for label,hand in event.get('hoi_data',{}).items():
                        if selected(label,hand):self._set_assembly_reference(hand,uid)
        self.frame_review=deepcopy(getattr(self,'frame_review',{}))
        ids={s['object_id'] for s in before.get('states',[])+updated['states']}
        for text,records in self.frame_review.items():
            for obj in ids:
                if state_at(before,int(text),obj)!=state_at(updated,int(text),obj):records.pop('O:'+str(obj),None)
        for box in self.raw_boxes:
            obj=box.get('id');frame=int(box.get('orig_frame',-1))+int(getattr(self,'start_offset',0))
            if obj in ids and state_at(before,frame,obj)!=state_at(updated,frame,obj):box['human_verified']=False
        self._bump_query_state_revision();self._refresh_boxes_for_frame(self._assembly_frame())
        if self.selected_hand_label:self._load_hand_draft_to_ui(self.selected_hand_label)
        self._refresh_assembly_caption();self._checkpoint_tracking()

    def _open_assembly_editor(self):
        if not getattr(self.player,'cap',None):QMessageBox.information(self,'Assembly','Load a video first.');return
        if hasattr(self.player,'pause'):self.player.pause()
        data=self._assembly_data();dlg=QDialog(self);dlg.setWindowTitle('Assembly composition by ID');layout=QVBoxLayout(dlg)
        layout.addWidget(QLabel('Edit one instance. Hands referencing this ID share its composition.'))
        frame=QSpinBox();frame.setObjectName('assemblyFrame');frame.setRange(0,max(0,self.player.frame_count-1));frame.setValue(self._assembly_frame());layout.addWidget(frame)
        objects=QComboBox();objects.setObjectName('assemblyInstance');objects.addItem('Create a new instance',None)
        for name,uid in self.global_object_map.items():objects.addItem(f'ID {uid}: {name}',uid)
        objects.setCurrentIndex(max(0,objects.findData(self._assembly_selected_id())));layout.addWidget(objects)
        status=QLabel();layout.addWidget(status)
        names=set(PROFILE.get('assembly_components',[]))|{x for s in data.get('states',[]) for x in s['components']};names.discard('assembly')
        model=model_from_trial(getattr(self,'_task_trial_id','')+' '+str(self.video_path))
        forbidden=set(PROFILE.get('model_component_rules',{}).get(model,{}).get('forbidden_components',[]))
        parts=QListWidget();parts.setObjectName('assemblyComponents');layout.addWidget(parts)
        for name in sorted(x for x in names if isinstance(x,str) and x and x.lower().replace(' ','_') not in {'hand','left_hand','right_hand'}):
            item=QListWidgetItem(name);item.setFlags(item.flags()|Qt.ItemIsUserCheckable);item.setToolTip(label_tooltip(name,'noun'));item.setCheckState(Qt.Unchecked)
            if name in forbidden:item.setFlags(item.flags() & ~Qt.ItemIsEnabled);item.setToolTip('Unavailable for this model')
            parts.addItem(item)
        table=self._assembly_slot_table(dlg);layout.addWidget(table);layout.addWidget(QLabel('Slots follow the fixed reference orientation.'))
        link=QCheckBox('Use this ID for the selected hand and as the default for new events');link.setObjectName('assemblyLink');link.setChecked(True);layout.addWidget(link)
        both=QCheckBox('Also use this ID for the other participating hand');both.setObjectName('assemblyBothHands');layout.addWidget(both)
        def load_state():
            uid=objects.currentData();initial=state_at(data,frame.value(),uid,follow_merges=False) if uid is not None else None;initial=initial or {}
            retired=uid is not None and active_id(data,uid,frame.value())!=uid
            status.setText(f'ID {uid} is already merged. Select its receiving ID or an earlier frame.' if retired else 'Changes start at this frame for this ID only.')
            for i in range(parts.count()):
                item=parts.item(i);item.setCheckState(Qt.Checked if item.text() in initial.get('components',[]) and item.text() not in forbidden else Qt.Unchecked)
            self._assembly_load_slots(table,initial.get('interfaces',{}))
        objects.currentIndexChanged.connect(load_state);frame.valueChanged.connect(load_state);load_state()
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons);buttons.rejected.connect(dlg.reject)
        def accept():
            components=[parts.item(i).text() for i in range(parts.count()) if parts.item(i).checkState()==Qt.Checked];interfaces=self._assembly_read_slots(table)
            try:
                self._assembly_validate_components(components,interfaces);uid=objects.currentData()
                if uid is None:
                    used={int(x) for x in self.global_object_map.values()}|{s['object_id'] for s in data.get('states',[])}|{b['id'] for b in self.raw_boxes if type(b.get('id')) is int}
                    uid=max(int(self.object_id_counter),max(used,default=-1)+1)
                updated=put_state(data,dict(frame=frame.value(),object_id=uid,components=components,interfaces=interfaces,composition_review_state='reviewed'))
            except ValueError as exc:QMessageBox.warning(dlg,'Invalid assembly',str(exc));return
            self._commit_assembly_update(updated,uid,link.isChecked(),both.isChecked());dlg.accept()
        buttons.accepted.connect(accept);dlg.resize(620,650);dlg.exec_()

    def _open_assembly_merge(self):
        if not getattr(self.player,'cap',None):QMessageBox.information(self,'Merge assemblies','Load a video first.');return
        if hasattr(self.player,'pause'):self.player.pause()
        data=self._assembly_data();dlg=QDialog(self);dlg.setWindowTitle('Merge assemblies');layout=QVBoxLayout(dlg)
        layout.addWidget(QLabel('Choose the completed-connection frame. The receiving ID remains active.\nCorrect its whole-assembly box at this frame, then propagate.'))
        frame=QSpinBox();frame.setObjectName('mergeFrame');frame.setRange(0,max(0,self.player.frame_count-1));frame.setValue(self._assembly_frame());layout.addWidget(frame)
        source=QComboBox();source.setObjectName('mergeSource');target=QComboBox();target.setObjectName('mergeTarget')
        layout.addWidget(QLabel('Assembly being inserted:'));layout.addWidget(source);layout.addWidget(QLabel('Receiving assembly (ID to keep):'));layout.addWidget(target)
        table=self._assembly_slot_table(dlg);layout.addWidget(table)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons);buttons.rejected.connect(dlg.reject)
        def preview():
            slots={}
            for combo in (target,source):
                state=state_at(data,frame.value(),combo.currentData()) if combo.currentData() is not None else None
                for key,values in (state or {}).get('interfaces',{}).items():slots.setdefault(key,deepcopy(values))
            self._assembly_load_slots(table,slots)
        def fill():
            old=(source.currentData(),target.currentData())
            for combo in (source,target):
                combo.blockSignals(True);combo.clear()
                for uid in active_object_ids(data,frame.value()):
                    state=state_at(data,frame.value(),uid)
                    if len(state['components'])>=2:combo.addItem(f"ID {uid}: {', '.join(state['components'])}",uid)
                combo.blockSignals(False)
            source.setCurrentIndex(max(0,source.findData(old[0])))
            target.setCurrentIndex(target.findData(old[1]) if target.findData(old[1])>=0 else min(1,target.count()-1));preview()
        frame.valueChanged.connect(fill);source.currentIndexChanged.connect(preview);target.currentIndexChanged.connect(preview);fill()
        def accept():
            try:
                updated=merge_instances(data,source.currentData(),target.currentData(),frame.value(),self._assembly_read_slots(table))
                state=state_at(updated,frame.value(),target.currentData());self._assembly_validate_components(state['components'],state.get('interfaces',{}))
            except ValueError as exc:QMessageBox.warning(dlg,'Invalid merge',str(exc));return
            self._commit_assembly_update(updated);dlg.accept()
        buttons.accepted.connect(accept);dlg.resize(620,440);dlg.exec_()

    def _refresh_assembly_caption(self):
        frame=self._assembly_frame();data=self._assembly_data();uid=self._assembly_selected_id();action=getattr(self,'act_shared_assembly',None)
        if action:
            ids=active_object_ids(data,frame);action.setText('Assembly composition...'+(f' [ID {uid}]' if uid is not None else f' [{len(ids)} active]' if ids else ''))
        for obj in active_object_ids(data,frame):
            for name in ('combo_target','combo_inline_noun'):
                combo=getattr(self,name,None)
                if combo is not None:
                    index=combo.findData(obj)
                    if index>=0:
                        old=combo.blockSignals(True);combo.setItemText(index,f'[{obj}] {noun_at(data,frame,obj)}');combo.blockSignals(old)
