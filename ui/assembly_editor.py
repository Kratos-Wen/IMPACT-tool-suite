"""Additive shared-composition editor; existing layout and hand fields stay intact."""
from copy import deepcopy
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog,QVBoxLayout,QLabel,QSpinBox,QComboBox,QListWidget,QListWidgetItem,QTableWidget,QTableWidgetItem,QCheckBox,QDialogButtonBox,QMessageBox,QPushButton
from core.assembly_timeline import state_at,put_state,noun_at,validate_timeline
from core.project_profile import PROFILE
from core.annotation_migration import model_from_trial

class AssemblyEditorMixin:
    def _unlink_shared_assembly(self):
        hand=self.event_draft.get(self.selected_hand_label) if self.selected_hand_label else None
        if hand is None:return
        self._push_undo()
        current=state_at(self._assembly_data(),self._assembly_frame())
        if hand.get('shared_assembly_ref') and current:
            hand['noun_object_id']=hand['target_object_id']=current['object_id']
        hand['shared_assembly_ref']=False
        for event in self.events:
            if event.get('event_id')==self.selected_event_id:
                data=event.get('hoi_data',{}).get(self.selected_hand_label)
                if data is not None:
                    data['shared_assembly_ref']=False
                    data['noun_object_id']=hand.get('noun_object_id')
                    data['target_object_id']=hand.get('target_object_id')
        self._bump_query_state_revision()
        if self.selected_hand_label:self._load_hand_draft_to_ui(self.selected_hand_label)
    def _assembly_data(self):
        return getattr(self,'shared_assembly',{'schema':'shared-assembly-1','states':[]})

    def _assembly_frame(self):
        return int(getattr(getattr(self,'player',None),'current_frame',0))

    def _open_assembly_editor(self):
        if not getattr(self.player,'cap',None):
            QMessageBox.information(self,'Shared assembly','Load a video first.');return
        data=self._assembly_data(); initial=state_at(data,self._assembly_frame()) or {}
        dlg=QDialog(self);dlg.setWindowTitle('Shared assembly timeline');layout=QVBoxLayout(dlg)
        layout.addWidget(QLabel('Changes apply from this frame until the next saved state. Both hands share this state.'))
        frame=QSpinBox();frame.setRange(0,max(0,self.player.frame_count-1));frame.setValue(self._assembly_frame());layout.addWidget(frame)
        objects=QComboBox();objects.addItem('Create shared object',None)
        for name,uid in self.global_object_map.items():objects.addItem(f'{uid}: {name}',uid)
        index=objects.findData(initial.get('object_id'));objects.setCurrentIndex(max(0,index));layout.addWidget(objects)
        parts=QListWidget();layout.addWidget(parts)
        names=set(PROFILE.get('assembly_components',[])) | set(initial.get('components',[]))
        names.discard('assembly')
        model=model_from_trial(getattr(self,'_task_trial_id','')+' '+str(self.video_path))
        forbidden=set(PROFILE.get('model_component_rules',{}).get(model,{}).get('forbidden_components',[]))
        for name in sorted(x for x in names if isinstance(x,str) and x and 'hand' not in x.lower()):
            item=QListWidgetItem(name);item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in initial.get('components',[]) else Qt.Unchecked)
            if name in forbidden:
                item.setCheckState(Qt.Unchecked);item.setFlags(item.flags() & ~Qt.ItemIsEnabled);item.setToolTip('Unavailable for this model')
            parts.addItem(item)
        specs=PROFILE.get('assembly_interfaces',[])
        table=QTableWidget(len(specs),3);table.setHorizontalHeaderLabels(['Interface','Slot 1 secured','Slot 2 secured']);layout.addWidget(table)
        for row,spec in enumerate(specs):
            cell=QTableWidgetItem(spec['id']);cell.setFlags(cell.flags() & ~Qt.ItemIsEditable);table.setItem(row,0,cell)
            for col,key in ((1,'1'),(2,'2')):
                check=QCheckBox();check.setTristate(True)
                value=initial.get('interfaces',{}).get(spec['id'],{}).get(key)
                check.setCheckState(Qt.PartiallyChecked if value is None else Qt.Checked if value else Qt.Unchecked)
                check.setToolTip('Checked: secured; unchecked: not secured; dash: unknown')
                table.setCellWidget(row,col,check)
        layout.addWidget(QLabel('Slots follow the fixed reference orientation, never the current camera view.'))
        link=QCheckBox('Use shared Object for the selected hand and new events');link.setChecked(True);layout.addWidget(link)
        both=QCheckBox('Also use shared Object for the other participating hand');layout.addWidget(both)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons);buttons.rejected.connect(dlg.reject)
        def accept():
            components=[parts.item(i).text() for i in range(parts.count()) if parts.item(i).checkState()==Qt.Checked]
            interfaces={spec['id']:{key:(None if table.cellWidget(row,col).checkState()==Qt.PartiallyChecked else table.cellWidget(row,col).isChecked()) for col,key in ((1,'1'),(2,'2'))} for row,spec in enumerate(specs)}
            try:
                for spec in specs:
                    if spec.get('completed_component') in components and spec.get('base_component') in components and not all(interfaces[spec['id']].values()):
                        raise ValueError(f"{spec['id']}: both slots must be secured before including the completed component.")
                uid=objects.currentData()
                if uid is None:
                    used=set(self.global_object_map.values())|{b.get('id') for b in self.raw_boxes if isinstance(b.get('id'),int)}
                    uid=max([int(self.object_id_counter),*(x+1 for x in used if type(x) is int)],default=0)
                updated=put_state(data,dict(frame=frame.value(),object_id=uid,components=components,interfaces=interfaces,composition_review_state='reviewed'))
            except ValueError as exc:QMessageBox.warning(dlg,'Invalid assembly',str(exc));return
            self._push_undo();self.shared_assembly=updated;self._assembly_default_reference=link.isChecked()
            reviews=deepcopy(getattr(self,'frame_review',{}))
            for text,records in reviews.items():
                old=state_at(data,int(text));new=state_at(updated,int(text))
                if old!=new:
                    for state in (old,new):
                        if state:records.pop('O:'+str(state['object_id']),None)
            self.frame_review=reviews
            controls=[getattr(self,n,None) for n in ('combo_target','combo_instrument','combo_inline_noun')]
            blocked=[(c,c.blockSignals(True)) for c in controls if c is not None]
            try:self._register_object_entry(uid,'assembly' if len(components)>1 else components[0])
            finally:
                for c,old in blocked:c.blockSignals(old)
            def link_hand(hand):
                hand['shared_assembly_ref']=link.isChecked()
                if link.isChecked():
                    start=hand.get('interaction_start')
                    current=state_at(updated,start if type(start) is int else frame.value())
                    hand['noun_object_id']=hand['target_object_id']=current['object_id'] if current else None
            def selected(label,hand):
                return label==self.selected_hand_label or (both.isChecked() and bool(hand.get('verb') or hand.get('interaction_start') is not None))
            for label,hand in self.event_draft.items():
                if selected(label,hand):link_hand(hand)
            for event in self.events:
                if event.get('event_id')==self.selected_event_id:
                    for label,hand in event.get('hoi_data',{}).items():
                        if selected(label,hand):link_hand(hand)
            self._refresh_boxes_for_frame(self.player.current_frame)
            if self.selected_hand_label:self._load_hand_draft_to_ui(self.selected_hand_label)
            self._bump_query_state_revision();dlg.accept()
        buttons.accepted.connect(accept);dlg.resize(620,600);dlg.exec_()

    def _refresh_assembly_caption(self):
        action=getattr(self,'act_shared_assembly',None)
        state=state_at(self._assembly_data(),self._assembly_frame())
        noun=noun_at(self._assembly_data(),self._assembly_frame())
        if action:
            action.setText('Shared assembly...' + (f' [{noun}]' if noun else ''))
        if state:
            for name in ('combo_target','combo_inline_noun'):
                combo=getattr(self,name,None)
                if combo is not None:
                    index=combo.findData(state['object_id'])
                    if index>=0:
                        old=combo.blockSignals(True)
                        combo.setItemText(index,f"[{state['object_id']}] {noun}")
                        combo.blockSignals(old)
