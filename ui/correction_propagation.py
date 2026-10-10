"""Selected event targets and bounded bidirectional background proposals."""
import os,copy
from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QMessageBox,QFileDialog,QDialog,QVBoxLayout,
    QLabel,QListWidget,QListWidgetItem,QSpinBox,QFormLayout,QDialogButtonBox)
from core.correction_propagation import plan
from core.assembly_timeline import next_change,state_at,resolve_object
from core.frame_review import valid_record
from ui.tracking_tasks import TrackingTasksMixin

class CorrectionPropagationMixin(TrackingTasksMixin):
    def _correction_targets(self):
        frame=int(self.player.current_frame);offset=int(self.start_offset);targets={}
        for actor,hand in (self.event_draft or {}).items():
            start,end=hand.get('interaction_start'),hand.get('interaction_end')
            if type(start) is not int or type(end) is not int or not start<=frame<=end:continue
            entities=[('H:'+actor,actor)]
            for uid in (resolve_object(hand,self._assembly_data(),frame),self._hand_instrument_object_id(hand)):
                if uid is not None:entities.append(('O:'+str(uid),None))
            for entity,hand_actor in entities:
                if entity in targets:continue
                candidates=[b for b in self.raw_boxes if int(b.get('orig_frame',-1))+offset==frame and
                    (self._normalize_hand_label(b.get('label'))==hand_actor if hand_actor else
                     not self._normalize_hand_label(b.get('label')) and str(b.get('id'))==entity[2:])]
                if len(candidates)!=1:continue
                anchor=copy.deepcopy(candidates[0])
                title=(actor if hand_actor else self._object_name_for_id(anchor['id'],fallback=anchor.get('label','Object')))
                targets[entity]=dict(entity=entity,anchor=anchor,start=start,end=end,
                    title=f'{title} [ID {anchor["id"]}] — frames {start}–{end}')
        return list(targets.values())

    def _choose_correction_batch(self,targets):
        frame=int(self.player.current_frame);fps=float(self.player.frame_rate or 15)
        limit=max(1,int(30*fps));selected=getattr(self,'_selected_edit_box',None) or {}
        actor=self._normalize_hand_label(selected.get('label'))
        selected_key='H:'+actor if actor else 'O:'+str(selected.get('id'))
        dialog=QDialog(self);dialog.setWindowTitle('Track correction');layout=QVBoxLayout(dialog)
        layout.addWidget(QLabel(f'Anchor frame: {frame}. Check each selected box before propagating.'))
        listing=QListWidget(dialog);listing.setObjectName('correctionTargets')
        for target in targets:
            item=QListWidgetItem(target['title']);item.setData(Qt.UserRole,target)
            item.setFlags(item.flags()|Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if target['entity']==selected_key else Qt.Unchecked)
            listing.addItem(item)
        layout.addWidget(listing)
        left=max(min(t['start'] for t in targets),frame-limit)
        right=min(max(t['end'] for t in targets),frame+limit)
        form=QFormLayout();begin=QSpinBox(dialog);begin.setObjectName('correctionStart')
        begin.setRange(left,frame);begin.setValue(max(left,frame-int(10*fps)))
        finish=QSpinBox(dialog);finish.setObjectName('correctionEnd')
        finish.setRange(frame,right);finish.setValue(min(right,frame+int(10*fps)))
        form.addRow('From frame',begin);form.addRow('To frame',finish);layout.addLayout(form)
        layout.addWidget(QLabel('Both sides of the anchor run in one batch. Each direction stops before human anchors or assembly changes.\nAutomatic boxes remain unverified.'))
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel,dialog)
        buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addWidget(buttons)
        if dialog.exec_()!=QDialog.Accepted:return None
        chosen=[listing.item(i).data(Qt.UserRole) for i in range(listing.count()) if listing.item(i).checkState()==Qt.Checked]
        return chosen,begin.value(),finish.value()

    def _plan_correction_batch(self,targets,begin,end):
        start=int(self.player.current_frame);requests=[];notes=[]
        if type(begin) is not int or type(end) is not int or not begin<=start<=end:
            raise ValueError('The interval must include the current anchor frame.')
        limit=max(1,int(30*float(self.player.frame_rate or 15)))
        if start-begin>limit or end-start>limit:raise ValueError('Use at most 30 seconds on each side per batch.')
        for target in targets:
            anchor=target['anchor'];actor=self._normalize_hand_label(anchor.get('label'))
            composition=state_at(self._assembly_data(),start,anchor.get('id')) if not actor else None
            for direction,desired in [(-1,begin),(1,end)]:
                stop=max(desired,target['start']) if direction<0 else min(desired,target['end'])
                if composition:
                    if direction<0:stop=max(stop,composition['frame'])
                    else:
                        change=next_change(self._assembly_data(),start,anchor['id'])
                        if change is not None:stop=min(stop,change-1)
                # Confirmed occlusion has no geometry, but is still a human anchor.
                entity='H:'+actor if actor else 'O:'+str(anchor['id'])
                for text,row in getattr(self,'frame_review',{}).items():
                    frame=int(text)
                    if 0<(frame-start)*direction<=(stop-start)*direction and valid_record(row.get(entity),self._boxes_for_review(entity,frame)):
                        stop=frame-direction
                if stop==start:continue
                try:
                    request=plan(self.raw_boxes,anchor,start,stop,int(self.start_offset),entity_kind='hand' if actor else 'object')
                except ValueError as exc:
                    notes.append(target['title']+': '+str(exc));continue
                requests.append(request)
                if request['end']!=desired:
                    notes.append(f'{target["entity"]}: {start} → {request["end"]} (bounded by event, human anchor or assembly state)')
        return requests,notes

    def _start_correction_propagation(self):
        if self.selected_event_id is None:
            QMessageBox.information(self,'Track','Select an event first.');return
        self._pause()
        targets=self._correction_targets()
        if not targets:
            QMessageBox.information(self,'Track','Draw or select unambiguous hand, Object or instrument boxes inside this event on the current frame.');return
        choice=self._choose_correction_batch(targets)
        if choice is None:return
        chosen,begin,end=choice
        if not chosen:
            QMessageBox.information(self,'Track','Select at least one target.');return
        try:requests,notes=self._plan_correction_batch(chosen,begin,end)
        except ValueError as exc:QMessageBox.information(self,'Track',str(exc));return
        if not requests:
            QMessageBox.information(self,'Track','No continuation within the chosen interval.\n'+'\n'.join(notes));return
        checkpoint=os.environ.get('IMPACT_SAM2_CHECKPOINT','')
        if not Path(checkpoint).is_file():
            checkpoint,_=QFileDialog.getOpenFileName(self,'Select SAM 2.1 small checkpoint','','Checkpoint (*.pt)')
            if not checkpoint:return
        request=dict(requests=requests,video=self.video_path,checkpoint=checkpoint,
            config=os.environ.get('IMPACT_SAM2_CONFIG','configs/sam2.1/sam2.1_hiera_s.yaml'),
            device=os.environ.get('IMPACT_SAM2_DEVICE','cpu'),
            gpu_memory_fraction=float(os.environ.get('IMPACT_SAM2_GPU_MEMORY_FRACTION','0.5')),
            cpu_threads=max(1,int(os.environ.get('IMPACT_SAM2_CPU_THREADS',str(min(4,max(1,(os.cpu_count() or 2)-2)))))))
        try:self._enqueue_tracking(request,notes)
        except (ValueError,OSError) as exc:QMessageBox.warning(self,'Tracking queue',str(exc))
