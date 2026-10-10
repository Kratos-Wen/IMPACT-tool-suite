"""Selected event targets, bounded bidirectional proposals and one atomic apply."""
import json,os,sys,tempfile,copy
from pathlib import Path
from PyQt5.QtCore import QProcess,Qt
from PyQt5.QtWidgets import (QMessageBox,QFileDialog,QProgressDialog,QDialog,QVBoxLayout,
    QLabel,QListWidget,QListWidgetItem,QSpinBox,QFormLayout,QDialogButtonBox)
from core.correction_propagation import plan,apply_batch
from core.assembly_timeline import next_change,state_at,resolve_object

class CorrectionPropagationMixin:
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
        if getattr(self,'_correction_process',None) is not None:
            QMessageBox.information(self,'Track','A propagation is already running.');return
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
            cpu_threads=max(1,int(os.environ.get('IMPACT_SAM2_CPU_THREADS','1'))))
        temporary=tempfile.TemporaryDirectory(prefix='impact_correction_')
        req=Path(temporary.name)/'request.json';out=Path(temporary.name)/'result.json'
        req.write_text(json.dumps(request),encoding='utf-8')
        snapshot=copy.deepcopy(self.raw_boxes);video=self.video_path;offset=int(self.start_offset)
        assembly_snapshot=copy.deepcopy(self._assembly_data());events_snapshot=copy.deepcopy(self.events)
        draft_snapshot=copy.deepcopy(self.event_draft)
        self._checkpoint_tracking()
        process=QProcess(self);self._correction_process=process
        progress=QProgressDialog(f'{len(chosen)} targets; {len(requests)} direction branches on {request["device"]}.\n'+'\n'.join(notes[:6]),'Cancel',0,0,self)
        progress.setWindowTitle('Correction propagation');progress.setMinimumDuration(0)
        canceled=[False]
        def cancel():canceled[0]=True;process.kill()
        progress.canceled.connect(cancel)
        stderr=bytearray()
        process.readyReadStandardError.connect(lambda:stderr.extend(bytes(process.readAllStandardError())))
        def finished(code,*args):
            self._correction_process=None;progress.blockSignals(True);progress.close()
            try:
                if canceled[0]:return
                if code!=0 or not out.exists():
                    QMessageBox.warning(self,'Tracking failed',bytes(stderr).decode(errors='replace')[-2000:] or 'Backend not installed or tracking failed.');return
                if self.video_path!=video or int(self.start_offset)!=offset or self.raw_boxes!=snapshot or self._assembly_data()!=assembly_snapshot or self.events!=events_snapshot or self.event_draft!=draft_snapshot:
                    QMessageBox.information(self,'Track','Annotations changed during tracking. Discarded proposals; rerun from the corrected frame.');return
                result=json.loads(out.read_text(encoding='utf-8'))
                updated,completed=apply_batch(self.raw_boxes,requests,result.get('results',[]))
                if not completed:
                    QMessageBox.information(self,'Track','No usable continuation. Correct the first uncertain frame and retry.');return
                lines=[f'{r["entity_kind"]} ID {r["id"]}: {r["start"]} → {r["end"]}, {len(rows.get("boxes",[]))} boxes'+(' — '+rows['stop_reason'] if rows.get('stop_reason') else '') for r,rows in completed]
                answer=QMessageBox.question(self,'Apply tracking proposals','\n'.join(lines)+'\nReplace automatic boxes in these intervals? Human anchors and other IDs are preserved.\nOne Undo restores the batch. Accepting does not mark frames human-verified.')
                if answer!=QMessageBox.Yes:return
                self._push_undo();self.raw_boxes=updated
                suppressed=set(getattr(self,'_suppressed_hand_boxes',[]))
                for r,rows in completed:
                    if r['entity_kind']!='hand':continue
                    actor=self._normalize_hand_label(r['label'])
                    suppressed.update(f'{actor}:{frame}' for frame in rows.get('empty_frames',[]))
                    for row in rows.get('boxes',[]):suppressed.discard(f'{actor}:{row["frame"]}')
                self._suppressed_hand_boxes=sorted(suppressed)
                self._rebuild_bboxes_from_raw();self._bump_bbox_revision();self._bump_query_state_revision()
                self._refresh_boxes_for_frame(self.player.current_frame);self._checkpoint_tracking()
                self._log('hoi_correction_batch',targets=len(chosen),branches=len(completed),begin=begin,end=end)
            except Exception as exc:QMessageBox.warning(self,'Track',str(exc))
            finally:temporary.cleanup();process.deleteLater()
        process.finished.connect(finished)
        def failed(error):
            if error==QProcess.FailedToStart:
                self._correction_process=None;progress.blockSignals(True);progress.close();temporary.cleanup();process.deleteLater()
                QMessageBox.warning(self,'Track','Could not start the configured SAM2 Python interpreter.')
        process.errorOccurred.connect(failed)
        worker=str(Path(__file__).resolve().parents[1]/'tools/sam2_correction_worker.py')
        process.start(os.environ.get('IMPACT_SAM2_PYTHON',sys.executable),[worker,str(req),str(out)])
