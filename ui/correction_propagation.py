"""Small addition to the existing English editor: corrected box -> bounded proposals."""
import json, os, sys, tempfile, copy
from pathlib import Path
from PyQt5.QtCore import QProcess
from PyQt5.QtWidgets import QPushButton, QMessageBox, QInputDialog, QFileDialog, QProgressDialog
from core.correction_propagation import plan, apply
from core.assembly_timeline import next_change,state_at

class CorrectionPropagationMixin:
    def _start_correction_propagation(self):
        if getattr(self, "_correction_process", None) is not None:
            QMessageBox.information(self,"Track","A propagation is already running."); return
        anchor=getattr(self,"_selected_edit_box",None)
        if not isinstance(anchor,dict):
            QMessageBox.information(self,"Track","First select and correct the object box on the current frame."); return
        actor=self._normalize_hand_label(anchor.get('label'))
        hand=self._selected_hand_data() or {}; start=int(self.player.current_frame)
        if actor and actor!=self.selected_hand_label:
            QMessageBox.information(self,'Track','Select this hand’s event before propagating its box.');return
        if not actor:
            from core.assembly_timeline import resolve_object
            allowed={resolve_object(hand,self._assembly_data(),start),self._hand_instrument_object_id(hand)}
            if anchor.get('id') not in allowed:
                QMessageBox.information(self,'Track','Select the event that uses this Object or instrument.');return
        try: event_start=int(hand["interaction_start"]); event_end=int(hand["interaction_end"])
        except (KeyError,TypeError,ValueError):
            QMessageBox.information(self,"Track","Set this event's Start and End first."); return
        if not event_start <= start <= event_end:
            QMessageBox.information(self,"Track","The corrected frame must be inside the selected event and within its interval."); return
        # The actual current-frame box, not a stale selection from another frame.
        candidates=[b for b in self.raw_boxes if (self._normalize_hand_label(b.get("label"))==actor if actor else b.get("id")==anchor.get("id") and not self._normalize_hand_label(b.get("label"))) and int(b.get("orig_frame",-1))+int(self.start_offset)==start]
        if len(candidates)!=1:
            QMessageBox.information(self,"Track","Select or draw one unambiguous box with this ID on the current frame."); return
        anchor=candidates[0]
        fps=float(getattr(self.player,"frame_rate",15) or 15)
        choices=[]
        if start<event_end:choices.append('Following frames')
        if start>event_start:choices.append('Preceding frames')
        if not choices:return
        choice,ok=QInputDialog.getItem(self,'Propagation direction','From the corrected frame:',choices,0,False)
        if not ok:return
        direction=1 if choice=='Following frames' else -1
        available=event_end-start if direction>0 else start-event_start
        following,ok=QInputDialog.getInt(self,'Track corrected instance',f"ID {anchor['id']} — {choice.lower()} ({fps:g} fps):",min(available,int(10*fps)),1,min(available,int(30*fps)))
        if not ok:return
        end=start+direction*following
        composition=state_at(self._assembly_data(),start,anchor.get('id')) if not actor else None
        shared=not actor and composition is not None and composition['object_id']==anchor.get('id')
        if shared:
            if direction>0:
                change=next_change(self._assembly_data(),start,anchor.get('id'))
                if change is not None:end=min(end,change-1)
            else:end=max(end,composition['frame'])
        if end==start:
            QMessageBox.information(self,'Track','No continuation within this composition. Correct the adjacent composition separately.');return
        try: request=plan(self.raw_boxes,anchor,start,end,int(self.start_offset),entity_kind='hand' if actor else 'object')
        except ValueError as exc: QMessageBox.information(self,"Track",str(exc)); return
        checkpoint=os.environ.get("IMPACT_SAM2_CHECKPOINT", "")
        if not Path(checkpoint).is_file():
            checkpoint,_=QFileDialog.getOpenFileName(self,"Select SAM 2.1 small checkpoint", "", "Checkpoint (*.pt)")
            if not checkpoint:return
        config=os.environ.get("IMPACT_SAM2_CONFIG","configs/sam2.1/sam2.1_hiera_s.yaml")
        device=os.environ.get("IMPACT_SAM2_DEVICE","cpu")
        request.update(video=self.video_path,checkpoint=checkpoint,config=config,device=device,gpu_memory_fraction=float(os.environ.get("IMPACT_SAM2_GPU_MEMORY_FRACTION","0.5")),cpu_threads=max(1,int(os.environ.get("IMPACT_SAM2_CPU_THREADS","1"))))
        temporary=tempfile.TemporaryDirectory(prefix="impact_correction_")
        req=Path(temporary.name)/"request.json"; out=Path(temporary.name)/"result.json"
        req.write_text(json.dumps(request),encoding="utf-8")
        snapshot=copy.deepcopy(self.raw_boxes); video=self.video_path; offset=int(self.start_offset)
        self._checkpoint_tracking()
        assembly_snapshot=copy.deepcopy(self._assembly_data())
        event_snapshot=copy.deepcopy(self.events)
        process=QProcess(self); self._correction_process=process
        dialog=QProgressDialog(f"Tracking ID {request['id']}: frames {start}–{request['end']} on {device}.\nResults are proposals, not reviewed annotations.","Cancel",0,0,self)
        dialog.setWindowTitle("Correction propagation"); dialog.setMinimumDuration(0)
        dialog.canceled.connect(process.kill)
        stderr=bytearray()
        process.readyReadStandardError.connect(lambda:stderr.extend(bytes(process.readAllStandardError())))
        def finished(code,*args):
            self._correction_process=None; dialog.close()
            try:
                if code!=0 or not out.exists():
                    QMessageBox.warning(self,"Tracking failed",bytes(stderr).decode(errors="replace")[-2000:] or "Cancelled / backend not installed.");return
                if self.video_path!=video or int(self.start_offset)!=offset or self.raw_boxes!=snapshot or self._assembly_data()!=assembly_snapshot or self.events!=event_snapshot:
                    QMessageBox.information(self,"Track","Annotations changed during tracking. Discarded proposals; rerun from the corrected frame.");return
                result=json.loads(out.read_text(encoding="utf-8"))
                if not min(start,request["end"]) <= int(result["end"]) <= max(start,request["end"]): raise ValueError("Backend returned invalid end frame")
                request["end"]=int(result["end"])
                if request["end"]==start:
                    QMessageBox.information(self,"Track",result.get("stop_reason") or "No usable continuation. Correct the first uncertain frame and retry.");return
                answer=QMessageBox.question(self,"Apply tracking proposals",f"ID {request['id']}: {len(result['boxes'])} predicted boxes; {len(result['empty_frames'])} empty-mask frames.\n{result.get('stop_reason', '')}\nReplace this ID's automatic boxes only in frames {min(start+direction,request['end'])}–{max(start+direction,request['end'])}?\nThe corrected frame, other IDs and human anchors are preserved. Undo restores this operation.\nAccepting does not mark these frames human-verified.")
                if answer!=QMessageBox.Yes:return
                updated=apply(self.raw_boxes,request,result["boxes"])
                self._push_undo(); self.raw_boxes=updated
                if actor:
                    predicted={str(row['frame']) for row in result['boxes']}
                    suppressed=set(getattr(self,'_suppressed_hand_boxes',[]))
                    for frame in result.get('empty_frames',[]):suppressed.add(f'{actor}:{frame}')
                    suppressed={key for key in suppressed if not (key.startswith(actor+':') and key.rsplit(':',1)[1] in predicted)}
                    self._suppressed_hand_boxes=sorted(suppressed)
                self._rebuild_bboxes_from_raw(); self._bump_bbox_revision(); self._bump_query_state_revision()
                self._refresh_boxes_for_frame(self.player.current_frame)
                self._checkpoint_tracking()
                self._log("hoi_correction_propagation",box_id=request["id"],start=start,end=request["end"],predicted=len(result["boxes"]))
            except Exception as exc: QMessageBox.warning(self,"Track",str(exc))
            finally:temporary.cleanup();process.deleteLater()
        process.finished.connect(finished)
        def failed(error):
            if error==QProcess.FailedToStart:
                self._correction_process=None;dialog.close();temporary.cleanup()
                QMessageBox.warning(self,"Track","Could not start the configured SAM2 Python interpreter.")
        process.errorOccurred.connect(failed)
        worker=str(Path(__file__).resolve().parents[1]/"tools/sam2_correction_worker.py")
        process.start(os.environ.get("IMPACT_SAM2_PYTHON",sys.executable),[worker,str(req),str(out)])
