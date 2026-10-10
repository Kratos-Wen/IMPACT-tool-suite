"""Non-blocking, durable local tracking queue with scoped result reconciliation."""
import copy,json,os,sys,time,uuid
from pathlib import Path
from PyQt5 import sip
from PyQt5.QtCore import QProcess,QProcessEnvironment,QTimer,Qt
from PyQt5.QtWidgets import QWidget,QHBoxLayout,QLabel,QProgressBar,QPushButton,QDialog,QVBoxLayout,QListWidget
from core.safe_storage import atomic_json,write_recovery
from core.correction_propagation import plan,protected
from core.tracking_jobs import scoped_boxes,frame_rows,merge_branch,bindings_valid
from core.assembly_timeline import resolve_object,state_at,next_change
from core.frame_review import valid_record

class TrackingTasksMixin:
    def _init_tracking_tasks(self):
        self._tracking_jobs={};self._tracking_active=None;self._tracking_service=None;self._correction_process=None
        self._tracking_service_stopping=False
        self._tracking_stdout=bytearray();self._tracking_stderr=bytearray();self._tracking_closing=False
        self._tracking_row=QWidget(self);row=QHBoxLayout(self._tracking_row);row.setContentsMargins(0,0,0,0)
        self._tracking_label=QLabel('Tracking: ready',self);self._tracking_label.setMinimumWidth(0);row.addWidget(self._tracking_label,1)
        self._tracking_progress=QProgressBar(self);self._tracking_progress.setMaximumWidth(180);self._tracking_progress.setMaximumHeight(18)
        row.addWidget(self._tracking_progress)
        self._tracking_cancel=QPushButton('Cancel',self);self._tracking_cancel.clicked.connect(self._cancel_tracking_job);row.addWidget(self._tracking_cancel)
        self._tracking_details=QPushButton('Details',self);self._tracking_details.clicked.connect(self._show_tracking_tasks);row.addWidget(self._tracking_details)
        self.layout().addWidget(self._tracking_row);self._tracking_row.hide()
        self._tracking_idle_timer=QTimer(self);self._tracking_idle_timer.setSingleShot(True);self._tracking_idle_timer.setInterval(120000)
        self._tracking_idle_timer.timeout.connect(self._release_tracking_service)
        self._tracking_edit_timer=QTimer(self);self._tracking_edit_timer.setSingleShot(True);self._tracking_edit_timer.setInterval(250)
        self._tracking_edit_timer.timeout.connect(self._tracking_check_seed_edits)

    def _tracking_video_identity(self):
        path=Path(self.video_path).resolve();stat=path.stat() if path.is_file() else None
        return dict(path=str(path),bytes=stat.st_size if stat else None,mtime_ns=stat.st_mtime_ns if stat else None,
            frames=int(self.player.frame_count),fps=float(self.player.frame_rate))

    def _tracking_save(self,job):
        atomic_json(Path(job['directory'])/'job.json',job,backup=False)

    def _tracking_hand(self,event_id,actor):
        if self.selected_event_id==event_id:return copy.deepcopy(self.event_draft.get(actor,{}))
        for event in self.events:
            if event['event_id']==event_id:return copy.deepcopy(event.get('hoi_data',{}).get(actor,{}))
        return None

    def _tracking_binding(self,request,event_id,actor):
        hand=self._tracking_hand(event_id,actor)
        if not hand or not hand.get('verb'):return None
        start,end=hand.get('interaction_start'),hand.get('interaction_end');frame=request['start']
        if type(start)is not int or type(end)is not int or not start<=frame<=end:return None
        role=None;identity=None
        if request['entity_kind']=='hand' and self._normalize_hand_label(request['label'])==actor:role='hand';identity=actor
        elif request['entity_kind']=='object':
            if resolve_object(hand,self._assembly_data(),frame)==request['id']:role='object';identity=request['id']
            elif self._hand_instrument_object_id(hand)==request['id']:role='instrument';identity=request['id']
        if role is None:return None
        return dict(event_id=event_id,actor=actor,role=role,identity=identity,start=start,end=end,
            object_name=self._object_name_for_id(identity,fallback=request['label']) if role!='hand' else actor)

    def _tracking_capture(self,request,bindings=None):
        bindings=bindings or [self._tracking_binding(request,self.selected_event_id,actor) for actor in self.event_draft]
        bindings=[b for b in bindings if b]
        entity='H:'+self._normalize_hand_label(request['label']) if request['entity_kind']=='hand' else 'O:'+str(request['id'])
        return dict(boxes=scoped_boxes(self.raw_boxes,request),bindings=bindings,
            assembly=state_at(self._assembly_data(),request['start'],request['id']) if request['entity_kind']=='object' else None,
            review={f:copy.deepcopy(row.get(entity)) for f,row in getattr(self,'frame_review',{}).items() if row.get(entity) is not None},
            suppressed=[s for s in getattr(self,'_suppressed_hand_boxes',[]) if s.startswith(self._normalize_hand_label(request['label'])+':')] if request['entity_kind']=='hand' else [])

    def _enqueue_tracking(self,request,notes=(),retry=0,bindings=None):
        if len([j for j in self._tracking_jobs.values() if j['status'] in ('queued','running')])>=24:
            raise ValueError('The local queue is full. Finish or cancel queued jobs before adding more.')
        identity=self._tracking_video_identity();ident=uuid.uuid4().hex
        directory=Path(self.video_path).resolve().parent/'.impact-recovery'/(Path(self.video_path).name+'.tracking')/ident
        directory.mkdir(parents=True,exist_ok=True)
        request=copy.deepcopy(request);request['cancel_file']=str(directory/'cancel.flag')
        captures=[self._tracking_capture(r,bindings[i] if bindings else None) for i,r in enumerate(request['requests'])]
        if any(not c['bindings'] for c in captures):raise ValueError('A tracking target no longer belongs to its event.')
        job=dict(id=ident,directory=str(directory),identity=identity,request=request,captures=captures,status='queued',
            notes=list(notes),created=time.time(),retry=retry,completed=0,total=sum(abs(r['end']-r['start']) for r in request['requests']))
        atomic_json(directory/'request.json',request,backup=False);self._tracking_save(job);self._tracking_jobs[ident]=job
        self._checkpoint_tracking();self._tracking_row.show();self._update_tracking_status();self._start_next_tracking_job()
        return job

    def _tracking_attach_video(self):
        if not hasattr(self,'_tracking_jobs') or not self.video_path:return
        directory=Path(self.video_path).resolve().parent/'.impact-recovery'/(Path(self.video_path).name+'.tracking')
        if directory.is_dir():
            for path in sorted(directory.glob('*/job.json')):
                try:
                    job=json.loads(path.read_text(encoding='utf-8'))
                    if job['id'] in self._tracking_jobs or job['identity']!=self._tracking_video_identity():continue
                    if job['status'] in ('running','interrupted'):job['status']='queued'
                    elif job['status']=='saving':job['status']='ready'
                    self._tracking_jobs[job['id']]=job
                    if job['status']=='queued':Path(job['request']['cancel_file']).unlink(missing_ok=True)
                except (OSError,ValueError,KeyError):continue
        for job in list(self._tracking_jobs.values()):
            if job['status']=='ready' and job['identity']==self._tracking_video_identity():self._merge_tracking_job(job)
        if self._tracking_jobs:self._tracking_row.show()
        self._update_tracking_status();self._start_next_tracking_job()

    def _ensure_tracking_service(self):
        service=self._tracking_service
        if service is not None and service.state()!=QProcess.NotRunning:return
        service=QProcess(self);self._tracking_service=service;self._tracking_stdout.clear();self._tracking_stderr.clear()
        self._tracking_service_stopping=False
        env=QProcessEnvironment.systemEnvironment()
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):env.insert(key,'1')
        env.insert('PYTHONUNBUFFERED','1');service.setProcessEnvironment(env)
        service.readyReadStandardOutput.connect(self._read_tracking_messages)
        service.readyReadStandardError.connect(self._read_tracking_errors)
        service.started.connect(self._send_tracking_job)
        service.finished.connect(lambda code,status:self._tracking_service_exited(service,code))
        service.errorOccurred.connect(lambda error:self._tracking_start_error(service,error))
        worker=self._tracking_worker_path()
        service.start(os.environ.get('IMPACT_SAM2_PYTHON',sys.executable),[worker])

    def _tracking_worker_path(self):
        return str(Path(__file__).resolve().parents[1]/'tools/sam2_tracking_service.py')

    def _start_next_tracking_job(self):
        if self._tracking_active or self._tracking_closing:return
        # A shutdown command is already in the pipe. Leave new jobs queued
        # until finished starts a fresh backend, rather than writing after it.
        if self._tracking_service_stopping:return
        queue=sorted((j for j in self._tracking_jobs.values() if j['status']=='queued'),key=lambda j:j['created'])
        if not queue:
            if self._tracking_service:self._tracking_idle_timer.start()
            self._update_tracking_status();return
        job=queue[0];self._tracking_idle_timer.stop();self._tracking_active=job['id'];job['status']='running';self._tracking_save(job)
        for field in ('tracking_started','tracking_initial_done','eta_seconds'):job.pop(field,None)
        if self._tracking_service is not None and self._tracking_service.state()==QProcess.Running:self._send_tracking_job()
        else:self._ensure_tracking_service()
        self._correction_process=self._tracking_service;self._update_tracking_status()

    def _send_tracking_job(self):
        if not self._tracking_active or self._tracking_service is None:return
        job=self._tracking_jobs[self._tracking_active]
        message=dict(job_id=job['id'],request=str(Path(job['directory'])/'request.json'),output=str(Path(job['directory'])/'result.json'))
        self._tracking_service.write((json.dumps(message)+'\n').encode('utf-8'))

    def _read_tracking_errors(self):
        if self._tracking_service is None or sip.isdeleted(self._tracking_service):return
        self._tracking_stderr.extend(bytes(self._tracking_service.readAllStandardError()));self._tracking_stderr=self._tracking_stderr[-8000:]

    def _read_tracking_messages(self):
        if self._tracking_service is None or sip.isdeleted(self._tracking_service):return
        self._tracking_stdout.extend(bytes(self._tracking_service.readAllStandardOutput()))
        while b'\n' in self._tracking_stdout:
            line,_,remaining=self._tracking_stdout.partition(b'\n');self._tracking_stdout=bytearray(remaining)
            try:message=json.loads(line)
            except ValueError:continue
            job=self._tracking_jobs.get(message.get('job_id'))
            if not job or job['id']!=self._tracking_active:continue
            if message['type']=='progress':
                job.update(phase=message['phase'],completed=message.get('completed',job['completed']),total=message.get('total',job['total']))
                if message['phase']=='tracking':
                    if not job.get('tracking_started'):job['tracking_started']=time.monotonic();job['tracking_initial_done']=job['completed']
                    processed=job['completed']-job['tracking_initial_done']
                    if processed>=5:job['eta_seconds']=(time.monotonic()-job['tracking_started'])*(job['total']-job['completed'])/processed
                self._update_tracking_status()
            elif message['type'] in ('completed','failed'):
                job['status']='ready' if message['type']=='completed' else 'failed';job['error']=message.get('error','')
                self._tracking_save(job);self._tracking_active=None;self._correction_process=None
                if job['status']=='ready' and not self._tracking_closing and not getattr(self,'_tracking_loading_video',False) and job['identity']==self._tracking_video_identity():self._merge_tracking_job(job)
                self._update_tracking_status();QTimer.singleShot(0,self._start_next_tracking_job)

    def _tracking_start_error(self,service,error):
        if error==QProcess.FailedToStart:self._tracking_service_exited(service,-1)

    def _tracking_service_exited(self,service,code):
        if service is not self._tracking_service:return
        self._tracking_service=None;self._correction_process=None;self._tracking_service_stopping=False
        if self._tracking_active:
            job=self._tracking_jobs[self._tracking_active]
            if job['status']=='running':job['status']='interrupted' if self._tracking_closing else 'failed';job['error']=bytes(self._tracking_stderr).decode(errors='replace')[-2000:] or 'Tracking backend stopped. Completed candidates are retained.'
            self._tracking_save(job);self._tracking_active=None
        if not sip.isdeleted(service):service.deleteLater()
        if not sip.isdeleted(self):
            self._update_tracking_status()
            if not self._tracking_closing:QTimer.singleShot(0,self._start_next_tracking_job)

    def _tracking_end(self,job,request,capture):
        if job['identity']!=self._tracking_video_identity():return None
        valid=[]
        for old in capture['bindings']:
            current=self._tracking_binding(request,old['event_id'],old['actor'])
            if bindings_valid(old,current):valid.append(current)
        if not valid:return None
        direction=request['direction'];end=max((b['end'] if direction>0 else b['start'] for b in valid),key=lambda f:(f-request['start'])*direction)
        if request['entity_kind']=='object':
            composition=state_at(self._assembly_data(),request['start'],request['id'])
            if composition!=capture['assembly']:return None
            if composition:
                if direction<0:end=max(end,composition['frame'])
                else:
                    change=next_change(self._assembly_data(),request['start'],request['id'])
                    if change is not None:end=min(end,change-1)
        return end

    def _merge_tracking_job(self,job):
        try:
            result=json.loads((Path(job['directory'])/'result.json').read_text(encoding='utf-8'))
            if result.get('request_key')!=job['request']:raise ValueError('Tracking result does not match its saved request')
            rows={r['job_index']:r for r in result['results']};requests=job['request']['requests']
            if len(rows)!=len(requests):raise ValueError('Incomplete tracking result')
            job['stops']=[r['stop_reason'] for r in rows.values() if r.get('stop_reason')]
            updated=copy.deepcopy(self.raw_boxes);accepted=[];conflicts=[];reruns=[]
            for index,original in enumerate(requests):
                request=copy.deepcopy(original);capture=job['captures'][index];allowed=self._tracking_end(job,request,capture)
                # A display crop changes storage offsets, not native video frames.
                baseline=copy.deepcopy(capture['boxes']);delta=int(self.start_offset)-request['offset']
                for box in baseline:box['orig_frame']-=delta
                request['offset']=int(self.start_offset)
                actor=self._normalize_hand_label(request['label']);entity='H:'+actor if request['entity_kind']=='hand' else 'O:'+str(request['id'])
                review={f:row.get(entity) for f,row in getattr(self,'frame_review',{}).items() if row.get(entity) is not None}
                changed={int(f) for f in set(review)|set(capture['review']) if review.get(f)!=capture['review'].get(f)}
                if request['entity_kind']=='hand':
                    suppressed={s for s in getattr(self,'_suppressed_hand_boxes',[]) if s.startswith(actor+':')}
                    changed.update(int(s.rsplit(':',1)[1]) for s in suppressed.symmetric_difference(capture['suppressed']))
                updated,branch,conflict=merge_branch(updated,baseline,request,rows[index],allowed,changed)
                if branch:accepted.append(branch)
                if conflict:
                    conflicts.append(dict(branch=index,**conflict))
                    if conflict.get('frame') is not None and allowed is not None and job.get('retry',0)<2:
                        frame=conflict['frame'];anchors=frame_rows(self.raw_boxes,request,frame)
                        if len(anchors)==1 and protected(anchors[0]) and (original['end']-frame)*request['direction']>0:
                            seed=dict(original,start=frame)
                            bindings=[self._tracking_binding(seed,b['event_id'],b['actor']) for b in capture['bindings']]
                            bindings=[b for b in bindings if b]
                            if bindings:
                                bound=max((b['end'] if request['direction']>0 else b['start'] for b in bindings),key=lambda f:(f-frame)*request['direction'])
                                bound=frame+request['direction']*min((original['end']-frame)*request['direction'],(bound-frame)*request['direction'])
                                composition=state_at(self._assembly_data(),frame,request['id']) if not actor else None
                                if composition:
                                    if request['direction']<0:bound=max(bound,composition['frame'])
                                    else:
                                        change=next_change(self._assembly_data(),frame,request['id'])
                                        if change is not None:bound=min(bound,change-1)
                                for text,row in getattr(self,'frame_review',{}).items():
                                    f=int(text)
                                    if 0<(f-frame)*request['direction']<=(bound-frame)*request['direction'] and valid_record(row.get(entity),self._boxes_for_review(entity,f)):bound=f-request['direction']
                                try:reruns.append((plan(self.raw_boxes,anchors[0],frame,bound,int(self.start_offset),request['entity_kind']),bindings))
                                except ValueError:pass
            if accepted:
                self._push_undo();self.raw_boxes=updated;suppressed=set(getattr(self,'_suppressed_hand_boxes',[]))
                for request,row in accepted:
                    if request['entity_kind']=='hand':
                        actor=self._normalize_hand_label(request['label']);suppressed.update(f'{actor}:{f}' for f in row['empty_frames'])
                        for box in row['boxes']:suppressed.discard(f'{actor}:{box["frame"]}')
                self._suppressed_hand_boxes=sorted(suppressed);self._rebuild_bboxes_from_raw();self._bump_bbox_revision();self._bump_query_state_revision()
                self._refresh_boxes_for_frame(self.player.current_frame)
            status=('applied_partial' if job['stops'] or conflicts or not result.get('complete',False) else 'applied') if accepted else 'needs_new_anchor'
            job.update(applied_branches=len(accepted),conflicts=conflicts)
            if accepted and getattr(self,'_autosave_path',None):self._save_tracking_merge(job,status)
            else:job['status']=status;self._tracking_save(job)
            self._log('hoi_background_tracking_merge',job_id=job['id'],branches=len(accepted),conflicts=len(conflicts))
            if reruns:
                new=dict(job['request'],requests=[r for r,b in reruns]);new.pop('cancel_file',None)
                self._enqueue_tracking(new,retry=job.get('retry',0)+1,bindings=[b for r,b in reruns])
        except Exception as exc:
            job.update(status='failed',error=str(exc));self._tracking_save(job)

    def _save_tracking_merge(self,job,status):
        """Commit recovery and job state on the existing save worker, never wait in Qt."""
        job['status']='saving';self._tracking_save(job)
        saved=copy.deepcopy(job);saved['status']=status
        envelope=dict(identity=self._autosave_identity(),state=self._autosave_snapshot(),saved_at=time.time())
        path=self._autosave_path;baseline=self._autosave_baseline;previous=self._autosave_future
        def commit():
            if previous:previous.result()
            changed=write_recovery(path,envelope,baseline)
            atomic_json(Path(saved['directory'])/'job.json',saved,backup=False)
            return changed
        future=self._autosave_executor.submit(commit);self._autosave_future=future
        def completed():
            if not future.done():QTimer.singleShot(25,completed);return
            try:future.result();job['status']=status
            except Exception as exc:job.update(status='failed',error='Recovery save failed: '+str(exc));self._tracking_save(job)
            if not self._tracking_closing and not sip.isdeleted(self):self._update_tracking_status()
        QTimer.singleShot(0,completed)

    def _tracking_check_seed_edits(self):
        if not self._tracking_active or self._tracking_closing or getattr(self,'_tracking_loading_video',False):return
        job=self._tracking_jobs[self._tracking_active]
        if job['identity']!=self._tracking_video_identity():return
        # Finish unaffected targets first. The scoped merge schedules new manual seeds.
        # Intermediate edits are not polled over the whole video in the UI thread.
        for request,capture in zip(job['request']['requests'],job['captures']):
            anchor=frame_rows(self.raw_boxes,dict(request,offset=int(self.start_offset)),request['start'])
            if not anchor:job['phase']='tracking; reference changed';self._update_tracking_status();break

    def _cancel_tracking_job(self,job_id=None):
        ident=job_id if isinstance(job_id,str) else self._tracking_active
        if not ident:return
        job=self._tracking_jobs[ident]
        if job['status'] not in ('queued','running'):return
        job['status']='cancelled';self._tracking_save(job)
        Path(job['request']['cancel_file']).touch()
        if ident==self._tracking_active and self._tracking_service:self._tracking_service.kill()
        self._update_tracking_status()

    def _release_tracking_service(self):
        if self._tracking_active:return
        if self._tracking_service:
            self._tracking_service_stopping=True;self._tracking_service.write(b'{"shutdown": true}\n')

    def _stop_tracking_tasks(self):
        self._tracking_closing=True;self._tracking_idle_timer.stop();self._tracking_edit_timer.stop()
        if self._tracking_active:
            job=self._tracking_jobs[self._tracking_active];job['status']='interrupted';self._tracking_save(job)
        if self._tracking_service:
            self._tracking_service.kill();self._tracking_service.waitForFinished(2000)

    def _resume_tracking_tasks(self):
        self._tracking_closing=False
        for job in self._tracking_jobs.values():
            if job['status']=='interrupted':job['status']='queued';self._tracking_save(job)
        self._start_next_tracking_job()

    def _update_tracking_status(self):
        queued=sum(j['status']=='queued' for j in self._tracking_jobs.values())
        job=self._tracking_jobs.get(self._tracking_active)
        self._tracking_cancel.setEnabled(bool(job))
        if job:
            phase=job.get('phase','Starting');done=job.get('completed',0);total=job['total']
            self._tracking_progress.setRange(0,total if phase=='tracking' else 0);self._tracking_progress.setValue(done)
            eta=f' · ~{int(job["eta_seconds"])} s left' if job.get('eta_seconds') is not None and phase=='tracking' else ''
            ids=', '.join(dict.fromkeys(('hand ' if r['entity_kind']=='hand' else 'ID ')+str(r['label'] if r['entity_kind']=='hand' else r['id']) for r in job['request']['requests']))
            self._tracking_label.setText(f'{Path(job["identity"]["path"]).stem} · {ids} · {phase} · {done}/{total} target-frames{eta} · {queued} queued')
        else:
            self._tracking_progress.setRange(0,1);self._tracking_progress.setValue(0)
            latest=max(self._tracking_jobs.values(),key=lambda j:j['created'],default=None)
            detail=' · open Details' if latest and (latest.get('stops') or latest.get('conflicts') or latest.get('error')) else ''
            self._tracking_label.setText('Tracking: '+(latest['status'].replace('_',' ') if latest else 'ready')+detail+f' · {queued} queued')
        self._tracking_label.setToolTip('Continue editing. Tracking results remain unverified; changed references and human decisions are preserved.')

    def _show_tracking_tasks(self):
        dialog=QDialog(self);dialog.setWindowTitle('Tracking tasks');dialog.setAttribute(Qt.WA_DeleteOnClose)
        layout=QVBoxLayout(dialog);listing=QListWidget(dialog);layout.addWidget(listing)
        ids=[]
        for job in sorted(self._tracking_jobs.values(),key=lambda j:j['created'],reverse=True):
            details=[job.get('error','')]+job.get('stops',[])+[f'Branch {c["branch"]}: {c["reason"]}'+(f' at frame {c["frame"]}' if 'frame' in c else '') for c in job.get('conflicts',[])]
            ids.append(job['id']);listing.addItem(f'{Path(job["identity"]["path"]).stem} · {job["status"]} · {job["completed"]}/{job["total"]}\n'+'\n'.join(t for t in details if t))
        row=QHBoxLayout();layout.addLayout(row)
        def selected():return self._tracking_jobs.get(ids[listing.currentRow()]) if 0<=listing.currentRow()<len(ids) else None
        def cancel():
            job=selected()
            if job:self._cancel_tracking_job(job['id']);dialog.close()
        def retry():
            job=selected()
            if job and job['status'] not in ('queued','running','saving','applied','applied_partial'):
                job['status']='queued';Path(job['request']['cancel_file']).unlink(missing_ok=True);self._tracking_save(job);self._start_next_tracking_job();dialog.close()
        def apply_saved():
            job=selected()
            if job and job['status'] not in ('running','queued','saving','applied','applied_partial') and job['identity']==self._tracking_video_identity():self._merge_tracking_job(job);self._update_tracking_status();dialog.close()
        for text,callback in [('Cancel selected',cancel),('Retry / resume',retry),('Apply saved candidates',apply_saved)]:
            button=QPushButton(text,dialog);button.clicked.connect(callback);row.addWidget(button)
        dialog.resize(650,320);dialog.show();self._tracking_tasks_dialog=dialog
