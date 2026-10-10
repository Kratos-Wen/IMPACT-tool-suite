"""Resident local SAM backend. JSON progress, durable results, shared video states."""
import contextlib,gc,json,os,shutil,sys,tempfile,time
from collections import OrderedDict,defaultdict
from pathlib import Path

def atomic_output(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(data,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)

class FrameCache:
    """At most two decoded intervals and a bounded disk cache; native frames only."""
    def __init__(self,budget=512*1024**2):
        self.temporary=tempfile.TemporaryDirectory(prefix='impact_sam_frames_');self.entries=OrderedDict();self.budget=budget
    def close(self):self.temporary.cleanup()
    def get(self,video,start,end,notify):
        import cv2
        source=Path(video).resolve();stat=source.stat();direction=1 if end>start else -1
        key=(str(source),stat.st_size,stat.st_mtime_ns,start,end)
        if key in self.entries:
            path,size=self.entries.pop(key);self.entries[key]=(path,size);return path
        while len(self.entries)>=2:
            _,(old,_) = self.entries.popitem(last=False);shutil.rmtree(old,ignore_errors=True)
        path=Path(self.temporary.name)/str(time.time_ns());path.mkdir();cap=cv2.VideoCapture(str(source));size=0
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES,min(start,end))
            for index in range(abs(end-start)+1):
                ok,image=cap.read()
                if not ok:raise RuntimeError('Could not decode frame '+str(min(start,end)+index))
                output=path/f'{index if direction>0 else abs(end-start)-index:06d}.jpg'
                if not cv2.imwrite(str(output),image,[cv2.IMWRITE_JPEG_QUALITY,100]):raise RuntimeError('Frame write failed')
                size+=output.stat().st_size
                notify('preparing',prepared=index+1,prepare_total=abs(end-start)+1)
        except Exception:
            shutil.rmtree(path,ignore_errors=True);raise
        finally:cap.release()
        self.entries[key]=(path,size)
        while len(self.entries)>1 and sum(v[1] for v in self.entries.values())>self.budget:
            _,(old,_) = self.entries.popitem(last=False);shutil.rmtree(old,ignore_errors=True)
        return path
    def trim(self):
        while self.entries and sum(v[1] for v in self.entries.values())>self.budget:
            _,(old,_) = self.entries.popitem(last=False);shutil.rmtree(old,ignore_errors=True)

class Backend:
    def __init__(self,request,predictor=None):
        import cv2,torch
        from sam2.build_sam import build_sam2_video_predictor
        device=request.get('device','cpu')
        if device not in ('cpu','cuda'):raise ValueError('Invalid device')
        threads=max(1,int(request.get('cpu_threads',1)))
        if os.environ.get('SLURM_CPUS_PER_TASK'):threads=min(threads,int(os.environ['SLURM_CPUS_PER_TASK']))
        torch.set_num_threads(threads);cv2.setNumThreads(1)
        if device=='cuda':torch.cuda.set_per_process_memory_fraction(float(request.get('gpu_memory_fraction',.5)))
        self.predictor=predictor if predictor is not None else build_sam2_video_predictor(request['config'],request['checkpoint'],device=device)
        self.frames=FrameCache();self.device=device
    def close(self):self.frames.close()

def model_key(request):
    checkpoint=Path(request['checkpoint']).resolve();stat=checkpoint.stat()
    return (str(checkpoint),stat.st_size,stat.st_mtime_ns,request['config'],request.get('device','cpu'),request.get('cpu_threads',1),request.get('gpu_memory_fraction',.5))

def run_batch(request,output,backend=None,notify=None):
    import numpy as np,torch
    jobs=request.get('requests',[])
    if not jobs:raise ValueError('Select at least one tracking branch')
    notify=notify or (lambda *args,**kwargs:None);started=time.perf_counter();owns=backend is None
    if owns:notify('loading');backend=Backend(request)
    predictor=backend.predictor;total=sum(abs(job['end']-job['start']) for job in jobs);done=0
    previous={};finished=set()
    path=Path(output)
    if path.is_file():
        old=json.loads(path.read_text(encoding='utf-8'))
        # Reuse complete branches only when the saved request identity is exact.
        if old.get('request_key')==request:
            previous={r['job_index']:r for r in old.get('results',[])};finished=set(old.get('branches_finished',[]))
    results={i:dict(boxes=[],empty_frames=[],start=j['start'],end=j['start'],requested_end=j['end'],stop_reason='',job_index=i) for i,j in enumerate(jobs)}
    for index in finished:
        if index not in results or index not in previous:raise ValueError('Invalid saved branch')
        results[index]=previous[index];done+=abs(results[index]['end']-jobs[index]['start'])
    def persist(complete=False):
        atomic_output(output,dict(results=[results[i] for i in range(len(jobs))],branches_finished=sorted(finished),complete=complete,request_key=request))
    def canceled():
        return bool(request.get('cancel_file') and Path(request['cancel_file']).exists())
    groups=defaultdict(list)
    for index,job in enumerate(jobs):
        if index not in finished:groups[(job['start'],1 if job['end']>job['start'] else -1)].append((index,job))
    try:
        persist()
        for (start,direction),branches in groups.items():
            if canceled():raise InterruptedError('Tracking cancelled; partial candidates retained')
            end=max((j['end'] for _,j in branches),key=lambda f:(f-start)*direction)
            directory=backend.frames.get(request['video'],start,end,lambda phase,**details:notify(phase,completed=done,total=total,**details))
            notify('initializing',completed=done,total=total)
            inference_started=time.perf_counter()
            with torch.inference_mode():
                state=predictor.init_state(str(directory),offload_video_to_cpu=True,offload_state_to_cpu=True)
                # Model IDs are internal: hand ID 0 and object ID 0 stay distinct.
                mapping={local+1:(index,job) for local,(index,job) in enumerate(branches)}
                for internal,(index,job) in mapping.items():
                    predictor.add_new_points_or_box(state,frame_idx=0,obj_id=internal,box=np.asarray(job['bbox'],dtype=np.float32))
                last={index:list(job['bbox']) for index,job in branches};active=set(mapping)
                for step,ids,masks in predictor.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=abs(end-start)):
                    if canceled():persist();raise InterruptedError('Tracking cancelled; partial candidates retained')
                    if step==0:continue
                    for internal in list(active):
                        index,job=mapping[internal];row=results[index];frame=start+direction*step
                        if (frame-start)*direction>abs(job['end']-start):active.remove(internal);finished.add(index);continue
                        mask=(masks[ids.index(internal)]>0).detach().cpu().numpy().squeeze();yy,xx=np.where(mask)
                        if not len(xx):row['empty_frames'].append(frame)
                        else:
                            coords=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1];prior=last[index]
                            area=(coords[2]-coords[0])*(coords[3]-coords[1]);prior_area=max(1,(prior[2]-prior[0])*(prior[3]-prior[1]))
                            distance=((coords[0]+coords[2]-prior[0]-prior[2])**2+(coords[1]+coords[3]-prior[1]-prior[3])**2)**.5/2
                            if area/prior_area>4 or prior_area/max(area,1)>4 or distance>max(32,3*max(prior[2]-prior[0],prior[3]-prior[1])):
                                row['stop_reason']=f'Paused before frame {frame}: abrupt box change. Correct this frame and retrack.'
                                finished.add(index);active.remove(internal);continue
                            row['boxes'].append(dict(frame=frame,**dict(zip(('x1','y1','x2','y2'),coords))));last[index]=coords
                        row['end']=frame;done+=1
                        if frame==job['end']:finished.add(index);active.remove(internal)
                    notify('tracking',completed=done,total=total,frame=start+direction*step)
                    if step%5==0 or not active:persist()
                    if not active:break
                for index,job in branches:
                    finished.add(index)
                    results[index].update(elapsed_seconds=time.perf_counter()-started,inference_seconds=time.perf_counter()-inference_started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if backend.device=='cuda' else None)
                predictor.reset_state(state);del state
            gc.collect();backend.frames.trim();persist()
        persist(complete=True);notify('finished',completed=done,total=total)
        return dict(results=[results[i] for i in range(len(jobs))])
    finally:
        if owns:backend.close()

def serve():
    backend=None;key=None;original_stdout=sys.stdout
    def emit(data):original_stdout.write(json.dumps(data,allow_nan=False)+'\n');original_stdout.flush()
    emit(dict(type='ready'))
    try:
        for line in sys.stdin:
            message=json.loads(line)
            if message.get('shutdown'):break
            job_id=message['job_id']
            try:
                request=json.loads(Path(message['request']).read_text(encoding='utf-8'))
                def notify(phase,**details):emit(dict(type='progress',job_id=job_id,phase=phase,**details))
                with contextlib.redirect_stdout(sys.stderr):
                    next_key=model_key(request)
                    if backend is None or next_key!=key:
                        if backend is not None:backend.close();backend=None;gc.collect()
                        notify('loading');backend=Backend(request);key=next_key
                    run_batch(request,message['output'],backend=backend,notify=notify)
                emit(dict(type='completed',job_id=job_id))
            except Exception as exc:emit(dict(type='failed',job_id=job_id,error=type(exc).__name__+': '+str(exc)))
    finally:
        if backend is not None:backend.close()

if __name__=='__main__':serve()
