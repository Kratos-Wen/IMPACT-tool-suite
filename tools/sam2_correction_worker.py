"""Optional SAM 2.1 worker; uses only explicitly configured local hardware."""
import json, os, sys, tempfile, time
from pathlib import Path

def run(request, output):
    started=time.perf_counter()
    import cv2, numpy as np, torch
    from sam2.build_sam import build_sam2_video_predictor
    threads=max(1,int(request.get("cpu_threads",1)))
    if os.environ.get("SLURM_CPUS_PER_TASK"): threads=min(threads,int(os.environ["SLURM_CPUS_PER_TASK"]))
    torch.set_num_threads(threads); cv2.setNumThreads(1)
    device=request.get("device", "cpu")
    if device not in ("cpu", "cuda"): raise ValueError("Invalid device")
    if device=="cuda":
        torch.cuda.set_per_process_memory_fraction(float(request.get("gpu_memory_fraction",0.5)))
        torch.cuda.reset_peak_memory_stats()
    predictor=build_sam2_video_predictor(request["config"], request["checkpoint"], device=device)
    start=request["start"]; end=request["end"]; rows=[]; empty=[]; completed_end=start; stop_reason=""; previous=list(request["bbox"])
    direction=1 if end>start else -1
    # Decode only the requested interval; never load the complete video into model memory.
    with tempfile.TemporaryDirectory(prefix="impact_sam2_") as temporary:
        cap=cv2.VideoCapture(request["video"]); cap.set(cv2.CAP_PROP_POS_FRAMES,min(start,end))
        try:
            for idx in range(abs(end-start)+1):
                ok, image=cap.read()
                if not ok: raise RuntimeError("Could not decode frame "+str(min(start,end)+idx))
                if not cv2.imwrite(str(Path(temporary)/f"{idx if direction>0 else abs(end-start)-idx:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY,100]): raise RuntimeError("Frame write failed")
        finally: cap.release()
        inference_started=time.perf_counter()
        with torch.inference_mode():
            state=predictor.init_state(temporary, offload_video_to_cpu=True, offload_state_to_cpu=True)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=request["id"], box=np.asarray(request["bbox"], dtype=np.float32))
            for idx, ids, masks in predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=abs(end-start)):
                if idx==0: continue
                mask=(masks[ids.index(request["id"])]>0).detach().cpu().numpy().squeeze()
                yy,xx=np.where(mask)
                if not len(xx): empty.append(start+direction*idx); completed_end=start+direction*idx; continue
                coords=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
                area=(coords[2]-coords[0])*(coords[3]-coords[1]); prior_area=max(1,(previous[2]-previous[0])*(previous[3]-previous[1]))
                distance=((coords[0]+coords[2]-previous[0]-previous[2])**2+(coords[1]+coords[3]-previous[1]-previous[3])**2)**.5/2
                if area/prior_area>4 or prior_area/max(area,1)>4 or distance>max(32,3*max(previous[2]-previous[0],previous[3]-previous[1])):
                    stop_reason="Paused before frame %d: box area changed >4x or center moved >max(32px,3x prior box side). Correct the first uncertain frame and retrack."%(start+direction*idx)
                    break
                rows.append(dict(frame=start+direction*idx,**dict(zip(("x1","y1","x2","y2"),coords)))); previous=coords; completed_end=start+direction*idx
    Path(output).write_text(json.dumps({"boxes":rows,"empty_frames":empty,"start":start,"end":completed_end,"requested_end":end,"stop_reason":stop_reason,"elapsed_seconds":time.perf_counter()-started,"inference_seconds":time.perf_counter()-inference_started,"peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else None}),encoding="utf-8")

if __name__=="__main__":
    try: run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), sys.argv[2])
    except Exception as exc:
        print(type(exc).__name__+": "+str(exc), file=sys.stderr); sys.exit(1)
