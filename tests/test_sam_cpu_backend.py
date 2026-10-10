"""Real CPU inference portability smoke test; synthetic frames, no project data."""
import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from tools.sam2_correction_worker import run,run_batch
checkpoint=Path(sys.argv[1]).resolve()
assert checkpoint.is_file()
with tempfile.TemporaryDirectory(prefix='SAM CPU space ') as temporary:
 root=Path(temporary);video=root/'video.mp4';output=root/'result.json'
 writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),15,(64,64))
 assert writer.isOpened(),'Video encoding unavailable'
 try:
  for i in range(3):
   frame=np.full((64,64,3),220,dtype=np.uint8)
   frame[15:45,15+i:40+i]=(30,80,200)
   writer.write(frame)
 finally:writer.release()
 run(dict(video=str(video),checkpoint=str(checkpoint),config='configs/sam2.1/sam2.1_hiera_s.yaml',device='cpu',cpu_threads=1,start=0,end=2,id=0,bbox=[15,15,40,45]),str(output))
 result=json.loads(output.read_text(encoding='utf-8'))
 assert result['requested_end']==2 and result['peak_cuda_allocated_bytes'] is None
 assert result['inference_seconds']>0
 assert result['boxes'] or result['empty_frames'] or result['stop_reason']
 for b in result['boxes']:
  assert 0<=b['x1']<b['x2']<=64 and 0<=b['y1']<b['y2']<=64
 run(dict(video=str(video),checkpoint=str(checkpoint),config='configs/sam2.1/sam2.1_hiera_s.yaml',device='cpu',cpu_threads=1,start=2,end=0,id=0,bbox=[17,15,42,45]),str(output))
 backward=json.loads(output.read_text(encoding='utf-8'));assert backward['requested_end']==0
 assert all(0<=row['frame']<2 for row in backward['boxes'])
 # Multiple targets and both directions share one model, with independent states.
 from unittest.mock import patch
 from sam2.build_sam import build_sam2_video_predictor
 batch=dict(video=str(video),checkpoint=str(checkpoint),config='configs/sam2.1/sam2.1_hiera_s.yaml',
     device='cpu',cpu_threads=1,requests=[dict(start=1,end=e,id=uid,bbox=[16,15,41,45]) for uid in [0,7] for e in [0,2]])
 with patch('sam2.build_sam.build_sam2_video_predictor',wraps=build_sam2_video_predictor) as build:
  run_batch(batch,str(output));assert build.call_count==1
 result_batch=json.loads(output.read_text(encoding='utf-8'))
 assert len(result_batch['results'])==4
 assert {row['job_index'] for row in result_batch['results']}=={0,1,2,3}
 for branch,row in zip(batch['requests'],result_batch['results']):
  assert row['requested_end']==branch['end'] and row['peak_cuda_allocated_bytes'] is None
 print('REAL_SAM_CPU_INFERENCE_PASS',sys.platform,round(result['elapsed_seconds'],2))
