"""Standalone compatibility entry point for the resident SAM tracking backend."""
import json,sys,tempfile
from pathlib import Path
try:
    from tools.sam2_tracking_service import Backend,run_batch,atomic_output
except ModuleNotFoundError:
    from sam2_tracking_service import Backend,run_batch,atomic_output

def run(request,output=None,predictor=None):
    batch=dict(request,requests=[{k:request[k] for k in ('start','end','id','bbox')}])
    with tempfile.TemporaryDirectory(prefix='impact_sam_result_') as directory:
        backend=Backend(request,predictor=predictor) if predictor is not None else None
        try:result=run_batch(batch,str(Path(directory)/'result.json'),backend=backend)['results'][0]
        finally:
            if backend is not None:backend.close()
    if output:atomic_output(output,result)
    return result

if __name__=='__main__':
    try:
        request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        (run_batch if 'requests' in request else run)(request,sys.argv[2])
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr);sys.exit(1)
