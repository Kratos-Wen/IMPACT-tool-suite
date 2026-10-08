"""Validate configured video orientation requirements."""
import json
from pathlib import Path
from core.project_profile import PROFILE

def check_ego_orientation(path):
 path=Path(path)
 prefixes=tuple(PROFILE.get('orientation_required_prefixes') or [])
 if not prefixes or not path.name.startswith(prefixes):return True,''
 sidecar=Path(str(path)+'.orientation.json')
 try:
  metadata=json.loads(sidecar.read_text(encoding='utf-8'))
  if metadata.get('pixel_orientation')=='upright_ccw90' and metadata.get('rotation_ccw_degrees')==90:return True,''
 except (OSError,ValueError):pass
 return False,'This native Ego video is encoded sideways. Open its verified upright copy from the upright Ego catalog. Old native boxes/gaze need the matching 90-degree coordinate conversion; do not reuse them unchanged.'
