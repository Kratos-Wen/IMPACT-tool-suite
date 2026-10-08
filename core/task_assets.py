"""Explicit portable task manifest. Never guess which annotation belongs to a video."""
import json
from pathlib import Path

def resolve_task(video):
    video = Path(video).expanduser().resolve()
    manifest = video.with_suffix(video.suffix + ".task.json")
    if not manifest.is_file():
        manifest = video.parent / "task.json"
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if data.get("schema") != "IMPACT-TASK-1":
        raise ValueError("Unsupported task manifest")
    if data.get("video") != video.name:
        raise ValueError("Task manifest belongs to a different video")
    base = manifest.parent
    root = (base / data.get("bundle_root", ".")).resolve()
    if root != base and root not in base.parents:
        raise ValueError("Bundle root must be a parent of the video folder")
    paths = {}
    for key in ("project_profile", "annotations", "resume_annotations", "sam_checkpoint", "yolo_checkpoint", "review"):
        value = data.get(key)
        if not value:
            continue
        if Path(value).is_absolute():
            raise ValueError("Task paths must be relative")
        path = (base / value).resolve()
        if path != root and root not in path.parents:
            raise ValueError("Task asset escapes its bundle")
        paths[key] = path
    return dict(manifest=manifest, data=data, paths=paths)
