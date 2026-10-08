"""Load a local project profile without embedding project annotation policy."""
import json, os
from pathlib import Path

def load_project_profile():
    path = os.environ.get("IMPACT_PROJECT_PROFILE", "").strip()
    if not path:
        return {}
    file = Path(path).expanduser()
    with file.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Project profile must be a JSON object")
    return data

PROFILE = load_project_profile()
