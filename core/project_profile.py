"""Use the versioned repository configuration across all project entry points."""
import json, os
from pathlib import Path

REPOSITORY_PROFILE = Path(__file__).resolve().parents[1] / 'project_profile.json'

def repository_profile_enabled():
    return os.environ.get('IMPACT_USE_BUNDLED_PROFILE','1') != '0'

def resolve_profile_path(path=None):
    if repository_profile_enabled():
        return REPOSITORY_PROFILE
    path = path or os.environ.get('IMPACT_PROJECT_PROFILE','').strip()
    return Path(path).expanduser().resolve() if path else None

def load_project_profile():
    file=resolve_profile_path()
    if file is None:return {}
    with file.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Project profile must be a JSON object")
    os.environ['IMPACT_PROJECT_PROFILE']=str(file)
    return data

PROFILE = load_project_profile()


def activate_project_profile(path):
    """Keep the shared dictionary identity so imported references stay current."""
    file = resolve_profile_path(path)
    if file is None:raise ValueError('No project profile was selected')
    data = json.loads(file.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("Project profile must be a JSON object")
    PROFILE.clear()
    PROFILE.update(data)
    os.environ["IMPACT_PROJECT_PROFILE"] = str(file)
    from core.noun_aliases import refresh_aliases
    refresh_aliases()
