import copy
import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple, Any


SHORTCUT_DEFINITIONS: List[Dict[str, str]] = [
    # --- HOI ---
    {
        "id": "hoi.step_prev",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Step frame -1",
        "default": "Left",
    },
    {
        "id": "hoi.step_next",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Step frame +1",
        "default": "Right",
    },
    {
        "id": "hoi.seek_prev_second",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Seek -1 second",
        "default": "Up",
    },
    {
        "id": "hoi.seek_next_second",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Seek +1 second",
        "default": "Down",
    },
    {
        "id": "hoi.play_pause",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Play / Pause",
        "default": "Space",
    },
    {
        "id": "hoi.pause",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Pause",
        "default": "K",
    },
    {
        "id": "hoi.detect",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Detect current frame",
        "default": "Ctrl+Shift+D",
    },
    {
        "id": "hoi.toggle_edit_boxes",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Toggle box edit mode",
        "default": "Ctrl+B",
    },
    {
        "id": "hoi.undo",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Undo",
        "default": "Ctrl+Z",
    },
    {
        "id": "hoi.redo",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Redo",
        "default": "Ctrl+Y",
    },
    {
        "id": "hoi.open_settings",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Open settings",
        "default": "Ctrl+,",
    },
    {
        "id": "hoi.open_quick_start",
        "section": "IMPACT HOI",
        "scope": "hoi",
        "label": "Open quick start",
        "default": "F1",
    },
]


def _shortcuts_dir() -> str:
    base = os.environ.get("IMPACT_HOI_SETTINGS_DIR") or os.environ.get("CVHCI_SETTINGS_DIR")
    if base:
        return os.path.abspath(base)
    home = os.path.expanduser("~")
    return os.path.join(home, ".impact_hoi")


def shortcuts_file_path() -> str:
    return os.path.join(_shortcuts_dir(), "shortcuts.json")


def shortcuts_backup_path() -> str:
    return os.path.join(_shortcuts_dir(), "shortcuts.backup.json")


def logging_policy_file_path() -> str:
    return os.path.join(_shortcuts_dir(), "logging_policy.json")


def logging_policy_backup_path() -> str:
    return os.path.join(_shortcuts_dir(), "logging_policy.backup.json")

def ui_preferences_file_path() -> str:
    return os.path.join(_shortcuts_dir(), "ui_preferences.json")


def ui_preferences_backup_path() -> str:
    return os.path.join(_shortcuts_dir(), "ui_preferences.backup.json")


def _definition_map() -> Dict[str, Dict[str, str]]:
    return {item["id"]: item for item in SHORTCUT_DEFINITIONS}


def default_shortcut_bindings() -> Dict[str, str]:
    return {item["id"]: item["default"] for item in SHORTCUT_DEFINITIONS}


def shortcut_value(
    bindings: Optional[Dict[str, Any]],
    defaults: Optional[Dict[str, str]],
    sid: str,
    default_key: str,
) -> str:
    if isinstance(bindings, dict) and sid in bindings:
        try:
            return str(bindings.get(sid, default_key) or "").strip()
        except Exception:
            return str(default_key or "").strip()
    if isinstance(defaults, dict):
        try:
            return str(defaults.get(sid, default_key) or "").strip()
        except Exception:
            return str(default_key or "").strip()
    return str(default_key or "").strip()


def set_shortcut_key(shortcut: Any, key: str, default_key: str) -> None:
    if shortcut is None:
        return
    try:
        from PyQt5.QtGui import QKeySequence  # lazy import to keep tools lightweight
    except Exception:
        return
    try:
        shortcut.setKey(QKeySequence(key))
    except Exception:
        try:
            shortcut.setKey(QKeySequence(default_key))
        except Exception:
            pass


def shortcut_definitions_by_section() -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for item in SHORTCUT_DEFINITIONS:
        grouped.setdefault(item["section"], []).append(copy.deepcopy(item))
    return grouped


def _normalize_bindings(bindings: Optional[Dict[str, Any]]) -> Dict[str, str]:
    defs = _definition_map()
    out = default_shortcut_bindings()
    if not isinstance(bindings, dict):
        return out
    for sid, val in bindings.items():
        if sid not in defs:
            continue
        txt = ""
        if isinstance(val, str):
            txt = val.strip()
        elif val is None:
            txt = ""
        else:
            txt = str(val).strip()
        out[sid] = txt
    return out


def default_logging_policy() -> Dict[str, bool]:
    return {
        "ops_csv_enabled": False,
        "validation_summary_enabled": True,
        "validation_comment_prompt_enabled": True,
    }

def default_ui_preferences() -> Dict[str, Any]:
    return {
        "ui_scale": 0.85,
        "show_quick_start_on_startup": True,
        "participant_code": "",
    }


def _coerce_ui_scale(value: Any, fallback: float) -> float:
    try:
        if isinstance(value, str):
            text = value.strip().replace("%", "")
            num = float(text)
        else:
            num = float(value)
        if num > 10.0:
            num = num / 100.0
        return max(0.80, min(1.25, num))
    except Exception:
        try:
            return max(0.80, min(1.25, float(fallback)))
        except Exception:
            return 0.85


def _normalize_ui_preferences(
    data: Optional[Dict[str, Any]], base: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    out = dict(base or default_ui_preferences())
    if not isinstance(data, dict):
        return out
    obj = data
    if isinstance(data.get("ui"), dict):
        obj = data.get("ui") or {}
    if "ui_scale" in obj:
        out["ui_scale"] = _coerce_ui_scale(obj.get("ui_scale"), out["ui_scale"])
    if "show_quick_start_on_startup" in obj:
        out["show_quick_start_on_startup"] = _coerce_bool(
            obj.get("show_quick_start_on_startup"),
            out["show_quick_start_on_startup"],
        )
    if "participant_code" in obj:
        out["participant_code"] = str(obj.get("participant_code") or "").strip()
    return out


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "enabled"}:
            return True
        if text in {"0", "false", "no", "off", "disabled"}:
            return False
    return bool(fallback)


def _normalize_logging_policy(
    data: Optional[Dict[str, Any]], base: Optional[Dict[str, bool]] = None
) -> Dict[str, bool]:
    out = dict(base or default_logging_policy())
    if not isinstance(data, dict):
        return out
    obj = data
    if isinstance(data.get("logging"), dict):
        obj = data.get("logging") or {}

    if "ops_csv_enabled" in obj:
        out["ops_csv_enabled"] = _coerce_bool(
            obj.get("ops_csv_enabled"), out["ops_csv_enabled"]
        )
    elif "oplog_enabled" in obj:
        out["ops_csv_enabled"] = _coerce_bool(
            obj.get("oplog_enabled"), out["ops_csv_enabled"]
        )
    elif "oplog" in obj:
        out["ops_csv_enabled"] = _coerce_bool(obj.get("oplog"), out["ops_csv_enabled"])
    elif "enabled" in obj:
        # Backward compatibility with old single-toggle files.
        out["ops_csv_enabled"] = _coerce_bool(
            obj.get("enabled"), out["ops_csv_enabled"]
        )

    if "validation_summary_enabled" in obj:
        out["validation_summary_enabled"] = _coerce_bool(
            obj.get("validation_summary_enabled"), out["validation_summary_enabled"]
        )
    if "validation_comment_prompt_enabled" in obj:
        out["validation_comment_prompt_enabled"] = _coerce_bool(
            obj.get("validation_comment_prompt_enabled"),
            out["validation_comment_prompt_enabled"],
        )

    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    folder = os.path.dirname(path) or "."
    _ensure_dir(folder)
    fd, tmp_path = tempfile.mkstemp(prefix="shortcuts_", suffix=".json", dir=folder)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def load_shortcut_bindings() -> Dict[str, str]:
    for path in (shortcuts_file_path(), shortcuts_backup_path()):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if isinstance(data.get("bindings"), dict):
                    return _normalize_bindings(data.get("bindings"))
                return _normalize_bindings(data)
        except Exception:
            continue
    return default_shortcut_bindings()


def load_logging_policy(
    default_ops_enabled: bool = False,
    default_validation_summary_enabled: bool = True,
) -> Dict[str, bool]:
    base = {
        "ops_csv_enabled": bool(default_ops_enabled),
        "validation_summary_enabled": bool(default_validation_summary_enabled),
        "validation_comment_prompt_enabled": True,
    }
    for path in (logging_policy_file_path(), logging_policy_backup_path()):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return _normalize_logging_policy(data, base=base)
        except Exception:
            continue

    # Backward compatibility: accept legacy logging keys if they were stored in
    # shortcut settings payloads.
    for path in (shortcuts_file_path(), shortcuts_backup_path()):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            if any(
                key in data
                for key in (
                    "logging",
                    "ops_csv_enabled",
                    "oplog_enabled",
                    "validation_summary_enabled",
                    "enabled",
                )
            ):
                return _normalize_logging_policy(data, base=base)
        except Exception:
            continue
    return base


def load_ui_preferences(default_ui_scale: float = 0.85) -> Dict[str, Any]:
    base = {
        "ui_scale": _coerce_ui_scale(default_ui_scale, 0.85),
    }
    for path in (ui_preferences_file_path(), ui_preferences_backup_path()):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return _normalize_ui_preferences(data, base=base)
        except Exception:
            continue

    for path in (logging_policy_file_path(), logging_policy_backup_path()):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "ui_scale" in data:
                return _normalize_ui_preferences(data, base=base)
        except Exception:
            continue
    return base


def save_shortcut_bindings(bindings: Dict[str, Any]) -> Tuple[bool, str]:
    payload = {
        "schema": "cvhci.shortcut_bindings.v1",
        "bindings": _normalize_bindings(bindings),
    }
    primary = shortcuts_file_path()
    backup = shortcuts_backup_path()
    try:
        _atomic_write_json(primary, payload)
        _atomic_write_json(backup, payload)
        return True, primary
    except Exception as ex:
        return False, str(ex)


def save_logging_policy(policy: Dict[str, Any]) -> Tuple[bool, str]:
    payload = {
        "schema": "cvhci.logging_policy.v1",
        "logging": _normalize_logging_policy(policy),
    }
    primary = logging_policy_file_path()
    backup = logging_policy_backup_path()
    try:
        _atomic_write_json(primary, payload)
        _atomic_write_json(backup, payload)
        return True, primary
    except Exception as ex:
        return False, str(ex)


def save_ui_preferences(prefs: Dict[str, Any]) -> Tuple[bool, str]:
    payload = {
        "schema": "cvhci.ui_preferences.v1",
        "ui": _normalize_ui_preferences(prefs),
    }
    primary = ui_preferences_file_path()
    backup = ui_preferences_backup_path()
    try:
        _atomic_write_json(primary, payload)
        _atomic_write_json(backup, payload)
        return True, primary
    except Exception as ex:
        return False, str(ex)


def detect_scope_conflicts(
    bindings: Optional[Dict[str, Any]],
) -> Dict[str, List[Tuple[str, List[str]]]]:
    defs = _definition_map()
    normalized = _normalize_bindings(bindings)
    scope_to_key_ids: Dict[str, Dict[str, List[str]]] = {}
    for sid, key in normalized.items():
        key = (key or "").strip()
        if not key:
            continue
        scope = defs.get(sid, {}).get("scope", "")
        if not scope:
            continue
        scope_to_key_ids.setdefault(scope, {}).setdefault(key, []).append(sid)
    conflicts: Dict[str, List[Tuple[str, List[str]]]] = {}
    for scope, mapping in scope_to_key_ids.items():
        for key, ids in mapping.items():
            if len(ids) > 1:
                conflicts.setdefault(scope, []).append((key, sorted(ids)))
    return conflicts


def conflict_messages(bindings: Optional[Dict[str, Any]]) -> List[str]:
    defs = _definition_map()
    messages: List[str] = []
    conflicts = detect_scope_conflicts(bindings)
    for scope, items in conflicts.items():
        for key, ids in items:
            labels = [defs.get(sid, {}).get("label", sid) for sid in ids]
            messages.append(f"[{scope}] {key}: " + ", ".join(labels))
    return messages
