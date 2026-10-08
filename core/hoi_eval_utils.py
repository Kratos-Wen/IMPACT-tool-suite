from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", safe_text(value).lower()).strip()


def normalize_hand(value: Any) -> str:
    text = normalize_text(value).replace("-", "_")
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return text or "unknown"


def is_annotation_json(path: str) -> bool:
    lower = str(path or "").lower()
    return (
        lower.endswith(".json")
        and not lower.endswith(".event_graph.json")
        and not lower.endswith(".validation.json")
    )


def iter_annotation_paths(path: str) -> List[str]:
    src = os.path.abspath(str(path or ""))
    if not src:
        return []
    out: List[str] = []
    if os.path.isfile(src):
        if is_annotation_json(src):
            out.append(src)
        return out
    if not os.path.isdir(src):
        return out
    for root, _dirs, files in os.walk(src):
        for name in files:
            candidate = os.path.join(root, name)
            if is_annotation_json(candidate):
                out.append(candidate)
    return sorted(out)


def normalized_annotation_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    # Remove common suffixes (potentially multiple in a row)
    stem = re.sub(r"(?:_(?:manual|assist|full_assist|fullassist|hoi_bbox|hoi|bbox|rgb))+$", "", stem)
    # Remove participant and condition prefixes
    stem = re.sub(r"^p\d+_(manual|assist|full_assist|fullassist)_", "", stem)
    stem = re.sub(r"^p\d+_", "", stem)
    stem = re.sub(r"^participant_\d+_", "", stem)
    # Extract clip ID if present, otherwise just return cleaned stem
    clip_match = re.search(r"(clip[_-]?\d+)", stem)
    if clip_match:
        return clip_match.group(1).replace("-", "_")
    return stem.replace("-", "_")


def load_json_dict(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


@dataclass
class EventRow:
    hand: str
    start: int
    onset: int
    end: int
    verb: str
    noun_id: Optional[int]
    noun_label: str
    source_path: str
    raw_event_id: str


def event_noun_label(event: Dict[str, Any], object_library: Dict[str, Any]) -> str:
    noun_id = safe_int(event.get("noun_object_id"))
    if noun_id is not None:
        info = object_library.get(str(noun_id)) or {}
        if isinstance(info, dict):
            label = safe_text(info.get("label")) or safe_text(info.get("category"))
            if label:
                return label
    interaction = event.get("interaction") or {}
    if isinstance(interaction, dict):
        label = safe_text(interaction.get("noun")) or safe_text(interaction.get("target"))
        if label:
            return label
    return ""


def load_event_rows(path: str) -> List[EventRow]:
    data = load_json_dict(path)
    object_library = data.get("object_library") or {}
    if not isinstance(object_library, dict):
        object_library = {}
    hoi_events = data.get("hoi_events") or {}
    if not isinstance(hoi_events, dict):
        return []
    rows: List[EventRow] = []
    for hand_key, items in hoi_events.items():
        if not isinstance(items, list):
            continue
        hand = normalize_hand(hand_key)
        for item in items:
            if not isinstance(item, dict):
                continue
            start = safe_int(item.get("start_frame"))
            onset = safe_int(item.get("contact_onset_frame"))
            end = safe_int(item.get("end_frame"))
            if start is None or onset is None or end is None:
                continue
            if end < start:
                start, end = end, start
            rows.append(
                EventRow(
                    hand=hand,
                    start=int(start),
                    onset=int(onset),
                    end=int(end),
                    verb=normalize_text(item.get("verb")),
                    noun_id=safe_int(item.get("noun_object_id")),
                    noun_label=normalize_text(event_noun_label(item, object_library)),
                    source_path=path,
                    raw_event_id=safe_text(item.get("event_id")),
                )
            )
    return rows
