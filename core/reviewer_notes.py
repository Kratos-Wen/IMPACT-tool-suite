"""Read-only task-specific reviewer notes; source video frames are zero based."""
import json
from pathlib import Path

def load_notes(video_path, frame_count):
    if not video_path:
        return []
    path = Path(video_path).parent / 'review_flags.json'
    if not path.is_file():
        return []
    if path.stat().st_size > 2_000_000:
        raise ValueError('Reviewer notes file is too large')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 'impact-review-flags' or data.get('frame_count') != frame_count:
        raise ValueError('Reviewer notes do not match this video frame count')
    if data.get('trial') != Path(video_path).parent.name:
        raise ValueError('Reviewer notes do not match this task folder')
    result = []
    for note in data.get('notes', []):
        if not isinstance(note.get('id'), str) or type(note.get('frame')) is not int:
            raise ValueError('Invalid reviewer note')
        if not 0 <= note['frame'] < frame_count:
            raise ValueError('Reviewer note is outside the video')
        result.append(note)
    return result

def match_event(events, note):
    hand = str(note.get('hand') or '').lower()
    source_id = str(note.get('event_id') or '')
    if not source_id:
        return None, None
    for event in events:
        for key, data in (event.get('hoi_data') or {}).items():
            if key.lower() == hand and str(data.get('_source_event_id') or '') == source_id:
                return event.get('event_id'), key
    return None, None
