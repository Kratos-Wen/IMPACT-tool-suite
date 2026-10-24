"""Small cached note navigation; never changes annotation or review truth values."""
import json, os, hashlib
from pathlib import Path
from core.reviewer_notes import load_notes, match_event

class ReviewerNotesMixin:
    def _reviewer_notes(self):
        key = (getattr(self, 'video_path', None), getattr(self.player, 'frame_count', None))
        if getattr(self, '_reviewer_notes_key', None) != key:
            self._reviewer_notes_key = key
            self._reviewer_note_index = -1
            self._reviewer_notes_error = ''
            try:
                self._reviewer_note_rows = load_notes(*key)
                path = Path(key[0]).parent / 'review_flags.resolved.json' if key[0] else None
                flagpath = Path(key[0]).parent / 'review_flags.json' if key[0] else None
                self._reviewer_flags_sha = hashlib.sha256(flagpath.read_bytes()).hexdigest() if flagpath and flagpath.is_file() else ''
                saved = json.loads(path.read_text(encoding='utf-8')) if path and path.is_file() else {}
                self._reviewer_note_resolved = set(saved.get('resolved_ids', [])) if saved.get('flags_sha256') == self._reviewer_flags_sha else set()
            except (OSError, ValueError, TypeError) as exc:
                self._reviewer_note_rows = []
                self._reviewer_note_resolved = set()
                self._reviewer_notes_error = str(exc)
        return [n for n in self._reviewer_note_rows if n['id'] not in self._reviewer_note_resolved]

    def _refresh_reviewer_notes(self):
        notes = self._reviewer_notes()
        if not hasattr(self, 'lbl_reviewer_notes'):
            return
        self.reviewer_note_widget.setVisible(bool(notes or self._reviewer_notes_error))
        self.btn_reviewer_note_next.setEnabled(bool(notes))
        if self._reviewer_notes_error:
            self.lbl_reviewer_notes.setText('Reviewer notes: ' + self._reviewer_notes_error)
        elif notes:
            self.lbl_reviewer_notes.setText(f'Reviewer notes: {len(notes)} pending. Next note jumps to the source frame.')
        self.btn_reviewer_note_done.setEnabled(False)

    def _jump_reviewer_note(self):
        notes = self._reviewer_notes()
        if not notes:
            return
        self._reviewer_note_index = (self._reviewer_note_index + 1) % len(notes)
        note = notes[self._reviewer_note_index]
        eid, hand = match_event(self.events, note)
        previous = getattr(self, '_review_navigation_active', False)
        self._review_navigation_active = True
        try:
            if eid is not None:
                self._set_selected_event(eid, hand)
        finally:
            self._review_navigation_active = previous
        self._jump_bbox_frame(note['frame'])
        self._active_reviewer_note_id = note['id']
        self._focus_inspector_tab('review')
        self.lbl_reviewer_notes.setText(f"Frame {note['frame']}: {note['message']}")
        self.lbl_reviewer_notes.setToolTip(note['message'])
        self.btn_reviewer_note_done.setEnabled(True)

    def _show_reviewer_note_at(self, frame):
        notes = [n for n in self._reviewer_notes() if n['frame'] == frame]
        if notes and hasattr(self, 'lbl_reviewer_notes'):
            self._active_reviewer_note_id = notes[0]['id']
            self.lbl_reviewer_notes.setText(f"Frame {frame}: " + '\n'.join(n['message'] for n in notes))
            self.btn_reviewer_note_done.setEnabled(True)

    def _resolve_reviewer_note(self):
        uid = getattr(self, '_active_reviewer_note_id', None)
        notes = self._reviewer_notes()
        if uid not in {n['id'] for n in notes}:
            return
        resolved = self._reviewer_note_resolved | {uid}
        path = Path(self.video_path).parent / 'review_flags.resolved.json'
        temp = path.with_name(path.name + '.tmp')
        try:
            with temp.open('w', encoding='utf-8') as stream:
                json.dump({'flags_sha256': self._reviewer_flags_sha, 'resolved_ids': sorted(resolved)}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        except OSError as exc:
            self.lbl_reviewer_notes.setText('Could not save note status: ' + str(exc))
            return
        self._reviewer_note_resolved = resolved
        self._reviewer_note_index -= 1
        self._refresh_reviewer_notes()
        self.hoi_timeline.refresh()

    def _reviewer_note_markers(self):
        notes = self._reviewer_notes()
        selected = self._selected_hand_data() or {}
        start, end = selected.get('interaction_start'), selected.get('interaction_end')
        if type(start) is not int or type(end) is not int:
            return []
        markers = []
        source = str(selected.get('_source_event_id') or '')
        for n in notes:
            matches = not n.get('event_id') or (str(n['event_id']) == source and str(n.get('hand','')).lower() == str(self.selected_hand_label).lower())
            if start <= n['frame'] <= end and matches:
                markers.append(dict(frame=n['frame'], role='review', entity='review:'+n['id'], state='reviewer'))
        return markers
