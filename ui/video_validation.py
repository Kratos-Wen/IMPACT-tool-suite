"""One explicit video-level validation action, reusing the existing editor."""
import copy
from datetime import datetime, timezone
from core.video_validation import annotation_signature, checked_record

class VideoValidationMixin:
    def _configure_review_view(self, task=None, payload=None):
        if task is not None:
            self._review_task_view = str(task.get('view') or '').lower()
        view = getattr(self, '_review_task_view', '')
        if not view and payload:
            saved = payload.get('review_view') or {}
            if saved.get('coordinate_space') == 'native_video':
                view = str(saved.get('view') or '').lower()
        self.player.set_review_rotation(180 if view == 'front' else 0)
        if view != 'front':
            self.player.set_review_labels([])

    def _review_view_metadata(self):
        return {'view': 'front' if self.player.review_rotation else getattr(self, '_review_task_view', ''),
                'rotation_degrees': self.player.review_rotation,
                'coordinate_space': 'native_video', 'transform_scope': 'display_only'}

    def _update_review_presentation(self, frame, selected_relations, current_boxes):
        if not self.validation_enabled:
            self.player.set_review_labels([])
            return
        from core.project_profile import PROFILE
        from core.anomaly_attributes import export_review
        from core.assembly_timeline import resolve_object
        from core.frame_review import valid_record
        names = PROFILE.get('anomaly_display_names') or {}
        active = {actor['id']: [] for actor in self.actors_config}
        for event in self.events:
            hands = (self.event_draft if event.get('event_id') == self.selected_event_id
                     else event.get('hoi_data', {}))
            for hand, data in hands.items():
                start, end = data.get('interaction_start'), data.get('interaction_end')
                if type(start) is int and type(end) is int and start <= frame <= end:
                    active.setdefault(hand, []).append(data)
        labels = []
        for actor in self.actors_config:
            hand = actor['id']; parts = []; anomalous = False
            for data in active.get(hand, []):
                verb = str(data.get('verb') or 'Choose verb').replace('_', ' ')
                uid = resolve_object(data, self._assembly_data(), frame)
                noun = self._object_name_for_id(uid, fallback='Object') if uid is not None else 'Choose Object'
                anomaly = export_review(data.get('anomaly_label'), PROFILE)['anomaly_labels']
                anomalous = anomalous or bool(anomaly)
                desc = ', '.join(names.get(x, x.replace('_', ' ')) for x in anomaly) or 'Normal'
                parts.append(f'{verb} → {noun}' + (f' [{uid}]' if uid is not None else '') + f' | {desc}')
            text = f"{self._get_actor_short_label(hand)}: " + (' / '.join(parts) if parts else '—')
            labels.append((text, '#fdba74' if anomalous else '#dbeafe'))
        missing = []
        invisible = lambda entity: valid_record((getattr(self, 'frame_review', {}).get(str(frame), {}) or {}).get(entity), [])
        for data in selected_relations:
            frames = {data.get(k) for k in ('interaction_start', 'functional_contact_onset', 'interaction_end')}
            frames.update(f for f in (data.get('required_review_frames') or []) if type(f) is int)
            if frame not in frames:
                continue
            hand = data.get('hand_id')
            if ('hand', hand) not in current_boxes and not invisible('H:' + hand):
                missing.append(self._get_actor_short_label(hand) + ' hand')
            for role, uid in [('Object', resolve_object(data, self._assembly_data(), frame)),
                              ('Instrument', self._hand_instrument_object_id(data))]:
                if uid is not None and uid not in current_boxes and not invisible('O:' + str(uid)):
                    missing.append(f'{role} [{uid}]')
        if missing:
            labels.append(('No keyframe box: ' + ', '.join(dict.fromkeys(missing)) + ' · Draw if visible', '#fbbf24'))
        self.player.set_review_labels(labels)

    def _refresh_video_validation_status(self):
        record=getattr(self,'video_validation',{}) or {}
        status=record.get('status','not_validated')
        captions={'validated':'Validated ✓','needs_review':'Needs revalidation','in_progress':'In validation','not_validated':'Not validated'}
        caption=captions.get(status,'Not validated')
        mode=bool(getattr(self,'validation_enabled',False))
        correction=getattr(self,'btn_correct_track',None)
        if correction is not None:
            local_tracking_allowed=not (mode or self.player.review_rotation==180)
            correction.setEnabled(local_tracking_allowed)
            correction.setToolTip('Optional local tracking.' if local_tracking_allowed else
                                 'Validation edits keyframes locally. Track propagation runs centrally after upload.')
        if getattr(self,'lbl_validation',None):
            self.lbl_validation.setText('Validated' if status=='validated' else 'Validate')
            self.lbl_validation.setVisible(not mode)
            self.lbl_validation.setToolTip(caption)
        self._set_status_chip(getattr(self,'lbl_validation_chip',None),caption,'ok' if status=='validated' else 'warn' if status in ('needs_review','in_progress') else 'neutral')
        button=getattr(self,'btn_mark_validated',None)
        if button is not None:
            button.setVisible(mode)
            button.setText('✓')
            button.setStyleSheet('color: #15803d; font-weight: bold;' if status=='validated' else '')
            button.setEnabled(mode and status!='validated' and bool(self.player.frame_count))
            button.setToolTip((f"Reviewed by {record.get('reviewer','')} at {record.get('validated_at','')}" if status=='validated' else 'Mark validated: this whole video has been reviewed. Ctrl+S saves the status with the annotation.'))

    def _mark_video_validated(self):
        if not self.validation_enabled or not str(getattr(self,'validator_name','')).strip() or not self.player.frame_count:
            return
        self._sync_live_edits_for_export()
        payload=self._build_payload_v2()
        self.video_validation={'schema':'impact-video-validation-1','scope':'video','status':'validated',
            'reviewer':self.validator_name.strip(),'validated_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'annotation_sha256':annotation_signature(payload)}
        self._refresh_video_validation_status()
        self._autosave_tick()

    def _invalidate_video_validation(self):
        record=getattr(self,'video_validation',{}) or {}
        if record.get('status')=='validated':
            self.video_validation={**record,'status':'needs_review','reason':'annotation_edited'}
            self._refresh_video_validation_status()

    def _export_video_validation(self,payload):
        self.video_validation=checked_record(getattr(self,'video_validation',{}),payload)
        self._refresh_video_validation_status()
        return copy.deepcopy(self.video_validation)
