"""One explicit video-level validation action, reusing the existing editor."""
import copy
from datetime import datetime, timezone
from core.video_validation import annotation_signature, checked_record

class VideoValidationMixin:
    def _refresh_video_validation_status(self):
        record=getattr(self,'video_validation',{}) or {}
        status=record.get('status','not_validated')
        captions={'validated':'Validated ✓','needs_review':'Needs revalidation','in_progress':'In validation','not_validated':'Not validated'}
        caption=captions.get(status,'Not validated')
        mode=bool(getattr(self,'validation_enabled',False))
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
