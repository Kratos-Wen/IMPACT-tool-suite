"""Remember local project locations and view state, without duplicating annotations."""
import json,os,time
from pathlib import Path
from PyQt5.QtWidgets import QMessageBox
from core.safe_storage import atomic_json

class ProjectSessionMixin:
    def _project_session_path(self):
        from utils.shortcut_settings import _shortcuts_dir
        return Path(_shortcuts_dir())/'last_project.json'

    def _remember_project_session(self,force=False):
        if getattr(self,'_restoring_project_session',False) or not self.video_path:
            return
        now=time.monotonic()
        if not force and now-getattr(self,'_session_saved_at',0)<10:
            return
        try:
            path=Path(self.video_path).resolve()
            if not path.is_file() or not getattr(self,'_autosave_path',None):return
            annotation=str(getattr(self,'current_annotation_path','') or '')
            record=dict(schema='IMPACT-SESSION-1',video=str(path),bytes=path.stat().st_size,
                frames=int(self.player.frame_count),fps=float(self.player.frame_rate),
                annotation=str(Path(annotation).resolve()) if annotation else '',
                annotation_existed=bool(annotation and Path(annotation).is_file()),
                profile=os.environ.get('IMPACT_PROJECT_PROFILE',''),
                mode=self._experiment_mode_key(),
                frame=int(self.player.current_frame),start=self.start_offset,end=self.end_frame,
                selected_event=self.selected_event_id,selected_hand=self.selected_hand_label,
                timeline_start=self.hoi_timeline.get_view_start(),timeline_span=self.hoi_timeline.get_view_span())
            if record!=getattr(self,'_last_session_record',None):
                atomic_json(self._project_session_path(),record,backup=False)
                self._last_session_record=record
            self._session_saved_at=now
        except (OSError,ValueError,TypeError) as exc:
            self._log('hoi_session_save_failed',error=str(exc))

    def _restore_last_project_session(self):
        if self.video_path or os.environ.get('IMPACT_RESTORE_LAST_PROJECT','1')=='0':
            return False
        self._restoring_project_session=True
        try:
            path=self._project_session_path()
            if not path.is_file():return False
            data=json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(data,dict) or data.get('schema')!='IMPACT-SESSION-1':raise ValueError('Unsupported saved project session')
            video=Path(data['video'])
            if not video.is_file():raise ValueError('The previous video was moved or is unavailable. Open its new location.')
            if video.stat().st_size!=data['bytes']:raise ValueError('The previous video has changed. Open it manually.')
            annotation=data.get('annotation','')
            if data.get('annotation_existed') and not Path(annotation).is_file():
                raise ValueError('The previous annotation file is unavailable. Open the project manually.')
            from core.task_assets import resolve_task
            task=resolve_task(video)
            profile=data.get('profile','')
            if not task and profile:
                if not Path(profile).is_file():raise ValueError('The previous project profile is unavailable.')
                from core.project_profile import activate_project_profile
                activate_project_profile(profile)
            if data.get('mode') in ('manual','full_assist'):
                self._set_experiment_mode_key(data['mode'])
            if not self._load_video(str(video),resume_annotation=annotation):return False
            if int(self.player.frame_count)!=data['frames'] or abs(float(self.player.frame_rate)-data['fps'])>0.01:
                raise ValueError('The previous video timeline has changed. Check the project before editing.')
            if not getattr(self,'_autosave_recovered',False):
                self._restore_editor_view(dict(start_frame=data.get('start',0),end_frame=data.get('end')))
                selected=data.get('selected_event');hand=data.get('selected_hand')
                event=self._find_event_by_id(selected) if selected is not None else None
                if event and hand in event.get('hoi_data',{}):self._set_selected_event(selected,hand)
            last_frame=self.player.current_frame if getattr(self,'_autosave_recovered',False) else int(data.get('frame',0))
            frame=max(self.start_offset,min(last_frame,self.end_frame if self.end_frame is not None else self.player.frame_count-1))
            self.player.seek(frame);self._refresh_boxes_for_frame(frame);self._set_frame_controls(frame)
            self.hoi_timeline._set_view(data.get('timeline_start',0),data.get('timeline_span',self.player.frame_count))
            self._log('hoi_last_project_restored',path=str(video),frame=frame)
            return True
        except (OSError,ValueError,KeyError,TypeError) as exc:
            QMessageBox.warning(self,'Previous project unavailable',str(exc))
            return False
        finally:
            self._restoring_project_session=False
