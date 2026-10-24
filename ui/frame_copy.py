"""Copy visible event evidence between exact source frames, without track propagation."""
import copy
from PyQt5.QtWidgets import QToolTip
from PyQt5.QtGui import QCursor
from core.frame_copy import paste_plan

class FrameCopyMixin:
    def _frame_copy_identity(self):
        return (str(self.video_path),int(self.player.frame_count),float(self.player.frame_rate),
                int(self.player._frame_w),int(self.player._frame_h))

    def _frame_copy_message(self,message):
        self._last_frame_copy_message=message
        QToolTip.showText(QCursor.pos(),message,self,self.rect(),2200)

    def _copy_frame_boxes(self):
        self._frame_box_clipboard=None
        if not self.selected_hand_label or self.selected_event_id is None:
            self._frame_copy_message('Select an event before copying boxes.');return
        frame=int(self.player.current_frame)
        entities=self._current_review_entities()
        copied={}
        for entity in entities:
            rows=self._boxes_for_review(entity,frame)
            if rows:
                copied[entity]=[dict(copy.deepcopy(box),_copy_frame=frame) for box in rows]
        if not copied:
            self._frame_box_clipboard=None
            self._frame_copy_message('No event boxes at this exact frame.');return
        self._frame_box_clipboard={'identity':self._frame_copy_identity(),'frame':frame,'boxes':copied}
        self._frame_copy_message(f'Copied {len(copied)} entities from frame {frame}. Go to the target frame and paste.')

    def _paste_frame_boxes(self):
        clip=getattr(self,'_frame_box_clipboard',None)
        if not clip:
            self._frame_copy_message('Copy boxes first.');return
        if clip['identity']!=self._frame_copy_identity():
            self._frame_copy_message('The copied boxes belong to another video.');return
        if not self.selected_hand_label or self.selected_event_id is None:
            self._frame_copy_message('Select the target event first.');return
        frame=int(self.player.current_frame)
        entities=self._current_review_entities()
        occupied={key for key in entities if self._boxes_for_review(key,frame)}
        additions,skipped=paste_plan(clip['boxes'],frame-int(self.start_offset),set(entities),occupied)
        if not additions:
            self._frame_copy_message('Nothing pasted: existing boxes, changed entities or an invalid source.');return
        for entity,box in additions:
            if entity.startswith('O:'):
                box['label']=self._object_name_for_id(box['id'],fallback=box.get('label',''))
        self._push_undo()
        # Copy the list once; undo retains every original row.
        self.raw_boxes=list(self.raw_boxes)+[row for _,row in additions]
        for entity,box in additions:
            self.frame_review.get(str(frame),{}).pop(entity,None)
            if entity.startswith('H:'):
                key=f"{entity[2:]}:{frame}"
                self._suppressed_hand_boxes=[s for s in getattr(self,'_suppressed_hand_boxes',[]) if s!=key]
        self._selected_edit_box=dict(additions[0][1])
        self._rebuild_bboxes_from_raw()
        self._bump_bbox_revision(refresh_completion=False)
        self._bump_query_state_revision()
        self.chk_edit_boxes.setChecked(True)
        self._refresh_boxes_for_frame(frame)
        self._refresh_frame_review_status()
        self.hoi_timeline.refresh()
        self._autosave_tick()
        self._log('hoi_copy_boxes',source_frame=clip['frame'],target_frame=frame,
                  count=len(additions),skipped=len(skipped))
        self._frame_copy_message(f'Pasted {len(additions)} boxes at frame {frame}; skipped {len(skipped)}. Drag to adjust.')
