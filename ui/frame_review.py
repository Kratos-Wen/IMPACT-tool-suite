"""Compact, explicit frame review without changing event semantics."""
import copy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QMessageBox,QShortcut,QInputDialog
from core.frame_review import signature,valid_record,missing_frames
from core.assembly_timeline import resolve_object
from core.correction_propagation import object_instance,protected

class FrameReviewMixin:
    def _install_frame_review(self):
        menu=self.file_menu.addMenu('Frame review')
        self._frame_review_actions={}
        for label,key,fn in [('Verify visible boxes','Ctrl+Return',self._verify_current_frame),('Next frame needing review','Alt+N',lambda:self._jump_frame_review(1)),('Previous frame needing review','Alt+P',lambda:self._jump_frame_review(-1))]:
            action=menu.addAction(label,fn);action.setShortcut(QKeySequence(key))
            sid={"Ctrl+Return":"hoi.verify_frame","Alt+N":"hoi.next_review_frame","Alt+P":"hoi.prev_review_frame"}[key]
            self._frame_review_actions[sid]=(action,key)
        menu.addAction('Mark selected entity not visible...',self._mark_not_visible)
        menu.addAction('Require review at current frame',self._require_current_frame)
        menu.addSeparator()
        menu.addAction('Clear automatic Object/tool track in this event...',lambda:self._clear_event_track(False))
        menu.addAction('Clear all Object/tool boxes in this event...',lambda:self._clear_event_track(True))
        action=menu.addAction('Skip automatic Object/tool boxes on import');action.setCheckable(True)
        action.toggled.connect(lambda on:setattr(self,'skip_auto_tracking_import',on))

    def _review_entities(self,hand,key,frame):
        entities=['H:'+key]
        uid=resolve_object(hand,self._assembly_data(),frame)
        if uid is not None:entities.append('O:'+str(uid))
        tool=self._hand_instrument_object_id(hand)
        if tool is not None:entities.append('O:'+str(tool))
        return list(dict.fromkeys(entities))

    def _boxes_for_review(self,entity,frame):
        return [b for b in self.raw_boxes if int(b.get('orig_frame',-1))+int(self.start_offset)==frame and
                (self._normalize_hand_label(b.get('label'))==entity[2:] if entity.startswith('H:') else
                 not self._normalize_hand_label(b.get('label')) and str(b.get('id'))==entity[2:])]

    def _current_review_entities(self):
        frame=int(self.player.current_frame);entities=[]
        for key,hand in self.event_draft.items():
            s=hand.get('interaction_start');e=hand.get('interaction_end')
            if type(s) is int and type(e) is int and s<=frame<=e and hand.get('verb'):
                entities.extend(self._review_entities(hand,key,frame))
        return list(dict.fromkeys(entities))

    def _require_current_frame(self):
        hand=self.event_draft.get(self.selected_hand_label,{})
        frame=int(self.player.current_frame);s=hand.get('interaction_start');e=hand.get('interaction_end')
        if type(s) is not int or type(e) is not int or not s<=frame<=e:return
        self._push_undo();hand['required_review_frames']=sorted(set(hand.get('required_review_frames',[])+[frame]))
        for event in self.events:
            if event['event_id']==self.selected_event_id:
                event['hoi_data'][self.selected_hand_label]['required_review_frames']=list(hand['required_review_frames'])
        self._bump_query_state_revision();self._refresh_events()

    def _verify_current_frame(self):
        entities=self._current_review_entities();frame=int(self.player.current_frame)
        boxes={k:self._boxes_for_review(k,frame) for k in entities}
        if not entities:return
        if any(len(v)>1 for v in boxes.values()):
            QMessageBox.warning(self,'Frame review','Duplicate boxes for the same entity. Resolve duplicates before verifying.');return
        if not any(boxes.values()):
            QMessageBox.information(self,'Frame review','No visible boxes. Use Mark selected entity not visible.');return
        try:signatures={entity:signature(rows) for entity,rows in boxes.items() if rows}
        except (ValueError,KeyError,TypeError):
            QMessageBox.warning(self,'Frame review','Invalid box geometry. Correct the box before verifying.');return
        self._push_undo()
        records=getattr(self,'frame_review',{});records=copy.deepcopy(records)
        for entity,rows in boxes.items():
            if rows:
                records.setdefault(str(frame),{})[entity]={'state':'verified','signature':signatures[entity]}
                rows[0]['human_verified']=True
        self.frame_review=records;self._refresh_boxes_for_frame(frame);self._refresh_frame_review_status()

    def _mark_not_visible(self):
        choices=self._current_review_entities()
        if not choices:return
        entity,ok=QInputDialog.getItem(self,'Frame review','Entity confirmed not visible:',choices,0,False)
        if not ok:return
        frame=int(self.player.current_frame)
        if self._boxes_for_review(entity,frame):
            QMessageBox.warning(self,'Frame review','Remove the incorrect current-frame box first; this action does not hide a box.');return
        self._push_undo();self.frame_review=copy.deepcopy(getattr(self,'frame_review',{}))
        self.frame_review.setdefault(str(frame),{})[entity]={'state':'not_visible'}
        self._refresh_frame_review_status()

    def _frame_review_issues(self):
        records=getattr(self,'frame_review',{});issues=[];fps=float(self.player.frame_rate or 15)
        cache_key=(getattr(self,'_bbox_revision',0),getattr(self,'_query_state_revision',0),id(records),len(self.raw_boxes),len(self.events),fps)
        if getattr(self,'_frame_review_cache_key',None)==cache_key:return list(self._frame_review_cache)
        index={}
        for box in self.raw_boxes:
            frame=int(box.get('orig_frame',-1))+int(self.start_offset)
            hand=self._normalize_hand_label(box.get('label'))
            entity='H:'+hand if hand else 'O:'+str(box.get('id'))
            index.setdefault((frame,entity),[]).append(box)
        for event in self.events:
            for key,hand in event.get('hoi_data',{}).items():
                s=hand.get('interaction_start');e=hand.get('interaction_end');o=hand.get('functional_contact_onset')
                if not hand.get('verb') or type(s) is not int or type(e) is not int or s>e:continue
                # Review only recorded candidate frames; never scan video pixels.
                valid=[]
                for text,row in records.items():
                    frame=int(text)
                    if s<=frame<=e and all(valid_record(row.get(entity),index.get((frame,entity),[])) for entity in self._review_entities(hand,key,frame)):valid.append(frame)
                extra=[st['frame'] for st in self._assembly_data().get('states',[]) if hand.get('shared_assembly_ref')]
                extra+=hand.get('required_review_frames',[])
                for frame in missing_frames(s,e,o,valid,fps,extra=extra):
                    issues.append(dict(event_id=event['event_id'],hand=key,field='bbox_evidence',code='human_frame_review_'+str(frame),frame=frame,missing=['Human frame review required']))
        result=sorted(issues,key=lambda x:(x['frame'],str(x['event_id']),x['hand']))
        self._frame_review_cache_key=cache_key;self._frame_review_cache=result
        self._frame_review_cache_records=records
        return list(result)

    def _jump_frame_review(self,direction):
        issues=self._frame_review_issues()
        if not issues:return
        frame=int(self.player.current_frame)
        after=[i for i in issues if i['frame']>frame] if direction>0 else [i for i in issues if i['frame']<frame]
        issue=(after[0] if direction>0 else after[-1]) if after else (issues[0] if direction>0 else issues[-1])
        self._set_selected_event(issue['event_id'],issue['hand']);self.player.seek(issue['frame']);self._set_frame_controls(issue['frame'])

    def _refresh_frame_review_status(self):
        if not getattr(self,'act_frame_review_status',None):return
        frame=int(self.player.current_frame);entities=self._current_review_entities();row=getattr(self,'frame_review',{}).get(str(frame),{})
        done=sum(valid_record(row.get(k),self._boxes_for_review(k,frame)) for k in entities)
        text=f'Frame {frame}: '+('Reviewed' if done==len(entities) else 'Needs review')+f' ({done}/{len(entities)})' if entities else 'Frame review: no active event'
        self.act_frame_review_status.setText(text)
        label=getattr(self,'lbl_frame_review_status',None)
        if label:
            label.setText(text);label.setStyleSheet('color: '+('#15803d' if entities and done==len(entities) else '#a16207')+';')

    def _clear_event_track(self,include_human):
        hand=self.event_draft.get(self.selected_hand_label,{})
        s=hand.get('interaction_start');e=hand.get('interaction_end')
        if type(s) is not int or type(e) is not int:return
        ids=[self._hand_noun_object_id(hand),self._hand_instrument_object_id(hand)];ids=list(dict.fromkeys(x for x in ids if x is not None))
        if not ids:return
        choice,ok=QInputDialog.getItem(self,'Clear event track','Object/tool ID:',[str(x) for x in ids],0,False)
        if not ok:return
        uid=next(x for x in ids if str(x)==choice)
        if QMessageBox.question(self,'Clear event track',f'Clear {"all" if include_human else "automatic"} boxes for ID {uid}, frames {s}–{e}?\nShared and overlapping events referencing this ID also see this change. Other IDs and frames are preserved; Undo restores it.')!=QMessageBox.Yes:return
        self._push_undo();self.raw_boxes=[b for b in self.raw_boxes if not (object_instance(b,uid) and s<=int(b.get('orig_frame',-1))+int(self.start_offset)<=e and (include_human or not protected(b)))]
        self._rebuild_bboxes_from_raw();self._bump_bbox_revision();self._refresh_boxes_for_frame(self.player.current_frame)
