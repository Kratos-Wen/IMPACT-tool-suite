"""Compact, explicit frame review without changing event semantics."""
import copy
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QMessageBox,QShortcut,QInputDialog,QActionGroup
from bisect import bisect_left,bisect_right
from core.bbox_markers import BoxIndex,state as marker_state
import os
from core.frame_review import signature,valid_record,missing_frames,box_observed
from core.project_profile import PROFILE
from core.anomaly_attributes import export_review
from core.hoi_query_controller import get_field_state
from ui.annotation_policy import onset_resolved
from core.assembly_timeline import resolve_object,reference_id,change_frames,track_segments
from core.correction_propagation import object_instance,protected

class FrameReviewMixin:
    def _install_frame_review(self):
        self._observed_frame_context=None
        self._observed_frame_timer=QTimer(self)
        self._observed_frame_timer.setSingleShot(True)
        self._observed_frame_timer.setInterval(400)
        self._observed_frame_timer.timeout.connect(self._observe_displayed_boxes)
        menu=self.file_menu.addMenu('Keyframes')
        self._frame_review_actions={}
        for label,key,fn in [('Verify visible boxes','Ctrl+Return',self._verify_current_frame),('Next missing keyframe','Alt+N',lambda:self._jump_frame_review(1)),('Previous missing keyframe','Alt+P',lambda:self._jump_frame_review(-1))]:
            action=menu.addAction(label,fn);action.setShortcut(QKeySequence(key))
            sid={"Ctrl+Return":"hoi.verify_frame","Alt+N":"hoi.next_review_frame","Alt+P":"hoi.prev_review_frame"}[key]
            self._frame_review_actions[sid]=(action,key)
        menu.addSeparator()
        for label,key,sid,fn in [
            ('Jump to Start','Alt+1','hoi.jump_start',self._jump_to_selected_start),
            ('Jump to Onset','Alt+2','hoi.jump_onset',self._jump_to_selected_onset),
            ('Jump to End','Alt+3','hoi.jump_end',self._jump_to_selected_end),
            ('Previous event keyframe','Alt+Left','hoi.prev_keyframe',lambda:self._jump_event_keyframe(-1)),
            ('Next event keyframe','Alt+Right','hoi.next_keyframe',lambda:self._jump_event_keyframe(1))]:
            action=menu.addAction(label,fn);action.setShortcut(QKeySequence(key))
            self._frame_review_actions[sid]=(action,key)
        menu.addAction('Mark selected entity not visible...',self._mark_not_visible)
        menu.addAction('Add anomaly keyframe',self._require_current_frame)
        menu.addSeparator()
        menu.addAction('Clear automatic track in this event...',lambda:self._clear_event_track(False))
        menu.addAction('Clear all boxes in this event...',lambda:self._clear_event_track(True))
        choices=menu.addMenu('Automatic boxes on next import')
        group=QActionGroup(self);group.setExclusive(True);self._box_import_actions={}
        for label,value in [('All machine boxes','all'),('Hands only','hands'),('No machine boxes','none')]:
            act=choices.addAction(label);act.setCheckable(True);group.addAction(act)
            act.setChecked(value==self._automatic_box_import_mode())
            act.triggered.connect(lambda checked,v=value:setattr(self,'_box_import_override',v) if checked else None)
            self._box_import_actions[value]=act
        for label,key,sid,direction in [('Previous frame with boxes','Alt+,','hoi.prev_bbox_frame',-1),('Next frame with boxes','Alt+.','hoi.next_bbox_frame',1)]:
            action=menu.addAction(label,lambda checked=False,d=direction:self._jump_bbox_direction(d));action.setShortcut(QKeySequence(key))
            self._frame_review_actions[sid]=(action,key)

    def _automatic_box_import_mode(self):
        if getattr(self,'skip_auto_tracking_import',False) or os.environ.get('IMPACT_SKIP_AUTOMATIC_TRACKS')=='1':return 'none'
        value=getattr(self,'_box_import_override',None) or os.environ.get('IMPACT_AUTOMATIC_TRACKS') or PROFILE.get('automatic_tracking_import','all')
        from core.box_import import constrained_import_mode
        return constrained_import_mode(value,PROFILE)

    def _refresh_box_import_actions(self):
        mode=self._automatic_box_import_mode()
        for value,action in getattr(self,'_box_import_actions',{}).items():
            action.setChecked(value==mode)
            from core.box_import import constrained_import_mode
            action.setEnabled(constrained_import_mode(value,PROFILE)==value)

    def _bbox_frame_markers(self):
        hand=self._selected_hand_data() or {};s=hand.get('interaction_start');e=hand.get('interaction_end')
        if type(s) is not int or type(e) is not int or not self.selected_hand_label:return []
        rows=self.raw_boxes;revision=(id(rows),len(rows),getattr(self,'_bbox_revision',0))
        if getattr(self,'_bbox_mark_index_key',None)!=revision:
            self._bbox_mark_index=BoxIndex(rows,self._normalize_hand_label);self._bbox_mark_index_key=revision
        records=getattr(self,'frame_review',{})
        key=(revision,id(records),getattr(self,'_query_state_revision',0),self.selected_event_id,self.selected_hand_label,s,e,self.start_offset)
        if getattr(self,'_bbox_mark_cache_key',None)==key:return self._bbox_mark_cache
        index=self._bbox_mark_index;offset=int(self.start_offset);markers=[]
        for raw_frame in index.frames[bisect_left(index.frames,s-offset):bisect_right(index.frames,e-offset)]:
            frame=raw_frame+offset;entities=self._review_entities(hand,self.selected_hand_label,frame)
            instrument=self._hand_instrument_object_id(hand)
            for entity in entities:
                boxes=index.by_frame[raw_frame].get(entity,[])
                if not boxes:continue
                role='H' if entity.startswith('H:') else 'I' if entity=='O:'+str(instrument) else 'O'
                markers.append(dict(frame=frame,role=role,entity=entity,state=marker_state(boxes,records.get(str(frame),{}).get(entity))))
        self._bbox_mark_cache_key=key;self._bbox_mark_cache=markers;return markers

    def _jump_bbox_frame(self,frame):
        self._pause();self.player.seek(int(frame));self._refresh_boxes_for_frame(int(frame));self._set_frame_controls(int(frame))
        self.hoi_timeline.set_current_frame(int(frame))

    def _jump_bbox_direction(self,direction):
        frames=sorted({row['frame'] for row in self._bbox_frame_markers()})
        options=[f for f in frames if (f-int(self.player.current_frame))*direction>0]
        if options:self._jump_bbox_frame(options[0] if direction>0 else options[-1])

    def _review_entities(self,hand,key,frame):
        entities=['H:'+key]
        uid=resolve_object(hand,self._assembly_data(),frame)
        if uid is not None:entities.append('O:'+str(uid))
        tool=self._hand_instrument_object_id(hand)
        if tool is not None:entities.append('O:'+str(tool))
        return list(dict.fromkeys(entities))

    def _jump_event_keyframe(self,direction):
        hand=self._selected_hand_data() or {}
        start,end=hand.get('interaction_start'),hand.get('interaction_end')
        if type(start) is not int or type(end) is not int:return
        frames={start,end}
        onset=hand.get('functional_contact_onset')
        if type(onset) is int:frames.add(onset)
        frames.update(hand.get('required_review_frames',[]))
        uid=reference_id(hand,self._assembly_data(),start)
        if hand.get('shared_assembly_ref') and uid is not None:
            frames.update(change_frames(self._assembly_data(),uid,start,end))
        for text,entities in getattr(self,'frame_review',{}).items():
            frame=int(text)
            if any(valid_record(entities.get(key),self._boxes_for_review(key,frame))
                   for key in self._review_entities(hand,self.selected_hand_label,frame)):
                frames.add(frame)
        current=int(self.player.current_frame)
        options=sorted(f for f in frames if type(f) is int and start<=f<=end and (f-current)*direction>0)
        if not options:return
        frame=options[0] if direction>0 else options[-1]
        self._pause();self.player.seek(frame);self._refresh_boxes_for_frame(frame);self._set_frame_controls(frame)

    def _boxes_for_review(self,entity,frame):
        # Retain raw row references: verification writes human_verified on them.
        # A rebuild, box edit, replacement or append invalidates this frame index.
        rows=self.raw_boxes
        key=(id(rows),len(rows),getattr(self,'_bbox_revision',0))
        if getattr(self,'_review_boxes_index_key',None)!=key:
            index={}
            for box in rows:
                index.setdefault(int(box.get('orig_frame',-1)),[]).append(box)
            self._review_boxes_index=index
            self._review_boxes_index_source=rows
            self._review_boxes_index_key=key
        candidates=self._review_boxes_index.get(int(frame)-int(self.start_offset),[])
        return [b for b in candidates if
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
        labels=export_review(hand.get('anomaly_label'),PROFILE)['anomaly_labels']
        evidence=hand.setdefault('_event_extra',{}).setdefault('anomaly_evidence',{})
        for label in labels:
            item=evidence.setdefault(label,{'view':'ego','intervals':[],'note':''})
            if not any(a<=frame<=b for a,b in item['intervals']):item['intervals'].append([frame,frame])
        for event in self.events:
            if event['event_id']==self.selected_event_id:
                event['hoi_data'][self.selected_hand_label]['required_review_frames']=list(hand['required_review_frames'])
                event['hoi_data'][self.selected_hand_label]['_event_extra']=copy.deepcopy(hand['_event_extra'])
        self._bump_query_state_revision();self._refresh_events()

    def _box_evidence_complete(self,entity,frame):
        return box_observed(getattr(self,'frame_review',{}).get(str(frame),{}).get(entity),self._boxes_for_review(entity,frame))

    def _observation_context(self):
        return (str(self.video_path),self.selected_event_id,self.selected_hand_label,
                int(self.player.current_frame),getattr(self,'_bbox_revision',0),id(self.raw_boxes))

    def _schedule_box_observation(self):
        timer=getattr(self,'_observed_frame_timer',None)
        if timer is None:return
        frame=int(self.player.current_frame)
        if (not self.isVisible() or not self.isActiveWindow() or self.player.is_playing or
                self.selected_event_id is None or self.selected_hand_label is None or
                getattr(self.player,'_displayed_frame',None)!=frame):
            timer.stop();self._observed_frame_context=None;return
        hand=self._selected_hand_data() or {};s=hand.get('interaction_start');e=hand.get('interaction_end')
        if type(s) is not int or type(e) is not int or not s<=frame<=e or not hand.get('verb'):return
        pending=[key for key in self._review_entities(hand,self.selected_hand_label,frame)
                 if self._boxes_for_review(key,frame) and not self._box_evidence_complete(key,frame)]
        suggested=any(get_field_state(hand,f).get('status')=='suggested' for f in ('interaction_start','functional_contact_onset','interaction_end','verb','noun_object_id'))
        if not pending and not suggested:return
        context=self._observation_context()
        if context!=self._observed_frame_context or not timer.isActive():
            self._observed_frame_context=context;timer.start()

    def _observe_displayed_boxes(self):
        context=self._observed_frame_context
        self._observed_frame_context=None
        if (context is None or context!=self._observation_context() or not self.isVisible() or
                not self.isActiveWindow() or self.player.is_playing or
                getattr(self.player,'_displayed_frame',None)!=self.player.current_frame):return
        frame=int(self.player.current_frame);hand=self._selected_hand_data() or {}
        records=copy.deepcopy(getattr(self,'frame_review',{}));changed=False
        for entity in self._review_entities(hand,self.selected_hand_label,frame):
            rows=self._boxes_for_review(entity,frame)
            if len(rows)!=1 or self._box_evidence_complete(entity,frame):continue
            # Only accept an entity actually drawn on the current canvas.
            visible=any((b.get('label')==self._get_actor_short_label(entity[2:]) if entity.startswith('H:')
                         else str(b.get('id'))==entity[2:] and not self._normalize_hand_label(b.get('label')))
                        and not b.get('synthetic') for b in self.player.overlay_boxes)
            if not visible:continue
            try:digest=signature(rows)
            except (ValueError,TypeError,KeyError):continue
            records.setdefault(str(frame),{})[entity]={'state':'verified','signature':digest,'source':'viewed_in_editor'}
            rows[0]['human_verified']=True;changed=True
        if changed:
            self.frame_review=records;self._bump_query_state_revision()
        accepted=self._accept_viewed_event_fields(hand)
        if changed or accepted:
            self._refresh_boxes_for_frame(frame,lightweight=True)
            self._refresh_frame_review_status()

    def _accept_viewed_event_fields(self,hand):
        # Visiting the event's keyframes accepts displayed values without another checklist.
        # Keep the original machine states for traceability; never accept missing values.
        frames={hand.get(f) for f in ('interaction_start','interaction_end')}
        onset=hand.get('functional_contact_onset')
        if type(onset) is int:frames.add(onset)
        elif not onset_resolved(hand):return False
        if (any(type(f) is not int for f in frames) or not hand.get('verb') or
                self._noun_required_for_verb(hand['verb']) and self._hand_noun_object_id(hand) is None or
                self._policy_missing(hand) or
                any(not self._box_evidence_complete(entity,f) for f in frames for entity in self._review_entities(hand,self.selected_hand_label,f))):return False
        changed=False
        for field in ('interaction_start','functional_contact_onset','interaction_end','verb','noun_object_id'):
            original=get_field_state(hand,field)
            if original.get('status')!='suggested':continue
            if hand.get(field) is None or hand.get(field)=='':continue
            hand.setdefault('_event_extra',{}).setdefault('source_field_states',{}).setdefault(field,original)
            self._set_hand_field_state(hand,field,source='viewed_in_editor',status='confirmed',note='Accepted through paused SOE views')
            changed=True
        if changed:
            self._apply_draft_to_selected_event();self._bump_query_state_revision();self._refresh_events()
        return changed

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
        cache_key=(getattr(self,'_bbox_revision',0),getattr(self,'_query_state_revision',0),
                   id(getattr(self,'frame_review',None)),id(self.raw_boxes),len(self.raw_boxes),
                   len(self.events),int(self.start_offset),fps)
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
                    if s<=frame<=e and all(box_observed(row.get(entity),index.get((frame,entity),[])) for entity in self._review_entities(hand,key,frame)):valid.append(frame)
                uid=reference_id(hand,self._assembly_data(),s)
                extra=list(hand.get('required_review_frames',[]))
                evidence=hand.get('_event_extra',{}).get('anomaly_evidence',{})
                extra += [f for label in export_review(hand.get('anomaly_label'),PROFILE)['anomaly_labels']
                          for pair in evidence.get(label,{}).get('intervals',[]) for f in pair]
                # Hand-drawn SOE/anomaly boxes count without a separate review click.
                required={s,e,*extra}
                if type(o) is int:required.add(o)
                valid += [f for f in required if s<=f<=e and all(box_observed(records.get(str(f),{}).get(entity),index.get((f,entity),[])) for entity in self._review_entities(hand,key,f))]
                for frame in missing_frames(s,e,o,valid,fps,extra=extra):
                    issues.append(dict(event_id=event['event_id'],hand=key,field='bbox_evidence',code='keyframe_boxes_'+str(frame),frame=frame,missing=['Keyframe boxes missing or not viewed']))
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
        self._pause()
        self._set_selected_event(issue['event_id'],issue['hand'])
        self.player.seek(issue['frame'])
        self._refresh_boxes_for_frame(issue['frame'])
        self._set_frame_controls(issue['frame'])

    def _refresh_frame_review_status(self):
        if not getattr(self,'act_frame_review_status',None):return
        frame=int(self.player.current_frame);entities=self._current_review_entities();row=getattr(self,'frame_review',{}).get(str(frame),{})
        done=sum(box_observed(row.get(k),self._boxes_for_review(k,frame)) for k in entities)
        text=f'Frame {frame}: '+('Boxes complete' if done==len(entities) else 'Complete the visible boxes')+f' ({done}/{len(entities)})' if entities else 'Keyframes: select an event'
        self.act_frame_review_status.setText(text)
        label=getattr(self,'lbl_frame_review_status',None)
        if label:
            label.setText(text)
            color='#15803d' if entities and done==len(entities) else '#a16207'
            if getattr(self,'_frame_review_status_color',None)!=color:
                label.setStyleSheet('color: '+color+';')
                self._frame_review_status_color=color

    def _clear_event_track(self,include_human):
        hand=self.event_draft.get(self.selected_hand_label,{})
        s=hand.get('interaction_start');e=hand.get('interaction_end')
        if type(s) is not int or type(e) is not int:return
        uid=reference_id(hand,self._assembly_data(),s)
        path=track_segments(self._assembly_data(),uid,s,e)
        noun_choice='Object: '+ ' → '.join(str(row['object_id']) for row in path) if path else None
        tool=self._hand_instrument_object_id(hand)
        choices=([noun_choice] if noun_choice else [])+([str(tool)] if tool is not None else [])+['H:'+self.selected_hand_label]
        choice,ok=QInputDialog.getItem(self,'Clear event track','Instance:',choices,0,False)
        if not ok:return
        actor=choice[2:] if choice.startswith('H:') else None
        uid=None if actor or choice==noun_choice else tool
        if QMessageBox.question(self,'Clear event track',f'Clear {"all" if include_human else "automatic"} boxes for {choice}, frames {s}–{e}?\nOverlapping events referencing this instance also see this change. Undo restores it.')!=QMessageBox.Yes:return
        def selected(box):
            if actor:return self._normalize_hand_label(box.get('label'))==actor
            frame=int(box.get('orig_frame',-1))+int(self.start_offset)
            target=resolve_object(hand,self._assembly_data(),frame) if choice==noun_choice else uid
            return object_instance(box,target)
        self._push_undo()
        kept_frames={int(b.get('orig_frame',-1))+int(self.start_offset) for b in self.raw_boxes if actor and selected(b) and protected(b) and not include_human}
        self.raw_boxes=[b for b in self.raw_boxes if not (selected(b) and s<=int(b.get('orig_frame',-1))+int(self.start_offset)<=e and (include_human or not protected(b)))]
        if actor:
            suppressed=set(getattr(self,'_suppressed_hand_boxes',[]))
            suppressed.update(f'{actor}:{frame}' for frame in range(s,e+1) if frame not in kept_frames)
            self._suppressed_hand_boxes=sorted(suppressed)
        self._rebuild_bboxes_from_raw();self._bump_bbox_revision();self._refresh_boxes_for_frame(self.player.current_frame)
