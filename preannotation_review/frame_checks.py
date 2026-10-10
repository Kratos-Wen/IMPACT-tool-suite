import copy
from PyQt5.QtWidgets import QMessageBox,QInputDialog
from core.frame_review import signature,valid_record,missing_frames
from core.assembly_timeline import resolve_object,reference_id,change_frames,track_segments

class ReviewFrameMixin:
    def require_review_current_frame(self):
        if not self.doc or self.current_index<0:return
        event=self.doc.events[self.current_index]['value'];frame=self.player.current_frame
        s=event.get('start_frame');e=event.get('end_frame')
        if type(s) is int and type(e) is int and s<=frame<=e:
            event['required_review_frames']=sorted(set(event.get('required_review_frames',[])+[frame]));self.dirty=True
    def _review_geometry(self,frame):
        boxes={f"EGO_T{t['track_id']:06d}":t for t in self.frames.get(frame,{}).get('tracks',[])}
        for uid,row in self.doc.data['box_overrides'].get(str(frame),{}).items():boxes.setdefault(uid,{}).update(row)
        return boxes
    def _review_roles(self,event,frame):
        obj=event.get('object_instance_id')
        if event.get('shared_assembly_ref'):
            uid=resolve_object(event,self.doc.data.get('shared_assembly',{}),frame)
            obj=f'EGO_T{uid:06d}' if uid is not None else None
        return ['H:'+event.get('hand','unknown')]+['O:'+x for x in dict.fromkeys([obj,event.get('instrument_instance_id')]) if x]
    def _role_boxes(self,role,frame,geometry=None):
        geometry=self._review_geometry(frame) if geometry is None else geometry
        ids=[role[2:]] if role.startswith('O:') else [uid for uid in geometry if self.doc.instance(uid,frame).get('anatomical_hand')==role[2:]]
        result=[]
        for uid in ids:
            row=geometry.get(uid,{})
            if row.get('visible') is True and row.get('bbox_xyxy'):
                b=row['bbox_xyxy'];result.append(dict(zip(('x1','y1','x2','y2'),b)))
        return result
    def verify_review_frame(self):
        if not self.doc or self.current_index<0 or not self.require_reviewer() or not self.apply_current():return
        frame=self.player.current_frame;event=self.doc.events[self.current_index]['value'];roles=self._review_roles(event,frame);geometry=self._review_geometry(frame)
        rows={r:self._role_boxes(r,frame,geometry) for r in roles}
        if any(len(v)>1 for v in rows.values()):QMessageBox.warning(self,'Frame review','Resolve multiple candidate boxes for one role before verifying.');return
        try:signatures={role:signature(boxes) for role,boxes in rows.items() if boxes}
        except (ValueError,KeyError,TypeError):QMessageBox.warning(self,'Frame review','Correct invalid box geometry first.');return
        for role,boxes in rows.items():
            if boxes:self.doc.data.setdefault('frame_review',{}).setdefault(str(frame),{})[role]={'state':'verified','signature':signatures[role]}
        self.dirty=True;self.frame_changed(frame)
    def mark_review_not_visible(self):
        if not self.doc or self.current_index<0 or not self.require_reviewer():return
        frame=self.player.current_frame;roles=self._review_roles(self.doc.events[self.current_index]['value'],frame)
        role,ok=QInputDialog.getItem(self,'Frame review','Entity not visible:',roles,0,False)
        if not ok:return
        if self._role_boxes(role,frame):QMessageBox.warning(self,'Frame review','Hide the incorrect current-frame box first.');return
        self.doc.data.setdefault('frame_review',{}).setdefault(str(frame),{})[role]={'state':'not_visible'};self.dirty=True;self.frame_changed(frame)
    def next_review_frame(self,direction=1):
        if not self.doc or self.current_index<0 or not self.apply_current():return
        event=self.doc.events[self.current_index]['value'];s=event.get('start_frame');e=event.get('end_frame')
        if type(s) is not int or type(e) is not int or s>e:return
        valid=[]
        for text,row in self.doc.data.get('frame_review',{}).items():
            f=int(text)
            if s<=f<=e:
                g=self._review_geometry(f)
                if all(valid_record(row.get(r),self._role_boxes(r,f,g)) for r in self._review_roles(event,f)):valid.append(f)
        data=self.doc.data.get('shared_assembly',{})
        uid=reference_id(event,data,s)
        extra=change_frames(data,uid,s,e) if event.get('shared_assembly_ref') and uid is not None else []
        extra += [f-1 for f in list(extra) if f>s]
        if event.get('shared_assembly_ref'):extra += [row['end_frame'] for row in track_segments(data,uid,s,e)]
        extra+=event.get('required_review_frames',[])
        missing=missing_frames(s,e,event.get('onset_frame'),valid,float(self.doc.data.get('source',{}).get('fps') or getattr(self.player,'frame_rate',15) or 15),extra=extra)
        if not missing:return
        current=self.player.current_frame;after=[f for f in missing if f>current] if direction>0 else [f for f in missing if f<current]
        self.seek((after[0] if direction>0 else after[-1]) if after else (missing[0] if direction>0 else missing[-1]))
    def clear_review_event_track(self):
        if not self.doc or self.current_index<0 or not self.require_reviewer() or not self.apply_current():return
        event=self.doc.events[self.current_index]['value'];s=event.get('start_frame');e=event.get('end_frame')
        if type(s) is not int or type(e) is not int:return
        data=self.doc.data.get('shared_assembly',{})
        anchor=reference_id(event,data,s);path=track_segments(data,anchor,s,e)
        noun='Object IDs: '+' → '.join(f"EGO_T{row['object_id']:06d}" for row in path) if path else None
        roles=([noun] if noun else [])+(['O:'+event['instrument_instance_id']] if event.get('instrument_instance_id') else [])
        if not roles:return
        role,ok=QInputDialog.getItem(self,'Clear automatic track','Object/tool:',roles,0,False)
        if not ok:return
        if QMessageBox.question(self,'Clear automatic track',f'Hide automatic {role}, frames {s}–{e}?\nHuman overrides remain. Shared overlapping events are affected. Undo is available.')!=QMessageBox.Yes:return
        previous={}
        for frame in range(s,e+1):
            number=resolve_object(event,data,frame) if role==noun else None
            uid=f'EGO_T{number:06d}' if number is not None else role[2:]
            overrides=self.doc.data['box_overrides'].setdefault(str(frame),{})
            if uid not in overrides:
                previous[str(frame)]=uid;overrides[uid]={'bbox_xyxy':None,'visible':False,'geometry_source':'human_event_track_rejection'}
        self._cleared_track_undo=(self.doc.data['base_snapshot_sha256'],previous);self.dirty=True;self.frame_changed(self.player.current_frame)
    def undo_review_track_clear(self):
        saved=getattr(self,'_cleared_track_undo',None)
        if not saved or not self.doc or saved[0]!=self.doc.data['base_snapshot_sha256']:return
        _,rows=saved
        for frame,uid in rows.items():
            current=self.doc.data['box_overrides'].get(frame,{}).get(uid,{})
            if current.get('geometry_source')=='human_event_track_rejection':self.doc.data['box_overrides'][frame].pop(uid,None)
        self._cleared_track_undo=None;self.dirty=True;self.frame_changed(self.player.current_frame)
