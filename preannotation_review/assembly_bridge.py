"""Editable shared composition outside the immutable machine snapshot."""
import copy
from types import SimpleNamespace
from PyQt5.QtWidgets import QWidget,QMessageBox
from ui.assembly_editor import AssemblyEditorMixin
from core.assembly_timeline import state_at,validate_timeline,empty_timeline

class ReviewAssemblyBridge(AssemblyEditorMixin,QWidget):
    def __init__(self,window):
        super().__init__(window);self.window=window;doc=window.doc
        self.shared_assembly=validate_timeline(doc.data.get('shared_assembly',empty_timeline()))
        self._assembly_default_reference=doc.data.get('assembly_default_reference',False)
        self.frame_review=copy.deepcopy(doc.data.get('frame_review',{}))
        self.player=SimpleNamespace(cap=True,current_frame=window.player.current_frame,frame_count=doc.frame_count)
        self.global_object_map={};self.id_to_category={};self.raw_boxes=[]
        for uid in sorted(doc.instance_ids()):
            if uid.startswith('EGO_T') and uid[5:].isdigit():
                n=int(uid[5:]);meta=doc.instance(uid,self.player.current_frame);name=meta.get('label') or ','.join(meta.get('category_candidates',[]))
                self.global_object_map[f'{name} [{uid}]']=n;self.id_to_category[f'{name} [{uid}]']=name
        self.object_id_counter=max(self.global_object_map.values(),default=0)+1
        row=doc.events[window.current_index] if window.current_index>=0 else None
        self.event_draft={'current':copy.deepcopy(row['value'])} if row else {}
        self.events=[];self.selected_event_id=None;self.selected_hand_label='current'
        self.video_path=doc.data.get('source',{}).get('video_path','')
        self._task_trial_id=doc.data.get('trial_id','')
        self.previous=None
    def _push_undo(self):
        self.previous=copy.deepcopy(self.window.doc.data)
    def _register_object_entry(self,uid,label):
        key=f'EGO_T{uid:06d}'
        if key not in self.window.doc.instance_ids():
            self.window.doc.data['human_instances'][key]=dict(track_uid=key,label=label,anatomical_hand='unknown',reviewer=self.window.reviewer.text().strip())
    def _refresh_boxes_for_frame(self,frame):pass
    def _load_hand_draft_to_ui(self,hand):pass
    def _checkpoint_tracking(self):pass
    def _bump_query_state_revision(self):
        doc=self.window.doc
        if self.previous is not None:
            self.window._assembly_undo=self.previous
        doc.data['shared_assembly']=copy.deepcopy(self.shared_assembly)
        # Reviewer roles use stable track UID keys instead of numeric object IDs.
        reviews=copy.deepcopy(doc.data.get('frame_review',{}))
        old_data=self.previous.get('shared_assembly',{}) if self.previous else {}
        for text,records in reviews.items():
            for uid in {s['object_id'] for s in old_data.get('states',[])+self.shared_assembly['states']}:
                if state_at(old_data,int(text),uid)!=state_at(self.shared_assembly,int(text),uid):
                    records.pop(f'O:EGO_T{uid:06d}',None)
        doc.data['frame_review']=reviews
        doc.data['assembly_default_reference']=self._assembly_default_reference
        if self.window.current_index>=0 and self.event_draft:
            value=doc.events[self.window.current_index]['value'];hand=self.event_draft['current']
            value['shared_assembly_ref']=bool(hand.get('shared_assembly_ref'))
            value['shared_assembly_id']=hand.get('shared_assembly_id')
        self.window.dirty=True;self.window.frame_changed(self.player.current_frame)
