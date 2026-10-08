"""Small controls for evidence and explicit missing-contact decisions."""
from copy import deepcopy
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QDialogButtonBox
from core.anomaly_attributes import export_review
from core.annotation_migration import model_from_trial
from core.assembly_timeline import state_at
from core.project_profile import PROFILE

def onset_resolved(hand):
    extra=hand.get('_event_extra',hand)
    return extra.get('onset_review_state') in ('unknown','no_contact') and bool(str(extra.get('onset_reason','')).strip())

class AnnotationPolicyMixin:
    def _install_annotation_policy(self):
        menu=self.file_menu.addMenu('Event review')
        menu.addAction('Review anomaly evidence...',self._edit_anomaly_evidence)
        menu.addAction('Record unobservable / absent contact...',self._record_contact_exception)
        menu.addAction('Clear contact exception',self._clear_contact_exception)

    def _record_contact_exception(self):
        hand=self._selected_hand_data()
        if not hand:return
        choice,ok=QInputDialog.getItem(self,'Contact','Contact decision:', ['Not observable','No contact occurred'],0,False)
        if not ok:return
        reason,ok=QInputDialog.getText(self,'Contact','Evidence / reason:')
        if not ok or not reason.strip():return
        self._push_undo();hand['functional_contact_onset']=None
        hand.setdefault('_event_extra',{}).update(onset_review_state='unknown' if choice=='Not observable' else 'no_contact',onset_reason=reason.strip())
        self._apply_draft_to_selected_event();self._load_hand_draft_to_ui(self.selected_hand_label);self._update_status_label()

    def _clear_contact_exception(self):
        hand=self._selected_hand_data()
        if not hand:return
        self._push_undo()
        for k in ('onset_review_state','onset_reason'):hand.setdefault('_event_extra',{}).pop(k,None)
        self._apply_draft_to_selected_event();self._update_status_label()

    def _edit_anomaly_evidence(self):
        hand=self._selected_hand_data()
        if not hand:return
        labels=export_review(hand.get('anomaly_label'),PROFILE)['anomaly_labels']
        if not labels:
            QMessageBox.information(self,'Evidence','Select an attribute first.');return
        old=hand.get('_event_extra',{}).get('anomaly_evidence',{})
        dlg=QDialog(self);dlg.setWindowTitle('Anomaly evidence');layout=QVBoxLayout(dlg)
        layout.addWidget(QLabel('Frame or inclusive frame interval, e.g. 120 or 120-135. Use the current video.'))
        table=QTableWidget(len(labels),3);table.setHorizontalHeaderLabels(['Attribute','Evidence frames','Note']);layout.addWidget(table)
        for row,label in enumerate(labels):
            title=QTableWidgetItem(PROFILE.get('anomaly_display_names',{}).get(label,label));title.setFlags(title.flags() & ~2);table.setItem(row,0,title)
            ranges=old.get(label,{}).get('intervals',[])
            text=', '.join(str(a) if a==b else f'{a}-{b}' for a,b in ranges)
            table.setItem(row,1,QTableWidgetItem(text));table.setItem(row,2,QTableWidgetItem(old.get(label,{}).get('note','')))
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel);layout.addWidget(buttons);buttons.rejected.connect(dlg.reject)
        def save():
            import re
            evidence={}
            try:
                for row,label in enumerate(labels):
                    intervals=[]
                    for text in table.item(row,1).text().split(','):
                        text=text.strip()
                        if not text:continue
                        match=re.fullmatch(r'(\d+)(?:\s*-\s*(\d+))?',text)
                        if not match:raise ValueError('Use integer frames or frame intervals.')
                        a=int(match[1]);b=int(match[2] or match[1])
                        if not 0<=a<=b<self.player.frame_count:raise ValueError('Evidence is outside the video.')
                        if type(hand.get('interaction_start')) is int and a<hand['interaction_start'] or type(hand.get('interaction_end')) is int and b>hand['interaction_end']:raise ValueError('Keep evidence inside this event; adjust the event boundary if needed.')
                        intervals.append([a,b])
                    evidence[label]={'view':'ego','intervals':intervals,'note':table.item(row,2).text().strip()}
            except ValueError as exc:QMessageBox.warning(dlg,'Evidence',str(exc));return
            self._push_undo();hand.setdefault('_event_extra',{})['anomaly_evidence']=evidence
            hand['required_review_frames']=sorted(set(hand.get('required_review_frames',[]))|{f for record in evidence.values() for pair in record['intervals'] for f in pair})
            self._apply_draft_to_selected_event();self._bump_query_state_revision();self._update_status_label();dlg.accept()
        buttons.accepted.connect(save);dlg.resize(700,260);dlg.exec_()

    def _policy_missing(self,hand):
        missing=[]
        target=self._hand_noun_object_id(hand)
        if target is not None and not hand.get('shared_assembly_ref'):
            categories={self._norm_category(name) for name,uid in self.global_object_map.items() if uid==target}
            if 'assembly' in categories:missing.append('assembly composition')
        model=model_from_trial(getattr(self,'_task_trial_id','')+' '+str(self.video_path))
        forbidden=set((PROFILE.get('model_component_rules',{}).get(model,{}) or {}).get('forbidden_components',[]))
        review=export_review(hand.get('anomaly_label'),PROFILE)
        if review['anomaly_review_state'] not in ('reviewed','partial'):missing.append('anomaly review')
        if review['anomaly_review_state']=='partial':missing.append('unknown attributes')
        if review['anomaly_labels']:
            evidence=hand.get('_event_extra',{}).get('anomaly_evidence',{})
            for label in review['anomaly_labels']:
                intervals=evidence.get(label,{}).get('intervals') or []
                s=hand.get('interaction_start');e=hand.get('interaction_end')
                if not intervals or any(not isinstance(pair,(list,tuple)) or len(pair)!=2 or any(type(f) is not int for f in pair) or pair[0]>pair[1] or type(s) is int and pair[0]<s or type(e) is int and pair[1]>e for pair in intervals):missing.append('anomaly evidence')
        if hand.get('shared_assembly_ref'):
            s=hand.get('interaction_start');e=hand.get('interaction_end')
            if type(s) is int and type(e) is int:
                states=[state_at(self._assembly_data(),s)]+[x for x in self._assembly_data().get('states',[]) if s<x['frame']<=e]
                if any(not x or x.get('composition_review_state','reviewed')!='reviewed' for x in states):missing.append('assembly composition')
                if any(x and forbidden.intersection(x.get('components',[])) for x in states):missing.append('model-incompatible composition')
        model=model_from_trial(getattr(self,'_task_trial_id','')+' '+str(self.video_path))
        forbidden=set((PROFILE.get('model_component_rules',{}).get(model,{}) or {}).get('forbidden_components',[]))
        for uid in (self._hand_noun_object_id(hand),self._hand_instrument_object_id(hand)):
            if uid is not None and any(self._canonical_label_name(name).rsplit('_',1)[0] in forbidden or self._canonical_label_name(name) in forbidden for name,value in self.global_object_map.items() if value==uid):missing.append('model-incompatible component')
        return list(dict.fromkeys(missing))

    def _policy_review_issues(self):
        issues=[]
        for event in self.events:
            for hand,data in event.get('hoi_data',{}).items():
                if not data.get('verb'):continue
                for name in self._policy_missing(data):
                    issues.append({'event_id':event['event_id'],'hand':hand,'frame':data.get('interaction_start') or 0,
                                   'field':'anomaly_label' if name.startswith(('anomaly','unknown')) else 'noun_object_id',
                                   'code':name.replace(' ','_'),'missing':[name]})
        return issues
