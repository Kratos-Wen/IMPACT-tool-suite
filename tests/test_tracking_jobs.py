"""Three-way merge protects manual edits, deletions, occlusion and namespaces."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.tracking_jobs import merge_branch,bindings_valid

class MergeTests(unittest.TestCase):
    def setUp(self):
        self.boxes=[dict(id=0,label='part',orig_frame=f,x1=1,y1=2,x2=11,y2=12,source='manual' if f==0 else 'auto') for f in range(6)]
        self.request=dict(id=0,label='part',entity_kind='object',start=0,end=5,offset=0,direction=1)
        self.result=dict(job_index=0,end=5,boxes=[dict(frame=f,x1=2,y1=3,x2=12,y2=13) for f in range(1,6)],empty_frames=[])
    def merge(self,current=None,end=5,review=()):return merge_branch(current or self.boxes,self.boxes,self.request,self.result,end,review)
    def test_other_entity_changes_do_not_discard(self):
        current=copy.deepcopy(self.boxes);other=dict(current[1],id=8,x1=3);current.append(other)
        out,branch,conflict=self.merge(current);self.assertIn(other,out);self.assertEqual(branch[0]['end'],5);self.assertIsNone(conflict)
        self.assertEqual(self.boxes[1]['x1'],1)
    def test_manual_edit_cuts_before_reference(self):
        current=copy.deepcopy(self.boxes);current[3].update(x1=4,source='manual_box_edit')
        out,branch,conflict=self.merge(current);self.assertEqual(branch[0]['end'],2);self.assertIn(current[3],out);self.assertEqual(conflict['frame'],3)
    def test_delete_is_not_resurrected(self):
        current=[copy.deepcopy(b) for b in self.boxes if b['orig_frame']!=3]
        out,branch,_=self.merge(current);self.assertEqual(branch[0]['end'],2);self.assertFalse(any(b['orig_frame']==3 for b in out))
    def test_seed_changed_invalidates_only_its_branch(self):
        current=copy.deepcopy(self.boxes);current[0]['x1']=3
        out,branch,conflict=self.merge(current);self.assertEqual(out,current);self.assertIsNone(branch);self.assertEqual(conflict['reason'],'anchor_changed')
    def test_interval_clipped(self):self.assertEqual(self.merge(end=2)[1][0]['end'],2)
    def test_identity_change_rejects(self):self.assertIsNone(self.merge(end=None)[1])
    def test_visibility_decision_preserved(self):self.assertEqual(self.merge(review={2})[1][0]['end'],1)
    def test_hand_object_id_collision(self):
        hand=dict(self.boxes[1],label='Left_hand');current=copy.deepcopy(self.boxes)+[hand]
        self.assertIn(hand,self.merge(current)[0])
    def test_nonfinite_and_incomplete_results_fail_without_mutation(self):
        before=copy.deepcopy(self.boxes);self.result['boxes'].pop()
        with self.assertRaises(ValueError):self.merge()
        self.assertEqual(before,self.boxes)
    def test_backward(self):
        self.request.update(start=5,end=0,direction=-1);self.boxes[0]['source']='auto';self.boxes[5]['source']='manual'
        self.result.update(end=0,boxes=[dict(frame=f,x1=2,y1=3,x2=12,y2=13) for f in range(5)])
        current=copy.deepcopy(self.boxes);current[3]['source']='manual_box_edit'
        self.assertEqual(self.merge(current,end=0)[1][0]['end'],4)
    def test_verb_and_onset_do_not_change_binding(self):
        old=dict(event_id=1,actor='Left_hand',role='object',identity=0,object_name='part',start=0,end=5)
        self.assertTrue(bindings_valid(old,dict(old,verb='align',onset=1,end=4)))
        self.assertFalse(bindings_valid(old,dict(old,identity=8)))
    def test_large_background_track_list_is_not_rescanned_per_frame(self):
        class ObservedList(list):
            passes=0
            def __iter__(self):
                type(self).passes+=1
                return super().__iter__()
        self.request['end']=60
        self.boxes=[dict(self.boxes[0],orig_frame=f,source='manual' if f==0 else 'auto') for f in range(61)]
        current=ObservedList(self.boxes+[dict(self.boxes[1],id=i+1,orig_frame=i%1000) for i in range(10000)])
        self.result.update(end=60,boxes=[dict(frame=f,x1=2,y1=3,x2=12,y2=13) for f in range(1,61)])
        out,branch,conflict=self.merge(current,end=60)
        self.assertEqual(len(out),len(current));self.assertEqual(branch[0]['end'],60)
        self.assertIsNone(conflict);self.assertLess(ObservedList.passes,8)

if __name__=='__main__':unittest.main()
