"""Exercise editor methods without importing optional model runtimes."""
import ast,copy,json,re,types
from pathlib import Path
root=Path(__file__).resolve().parent.parent
results=[]
for name in ['']:
 p=root/name/'ui/hoi_window.py';tree=ast.parse(p.read_text(encoding='utf-8-sig'))
 wanted={'_on_box_edited','_event_visible_object_ids','_event_box_visible','_frame_boxes_with_cached_hands','_frame_hand_box'}
 methods=[n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
 for n in methods:
  n.returns=None
  for a in n.args.args:a.annotation=None
 cls=ast.ClassDef(name='Editor',bases=[],keywords=[],body=methods,decorator_list=[])
 mod=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]));ns={'re':re}
 exec(compile(mod,str(p),'exec'),ns);E=ns['Editor']
 def make():
  e=E();e.selected_event_id=1;e.selected_hand_label='left_hand';e.start_offset=0
  e.event_draft={'left_hand':{'noun_object_id':2,'instrument_object_id':0},'right_hand':{}}
  e.player=types.SimpleNamespace(current_frame=10)
  e.raw_boxes=[dict(id=i,orig_frame=f,label='screw',x1=1.,y1=2.,x2=20.,y2=30.) for i,f in [(1,10),(2,10),(2,11)]]
  e._selected_edit_box=copy.deepcopy(e.raw_boxes[1]);e._pending_draw_box_role='noun';e.object_id_counter=3;e.box_id_counter=100;e.class_map={7:'screw'}
  e._normalize_hand_label=lambda s: s if s in ['left_hand','right_hand'] else ''
  e._hand_noun_object_id=lambda d:d.get('noun_object_id')
  e._hand_instrument_object_id=lambda d:d.get('instrument_object_id')
  e._box_locked_for_action=lambda *a,**k:False
  e._same_box_identity=lambda a,b:bool(a and b and a.get('id')==b.get('id') and a.get('orig_frame')==b.get('orig_frame'))
  e._resolve_label_and_id=lambda s:(s,7,True)
  e._is_hand_label=lambda s:bool(e._normalize_hand_label(s))
  e._default_object_id_for_label=lambda s:1
  e._object_name_for_id=lambda i,**kw:'screw' if i in [0,1,2] else ''
  e._selected_hand_data=lambda:e.event_draft['left_hand']
  for method in ['_push_undo','_rebuild_bboxes_from_raw','_bump_bbox_revision','_bump_query_state_revision','_refresh_boxes_for_frame','_sync_selected_hand_noun_after_box_relabel','_log','_register_object_entry','_assign_drawn_object_to_current_hand']:
   setattr(e,method,lambda *a,**kw:None)
  return e
 e=make();assert e._event_visible_object_ids()=={'0','2'}
 assert e._event_box_visible({'id':0,'label':'tool'})
 assert not e._event_box_visible({'id':1,'label':'screw'})
 assert e._event_box_visible({'id':123,'label':'right_hand'})
 e.selected_event_id=None;assert not e._event_box_visible({'id':2,'label':'screw'})
 e=make();b=dict(e.raw_boxes[1],frame=10,x1=5.);e._on_box_edited(2,b)
 assert e.raw_boxes[1]['id']==2 and e.raw_boxes[1]['x1']==5.
 assert e.raw_boxes[0]['x1']==1. and e.raw_boxes[2]['x1']==1.
 e=make();e._on_box_edited(2,dict(e.raw_boxes[1],frame=10,_action='delete'))
 assert [(b['id'],b['orig_frame']) for b in e.raw_boxes]==[(1,10),(2,11)]
 e=make();e._on_box_edited(None,dict(label='screw',frame=12,_action='add',x1=1,y1=2,x2=20,y2=30))
 assert e.raw_boxes[-1]['id']==2
 e=make();e._pending_draw_box_role=None;e._on_box_edited(None,dict(label='screw',frame=12,_action='add',x1=1,y1=2,x2=20,y2=30))
 assert e.raw_boxes[-1]['id']==3
 e=make();e._pending_draw_box_role=None;e._on_box_edited(None,dict(label='2',frame=12,_action='add',x1=1,y1=2,x2=20,y2=30))
 assert e.raw_boxes[-1]['id']==2
 e=make();e._suppressed_hand_boxes=['left_hand:10'];assert e._frame_hand_box('left_hand',10)=={}
 e=make();e.raw_boxes[1]['locked']=True;e._box_locked_for_action=lambda *a,**kw:True
 e._on_box_edited(2,dict(e.raw_boxes[1],frame=10,x1=5.));assert e.raw_boxes[1]['x1']==5.
 e=make();e.raw_boxes[1]['locked']=True;e._box_locked_for_action=lambda *a,**kw:True
 e._on_box_edited(2,dict(e.raw_boxes[1],frame=10,_action='delete'));assert len(e.raw_boxes)==2
 results.append({'variant':name,'passed':['selected_event_ID_filter_including_zero','no_background_when_no_event','geometry_keeps_instance_ID','delete_current_frame_only','noun_add_exact_ID','manual_name_creates_new_instance','numeric_add_reuses_ID','deleted_cached_hand_not_regenerated']})

for name in ['']:
 tree=ast.parse((root/name/'preannotation_review/review_gui.py').read_text())
 methods=[n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name in ['box_changed','hide_box']]
 cls=ast.ClassDef(name='Reviewer',bases=[],keywords=[],body=methods,decorator_list=[])
 ns={'copy':copy,'now':lambda:'test'}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),'reviewer','exec'),ns)
 e=ns['Reviewer']();e.doc=types.SimpleNamespace(data={'audit_log':[],'box_overrides':{'11':{'EGO_T000002':{'bbox_xyxy':[1,2,20,30],'visible':True}}}})
 e.player=types.SimpleNamespace(current_frame=10);e.reviewer=types.SimpleNamespace(text=lambda:'test');e.require_reviewer=lambda:True
 e.frame_changed=lambda *a:None;e.refresh_summary=lambda *a:None
 e.box_changed('EGO_T000002',{'_action':'delete','x1':1,'y1':2,'x2':20,'y2':30})
 assert e.doc.data['box_overrides']['10']['EGO_T000002']['bbox_xyxy'] is None
 assert e.doc.data['box_overrides']['11']['EGO_T000002']['visible'] is True
print('Reviewer current-frame deletion checks passed for the application')
print(json.dumps(results,indent=2))
