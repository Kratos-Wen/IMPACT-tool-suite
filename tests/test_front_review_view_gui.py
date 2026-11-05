"""Rotated review pixels, native editing coordinates, HUD and permissive saves."""
import copy,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
import numpy as np
from PyQt5.QtCore import Qt,QPoint,QRect,QEvent
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication,QMessageBox,QFileDialog
from ui.video_player import VideoPlayer
app=QApplication([])
for name in ('information','warning','question','critical'):
    setattr(QMessageBox,name,lambda *a,**k:QMessageBox.No)

player=VideoPlayer();player.resize(640,480)
image=np.zeros((240,320,3),np.uint8);image[:120,:160]=(0,0,255)
player._show_native_frame(image,0);player.set_review_rotation(180);player.show();app.processEvents()
grab=player.grab();ratio=grab.devicePixelRatio()
assert grab.toImage().pixelColor(round(600*ratio),round(440*ratio)).red()>220,'Native top-left pixels rotate to bottom-right'
assert np.array_equal(player._last_frame_bgr,image),'Detection and stored coordinates remain native'
for zoom,pan in [(1,(0,0)),(2.1,(-80,-110)),(.7,(20,30))]:
    player._zoom=zoom;player._pan_x,player._pan_y=pan;player._ensure_pan_clamped();player.repaint()
    for ix,iy in [(0,0),(30,50),(320,240),(159.5,119.5)]:
        point=player._image_to_screen(ix,iy);back=player._screen_to_image(*point)
        assert abs(back[0]-ix)<1e-7 and abs(back[1]-iy)<1e-7
player._zoom=1;player._recenter_keep_zoom();player.repaint()
box=dict(id=17,orig_frame=0,label='part',x1=40,y1=40,x2=160,y2=160)
events=[]
player.set_edit_context([box],on_change=lambda uid,data:events.append((uid,copy.deepcopy(data))),selected_box=box)
player.set_overlay_boxes([dict(box,color='#22ff22')]);player.set_overlay_relations([dict(x1=40,y1=40,x2=160,y2=160,label='hold')]);player.repaint()
rect=player._box_screen_rect(box);assert (rect.x(),rect.y(),rect.width(),rect.height())==(320,160,240,240)
def move(a,b,mod=Qt.NoModifier):
    QTest.mousePress(player,Qt.LeftButton,mod,QPoint(*a))
    player.mouseMoveEvent(QMouseEvent(QEvent.MouseMove,QPoint(*b),Qt.NoButton,Qt.LeftButton,mod))
    QTest.mouseRelease(player,Qt.LeftButton,mod,QPoint(*b));app.processEvents()
center=tuple(round(x) for x in player._image_to_screen(100,100))
move(center,(center[0]+20,center[1]+10))
assert events[-1][0]==17 and box['x1']==30 and box['y1']==35,'Dragging right/down moves native coordinates left/up'
corner=tuple(round(x) for x in player._image_to_screen(box['x1'],box['y1']))
move(corner,(corner[0]+10,corner[1]+8))
assert (box['x1'],box['y1'],box['x2'],box['y2'])==(25,31,150,155),'Rotated corner handles resize the correct native edge'
player.set_edit_context([],on_change=lambda uid,data:events.append((uid,copy.deepcopy(data))),allow_add=True)
player.edit_auto_label_fetcher=lambda:('part',2)
a=tuple(round(x) for x in player._image_to_screen(20,30));b=tuple(round(x) for x in player._image_to_screen(80,90))
move(a,b,Qt.ControlModifier)
added=events[-1][1];assert added['_action']=='add'
assert all(abs(added[k]-value)<=1 for k,value in dict(x1=20,y1=30,x2=80,y2=90).items())
player.set_edit_context([box],on_change=lambda uid,data:events.append((uid,copy.deepcopy(data))),selected_box=box)
center=QPoint(*(round(x) for x in player._image_to_screen((box['x1']+box['x2'])/2,(box['y1']+box['y2'])/2)))
QTest.mouseClick(player,Qt.RightButton,Qt.NoModifier,center)
assert events[-1][0]==17 and events[-1][1]['_action']=='delete'
player._zoom=2;player._pan_x=-80;player._pan_y=-110;player._ensure_pan_clamped();player.repaint()
selection=QRect(200,150,120,80)
anchor=player._screen_to_image((selection.left()+selection.right())/2,(selection.top()+selection.bottom())/2)
player._zoom_to_selection(selection);player.repaint()
center_native=player._screen_to_image(player.width()/2,player.height()/2)
assert all(abs(a-b)<1e-6 for a,b in zip(anchor,center_native)),'Magnifier selection remains centered after rotation, zoom and pan'
player._zoom=1;player._recenter_keep_zoom();player.repaint()
player.set_review_rotation(0);player.repaint();assert player._screen_to_image(40,60)==(20,30)
player.release_media();player.close();player.deleteLater();app.processEvents()

from core.project_profile import PROFILE
PROFILE.clear();PROFILE.update(verbs=['hold','pick_up'],noun_classes=['part','tool'],anomaly_labels=['repeated_attempts','control_loss'],anomaly_display_names={'repeated_attempts':'Repeated attempts','control_loss':'Control loss'})
from ui.hoi_window import HOIWindow
from core.video_validation import annotation_signature
class Capture:
    def set(self,*args):return True
    def read(self):return True,np.zeros((240,320,3),np.uint8)
    def release(self):pass
w=HOIWindow();w._autosave_timer.stop();w._show_quick_start_on_startup=False
w.player.cap=Capture();w.player.frame_count=40;w.player.frame_rate=30;w.player.crop_start=0;w.player.crop_end=39;w.player._frame_w=320;w.player._frame_h=240
payload=dict(schema='hoi-annotation',video_id='front_fixture',fps=30,frame_count=40,frame_size=[320,240],
    object_library={'4':dict(label='part',category='part'),'8':dict(label='tool',category='tool')},
    tracks={'T_LEFT_HAND':dict(category='left_hand',boxes=[dict(frame=10,bbox=[1,2,40,50],source='manual_box_add')])},
    hoi_events={'left_hand':[dict(event_id='L_001',start_frame=10,contact_onset_frame=10,end_frame=20,verb='hold',noun_object_id=4,anomaly_labels=['repeated_attempts'])],
                'right_hand':[dict(event_id='R_001',start_frame=10,contact_onset_frame=10,end_frame=30,verb='pick_up',noun_object_id=8,anomaly_labels=['control_loss'])]})
w._configure_review_view(task={'view':'front'});w._load_annotations_v2(payload,annotation_path='reviewed.json')
assert w.player.review_rotation==0,'Front in ordinary annotation mode stays native'
w._set_selected_event(w.events[0]['event_id'],'Left_hand');w.validator_name='Test reviewer';w.btn_validation.setChecked(True)
w.player.current_frame=10;w._refresh_boxes_for_frame(10)
labels='\n'.join(text for text,_ in w.player.review_labels)
assert 'L: hold' in labels and 'R: pick up' in labels
assert 'Repeated attempts' in labels and 'Control loss' in labels and 'Object [4]' in labels
assert w.player.review_rotation==180
original_review=copy.deepcopy(w.frame_review)
w.frame_review={'10':{'O:4':{'state':'not_visible'}}};w._refresh_boxes_for_frame(10)
assert 'No keyframe box:' not in '\n'.join(text for text,_ in w.player.review_labels),'Recorded invisibility does not require an invented box'
w.frame_review=original_review
assert not w.btn_correct_track.isEnabled()
with patch.object(w,'_correction_targets',side_effect=AssertionError('Local propagation must not start in validation')):
    w._start_correction_propagation()
w.player.current_frame=25;w._refresh_boxes_for_frame(25)
labels='\n'.join(text for text,_ in w.player.review_labels)
assert 'L: —' in labels and 'R: pick up' in labels and 'Repeated attempts' not in labels,'HUD follows the playhead, not a selected inactive event'
before=w._build_payload_v2();w.player.set_review_rotation(0);unrotated=w._build_payload_v2()
assert annotation_signature(before)==annotation_signature(unrotated),'Changing a display rotation never edits annotation geometry'
w._configure_review_view(task={'view':'front'})
prompts=[]
def approve(parent,title,message,*args,**kwargs):prompts.append(message);return QMessageBox.Yes
with tempfile.TemporaryDirectory() as directory:
    target=Path(directory)/'reviewed.json';w._task_resume_path=str(target)
    with patch.object(QFileDialog,'getSaveFileName',return_value=(str(target),'JSON Files (*.json)')),patch.object(w,'_require_participant_code_for_study_save',return_value=True),patch.object(QMessageBox,'question',side_effect=approve):
        w._save_annotations_json()
    assert target.exists() and any('Continue anyway' in p for p in prompts),'Missing boxes warn but do not prevent an explicit save'
    saved=json.loads(target.read_text());assert saved['review_view']['coordinate_space']=='native_video' and saved['review_view']['rotation_degrees']==180
    w._load_annotations_v2(saved,annotation_path=str(target));assert w.player.review_rotation==180
    assert saved['tracks']['T_LEFT_HAND']['boxes'][0]['bbox']==[1,2,40,50]
for view in ('ego','left','right','unknown',''):
    w._configure_review_view(task={'view':view})
    assert w.player.review_rotation==0,view+' must not rotate even in Validation'
w._configure_review_view(task={'view':'front'});assert w.player.review_rotation==180
for _ in range(3):
    w.btn_validation.setChecked(False)
    assert w.player.review_rotation==0 and not w.player.review_labels,'Leaving Validation restores native Front and clears HUD'
    assert w._review_view_metadata()['view']=='front','Disabling Validation must retain the source view'
    w.btn_validation.setChecked(True)
    assert w.player.review_rotation==180,'Reentering Validation rotates Front again'
w.btn_validation.setChecked(False)
w._configure_review_view(task={})
w._configure_review_view(payload={'review_view':{'view':'front','coordinate_space':'native_video','rotation_degrees':180}})
assert w.player.review_rotation==0,'Saved display rotation cannot bypass ordinary annotation mode'
w.btn_validation.setChecked(True);assert w.player.review_rotation==180,'Standalone saved Front keeps its view through mode toggles'
w._configure_review_view(task={'view':'ego'},payload={'review_view':{'view':'front','coordinate_space':'native_video'}})
assert w.player.review_rotation==0,'Explicit current Ego task overrides saved Front view'

w._stop_autosave();w._mark_hoi_saved();w._close_request_approved=True;w.close();w.player.release_media();w.deleteLater();app.processEvents()
print('FRONT_ROTATION_NATIVE_GEOMETRY_GESTURES_HUD_MISSING_BOXES_FORCE_SAVE_AND_RELOAD_PASS')
