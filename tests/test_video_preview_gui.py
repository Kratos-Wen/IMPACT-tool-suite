"""Delayed decoding must never block Qt or overwrite committed frame/video state."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
os.environ['QT_QPA_PLATFORM']='offscreen'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import _bootstrap_qt_runtime
_bootstrap_qt_runtime()
import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from ui.video_player import VideoPlayer

app=QApplication([])

def wait_until(condition, milliseconds=3000):
    start=time.monotonic()
    while not condition() and (time.monotonic()-start)*1000<milliseconds:
        QTest.qWait(10)
    assert condition(),'Timed out waiting for a preview'

class Capture:
    def __init__(self):self.position=0
    def set(self,key,frame):self.position=int(frame);return True
    def grab(self):self.position+=1;return True
    def read(self):
        image=np.full((32,32,3),self.position,np.uint8);self.position+=1
        return True,image
    def release(self):pass

p=VideoPlayer();p.cap=Capture();p._media_path='first';p.frame_count=150;p.crop_end=149
p.seek(0)
gate=threading.Event();entered=threading.Event();calls=[];callbacks=[];main_thread=threading.get_ident()
def decode(path,frame):
    assert threading.get_ident()!=main_thread
    calls.append((path,frame));entered.set()
    assert gate.wait(3),'Test decoder did not receive release'
    return np.full((32,32,3),frame+(100 if path=='second' else 0),np.uint8)
def ready(frame):
    assert threading.get_ident()==main_thread
    callbacks.append(frame)
p._preview_decoder.decode=decode
ticks=[];heartbeat=QTimer();heartbeat.setInterval(10);heartbeat.timeout.connect(lambda:ticks.append(time.monotonic()))
heartbeat.start()
start=time.monotonic();p.request_preview(70,ready)
assert time.monotonic()-start<.2
wait_until(entered.is_set)
QTest.qWait(80)
assert len(ticks)>=4 and p.current_frame==p._displayed_frame==0
for frame in range(71,91):p.request_preview(frame,ready)
assert len(calls)==1,'Intermediate preview requests were queued'
gate.set();wait_until(lambda:p._displayed_frame==90)
assert calls==[('first',70),('first',90)] and callbacks==[90]
assert p.current_frame==0
# A committed seek discards a result already decoding on another thread.
gate.clear();entered.clear();p.request_preview(110,ready);wait_until(entered.is_set)
p.seek(20);gate.set();wait_until(lambda:p._preview_future is None)
assert p.current_frame==p._displayed_frame==20 and callbacks==[90]
# Release/switch remains nonblocking and discards old-source results.
gate.clear();entered.clear();p.request_preview(120,ready);wait_until(entered.is_set)
start=time.monotonic();p.release_media();assert time.monotonic()-start<.2
p.cap=Capture();p._media_path='second';p.frame_count=150;p.crop_end=149;p.seek(0)
p.request_preview(25,ready);gate.set();wait_until(lambda:p._displayed_frame==25)
assert int(p.get_current_frame_bgr()[0,0,0])==125 and p.current_frame==0
assert callbacks==[90,25]
# A preview error retains the current image and committed position.
def fail(*a):raise ValueError('simulated decode failure')
p._preview_decoder.decode=fail;p.request_preview(40,ready)
wait_until(lambda:p._preview_future is None)
assert p._displayed_frame==25 and p.current_frame==0 and callbacks==[90,25]
p.release_media();heartbeat.stop();p.deleteLater()
app.processEvents()

with tempfile.TemporaryDirectory() as directory:
    path=Path(directory)/'clip.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),15,(64,64))
    assert writer.isOpened()
    for frame in range(40):writer.write(np.full((64,64,3),frame*5,np.uint8))
    writer.release();reference=cv2.VideoCapture(str(path));p=VideoPlayer();assert p.load(str(path))
    callbacks=[]
    for frame in (20,30,5,6,35):
        p.request_preview(frame,callbacks.append);wait_until(lambda:p._displayed_frame==frame)
        reference.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=reference.read()
        assert ok and np.array_equal(image,p.get_current_frame_bgr()) and p.current_frame==0
    assert callbacks==[20,30,5,6,35]
    p.seek(35);p.play();p.timer.stop();p._on_timer()
    reference.set(cv2.CAP_PROP_POS_FRAMES,36);ok,image=reference.read()
    assert p.current_frame==36 and ok and np.array_equal(image,p.get_current_frame_bgr())
    p.release_media();p.deleteLater();reference.release()
print('BACKGROUND_PREVIEW_HEARTBEAT_COALESCING_COMMIT_VIDEO_SWITCH_ERROR_NATIVE_PIXELS_PASS')
