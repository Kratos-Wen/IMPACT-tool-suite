"""A decoder used only by the serial preview worker; no Qt objects cross threads."""
import cv2


def read_frame(capture, next_frame, target, forward_limit=30):
    if next_frame is not None and 0 <= target-next_frame <= forward_limit and callable(getattr(capture,'grab',None)):
        for _ in range(target-next_frame):
            if not capture.grab():
                return None, None
    elif next_frame != target:
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, image = capture.read()
    return (image, target+1) if ok else (None, None)


class PreviewDecoder:
    def __init__(self):
        self.path = None
        self.capture = None
        self.next_frame = None

    def decode(self, path, frame):
        if path != self.path or self.capture is None:
            self.close()
            self.capture = cv2.VideoCapture(path)
            self.path = path
            if not self.capture.isOpened():
                self.close()
                raise ValueError('Preview decoder could not open the video')
        image, self.next_frame = read_frame(self.capture, self.next_frame, frame)
        if image is None:
            raise ValueError('Preview frame could not be decoded')
        return image

    def close(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self.path = None
        self.next_frame = None


def shutdown_preview(executor, decoder):
    # Close on the same worker after any in-flight decode, without blocking Qt.
    try:
        executor.submit(decoder.close)
        executor.shutdown(wait=False)
    except RuntimeError:
        pass  # The player already shut this executor down.
