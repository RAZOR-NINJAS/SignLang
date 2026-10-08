import queue
import threading
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

from .config import (
    CAPTURE_HEIGHT,
    CAPTURE_WIDTH,
    DETECT_EVERY,
    DETECT_HEIGHT,
    GPU_DELEGATE,
    LANDMARKER_PATH,
    MAX_HANDS,
    MIRROR,
    SOURCE,
)


def _create_landmarker(delegate) -> "vision.HandLandmarker":
    """Build the landmarker. `delegate` is mpp.BaseOptions.Delegate (CPU or GPU)."""
    return vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(
                model_asset_path=str(LANDMARKER_PATH), delegate=delegate
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


class HandPipeline:
    def __init__(self, source=None, mirror=MIRROR, num_hands=MAX_HANDS,
                 gpu=GPU_DELEGATE, detect_every=DETECT_EVERY):
        if not LANDMARKER_PATH.exists():
            raise FileNotFoundError(
                f"missing {LANDMARKER_PATH}. Run scripts/fetch_models.sh first."
            )
        self.source = SOURCE if source is None else source
        self.mirror = mirror
        self._frames: "queue.Queue" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._ts = 0
        self.started = threading.Event()
        self.error: "BaseException | None" = None
        self.detect_every = max(1, detect_every)
        self._skip_counter = 0
        # Cached landmarks reused on skipped frames, so recognition keeps running
        # while detection sleeps between frames.
        self._last_landmarks = None
        self._last_handedness = None

        self._landmarker = None
        self.delegate = "CPU"
        if gpu:
            try:
                self._landmarker = _create_landmarker(mpp.BaseOptions.Delegate.GPU)
                self.delegate = "GPU"
            except Exception as exc:  # GPU delegate unavailable -> CPU fallback
                print(f"[hands] GPU delegate unavailable ({exc}); using CPU")
        if self._landmarker is None:
            self._landmarker = _create_landmarker(mpp.BaseOptions.Delegate.CPU)

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _open(self):
        if isinstance(self.source, int) or str(self.source).isdigit():
            cap = cv2.VideoCapture(int(self.source))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _capture_loop(self):
        cap = None
        try:
            cap = self._open()
            if not cap.isOpened():
                self.error = RuntimeError(f"cannot open video source: {self.source}")
                self.started.set()
                return
            self.started.set()
            misses = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    misses += 1
                    if misses > 120:
                        self.error = RuntimeError(f"source stopped delivering frames: {self.source}")
                        return
                    time.sleep(0.005)
                    continue
                misses = 0
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                if frame.shape[0] != DETECT_HEIGHT:
                    scale = DETECT_HEIGHT / frame.shape[0]
                    frame = cv2.resize(
                        frame, (int(frame.shape[1] * scale), DETECT_HEIGHT)
                    )
                with self._lock:
                    try:
                        self._frames.get_nowait()
                    except queue.Empty:
                        pass
                    self._frames.put_nowait((time.monotonic(), frame))
        except BaseException as exc:
            self.error = exc
        finally:
            if cap is not None:
                cap.release()
            self.started.set()

    def read(self, timeout=2.0):
        if self.error is not None:
            raise self.error
        try:
            stamp, frame = self._frames.get(timeout=timeout)
        except queue.Empty:
            if self.error is not None:
                raise self.error
            return None, None, None
        self._ts += 1

        # Frame-skip: run the (expensive) hand detector only every DETECT_EVERY-th
        # call, reusing the last landmarks in between. The dwell timer in engine.py
        # is wall-clock based, so recognition timing is unaffected; only the rate at
        # which the landmarks refresh changes. Cheap frames short-circuit before the
        # RGB conversion and mp.Image build, which are the second-largest cost.
        if self.detect_every > 1 and self._last_landmarks is not None:
            self._skip_counter += 1
            if self._skip_counter >= self.detect_every:
                self._skip_counter = 0
                result = self._detect(frame)
                return self._emit(frame, result)
            return frame, self._last_landmarks, self._last_handedness

        result = self._detect(frame)
        return self._emit(frame, result)

    def _detect(self, frame):
        """Convert the BGR frame to an mp.Image and run MediaPipe detection."""
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        return self._landmarker.detect_for_video(image, self._ts)

    def _emit(self, frame, result):
        """Cache landmarks and return the standard (frame, landmarks, handedness)."""
        if not result.hand_landmarks:
            self._last_landmarks = None
            self._last_handedness = None
            return frame, None, None
        handedness = result.handedness[0][0].category_name
        lm = result.hand_landmarks[0]
        self._last_landmarks = lm
        self._last_handedness = handedness
        return frame, lm, handedness

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._landmarker.close()


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def draw_landmarks(frame, landmarks, size=2):
    h, w = frame.shape[:2]
    for a, b in HAND_CONNECTIONS:
        p1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
        p2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
        cv2.line(frame, p1, p2, (0, 200, 255), size, cv2.LINE_AA)
    for i, lm in enumerate(landmarks):
        cv2.circle(
            frame,
            (int(lm.x * w), int(lm.y * h)),
            size + 2 if i in (4, 8, 12, 16, 20) else size,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )
    return frame
