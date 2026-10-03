"""Background hand classification for the game.

One thread reads frames from the existing HandPipeline and publishes the
current best guess. Nothing else in the game waits on it.

This module does NOT modify hands.py. It instantiates the pipeline, reads
frames, normalises landmarks with features.normalize_hand, and classifies
them with features.landmark_features + model.Predictor - all existing code
called the existing way. Capture behaviour is whatever hands.py already does.

Threading contract:
    * one lock, held for microseconds only
    * results are plain scalars, so the 60fps animation never blocks on us
    * a failed frame leaves the previous reading in place rather than
      reporting a blank, so a dropped detection does not look like the
      player leaving the frame
"""

import threading
import time

import cv2
import numpy as np

from ..config import SOURCE
from ..features import landmark_features, normalize_hand
from ..hands import draw_landmarks
from ..model import Predictor

# Width of the preview copy kept for the browser. Small on purpose: the page
# only shows it in a corner, and encoding a big JPEG every frame would steal
# CPU from the detector on a weak laptop.
PREVIEW_WIDTH = 320


class LetterReader:
    """Classifies hand shapes on a background thread.

    The web page polls .reading() rather than the other way round: the game
    loop must never be held up by MediaPipe running at 10fps.
    """

    def __init__(self, source=None, predictor=None, hold_ms=420, min_conf=0.0):
        self.source = SOURCE if source is None else source
        self.hold_ms = int(hold_ms)
        self.min_conf = float(min_conf)
        self._predictor = predictor
        self._lock = threading.Lock()
        self._letter = None
        self._conf = 0.0
        self._since = 0.0
        self._last_frame_at = 0.0
        self._frames = 0
        self._hands_seen = 0
        self._fps = 0.0
        self._error = None
        self._stop = threading.Event()
        self._thread = None
        self._pipeline = None
        self._preview = None       # latest JPEG bytes for the corner window
        self._preview_at = 0.0
        self._preview_seq = 0

    # -- published state -------------------------------------------------

    def reading(self):
        """(letter, confidence, fps, stale_seconds) - never blocks meaningfully."""
        with self._lock:
            letter, conf = self._letter, self._conf
            last = self._last_frame_at
            fps = self._fps
            frames, hands = self._frames, self._hands_seen
        stale = (time.monotonic() - last) if last else 0.0
        return letter, conf, fps, stale, frames, hands

    def error(self):
        return self._error

    def preview(self):
        """(jpeg_bytes, sequence) for the browser's corner window.

        The camera can only be opened by one process, and this one already has
        it, so the page cannot use getUserMedia. It gets these frames instead.
        """
        with self._lock:
            return self._preview, self._preview_seq

    # -- lifecycle -------------------------------------------------------

    def start(self):
        from ..hands import HandPipeline  # imported late: pulls in cv2

        if self._predictor is None:
            self._predictor = Predictor()
        self._predictor_labels = set(self._predictor.labels)

        self._pipeline = HandPipeline(source=self.source)
        self._pipeline.started.wait(timeout=15)
        if self._pipeline.error:
            self._error = self._pipeline.error
            self._pipeline.close()
            self._pipeline = None
            return False

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        mark = time.monotonic()
        window_start = mark
        window_frames = 0
        while not self._stop.is_set():
            try:
                frame, landmarks, handedness = self._pipeline.read(timeout=2.0)
            except BaseException as exc:
                self._error = exc
                return
            if frame is None:
                # No frame this poll. Keep the last reading: a stutter is not
                # the same thing as the player removing their hand.
                continue

            now = time.monotonic()
            window_frames += 1
            if now - window_start >= 1.0:
                fps = window_frames / (now - window_start)
                with self._lock:
                    self._fps = fps
                window_start, window_frames = now, 0

            # Draw the hand skeleton on the preview copy, the same overlay
            # `signlang live` shows, so the player can see what the camera is
            # tracking. The frame is only used for the preview; classification
            # reads the landmarks, so drawing on it changes nothing else.
            if landmarks is not None:
                draw_landmarks(frame, landmarks, size=3)
            self._encode_preview(frame, now)

            if landmarks is None:
                with self._lock:
                    self._letter = None
                    self._conf = 0.0
                    self._since = 0.0
                    self._last_frame_at = now
                    self._frames += 1
                continue

            letter, conf = self._classify(landmarks, handedness)
            with self._lock:
                self._frames += 1
                self._hands_seen += 1
                self._last_frame_at = now
                if letter != self._letter:
                    self._letter = letter
                    self._conf = conf
                    self._since = now
                else:
                    # Smooth the confidence a little so a single jittery frame
                    # cannot swing the displayed probability.
                    self._conf = 0.7 * self._conf + 0.3 * conf
                    self._since = self._since or now

    def _encode_preview(self, frame, now):
        """Downscale and JPEG-encode a preview, at most ~12 times a second."""
        if now - self._preview_at < 0.08:
            return
        self._preview_at = now
        try:
            h, w = frame.shape[:2]
            if w > PREVIEW_WIDTH:
                scale = PREVIEW_WIDTH / w
                small = cv2.resize(
                    frame, (PREVIEW_WIDTH, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small = frame
            ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if not ok:
                return
            with self._lock:
                self._preview = buf.tobytes()
                self._preview_seq += 1
        except Exception:
            # A preview failure must never disturb detection.
            pass

    def _classify(self, landmarks, handedness):
        try:
            pts = normalize_hand(
                [(lm.x, lm.y, lm.z) for lm in landmarks], handedness
            )
            feats = landmark_features(pts)
            probs = self._predictor.probs(feats)
        except Exception:
            # A single bad frame must not kill the thread.
            return None, 0.0
        i = int(np.argmax(probs))
        return self._predictor.labels[i], float(probs[i])

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._pipeline is not None:
            try:
                self._pipeline.close()
            except Exception:
                pass
            self._pipeline = None