import time
from collections import deque

import numpy as np

from .config import (
    CONFIRM_THRESHOLD,
    DWELL_MS,
    REPEAT_COOLDOWN_MS,
    SMOOTH_MS,
    STABLE_MARGIN,
    SWIPE_MAX_MS,
    SWIPE_MIN_PX,
    WRIST,
    WORD_GLOSS,
)


def now_ms():
    return time.monotonic() * 1000.0


class DwellRecognizer:
    def __init__(self, predictor, smooth_ms=SMOOTH_MS, dwell_ms=DWELL_MS,
                 threshold=CONFIRM_THRESHOLD, margin=STABLE_MARGIN):
        self.predictor = predictor
        self.smooth_ms = smooth_ms
        self.dwell_ms = dwell_ms
        self.threshold = threshold
        self.margin = margin

        self.buffer = ""
        self._hist = deque()
        self._trail = deque()
        self._candidate = None
        self._stable_since = None
        self._dwell_start = None
        self._last_emit = 0.0
        self._last_gesture = 0.0
        self._just_emitted = None
        self._just_emitted_at = 0.0

    def reset(self):
        self.buffer = ""
        self._hist.clear()
        self._trail.clear()
        self._candidate = None
        self._stable_since = None
        self._dwell_start = None
        self._just_emitted = None

    def backspace(self):
        if self.buffer:
            self.buffer = self.buffer[:-1]

    def clear(self):
        self.buffer = ""
        self._hist.clear()
        self._candidate = None
        self._dwell_start = None
        self._just_emitted = None

    def set_dwell(self, ms):
        self.dwell_ms = int(max(200, min(2500, ms)))

    def set_threshold(self, t):
        self.threshold = float(max(0.3, min(0.99, t)))

    def _gesture(self, t, x, y):
        self._trail.append((t, x, y))
        while self._trail and t - self._trail[0][0] > SWIPE_MAX_MS * 1.6:
            self._trail.popleft()
        if len(self._trail) < 4 or t - self._last_gesture < 1100:
            return None
        t0, x0, y0 = self._trail[0]
        dt = t - t0
        if dt < 40 or dt > SWIPE_MAX_MS:
            return None
        dx, dy = x - x0, y - y0
        if abs(dx) < SWIPE_MIN_PX:
            return None
        if abs(dx) < abs(dy) * 1.8:
            return None
        self._last_gesture = t
        self._dwell_start = None
        self._candidate = None
        self._hist.clear()
        if dx > 0:
            return "space"
        return "backspace"

    def update(self, landmarks_norm, handedness, width, height):
        t = now_ms()
        if landmarks_norm is None:
            self._hist.clear()
            self._trail.clear()
            self._candidate = None
            self._dwell_start = None
            self._stable_since = None
            return {"gesture": None, "emitted": None, "gesture_ms": None}

        norm = self.predictor.probs_from_landmarks(landmarks_norm)
        wx = float(landmarks_norm[WRIST][0]) * width
        wy = float(landmarks_norm[WRIST][1]) * height

        gesture = self._gesture(t, wx, wy)
        if gesture == "space":
            self.buffer += " "
            return {"gesture": "space", "emitted": " ", "gesture_ms": t}
        if gesture == "backspace":
            self.backspace()
            return {"gesture": "backspace", "emitted": None, "gesture_ms": t}

        self._hist.append((t, norm))
        while self._hist and t - self._hist[0][0] > self.smooth_ms:
            self._hist.popleft()

        if len(self._hist) < 2:
            return self._status(None, 0.0, 0.0, t, None, None)

        stack = np.stack([h[1] for h in self._hist])
        mean = stack.mean(0)
        order = np.argsort(mean)[::-1]
        top_i = int(order[0])
        label = self.predictor.labels[top_i]
        conf = float(mean[top_i])
        runner = float(mean[int(order[1])])

        agreement = float((stack.argmax(1) == top_i).mean())
        stable = agreement >= 0.7
        if label != self._candidate:
            self._candidate = label
            self._stable_since = t
            self._dwell_start = None
        elif self._stable_since is None:
            self._stable_since = t

        moving = self._movement(200.0)
        confident = conf >= self.threshold and (conf - runner) >= self.margin * 0.5

        if not stable or moving > 26.0 or not confident:
            self._dwell_start = None
        elif self._dwell_start is None:
            self._dwell_start = t
        else:
            if t - self._last_emit >= REPEAT_COOLDOWN_MS and t - self._dwell_start >= self.dwell_ms:
                self._emit(label)
                self._last_emit = t
                self._dwell_start = t
                self._just_emitted = label
                self._just_emitted_at = t

        return self._status(label, conf, runner, t, mean, order, moving, agreement)

    def _movement(self, window_ms):
        if len(self._trail) < 2:
            return 0.0
        t, x, y = self._trail[-1]
        prior = [p for p in self._trail if t - p[0] <= window_ms]
        if len(prior) < 2:
            return 0.0
        x0, y0 = prior[0][1], prior[0][2]
        return float(np.hypot(x - x0, y - y0))

    def _emit(self, label):
        if label in WORD_GLOSS:
            self.buffer += WORD_GLOSS[label] + " "
        else:
            self.buffer += label

    def _status(self, label, conf, runner, t, mean, order, moving=0.0, agreement=1.0):
        dwell = 0.0
        if self._dwell_start is not None:
            dwell = min(1.0, (t - self._dwell_start) / max(self.dwell_ms, 1))
        top = []
        if mean is not None:
            for i in order[:5]:
                top.append(
                    {
                        "label": self.predictor.labels[int(i)],
                        "prob": float(mean[int(i)]),
                    }
                )
        just = (
            self._just_emitted
            if t - self._just_emitted_at < 500
            else None
        )
        return {
            "gesture": None,
            "emitted": just,
            "gesture_ms": self._last_gesture,
            "candidate": label,
            "confidence": conf,
            "margin": conf - runner,
            "dwell": dwell,
            "moving": moving,
            "agreement": agreement,
            "top": top,
            "buffer": self.buffer,
        }
