import math
import time
from collections import deque

import numpy as np

from .config import (
    CONFIRM_THRESHOLD,
    DWELL_MS,
    REPEAT_COOLDOWN_MS,
    SMOOTH_MS,
    SPACE_COOLDOWN_MS,
    SPACE_DWELL_MS,
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
                 threshold=CONFIRM_THRESHOLD, margin=STABLE_MARGIN,
                 space_dwell_ms=SPACE_DWELL_MS,
                 repeat_cooldown_ms=REPEAT_COOLDOWN_MS,
                 space_cooldown_ms=SPACE_COOLDOWN_MS):
        self.predictor = predictor
        self.smooth_ms = smooth_ms
        self.dwell_ms = dwell_ms
        self.threshold = threshold
        self.margin = margin
        self.space_dwell_ms = space_dwell_ms
        self.repeat_cooldown_ms = repeat_cooldown_ms
        self.space_cooldown_ms = space_cooldown_ms

        self.buffer = ""
        self._hist = deque()
        self._trail = deque(maxlen=128)
        self._candidate = None
        self._stable_since = None
        self._dwell_start = None
        self._last_emit = 0.0
        self._last_gesture = 0.0
        self._just_emitted = None
        self._just_emitted_at = 0.0
        self._hold_emitted = False
        self._unstable_since = None
        self._none_since = None
        self._last_emit_time = {}

    def reset(self):
        self.buffer = ""
        self._hist.clear()
        self._trail.clear()
        self._candidate = None
        self._stable_since = None
        self._dwell_start = None
        self._just_emitted = None
        self._hold_emitted = False
        self._unstable_since = None
        self._none_since = None
        self._last_emit_time.clear()

    def backspace(self):
        if self.buffer:
            self.buffer = self.buffer[:-1]

    def clear(self):
        self.buffer = ""
        self._hist.clear()
        self._candidate = None
        self._dwell_start = None
        self._just_emitted = None
        self._hold_emitted = False
        self._unstable_since = None
        self._none_since = None
        self._last_emit_time.clear()

    def set_dwell(self, ms):
        self.dwell_ms = int(max(200, min(2500, ms)))

    def set_space_dwell(self, ms):
        self.space_dwell_ms = int(max(200, min(2500, ms)))

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

    def update(self, landmarks_norm, handedness, width, height, raw_landmarks=None, t_ms=None):
        t = now_ms() if t_ms is None else float(t_ms)
        if landmarks_norm is None:
            if self._none_since is None:
                self._none_since = t
            # Genuine release when absent for >= 180ms
            if t - self._none_since >= 180.0:
                self._hist.clear()
                self._trail.clear()
                self._candidate = None
                self._dwell_start = None
                self._stable_since = None
                self._hold_emitted = False
                self._unstable_since = None
            else:
                self._dwell_start = None
            return {"gesture": None, "emitted": None, "gesture_ms": None}

        if self._none_since is not None:
            if t - self._none_since >= 180.0:
                self._hist.clear()
                self._trail.clear()
                self._candidate = None
                self._dwell_start = None
                self._stable_since = None
                self._hold_emitted = False
                self._unstable_since = None
            self._none_since = None

        norm = self.predictor.probs_from_landmarks(landmarks_norm)
        if raw_landmarks is not None:
            wx = float(raw_landmarks[WRIST][0]) * width
            wy = float(raw_landmarks[WRIST][1]) * height
        else:
            wx = float(landmarks_norm[WRIST][0]) * width
            wy = float(landmarks_norm[WRIST][1]) * height

        gesture = self._gesture(t, wx, wy)
        if gesture == "space":
            self.buffer += " "
            self._last_emit = t
            self._last_emit_time["SPACE"] = t
            self._hold_emitted = True
            return {"gesture": "space", "emitted": " ", "gesture_ms": t}
        if gesture == "backspace":
            self.backspace()
            self._last_emit = t
            self._hold_emitted = True
            return {"gesture": "backspace", "emitted": None, "gesture_ms": t}

        self._hist.append((t, norm))
        while self._hist and t - self._hist[0][0] > self.smooth_ms:
            self._hist.popleft()

        if len(self._hist) < 2:
            return self._status(None, 0.0, 0.0, t, None, None)

        stack = np.array([h[1] for h in self._hist], dtype=np.float32)
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
            self._hold_emitted = False
            self._unstable_since = None
        elif self._stable_since is None:
            self._stable_since = t

        moving = self._movement(200.0)
        confident = conf >= self.threshold and (conf - runner) >= self.margin * 0.5
        is_stable_hold = stable and (moving <= 26.0) and confident

        if not is_stable_hold:
            self._dwell_start = None
            if self._unstable_since is None:
                self._unstable_since = t
            if moving > 40.0 or (t - self._unstable_since >= 150.0):
                self._hold_emitted = False
        else:
            self._unstable_since = None
            if self._dwell_start is None:
                self._dwell_start = t
            elif not self._hold_emitted:
                target_dwell = self.space_dwell_ms if label == "SPACE" else self.dwell_ms
                cooldown = self.space_cooldown_ms if label == "SPACE" else self.repeat_cooldown_ms
                last_label_emit = self._last_emit_time.get(label, 0.0)

                if (t - self._dwell_start >= target_dwell) and (t - last_label_emit >= cooldown):
                    self._emit(label)
                    self._last_emit = t
                    self._last_emit_time[label] = t
                    self._hold_emitted = True
                    self._just_emitted = " " if label == "SPACE" else label
                    self._just_emitted_at = t

        return self._status(label, conf, runner, t, mean, order, moving, agreement)

    def _movement(self, window_ms):
        if len(self._trail) < 2:
            return 0.0
        t, x, y = self._trail[-1]
        # Scan from the front to find the oldest entry within the window.
        # O(relevant) instead of building a new list every frame.
        x0, y0 = x, y
        for i in range(len(self._trail)):
            if t - self._trail[i][0] <= window_ms:
                x0, y0 = self._trail[i][1], self._trail[i][2]
                break
        return math.hypot(x - x0, y - y0)

    def _emit(self, label):
        if label == "SPACE":
            self.buffer += " "
        elif label in WORD_GLOSS:
            self.buffer += WORD_GLOSS[label] + " "
        else:
            self.buffer += label

    def _status(self, label, conf, runner, t, mean, order, moving=0.0, agreement=1.0):
        target_dwell = self.space_dwell_ms if label == "SPACE" else self.dwell_ms
        if self._hold_emitted:
            dwell = 1.0
        elif self._dwell_start is not None:
            dwell = min(1.0, (t - self._dwell_start) / max(target_dwell, 1))
        else:
            dwell = 0.0
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
