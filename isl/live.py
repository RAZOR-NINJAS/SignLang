"""Live Indian Sign Language (ISL) recognition HUD view.

Tracks both hands concurrently, performs live two-handed landmark feature
extraction, classifies signs via lightweight pure NumPy ISL_MLP, and confirms
signs using a dwell timer state machine with audio TTS read-aloud and live transcript.
"""

import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from signlang.hands import HandPipeline, draw_landmarks
from signlang.tts import get_tts

from .config import (
    CONFIRM_THRESHOLD,
    DOMINANT_HAND,
    DWELL_MS,
    EXPECTED_HANDS,
    REPEAT_COOLDOWN_MS,
    SMOOTH_MS,
    SPACE_COOLDOWN_MS,
    SPACE_DWELL_MS,
    STABLE_MARGIN,
    WEIGHTS_PATH,
)
from .features import extract_features
from .model import Predictor, load_numpy_predictor

WINDOW = "signlang - ISL live"

# ---- BGR Palette ----
def _bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


C_PANEL = _bgr("#080b10")
C_BORDER = _bgr("#1a1f2e")
C_BORDER_HI = _bgr("#2d3748")
C_TRACK = _bgr("#151a24")
C_INK = _bgr("#f1f5f9")
C_BRIGHT = _bgr("#ffffff")
C_DIM = _bgr("#8b9cb3")
C_MUTED = _bgr("#5a6a80")
C_GOOD = _bgr("#22c55e")
C_WARN = _bgr("#eab308")
C_ACCENT = _bgr("#0ea5e9")
C_ACCENT_HI = _bgr("#38bdf8")
C_GOLD = _bgr("#fbbf24")
C_ERROR = _bgr("#ef4444")

F_BIG = cv2.FONT_HERSHEY_DUPLEX
F_SMALL = cv2.FONT_HERSHEY_SIMPLEX
RECENT_MAX = 10
FLASH_MS = 420


class ISLDwellRecognizer:
    """State machine for ISL sign dwell confirmation and transcript accumulation."""

    def __init__(
        self,
        predictor: Optional[Predictor] = None,
        dwell_ms: int = DWELL_MS,
        space_dwell_ms: int = SPACE_DWELL_MS,
        threshold: float = CONFIRM_THRESHOLD,
    ):
        self.predictor = predictor
        self.dwell_ms = dwell_ms
        self.space_dwell_ms = space_dwell_ms
        self.threshold = threshold
        self.buffer = ""
        self.cur_cand: Optional[str] = None
        self.cur_cand_prob = 0.0
        self.cur_dwell_start: Optional[float] = None
        self.dwell_progress = 0.0
        self.last_emitted: Optional[str] = None
        self.cooldown_until = 0.0
        self.top_candidates: List[Dict[str, object]] = []
        self.smoothed_probs: Optional[np.ndarray] = None
        self.status_note = "Show hands to begin"

    def set_dwell(self, ms: int) -> None:
        self.dwell_ms = max(200, ms)

    def set_threshold(self, th: float) -> None:
        self.threshold = min(0.99, max(0.10, th))

    def clear(self) -> None:
        self.buffer = ""
        self.cur_cand = None
        self.cur_dwell_start = None
        self.dwell_progress = 0.0
        self.status_note = "Cleared"

    def backspace(self) -> None:
        if self.buffer:
            self.buffer = self.buffer[:-1]
            self.status_note = "Backspace"

    def update(
        self,
        landmarks: Optional[object],
        handedness: Optional[object],
        visible_hands: int,
    ) -> Dict[str, object]:
        now = time.monotonic()
        emitted: Optional[str] = None

        if self.predictor is None:
            return {
                "emitted": None,
                "cand": None,
                "prob": 0.0,
                "dwell": 0.0,
                "top": [],
                "note": "No model loaded",
                "buffer": self.buffer,
            }

        if landmarks is None or visible_hands == 0:
            self.cur_cand = None
            self.cur_dwell_start = None
            self.dwell_progress = 0.0
            self.status_note = "Show hands clearly"
            return {
                "emitted": None,
                "cand": None,
                "prob": 0.0,
                "dwell": 0.0,
                "top": [],
                "note": self.status_note,
                "buffer": self.buffer,
            }

        # Normalize landmarks and handedness into lists
        if isinstance(landmarks, (list, tuple)) and len(landmarks) > 0:
            if isinstance(landmarks[0], (list, tuple, np.ndarray)) and not hasattr(landmarks[0], "x"):
                lms_list = list(landmarks)
            else:
                lms_list = [landmarks]
        else:
            lms_list = [landmarks] if landmarks is not None else []

        hnd_list = (
            list(handedness)
            if isinstance(handedness, (list, tuple))
            else [handedness]
        )

        feats = extract_features(lms_list, hnd_list, dominant=DOMINANT_HAND)
        probs = self.predictor.probs(feats)

        # Exponential moving average smoothing
        if self.smoothed_probs is None or len(self.smoothed_probs) != len(probs):
            self.smoothed_probs = probs.copy()
        else:
            alpha = 0.40
            self.smoothed_probs = alpha * probs + (1.0 - alpha) * self.smoothed_probs

        top_indices = np.argsort(self.smoothed_probs)[::-1]
        self.top_candidates = [
            {"label": self.predictor.labels[i], "prob": float(self.smoothed_probs[i])}
            for i in top_indices[:5]
        ]

        cand = self.top_candidates[0]["label"]
        cand_prob = float(self.top_candidates[0]["prob"])
        margin = (
            (cand_prob - float(self.top_candidates[1]["prob"]))
            if len(self.top_candidates) > 1
            else cand_prob
        )
        self.cur_cand_prob = cand_prob

        expected = EXPECTED_HANDS.get(cand, 2)
        if visible_hands < expected:
            self.cur_cand = cand
            self.cur_dwell_start = None
            self.dwell_progress = 0.0
            self.status_note = (
                f"Show BOTH hands for '{cand}'" if expected == 2 else f"Show 1 hand for '{cand}'"
            )
            return {
                "emitted": None,
                "cand": cand,
                "prob": cand_prob,
                "dwell": 0.0,
                "top": self.top_candidates,
                "note": self.status_note,
                "buffer": self.buffer,
            }

        # Check confidence and margin
        if cand_prob < self.threshold or margin < STABLE_MARGIN:
            self.cur_cand = cand
            self.cur_dwell_start = None
            self.dwell_progress = 0.0
            self.status_note = "Holding steady..." if cand_prob >= 0.50 else "Not sure yet"
            return {
                "emitted": None,
                "cand": cand,
                "prob": cand_prob,
                "dwell": 0.0,
                "top": self.top_candidates,
                "note": self.status_note,
                "buffer": self.buffer,
            }

        # Check cooldown to avoid runaway repeating
        if now < self.cooldown_until and cand == self.last_emitted:
            self.cur_cand = cand
            self.cur_dwell_start = None
            self.dwell_progress = 0.0
            self.status_note = "Release sign to repeat"
            return {
                "emitted": None,
                "cand": cand,
                "prob": cand_prob,
                "dwell": 0.0,
                "top": self.top_candidates,
                "note": self.status_note,
                "buffer": self.buffer,
            }

        # Dwell progress accumulation
        target_dwell = self.space_dwell_ms if cand == "SPACE" else self.dwell_ms
        if self.cur_cand != cand or self.cur_dwell_start is None:
            self.cur_cand = cand
            self.cur_dwell_start = now
            self.dwell_progress = 0.0
            self.status_note = "Holding..."
        else:
            elapsed_ms = (now - self.cur_dwell_start) * 1000.0
            self.dwell_progress = min(1.0, elapsed_ms / target_dwell)
            if self.dwell_progress >= 1.0:
                # Sign confirmed
                emitted = " " if cand == "SPACE" else cand
                self.buffer += emitted
                self.last_emitted = cand
                self.cooldown_until = now + (
                    SPACE_COOLDOWN_MS if cand == "SPACE" else REPEAT_COOLDOWN_MS
                ) / 1000.0
                self.cur_dwell_start = None
                self.dwell_progress = 0.0
                self.status_note = "Confirmed!"

        return {
            "emitted": emitted,
            "cand": cand,
            "prob": cand_prob,
            "dwell": self.dwell_progress,
            "top": self.top_candidates,
            "note": self.status_note,
            "buffer": self.buffer,
        }


def _draw_hud(
    frame: np.ndarray,
    st: Dict[str, object],
    left_ok: bool,
    right_ok: bool,
    fps: float,
    has_model: bool,
    recent: List[str],
    flash_cand: Optional[str],
    flash_age_ms: Optional[float],
) -> None:
    """Draw high-contrast translucent ISL HUD overlay on frame."""
    h, w = frame.shape[:2]
    cand = st.get("cand")
    prob = float(st.get("prob", 0.0))
    dwell = float(st.get("dwell", 0.0))
    top = st.get("top", [])
    note = str(st.get("note", ""))
    buf = str(st.get("buffer", ""))

    # Top header bar
    header_h = 44
    cv2.rectangle(frame, (0, 0), (w, header_h), C_PANEL, -1)
    cv2.line(frame, (0, header_h), (w, header_h), C_BORDER_HI, 1, cv2.LINE_AA)

    # Wordmark and badge
    cv2.putText(frame, "SIGNLANG", (16, 28), F_BIG, 0.65, C_BRIGHT, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (140, 10), (280, 34), C_BORDER, -1)
    cv2.rectangle(frame, (140, 10), (280, 34), C_ACCENT_HI, 1, cv2.LINE_AA)
    cv2.putText(frame, "ISL MODE (ISLRTC)", (146, 27), F_SMALL, 0.42, C_INK, 1, cv2.LINE_AA)

    # Hand presence badges
    lx = 300
    cv2.rectangle(frame, (lx, 10), (lx + 82, 34), C_BORDER, -1)
    lcol = C_GOOD if left_ok else C_MUTED
    cv2.putText(frame, f"L: {'OK' if left_ok else '--'}", (lx + 10, 27), F_SMALL, 0.45, lcol, 1, cv2.LINE_AA)

    rx = lx + 92
    cv2.rectangle(frame, (rx, 10), (rx + 82, 34), C_BORDER, -1)
    rcol = C_GOOD if right_ok else C_MUTED
    cv2.putText(frame, f"R: {'OK' if right_ok else '--'}", (rx + 10, 27), F_SMALL, 0.45, rcol, 1, cv2.LINE_AA)

    # FPS counter
    fps_txt = f"{fps:4.1f} FPS"
    cv2.putText(frame, fps_txt, (w - 100, 28), F_SMALL, 0.45, C_DIM, 1, cv2.LINE_AA)

    # Candidate Sign Card (top-left)
    card_w, card_h = 160, 180
    card_x, card_y = 20, 60

    if not has_model:
        # Warning card if no model exists
        warn_w, warn_h = 360, 110
        cv2.rectangle(frame, (card_x, card_y), (card_x + warn_w, card_y + warn_h), C_PANEL, -1)
        cv2.rectangle(frame, (card_x, card_y), (card_x + warn_w, card_y + warn_h), C_WARN, 2, cv2.LINE_AA)
        cv2.putText(frame, "NO TRAINED ISL MODEL FOUND", (card_x + 14, card_y + 32), F_BIG, 0.58, C_WARN, 1, cv2.LINE_AA)
        cv2.putText(frame, "1. Run: signlang isl collect", (card_x + 14, card_y + 64), F_SMALL, 0.50, C_INK, 1, cv2.LINE_AA)
        cv2.putText(frame, "2. Run: signlang isl train", (card_x + 14, card_y + 92), F_SMALL, 0.50, C_INK, 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), C_PANEL, -1)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), C_BORDER, 1, cv2.LINE_AA)

        disp_sign = ("SPACE" if cand == "SPACE" else (cand or "--"))
        col_sign = C_BRIGHT if dwell > 0.0 else C_DIM
        font_scale = 1.6 if len(disp_sign) <= 2 else 0.8
        (tw, th), _ = cv2.getTextSize(disp_sign, F_BIG, font_scale, 2)
        cv2.putText(frame, disp_sign, (card_x + (card_w - tw) // 2, card_y + 70), F_BIG, font_scale, col_sign, 2, cv2.LINE_AA)

        # Confidence percentage
        conf_str = f"{prob * 100:.0f}%" if cand else "--%"
        (cw, _), _ = cv2.getTextSize(conf_str, F_SMALL, 0.55, 1)
        cv2.putText(frame, conf_str, (card_x + (card_w - cw) // 2, card_y + 105), F_SMALL, 0.55, C_DIM, 1, cv2.LINE_AA)

        # Note / hint
        (nw, _), _ = cv2.getTextSize(note, F_SMALL, 0.40, 1)
        cv2.putText(frame, note, (max(card_x + 8, card_x + (card_w - nw) // 2), card_y + 130), F_SMALL, 0.40, C_GOLD, 1, cv2.LINE_AA)

        # Progress bar
        bar_x0, bar_x1 = card_x + 12, card_x + card_w - 12
        bar_y, bar_h = card_y + card_h - 18, 8
        cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), C_TRACK, -1)
        if dwell > 0.0:
            fill_w = int((bar_x1 - bar_x0) * min(1.0, dwell))
            cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + fill_w, bar_y + bar_h), C_GOOD, -1)
            cv2.line(frame, (bar_x0 + fill_w - 1, bar_y), (bar_x0 + fill_w - 1, bar_y + bar_h), C_BRIGHT, 1)

        # Runners-up panel
        if top and len(top) > 1:
            alt_x, alt_y = card_x + card_w + 10, card_y
            alt_w, alt_h = 130, card_h
            cv2.rectangle(frame, (alt_x, alt_y), (alt_x + alt_w, alt_y + alt_h), C_PANEL, -1)
            cv2.rectangle(frame, (alt_x, alt_y), (alt_x + alt_w, alt_y + alt_h), C_BORDER, 1, cv2.LINE_AA)

            for i, item in enumerate(top[1:5]):
                lbl = str(item["label"])
                p = float(item["prob"])
                ry = alt_y + 28 + i * 34
                cv2.putText(frame, lbl[:5], (alt_x + 10, ry), F_SMALL, 0.45, C_DIM, 1, cv2.LINE_AA)
                bx0, bx1 = alt_x + 55, alt_x + alt_w - 10
                by0, by1 = ry - 10, ry - 4
                cv2.rectangle(frame, (bx0, by0), (bx1, by1), C_TRACK, -1)
                bw = bx0 + int((bx1 - bx0) * p)
                cv2.rectangle(frame, (bx0, by0), (bw, by1), C_ACCENT, -1)

    # Confirmation Flash Ring & Letter
    if flash_cand and flash_age_ms is not None and flash_age_ms < FLASH_MS:
        prog = flash_age_ms / FLASH_MS
        fade = (1.0 - prog) ** 2
        cx, cy = w // 2, h // 2
        rad = int(35 + 50 * prog)
        ring_col = tuple(int(c * fade) for c in C_GOOD)
        cv2.circle(frame, (cx, cy), rad, ring_col, max(1, int(4 * fade)), cv2.LINE_AA)
        fl_sign = "SPACE" if flash_cand == " " else flash_cand
        fs_scale = 2.2
        (fw, fh), _ = cv2.getTextSize(fl_sign, F_BIG, fs_scale, 3)
        cv2.putText(frame, fl_sign, (cx - fw // 2, cy + fh // 2), F_BIG, fs_scale, ring_col, 3, cv2.LINE_AA)

    # Bottom Panel: Recent strip, Transcript, Status line
    bot_h = 110
    bot_y = h - bot_h
    cv2.rectangle(frame, (0, bot_y), (w, h), C_PANEL, -1)
    cv2.line(frame, (0, bot_y), (w, bot_y), C_BORDER_HI, 1, cv2.LINE_AA)

    # Recent signs strip
    rec_y = bot_y + 24
    cv2.putText(frame, "RECENT:", (16, rec_y), F_SMALL, 0.42, C_MUTED, 1, cv2.LINE_AA)
    rx = 85
    for i, s in enumerate(recent[:RECENT_MAX]):
        badge_col = C_GOOD if i == 0 else C_DIM
        cv2.putText(frame, s, (rx, rec_y), F_SMALL, 0.50, badge_col, 1, cv2.LINE_AA)
        rx += 26

    # Transcript line
    trans_y = bot_y + 58
    cv2.putText(frame, "TRANSCRIPT:", (16, trans_y), F_SMALL, 0.46, C_MUTED, 1, cv2.LINE_AA)
    disp_buf = (buf[-45:] if len(buf) > 45 else buf) + "|"
    cv2.putText(frame, disp_buf, (130, trans_y), F_BIG, 0.65, C_BRIGHT, 2, cv2.LINE_AA)

    # Help keybindings status
    help_y = bot_y + 92
    help_str = "Q quit · C clear · R speak transcript · Backspace delete · F toggle fullscreen"
    cv2.putText(frame, help_str, (16, help_y), F_SMALL, 0.40, C_MUTED, 1, cv2.LINE_AA)


def _selftest() -> int:
    """Headless selftest for ISL dwell recognition state machine and HUD drawing."""
    from .collect import _synthetic_hand

    # 1. Test recognizer without model
    rec_nomodel = ISLDwellRecognizer(predictor=None)
    st = rec_nomodel.update(None, None, 0)
    assert st["emitted"] is None

    # 2. Mock predictor with labels
    class MockPredictor:
        def __init__(self):
            self.labels = ["A", "B", "C", "SPACE"]

        def probs(self, feats):
            # Deterministic: class 0 ("A") gets high probability
            return np.array([0.92, 0.04, 0.02, 0.02], dtype=np.float32)

    rec = ISLDwellRecognizer(predictor=MockPredictor(), dwell_ms=200, threshold=0.70)

    # Feed synthetic 2-hand frames
    h_left = _synthetic_hand(0.0)
    h_right = _synthetic_hand(1.0)
    lms = [h_left, h_right]
    hnds = ["Left", "Right"]

    # Initial frame
    st = rec.update(lms, hnds, visible_hands=2)
    assert st["cand"] == "A"
    assert st["emitted"] is None

    # Sleep slightly past dwell_ms to trigger confirmation
    time.sleep(0.25)
    st2 = rec.update(lms, hnds, visible_hands=2)
    assert st2["emitted"] == "A"
    assert rec.buffer == "A"

    # Test backspace and clear
    rec.backspace()
    assert rec.buffer == ""
    rec.buffer = "HELLO"
    rec.clear()
    assert rec.buffer == ""

    # Test drawing on a test canvas
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    _draw_hud(
        canvas,
        st2,
        left_ok=True,
        right_ok=True,
        fps=30.0,
        has_model=True,
        recent=["A"],
        flash_cand="A",
        flash_age_ms=100.0,
    )
    assert canvas.shape == (480, 640, 3)

    print("ISL LIVE SELFTEST OK: state machine, dwell timing, and HUD overlay verified.")
    return 0


def main(argv=None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in args_list:
        return _selftest()

    parser = argparse.ArgumentParser(prog="signlang isl live")
    parser.add_argument("--dwell", type=int, default=DWELL_MS, help=f"Dwell time in ms (default: {DWELL_MS})")
    parser.add_argument("--threshold", type=float, default=CONFIRM_THRESHOLD, help=f"Confidence threshold (default: {CONFIRM_THRESHOLD})")
    parser.add_argument("--windowed", action="store_true", help="Launch in windowed mode instead of fullscreen")
    args = parser.parse_args(args_list)

    # Check for trained model weights
    has_model = WEIGHTS_PATH.exists()
    predictor = None
    if has_model:
        try:
            predictor = Predictor(WEIGHTS_PATH)
            print(f"Loaded ISL model: {len(predictor.labels)} signs ({' '.join(predictor.labels)})")
        except Exception as exc:
            print(f"Warning loading model ({exc}); running HUD without predictions.")
            has_model = False
    else:
        print("=" * 64)
        print("  NOTICE: No trained ISL model found at isl/models/isl_mlp.pt.")
        print("  Running camera and dual-hand skeleton tracking.")
        print("  To train a model:")
        print("    1. Record signs:  signlang isl collect")
        print("    2. Train model:   signlang isl train")
        print("=" * 64)

    rec = ISLDwellRecognizer(predictor=predictor, dwell_ms=args.dwell, threshold=args.threshold)
    tts = get_tts()
    if tts.enabled:
        print("  TTS: enabled (voice: en_US-lessac-medium)")
    else:
        print("  TTS: disabled")

    # Start 2-hand pipeline
    pipe = HandPipeline(num_hands=2)
    pipe.started.wait(timeout=10)
    if pipe.error is not None:
        print(f"Camera error: {pipe.error}", file=sys.stderr)
        return 1

    fullscreen = not args.windowed
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WINDOW, 1280, 720)
    except cv2.error:
        pass

    fps_t = time.monotonic()
    fps_n = 0
    fps_v = 0.0
    recent: List[str] = []
    flash_cand: Optional[str] = None
    flash_t: float = 0.0

    try:
        while True:
            frame, landmarks, handedness = pipe.read(timeout=3.0)
            if frame is None:
                continue

            fps_n += 1
            now = time.monotonic()
            if now - fps_t > 0.5:
                fps_v = fps_n / (now - fps_t)
                fps_t, fps_n = now, 0

            h, w = frame.shape[:2]

            # Detect hand presence and handedness
            left_ok = False
            right_ok = False
            visible_hands = 0

            if landmarks is not None:
                if isinstance(landmarks, list) and len(landmarks) > 0 and isinstance(landmarks[0], list):
                    visible_hands = len(landmarks)
                    draw_landmarks(frame, landmarks)
                    h_list = handedness if isinstance(handedness, list) else [handedness]
                    for h_label in h_list:
                        if str(h_label).lower() == "left":
                            left_ok = True
                        elif str(h_label).lower() == "right":
                            right_ok = True
                else:
                    visible_hands = 1
                    draw_landmarks(frame, landmarks)
                    h_label = str(handedness)
                    if h_label.lower() == "left":
                        left_ok = True
                    elif h_label.lower() == "right":
                        right_ok = True

                # Draw wrist reticle and fingertip halos on visible hands
                lms_iter = (
                    landmarks
                    if (isinstance(landmarks, list) and len(landmarks) > 0 and isinstance(landmarks[0], list))
                    else [landmarks]
                )
                for hand_pts in lms_iter:
                    wx, wy = int(hand_pts[0].x * w), int(hand_pts[0].y * h)
                    cv2.circle(frame, (wx, wy), 4, C_ACCENT_HI, -1, cv2.LINE_AA)
                    cv2.circle(frame, (wx, wy), 8, C_ACCENT, 1, cv2.LINE_AA)
                    for tip in (4, 8, 12, 16, 20):
                        tx, ty = int(hand_pts[tip].x * w), int(hand_pts[tip].y * h)
                        cv2.circle(frame, (tx, ty), 4, C_GOOD, 1, cv2.LINE_AA)

            st = rec.update(landmarks, handedness, visible_hands)
            emitted = st.get("emitted")
            if emitted:
                flash_cand = emitted
                flash_t = time.monotonic()
                recent.insert(0, "_" if emitted == " " else emitted)
                del recent[RECENT_MAX:]
                if tts.enabled:
                    tts.say_letter(emitted)

            flash_age_ms = ((time.monotonic() - flash_t) * 1000.0) if flash_cand else None
            _draw_hud(
                frame,
                st,
                left_ok=left_ok,
                right_ok=right_ok,
                fps=fps_v,
                has_model=has_model,
                recent=recent,
                flash_cand=flash_cand,
                flash_age_ms=flash_age_ms,
            )

            cv2.imshow(WINDOW, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k in (ord("c"), ord("C")):
                rec.clear()
                recent.clear()
            elif k in (ord("r"), ord("R")):
                if tts.enabled:
                    tts.read_transcript(rec.buffer)
            elif k == 8:
                rec.backspace()
            elif k in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                try:
                    if fullscreen:
                        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(WINDOW, 1280, 720)
                except cv2.error:
                    pass
    finally:
        pipe.close()
        tts.close()
        cv2.destroyAllWindows()

    print(f"\nFinal Transcript: {rec.buffer!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
