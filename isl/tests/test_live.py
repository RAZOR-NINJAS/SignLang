"""Unit tests for ISL live recognition and HUD state machine."""

import time
import numpy as np
import pytest

from isl.collect import _synthetic_hand
from isl.live import ISLDwellRecognizer, _draw_hud, _selftest


class DummyPredictor:
    def __init__(self, labels=None):
        self.labels = labels or ["A", "C", "SPACE"]

    def probs(self, feats):
        # A has high prob
        return np.array([0.90, 0.05, 0.05], dtype=np.float32)


class DummySpacePredictor:
    def __init__(self):
        self.labels = ["A", "C", "SPACE"]

    def probs(self, feats):
        # SPACE has high prob
        return np.array([0.05, 0.05, 0.90], dtype=np.float32)


def test_dwell_recognizer_no_model():
    rec = ISLDwellRecognizer(predictor=None)
    st = rec.update(None, None, 0)
    assert st["emitted"] is None
    assert st["note"] == "No model loaded"


def test_dwell_recognizer_no_hands():
    rec = ISLDwellRecognizer(predictor=DummyPredictor())
    st = rec.update(None, None, 0)
    assert st["emitted"] is None
    assert st["cand"] is None
    assert "Show hands" in st["note"]


def test_dwell_recognizer_hand_count_enforcement():
    # 'A' requires 2 hands in ISL
    rec = ISLDwellRecognizer(predictor=DummyPredictor())
    h1 = _synthetic_hand(0.0)
    st = rec.update([h1], ["Right"], visible_hands=1)
    assert st["cand"] == "A"
    assert st["emitted"] is None
    assert st["dwell"] == 0.0
    assert "Show BOTH hands" in st["note"]


def test_dwell_recognizer_confirmation_and_transcript():
    rec = ISLDwellRecognizer(predictor=DummyPredictor(), dwell_ms=100)
    h_l = _synthetic_hand(0.0)
    h_r = _synthetic_hand(1.0)
    lms = [h_l, h_r]
    hnds = ["Left", "Right"]

    st1 = rec.update(lms, hnds, visible_hands=2)
    assert st1["cand"] == "A"
    assert st1["emitted"] is None

    time.sleep(0.15)
    st2 = rec.update(lms, hnds, visible_hands=2)
    assert st2["emitted"] == "A"
    assert rec.buffer == "A"

    # Immediate next frame should not repeat due to cooldown
    st3 = rec.update(lms, hnds, visible_hands=2)
    assert st3["emitted"] is None
    assert rec.buffer == "A"
    assert "Release sign to repeat" in st3["note"]


def test_dwell_recognizer_space_gesture():
    # SPACE is a 1-handed sign
    rec = ISLDwellRecognizer(predictor=DummySpacePredictor(), space_dwell_ms=100)
    h1 = _synthetic_hand(2.0)
    st1 = rec.update([h1], ["Right"], visible_hands=1)
    assert st1["cand"] == "SPACE"

    time.sleep(0.15)
    st2 = rec.update([h1], ["Right"], visible_hands=1)
    assert st2["emitted"] == " "
    assert rec.buffer == " "


def test_dwell_recognizer_editing():
    rec = ISLDwellRecognizer(predictor=DummyPredictor())
    rec.buffer = "HELLO"
    rec.backspace()
    assert rec.buffer == "HELL"
    rec.clear()
    assert rec.buffer == ""


def test_draw_hud_without_model():
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    st = {"cand": None, "prob": 0.0, "dwell": 0.0, "top": [], "note": "No model loaded", "buffer": ""}
    _draw_hud(
        canvas,
        st,
        left_ok=False,
        right_ok=False,
        fps=25.0,
        has_model=False,
        recent=[],
        flash_cand=None,
        flash_age_ms=None,
    )
    assert canvas.shape == (480, 640, 3)


def test_live_selftest():
    assert _selftest() == 0
