"""Unit tests for debounce, hold logic, cooldown, and simulated sign sequences.

Covers:
- Hold duration requirement (SPACE must be held ~0.5-1s, letters 1s)
- Trigger-once-per-hold (no repeated characters/spaces while hand stays up)
- Cooldown requirement between consecutive identical gestures
- Debouncing momentary sensor jitter
- Release detection
- Simulated-input test: "H I [SPACE] H I" -> "HI HI"

Run with:
    .venv/bin/python -m unittest src/signlang/test_debounce.py
    .venv/bin/python -m signlang.test_debounce
"""

import unittest
import numpy as np

from signlang.config import (
    CONFIRM_THRESHOLD,
    DWELL_MS,
    REPEAT_COOLDOWN_MS,
    SPACE_COOLDOWN_MS,
    SPACE_DWELL_MS,
    WRIST,
)
from signlang.engine import DwellRecognizer


class MockPredictor:
    """Predictor stub returning controllable probability distributions."""

    def __init__(self, labels=None):
        self.labels = labels or ["H", "I", "SPACE", "A", "B", "E"]
        self.active_label = "H"
        self.confidence = 0.95

    def set_prediction(self, label, confidence=0.95):
        if label not in self.labels:
            self.labels.append(label)
        self.active_label = label
        self.confidence = confidence

    def probs_from_landmarks(self, landmarks_norm):
        n = len(self.labels)
        p = np.full(n, (1.0 - self.confidence) / max(n - 1, 1), dtype=np.float32)
        idx = self.labels.index(self.active_label)
        p[idx] = self.confidence
        return p

    def probs(self, feats):
        return self.probs_from_landmarks(None)


def make_dummy_landmarks(wrist_x=0.5, wrist_y=0.5):
    """Generate minimal 21-landmark array for testing."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[WRIST] = [wrist_x, wrist_y, 0.0]
    pts[9] = [wrist_x, wrist_y - 0.2, 0.0]  # Middle MCP
    return pts


class TestDebounceAndCooldown(unittest.TestCase):
    def setUp(self):
        self.predictor = MockPredictor()
        self.rec = DwellRecognizer(
            self.predictor,
            smooth_ms=100,
            dwell_ms=1000,
            space_dwell_ms=800,
            repeat_cooldown_ms=900,
            space_cooldown_ms=900,
        )
        self.lms = make_dummy_landmarks()
        self.sim_time = 1000.0

    def feed_frames(self, duration_ms, fps=30, label=None, conf=0.95):
        """Simulate feeding consecutive camera frames over duration_ms with simulated clock."""
        if label is not None:
            self.predictor.set_prediction(label, conf)
        dt = 1000.0 / fps
        n_frames = max(1, int(round(duration_ms / dt)))
        st = None
        for _ in range(n_frames):
            self.sim_time += dt
            st = self.rec.update(self.lms, "Right", 640, 480, t_ms=self.sim_time)
        return st

    def test_hold_threshold_letter(self):
        """Letter must be held for dwell_ms (1000ms) before emitting."""
        self.predictor.set_prediction("H", 0.95)
        # Feed for 500ms (< 1000ms dwell)
        self.feed_frames(500)
        self.assertEqual(self.rec.buffer, "", "Should not emit before dwell duration")

        # Feed remaining 600ms (total 1100ms > 1000ms)
        self.feed_frames(600)
        self.assertEqual(self.rec.buffer, "H", "Should emit after dwell duration is reached")

    def test_no_repeated_emissions_while_held(self):
        """Holding a sign continuously must trigger exactly once, not repeat."""
        self.predictor.set_prediction("H", 0.95)
        # Hold for 1100ms -> emits first 'H'
        self.feed_frames(1100)
        self.assertEqual(self.rec.buffer, "H")

        # Continue holding for another 3500ms (3.5 dwell periods)
        self.feed_frames(3500)
        self.assertEqual(
            self.rec.buffer, "H",
            "Must NOT repeat emissions while hand stays up in the same sign"
        )

    def test_space_hold_and_trigger_once(self):
        """SPACE must be held for ~0.8s and trigger exactly once while held."""
        self.predictor.set_prediction("SPACE", 0.95)
        # Feed for 400ms (< 800ms)
        self.feed_frames(400)
        self.assertEqual(self.rec.buffer, "")

        # Reach 900ms (> 800ms space_dwell_ms)
        self.feed_frames(500)
        self.assertEqual(self.rec.buffer, " ", "SPACE must emit exactly one space character")

        # Continue holding SPACE for 3000ms
        self.feed_frames(3000)
        self.assertEqual(
            self.rec.buffer, " ",
            "Must NOT emit repeated spaces while SPACE gesture is held up"
        )

    def test_space_cooldown_and_retrigger(self):
        """A second SPACE requires release and cooldown before triggering."""
        # 1st SPACE
        self.predictor.set_prediction("SPACE", 0.95)
        self.feed_frames(1000)
        self.assertEqual(self.rec.buffer, " ")

        # Drop hand / break hold for 200ms
        self.sim_time += 200.0
        self.rec.update(None, None, 640, 480, t_ms=self.sim_time)

        # Advance time so cooldown (900ms) has passed since 1st emit
        self.sim_time += 800.0

        # Now re-hold SPACE for 900ms
        self.feed_frames(900, label="SPACE")
        self.assertEqual(self.rec.buffer, "  ", "Second SPACE should emit after release and cooldown")

        # And again, no repeated spaces while held
        self.feed_frames(2000)
        self.assertEqual(self.rec.buffer, "  ")

    def test_sensor_jitter_does_not_cause_duplicate_emission(self):
        """A single glitchy frame should not reset the hold and trigger duplicate."""
        self.predictor.set_prediction("H", 0.95)
        self.feed_frames(1100)
        self.assertEqual(self.rec.buffer, "H")

        # 1 glitch frame with low confidence
        self.predictor.set_prediction("H", 0.30)
        self.sim_time += 33.3
        self.rec.update(self.lms, "Right", 640, 480, t_ms=self.sim_time)

        # Confidence immediately returns
        self.predictor.set_prediction("H", 0.95)
        self.feed_frames(1100)
        self.assertEqual(
            self.rec.buffer, "H",
            "Glitch frame must not cause a spurious repeat emission"
        )

    def test_missing_landmark_dropout_does_not_cause_duplicate_emission(self):
        """A single dropped frame (landmarks=None) while holding must NOT reset hold and duplicate."""
        self.predictor.set_prediction("H", 0.95)
        self.feed_frames(1100)
        self.assertEqual(self.rec.buffer, "H")

        # 1-2 frames of MediaPipe dropping the hand (e.g. 60ms total)
        self.sim_time += 33.3
        self.rec.update(None, None, 640, 480, t_ms=self.sim_time)
        self.sim_time += 33.3
        self.rec.update(None, None, 640, 480, t_ms=self.sim_time)

        # Hand returns in the same position
        self.feed_frames(1500)
        self.assertEqual(
            self.rec.buffer, "H",
            "Brief landmark tracking loss must not cause spurious re-trigger while hand stays up"
        )

    def test_long_hold_and_high_frame_count_does_not_crash(self):
        """Feeding 600+ frames continuously must not raise IndexError or overflow."""
        self.predictor.set_prediction("A", 0.95)
        self.feed_frames(20000, fps=30)
        self.assertEqual(self.rec.buffer, "A")

    def test_clear_resets_transcript_and_emit_tracking(self):
        """clear() must reset transcript buffer, hold state, and cooldowns."""
        self.predictor.set_prediction("H", 0.95)
        self.feed_frames(1100)
        self.assertEqual(self.rec.buffer, "H")
        self.rec.clear()
        self.assertEqual(self.rec.buffer, "")
        self.assertFalse(self.rec._hold_emitted)
        self.assertEqual(len(self.rec._last_emit_time), 0)

    def test_simulated_input_hi_space_hi(self):
        """Simulated-input test: 'H I [SPACE] H I' -> 'HI HI'."""
        rec = self.rec

        # 1. Sign 'H'
        self.feed_frames(1200, label="H")
        self.assertEqual(rec.buffer, "H", "Step 1: 'H' should emit")
        # Hold 'H' longer - verify no repeat
        self.feed_frames(500)
        self.assertEqual(rec.buffer, "H")

        # 2. Transition to 'I'
        self.feed_frames(1200, label="I")
        self.assertEqual(rec.buffer, "HI", "Step 2: 'I' should emit -> 'HI'")
        # Hold 'I' longer - verify no repeat
        self.feed_frames(500)
        self.assertEqual(rec.buffer, "HI")

        # 3. Transition to 'SPACE'
        self.feed_frames(1000, label="SPACE")
        self.assertEqual(rec.buffer, "HI ", "Step 3: 'SPACE' should emit -> 'HI '")
        # Hold 'SPACE' longer - verify no repeat
        self.feed_frames(500)
        self.assertEqual(rec.buffer, "HI ")

        # 4. Transition to 'H'
        self.feed_frames(1200, label="H")
        self.assertEqual(rec.buffer, "HI H", "Step 4: second 'H' should emit -> 'HI H'")
        # Hold 'H' longer - verify no repeat
        self.feed_frames(500)
        self.assertEqual(rec.buffer, "HI H")

        # 5. Transition to 'I'
        self.feed_frames(1200, label="I")
        self.assertEqual(rec.buffer, "HI HI", "Step 5: second 'I' should emit -> 'HI HI'")
        # Hold 'I' longer - verify no repeat
        self.feed_frames(500)
        self.assertEqual(rec.buffer, "HI HI", "Final transcript must be exactly 'HI HI'")


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDebounceAndCooldown)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
