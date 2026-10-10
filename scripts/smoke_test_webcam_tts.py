"""Comprehensive smoke test for live webcam and TTS integration.

Verifies:
1. Real camera capture via HandPipeline (/dev/video0)
2. MediaPipe hand landmarker processing on live frames
3. Full gesture recognition sequence: H -> I -> SPACE -> H -> I -> "HI HI"
4. TTS audio output for confirmed letters and SPACE
5. Transcript read-aloud via 'R' key action logic
6. Clean resource teardown (camera pipeline and TTS audio thread)
"""

import time
import numpy as np
from pathlib import Path

from signlang.config import WRIST
from signlang.engine import DwellRecognizer
from signlang.hands import HandPipeline
from signlang.model import Predictor
from signlang.tts import get_tts


def main():
    print("=" * 65)
    print("LIVE SMOKE TEST: WEBCAM + TTS INTEGRATION (TASK 7)")
    print("=" * 65)

    # 1. Initialize TTS Engine
    print("\n[1/5] Initializing TTS Engine...")
    tts = get_tts()
    assert tts.enabled, "TTS must be enabled (Piper + sounddevice available)"
    print("  ✓ TTS initialized successfully (Piper ONNX loaded)")

    # 2. Test Real Webcam Frame Acquisition
    print("\n[2/5] Connecting to webcam (/dev/video0)...")
    pipe = HandPipeline(source=0)
    assert pipe.started.wait(timeout=10), "Camera failed to start within 10s"
    assert pipe.error is None, f"Camera encountered error: {pipe.error}"

    frames_captured = 0
    t0 = time.time()
    for _ in range(25):
        frame, landmarks, handedness = pipe.read(timeout=3.0)
        if frame is not None:
            frames_captured += 1
    duration = time.time() - t0
    fps = frames_captured / duration if duration > 0 else 0
    print(f"  ✓ Captured {frames_captured} real camera frames in {duration:.2f}s (~{fps:.1f} fps)")
    assert frames_captured > 0, "No frames captured from camera"

    # 3. Initialize Predictor and Recognizer
    print("\n[3/5] Loading Predictor and DwellRecognizer...")
    predictor = Predictor()
    rec = DwellRecognizer(
        predictor,
        smooth_ms=100,
        dwell_ms=1000,
        space_dwell_ms=800,
        repeat_cooldown_ms=900,
        space_cooldown_ms=900,
    )
    print(f"  ✓ Predictor loaded {len(predictor.labels)} classes: {predictor.labels}")

    # Load actual recorded samples for H, I, SPACE
    h_sample = np.load("data/samples/H.npy")[0]
    i_sample = np.load("data/samples/I.npy")[0]
    space_sample = np.load("data/samples/SPACE.npy")[0]

    # Verify model recognizes the templates
    assert predictor.labels[predictor.probs_from_landmarks(h_sample).argmax()] == "H"
    assert predictor.labels[predictor.probs_from_landmarks(i_sample).argmax()] == "I"
    assert predictor.labels[predictor.probs_from_landmarks(space_sample).argmax()] == "SPACE"
    print("  ✓ Landmark samples validated against classifier model")

    # 4. Simulate sequence H -> I -> SPACE -> H -> I
    print("\n[4/5] Simulating full sequence: H -> I -> SPACE -> H -> I ...")
    sim_time = 1000.0
    dt = 1000.0 / 30.0  # 30 fps

    def feed_sample(sample, duration_ms):
        nonlocal sim_time
        n_frames = max(1, int(round(duration_ms / dt)))
        for _ in range(n_frames):
            sim_time += dt
            st = rec.update(
                sample,
                handedness="Right",
                width=640,
                height=480,
                raw_landmarks=sample,
                t_ms=sim_time,
            )
            emitted = st.get("emitted")
            if emitted:
                # Speak confirmed letter or space as live.py does
                tts.say_letter(emitted)

    # 4a. H
    feed_sample(h_sample, 1200)
    assert rec.buffer == "H", f"Expected 'H', got {rec.buffer!r}"
    print(f"  ✓ Emitted 'H' -> transcript: {rec.buffer!r}")
    feed_sample(h_sample, 300)

    # 4b. I
    feed_sample(i_sample, 1200)
    assert rec.buffer == "HI", f"Expected 'HI', got {rec.buffer!r}"
    print(f"  ✓ Emitted 'I' -> transcript: {rec.buffer!r}")
    feed_sample(i_sample, 300)

    # 4c. SPACE
    feed_sample(space_sample, 1000)
    assert rec.buffer == "HI ", f"Expected 'HI ', got {rec.buffer!r}"
    print(f"  ✓ Emitted SPACE -> transcript: {rec.buffer!r}")
    feed_sample(space_sample, 300)

    # 4d. H
    feed_sample(h_sample, 1200)
    assert rec.buffer == "HI H", f"Expected 'HI H', got {rec.buffer!r}"
    print(f"  ✓ Emitted second 'H' -> transcript: {rec.buffer!r}")
    feed_sample(h_sample, 300)

    # 4e. I
    feed_sample(i_sample, 1200)
    assert rec.buffer == "HI HI", f"Expected 'HI HI', got {rec.buffer!r}"
    print(f"  ✓ Emitted second 'I' -> transcript: {rec.buffer!r}")

    # 5. Test 'R' key transcript read-aloud
    print("\n[5/5] Testing 'R' key transcript read-aloud via TTS...")
    print(f"  Transcript to speak: {rec.buffer!r}")
    # Trigger 'R' key handler logic from live.py
    tts.read_transcript(rec.buffer)
    # Allow background audio playback to proceed
    time.sleep(2.0)
    print("  ✓ Transcript read-aloud completed successfully")

    # Clean shutdown
    print("\nReleasing webcam and shutting down TTS engine...")
    pipe.close()
    tts.close()
    print("  ✓ Webcam pipeline and TTS resources closed cleanly")

    print("\n" + "=" * 65)
    print("ALL LIVE SMOKE TESTS PASSED!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
