"""Real-time live ASL Words recognition using MediaPipe Holistic and DTW-kNN.

Features:
- Optimized for low-power dual-core CPU: 640x480 capture, FRAME_SKIP=2, model_complexity=0.
- Motion energy and sliding-window sign start/end detection.
- High-contrast HUD overlay displaying live status, recognized word, confidence, and transcript.
"""

import argparse
from collections import deque
from pathlib import Path
import sys
import time
from typing import Any, Deque, List, Optional, Tuple
import cv2
import numpy as np

from .classifier import DTWKNNClassifier
from .config import (
    CONFIDENCE_THRESHOLD,
    COOLDOWN_FRAMES,
    DEFAULT_SOURCE,
    ENERGY_ACTIVE_THRESHOLD,
    ENERGY_QUIET_THRESHOLD,
    FRAME_HEIGHT,
    FRAME_SKIP,
    FRAME_WIDTH,
    KNN_NEIGHBORS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_COMPLEXITY,
    MODEL_DIR,
    SEQUENCE_LENGTH,
    TOTAL_LANDMARKS,
    WORD_GLOSS,
    WORDS,
)
from .dataset import generate_synthetic_dataset, load_dataset
from .normalize import extract_landmarks, normalize_sequence

WINDOW_TITLE = "SignLang Words - Live Recognition"


def resample_sequence(sequence: np.ndarray, target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Linearly interpolate sequence of shape (M, ...) to (target_length, ...)."""
    seq = np.asarray(sequence, dtype=np.float32)
    M = len(seq)
    if M == target_length:
        return seq
    if M == 1:
        return np.repeat(seq, target_length, axis=0)

    orig_indices = np.linspace(0.0, 1.0, M)
    target_indices = np.linspace(0.0, 1.0, target_length)

    flat = seq.reshape(M, -1)
    D = flat.shape[1]
    resampled = np.empty((target_length, D), dtype=np.float32)

    for d in range(D):
        resampled[:, d] = np.interp(target_indices, orig_indices, flat[:, d])

    return resampled.reshape((target_length,) + seq.shape[1:])


def compute_motion_energy(prev_lms: np.ndarray, curr_lms: np.ndarray) -> float:
    """Compute average landmark motion velocity/energy between consecutive frames.

    Focuses on wrists (indices 5, 6) and hand landmarks (indices 9..50).
    """
    # Active landmark indices: wrists and hands
    active_indices = [5, 6] + list(range(9, min(51, len(curr_lms))))
    diff = curr_lms[active_indices] - prev_lms[active_indices]
    dists = np.linalg.norm(diff, axis=-1)
    return float(np.mean(dists))


def draw_hud(
    frame: np.ndarray,
    status: str,
    last_word: Optional[str],
    last_conf: float,
    energy: float,
    transcript: List[str],
    fps: float,
) -> None:
    """Draw high-contrast, modern HUD overlay on live frame."""
    h, w = frame.shape[:2]

    # 1. Top bar
    cv2.rectangle(frame, (0, 0), (w, 56), (11, 14, 20), -1)
    cv2.line(frame, (0, 56), (w, 56), (30, 41, 59), 1)

    # Wordmark & live badge
    cv2.putText(frame, "SIGNLANG", (16, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (248, 250, 252), 1, cv2.LINE_AA)
    cv2.putText(frame, "WORDS", (122, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, (56, 189, 248), 1, cv2.LINE_AA)

    # Status badge
    if status == "SIGNING":
        badge_col = (56, 189, 248)  # Sky blue
        badge_txt = "SIGNING..."
    elif status == "CONFIRMED":
        badge_col = (16, 185, 129)  # Emerald green
        badge_txt = "CONFIRMED"
    else:
        badge_col = (100, 116, 139)  # Slate
        badge_txt = "READY"

    cv2.putText(frame, f"[{badge_txt}]", (220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_col, 2, cv2.LINE_AA)

    # Energy meter
    energy_norm = min(1.0, energy / (ENERGY_ACTIVE_THRESHOLD * 2.5))
    bar_w = 120
    bar_x = w - 145
    cv2.rectangle(frame, (bar_x, 14), (bar_x + bar_w, 26), (25, 34, 49), -1)
    fill_col = (16, 185, 129) if energy < ENERGY_ACTIVE_THRESHOLD else (245, 158, 11)
    cv2.rectangle(frame, (bar_x, 14), (bar_x + int(bar_w * energy_norm), 26), fill_col, -1)
    cv2.putText(frame, "MOTION", (bar_x - 55, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)

    # 2. Reading Card (Top-Left)
    if last_word:
        gloss = WORD_GLOSS.get(last_word, last_word.lower())
        card_w, card_h = 240, 75
        card_x, card_y = 16, 70
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (11, 14, 20), -1)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (51, 65, 85), 1)

        cv2.putText(frame, last_word, (card_x + 14, card_y + 36), cv2.FONT_HERSHEY_DUPLEX, 0.9, (250, 204, 21), 2, cv2.LINE_AA)
        cv2.putText(frame, f"'{gloss}'", (card_x + 14, card_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1, cv2.LINE_AA)

        # Confidence tag
        conf_pct = int(last_conf * 100)
        cv2.putText(frame, f"{conf_pct}%", (card_x + card_w - 55, card_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (16, 185, 129), 2, cv2.LINE_AA)

        # Confidence micro bar
        cv2.rectangle(frame, (card_x + card_w - 65, card_y + 46), (card_x + card_w - 15, card_y + 52), (25, 34, 49), -1)
        cv2.rectangle(frame, (card_x + card_w - 65, card_y + 46), (card_x + card_w - 65 + int(50 * last_conf), card_y + 52), (16, 185, 129), -1)

    # 3. Bottom Panel
    bot_h = 60
    cv2.rectangle(frame, (0, h - bot_h), (w, h), (11, 14, 20), -1)
    cv2.line(frame, (0, h - bot_h), (w, h - bot_h), (30, 41, 59), 1)

    # Transcript line
    trans_text = " ".join(transcript[-8:]) if transcript else "(start signing to form a sentence)"
    trans_col = (248, 250, 252) if transcript else (100, 116, 139)
    cv2.putText(frame, f"TRANSCRIPT: {trans_text}", (16, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, trans_col, 1, cv2.LINE_AA)

    # Bottom helper & FPS
    cv2.putText(frame, f"{fps:.1f} FPS", (w - 75, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1, cv2.LINE_AA)
    cv2.putText(frame, "Controls: [Q] Quit   [C] Clear transcript", (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 116, 139), 1, cv2.LINE_AA)


def run_live(
    source: str = DEFAULT_SOURCE,
    classifier: Optional[DTWKNNClassifier] = None,
    max_frames: Optional[int] = None,
    headless: bool = False,
    display: bool = True,
) -> int:
    """Run real-time ASL Words live recognition.

    Args:
        source: Video capture source (device index '0' or video file path).
        classifier: Fitted DTWKNNClassifier. If None, loaded from disk.
        max_frames: Optional maximum frames to process (useful for tests).
        headless: If True, skips cv2 window display.
        display: If False, skips cv2.imshow.

    Returns:
        Exit code (0 for clean termination).
    """
    try:
        import mediapipe as mp
    except ImportError:
        print("Error: mediapipe is not installed.")
        return 1

    # Load or initialize classifier
    if classifier is None:
        model_path = MODEL_DIR / "words_classifier.npz"
        if model_path.exists():
            print(f"Loading classifier from {model_path}...")
            classifier = DTWKNNClassifier.load(model_path)
        else:
            print("No saved classifier found. Loading dataset to train classifier...")
            X, y = load_dataset()
            if len(X) == 0:
                print("No samples found. Generating synthetic dataset...")
                generate_synthetic_dataset(samples_per_word=10)
                X, y = load_dataset()
            classifier = DTWKNNClassifier(n_neighbors=KNN_NEIGHBORS, confidence_threshold=CONFIDENCE_THRESHOLD)
            classifier.fit(X, y)
            classifier.save(model_path)

    # Open Video Source
    src_val = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src_val)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source!r}")
        print("Tip: If running without a webcam, test with a recorded video or words.eval.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    mp_draw = mp.solutions.drawing_utils

    if not headless and display:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TITLE, FRAME_WIDTH, FRAME_HEIGHT)

    # State variables
    frame_count = 0
    fps_time = time.time()
    fps = 0.0
    fps_counter = 0

    sliding_window: Deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    active_segment: List[np.ndarray] = []
    state = "REST"  # REST, SIGNING, CONFIRMED
    cooldown = 0
    smooth_energy = 0.0
    prev_landmarks: Optional[np.ndarray] = None

    last_word: Optional[str] = None
    last_conf: float = 0.0
    transcript: List[str] = []

    print("=" * 60)
    print("  SIGNLANG WORDS - Live Recognition")
    print(f"  Vocabulary ({len(classifier.classes_)} words): {', '.join(classifier.classes_)}")
    print(f"  Confidence threshold: {classifier.confidence_threshold:.2f}")
    print("  Controls: [Q] Quit   [C] Clear transcript")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            fps_counter += 1
            now = time.time()
            if now - fps_time >= 1.0:
                fps = fps_counter / (now - fps_time)
                fps_counter = 0
                fps_time = now

            if max_frames is not None and frame_count >= max_frames:
                break

            frame = cv2.flip(frame, 1)  # Selfie mirror

            # Process detection at FRAME_SKIP rate
            if frame_count % FRAME_SKIP == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)

                raw_lms = extract_landmarks(results)
                sliding_window.append(raw_lms)

                # Draw landmark skeleton preview
                if results.pose_landmarks:
                    mp_draw.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                        mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
                    )
                if results.left_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1),
                    )
                if results.right_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1),
                    )

                # Motion energy calculation
                if prev_landmarks is not None:
                    inst_energy = compute_motion_energy(prev_landmarks, raw_lms)
                    smooth_energy = 0.7 * smooth_energy + 0.3 * inst_energy
                prev_landmarks = raw_lms

                # Gesture state machine
                if cooldown > 0:
                    cooldown -= 1
                    if cooldown == 0 and state == "CONFIRMED":
                        state = "REST"

                elif state == "REST":
                    if smooth_energy >= ENERGY_ACTIVE_THRESHOLD:
                        state = "SIGNING"
                        # Seed with recent frames
                        active_segment = list(sliding_window)[-5:]

                elif state == "SIGNING":
                    active_segment.append(raw_lms)

                    # Trigger condition: motion has slowed down or reached max length
                    motion_stopped = (smooth_energy < ENERGY_QUIET_THRESHOLD and len(active_segment) >= 12)
                    buffer_full = (len(active_segment) >= SEQUENCE_LENGTH + 6)

                    if motion_stopped or buffer_full:
                        # Extract and resample sequence
                        seg_arr = np.stack(active_segment, axis=0)
                        resampled = resample_sequence(seg_arr, target_length=SEQUENCE_LENGTH)
                        norm_seq = normalize_sequence(resampled, per_frame=True)

                        pred, conf, details = classifier.predict_single(norm_seq)

                        if pred is not None and conf >= classifier.confidence_threshold:
                            last_word = pred
                            last_conf = conf
                            transcript.append(pred)
                            state = "CONFIRMED"
                            cooldown = COOLDOWN_FRAMES
                            print(f"Recognized word: {pred} ({conf * 100:.1f}%)")
                        else:
                            state = "REST"

                        active_segment = []

            # Draw HUD
            draw_hud(
                frame=frame,
                status=state,
                last_word=last_word,
                last_conf=last_conf,
                energy=smooth_energy,
                transcript=transcript,
                fps=fps,
            )

            if not headless and display:
                cv2.imshow(WINDOW_TITLE, frame)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), ord("Q"), 27):
                    break
                elif k in (ord("c"), ord("C")):
                    transcript.clear()
                    last_word = None
                    last_conf = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if not headless and display:
            cv2.destroyAllWindows()
        holistic.close()

    print(f"\nFinal transcript: {' '.join(transcript)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="words.live",
        description="Real-time ASL Words live recognition.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Video capture source (device index '0' or file path, default: 0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold for recognition (default: 0.65).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process before exiting (useful for testing).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window.",
    )

    args = parser.parse_args(argv)

    return run_live(
        source=args.source,
        max_frames=args.max_frames,
        headless=args.headless,
    )


if __name__ == "__main__":
    sys.exit(main())
