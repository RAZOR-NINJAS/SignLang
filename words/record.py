"""Interactive recording tool and synthetic data generator for ASL Words.

Supports:
1. Webcam recording: captures 30-frame MediaPipe Holistic sequences with countdown and HUD.
2. Synthetic generation: generates realistic 30-frame motion sequences without requiring a camera.
"""

import argparse
import sys
import time
from typing import List, Optional
import cv2
import numpy as np

from .config import (
    DEFAULT_SOURCE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_COMPLEXITY,
    SEQUENCE_LENGTH,
    TOTAL_LANDMARKS,
    WORD_GLOSS,
    WORDS,
)
from .dataset import (
    count_samples,
    generate_synthetic_dataset,
    save_word_samples,
)
from .normalize import extract_landmarks, normalize_landmarks, normalize_sequence


def _draw_text_with_shadow(
    img: np.ndarray,
    text: str,
    org: tuple,
    font_face: int,
    font_scale: float,
    color: tuple,
    thickness: int = 1,
) -> None:
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), font_face, font_scale, (10, 10, 10), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)


def record_webcam(
    source: str = DEFAULT_SOURCE,
    target_words: Optional[List[str]] = None,
    samples_per_word: int = 10,
    windowed: bool = True,
) -> int:
    """Run interactive webcam recording session for words."""
    try:
        import mediapipe as mp
    except ImportError:
        print("Error: mediapipe is not installed.")
        return 1

    vocab = [w.upper() for w in target_words] if target_words else WORDS
    src_val = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src_val)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source!r}")
        print("Tip: Run with --synthetic to generate sequences without a physical camera.")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    mp_draw = mp.solutions.drawing_utils

    word_idx = 0
    state = "IDLE"  # IDLE, COUNTDOWN, RECORDING
    countdown_start = 0.0
    recorded_frames: List[np.ndarray] = []
    window_name = "SignLang Words - Recording Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, FRAME_WIDTH, FRAME_HEIGHT)

    print("=" * 60)
    print("  ASL Words Interactive Recorder")
    print(f"  Target words: {', '.join(vocab)}")
    print("  Controls:")
    print("    [SPACE]   Start countdown & record 30-frame sequence")
    print("    [N]       Next word")
    print("    [B]       Previous word")
    print("    [Q]/[ESC] Quit and exit")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("End of video stream or failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)  # Selfie mirror
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Draw landmarks preview
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

            curr_word = vocab[word_idx]
            counts = count_samples()
            curr_count = counts.get(curr_word, 0)

            # HUD Bar at top
            cv2.rectangle(frame, (0, 0), (w, 55), (15, 23, 42), -1)
            cv2.line(frame, (0, 55), (w, 55), (51, 65, 85), 1)

            _draw_text_with_shadow(
                frame,
                f"WORD [{word_idx + 1}/{len(vocab)}]: {curr_word} ('{WORD_GLOSS.get(curr_word, '')}')",
                (14, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (248, 250, 252),
                2,
            )
            _draw_text_with_shadow(
                frame,
                f"Recorded: {curr_count} / {samples_per_word}",
                (14, 46),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (148, 163, 184),
                1,
            )

            # State handling
            now = time.time()
            if state == "IDLE":
                _draw_text_with_shadow(
                    frame,
                    "Press [SPACE] to record",
                    (w - 230, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (56, 189, 248),
                    2,
                )

            elif state == "COUNTDOWN":
                elapsed = now - countdown_start
                rem = max(0.0, 2.0 - elapsed)
                count_val = int(np.ceil(rem))
                # Big countdown indicator
                cv2.circle(frame, (w // 2, h // 2), 65, (15, 23, 42), -1)
                cv2.circle(frame, (w // 2, h // 2), 65, (245, 158, 11), 3)
                _draw_text_with_shadow(
                    frame,
                    str(count_val),
                    (w // 2 - 18, h // 2 + 20),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.8,
                    (245, 158, 11),
                    3,
                )
                if elapsed >= 2.0:
                    state = "RECORDING"
                    recorded_frames = []

            elif state == "RECORDING":
                # Extract landmark frame
                lms = extract_landmarks(results)
                recorded_frames.append(lms)
                n_rec = len(recorded_frames)

                # Recording progress indicator
                cv2.circle(frame, (25, h - 25), 10, (0, 0, 230), -1)
                _draw_text_with_shadow(
                    frame,
                    f"REC: {n_rec}/{SEQUENCE_LENGTH} frames",
                    (45, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 240),
                    2,
                )

                # Progress bar at bottom
                prog_w = int((w - 20) * (n_rec / SEQUENCE_LENGTH))
                cv2.rectangle(frame, (10, h - 8), (10 + prog_w, h - 4), (0, 0, 240), -1)

                if n_rec >= SEQUENCE_LENGTH:
                    # Completed recording sequence
                    raw_seq = np.stack(recorded_frames[:SEQUENCE_LENGTH], axis=0)
                    norm_seq = normalize_sequence(raw_seq, per_frame=True)
                    new_tot = save_word_samples(
                        word=curr_word,
                        sequences=norm_seq[None, ...],
                        source=f"webcam{source}",
                        append=True,
                    )
                    print(f"Saved sample for {curr_word} (total: {new_tot})")
                    state = "IDLE"
                    recorded_frames = []

            # Bottom help line
            _draw_text_with_shadow(
                frame,
                "[SPACE] Record   [N] Next Word   [B] Prev Word   [Q] Quit",
                (14, h - 35 if state != "RECORDING" else h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (100, 116, 139),
                1,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == ord(" ") and state == "IDLE":
                state = "COUNTDOWN"
                countdown_start = time.time()
            elif key in (ord("n"), ord("N")) and state == "IDLE":
                word_idx = (word_idx + 1) % len(vocab)
            elif key in (ord("b"), ord("B")) and state == "IDLE":
                word_idx = (word_idx - 1) % len(vocab)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()

    print("\nRecording session ended.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="words.record",
        description="Record or synthetically generate 30-frame ASL word sequences.",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        default=True,
        help="Record interactively from webcam (default).",
    )
    parser.add_argument(
        "--synthetic",
        "--generate",
        dest="synthetic",
        action="store_true",
        help="Generate synthetic samples instead of opening webcam.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to record or generate per word (default: 10).",
    )
    parser.add_argument(
        "--word",
        default=None,
        help="Specific word(s) to process, comma-separated (e.g. HELLO,YES). Defaults to all.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Camera device index or video path (default: 0).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List current sample counts for each word and exit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic generation (default: 42).",
    )

    args = parser.parse_args(argv)

    if args.list:
        counts = count_samples()
        print("Sample counts for ASL Words vocabulary:")
        total = 0
        for w in WORDS:
            c = counts.get(w, 0)
            total += c
            print(f"  {w:<12} {c:>5} samples")
        print(f"Total: {total} samples across {len(WORDS)} words.")
        return 0

    target_words = None
    if args.word:
        target_words = [w.strip().upper() for w in args.word.split(",") if w.strip()]
        for w in target_words:
            if w not in WORDS:
                print(f"Warning: '{w}' is not in standard WORDS vocabulary.")

    if args.synthetic:
        print(f"Generating {args.samples} synthetic samples per word...")
        counts = generate_synthetic_dataset(
            samples_per_word=args.samples,
            seed=args.seed,
            overwrite=True,
            words=target_words,
        )
        for w, c in counts.items():
            print(f"  {w:<12}: {c} total samples")
        print(f"Synthetic generation complete for {len(counts)} words.")
        return 0

    return record_webcam(
        source=args.source,
        target_words=target_words,
        samples_per_word=args.samples,
    )


if __name__ == "__main__":
    sys.exit(main())
