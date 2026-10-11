"""Geometric landmark templates for static Indian Sign Language (ISL) word gestures.

Generates realistic 3D landmark configurations for iconic, movement-free ISL signs:
  - NAMASTE: Both flat palms joined at chest in prayer/greeting (2 hands)
  - GOOD: Fist with thumb extended upright / thumbs up (1 hand)
  - BAD: Fist with pinky finger extended upright / katti sign (1 hand)
  - PEACE: V-sign with index and middle fingers extended (1 hand)
  - OK: Thumb and index forming a circle, other 3 fingers extended (1 hand)
  - HOUSE: Fingertips of both flat hands touching at an angle forming a roof (2 hands)
  - BOOK: Both open palms facing upward touching pinky edges side-by-side (2 hands)
  - TIME: Dominant index finger touching opposite wrist (2 hands)
  - LOVE: Thumb, index, and pinky extended upright (1 hand)
  - STOP: Open flat palm facing forward towards camera (1 hand)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from . import dataset
from .config import DOMINANT_HAND, EXPECTED_HANDS, STATIC_WORDS
from .features import extract_features


def _build_hand(
    wrist: Tuple[float, float, float] = (0.50, 0.70, 0.0),
    thumb: str = "curled",
    index: str = "curled",
    middle: str = "curled",
    ring: str = "curled",
    pinky: str = "curled",
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    is_left: bool = False,
) -> np.ndarray:
    """Build a 21-landmark 3D hand point cloud based on finger state specifications."""
    wx, wy, wz = wrist
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [wx, wy, wz]

    # Finger MCP base offsets from wrist (relaxed hand coordinate frame)
    mcps = {
        "thumb":  [0.038, -0.048, 0.000],
        "index":  [0.028, -0.098, 0.000],
        "middle": [0.000, -0.106, 0.000],
        "ring":   [-0.028, -0.098, 0.000],
        "pinky":  [-0.048, -0.086, 0.000],
    }
    indices = {
        "thumb":  [1, 2, 3, 4],
        "index":  [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring":   [13, 14, 15, 16],
        "pinky":  [17, 18, 19, 20],
    }
    states = {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
    }

    for finger, (mcp_ox, mcp_oy, mcp_oz) in mcps.items():
        base = np.array([wx + mcp_ox, wy + mcp_oy, wz + mcp_oz], dtype=np.float32)
        idx_list = indices[finger]
        pts[idx_list[0]] = base
        st = states[finger]

        if finger == "thumb":
            if st == "up":  # Thumbs up
                pts[idx_list[1]] = base + [0.008, -0.032, 0.000]
                pts[idx_list[2]] = base + [0.010, -0.065, 0.000]
                pts[idx_list[3]] = base + [0.012, -0.098, 0.000]
            elif st == "extended":
                pts[idx_list[1]] = base + [0.030, -0.022, 0.000]
                pts[idx_list[2]] = base + [0.052, -0.042, 0.000]
                pts[idx_list[3]] = base + [0.074, -0.058, 0.000]
            elif st == "ring":  # OK sign touching index tip
                pts[idx_list[1]] = base + [0.000, -0.018, 0.000]
                pts[idx_list[2]] = base + [0.000, -0.035, 0.000]
                pts[idx_list[3]] = base + [0.000, -0.050, 0.000]
            else:  # curled
                pts[idx_list[1]] = base + [-0.012, 0.012, -0.020]
                pts[idx_list[2]] = base + [-0.022, 0.022, -0.030]
                pts[idx_list[3]] = base + [-0.030, 0.024, -0.032]
        else:
            if st == "extended":  # straight up
                # Fan fingers outward slightly
                spread_x = mcp_ox * 0.45
                pts[idx_list[1]] = base + [spread_x * 0.35, -0.038, 0.000]
                pts[idx_list[2]] = base + [spread_x * 0.70, -0.074, 0.000]
                pts[idx_list[3]] = base + [spread_x * 1.00, -0.108, 0.000]
            elif st == "v_index":  # V-sign index (angled left)
                pts[idx_list[1]] = base + [0.015, -0.038, 0.000]
                pts[idx_list[2]] = base + [0.030, -0.074, 0.000]
                pts[idx_list[3]] = base + [0.045, -0.108, 0.000]
            elif st == "v_middle":  # V-sign middle (angled right)
                pts[idx_list[1]] = base + [-0.015, -0.038, 0.000]
                pts[idx_list[2]] = base + [-0.030, -0.074, 0.000]
                pts[idx_list[3]] = base + [-0.045, -0.108, 0.000]
            elif st == "ring":  # index curled touching thumb in OK sign
                pts[idx_list[1]] = base + [0.005, -0.015, -0.010]
                pts[idx_list[2]] = base + [0.010, -0.008, -0.005]
                pts[idx_list[3]] = base + [0.010, 0.000, 0.000]
            else:  # curled into palm
                pts[idx_list[1]] = base + [0.000, 0.012, -0.022]
                pts[idx_list[2]] = base + [0.000, 0.024, -0.034]
                pts[idx_list[3]] = base + [0.000, 0.032, -0.026]

    if is_left:
        # Mirror x coordinates laterally across wrist x for authentic Left hand anatomy
        pts[:, 0] = 2 * wx - pts[:, 0]

    # Optional 3D Rotation
    if abs(roll) > 1e-4 or abs(pitch) > 1e-4 or abs(yaw) > 1e-4:
        center = pts[0].copy()
        pts -= center
        cr, sr = np.cos(roll), np.sin(roll)
        rot_z = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
        pts = pts @ rot_z.T
        pts += center

    return pts


def generate_word_landmarks(word: str) -> Tuple[List[np.ndarray], List[str]]:
    """Produce base landmark point cloud(s) and handedness list for an ISL static word."""
    w = word.upper().strip()

    if w == "NAMASTE":
        # Both hands vertically upright, flat palms pressed together at chest
        left = _build_hand(
            wrist=(0.48, 0.65, 0.0),
            thumb="extended", index="extended", middle="extended", ring="extended", pinky="extended",
            is_left=True,
        )
        right = _build_hand(
            wrist=(0.52, 0.65, 0.0),
            thumb="extended", index="extended", middle="extended", ring="extended", pinky="extended",
            is_left=False,
        )
        return [left, right], ["Left", "Right"]

    elif w == "GOOD":
        # Thumbs up gesture (Right hand)
        right = _build_hand(
            wrist=(0.55, 0.66, 0.0),
            thumb="up", index="curled", middle="curled", ring="curled", pinky="curled",
        )
        return [right], ["Right"]

    elif w == "BAD":
        # Pinky extended upright / Indian katti gesture
        right = _build_hand(
            wrist=(0.55, 0.66, 0.0),
            thumb="curled", index="curled", middle="curled", ring="curled", pinky="extended",
        )
        return [right], ["Right"]

    elif w == "PEACE":
        # V-sign with index and middle fingers
        right = _build_hand(
            wrist=(0.55, 0.66, 0.0),
            thumb="curled", index="v_index", middle="v_middle", ring="curled", pinky="curled",
        )
        return [right], ["Right"]

    elif w == "OK":
        # OK sign: thumb and index tips form circle, 3 fingers extended
        right = _build_hand(
            wrist=(0.55, 0.66, 0.0),
            thumb="ring", index="ring", middle="extended", ring="extended", pinky="extended",
        )
        return [right], ["Right"]

    elif w == "HOUSE":
        # Two hands with fingertips touching forming triangular roof
        left = _build_hand(
            wrist=(0.40, 0.68, 0.0),
            thumb="extended", index="extended", middle="extended", ring="extended", pinky="extended",
            roll=0.55,
            is_left=True,
        )
        right = _build_hand(
            wrist=(0.60, 0.68, 0.0),
            thumb="extended", index="extended", middle="extended", ring="extended", pinky="extended",
            roll=-0.55,
            is_left=False,
        )
        return [left, right], ["Left", "Right"]

    elif w == "BOOK":
        # Both open palms facing upward with pinky edges touching side-by-side
        left = _build_hand(
            wrist=(0.46, 0.66, 0.0),
            thumb="extended", index="extended", middle="extended", ring="extended", pinky="extended",
            is_left=True,
        )
        right = _build_hand(
            wrist=(0.54, 0.66, 0.0),
            thumb="extended", index="extended", middle="extended", ring="extended", pinky="extended",
            is_left=False,
        )
        return [left, right], ["Left", "Right"]

    elif w == "TIME":
        # Dominant index finger pointing to back of base wrist
        left = _build_hand(
            wrist=(0.48, 0.68, 0.0),
            thumb="curled", index="curled", middle="curled", ring="curled", pinky="curled",
            roll=-1.40,  # horizontal base wrist
            is_left=True,
        )
        right = _build_hand(
            wrist=(0.52, 0.48, 0.0),
            thumb="curled", index="extended", middle="curled", ring="curled", pinky="curled",
            roll=3.14,  # pointing down touching base wrist
            is_left=False,
        )
        return [left, right], ["Left", "Right"]

    elif w == "LOVE":
        # I-Love-You gesture (thumb, index, pinky extended)
        right = _build_hand(
            wrist=(0.55, 0.66, 0.0),
            thumb="extended", index="extended", middle="curled", ring="curled", pinky="extended",
        )
        return [right], ["Right"]

    elif w == "STOP":
        # Open flat palm facing forward
        right = _build_hand(
            wrist=(0.55, 0.66, 0.0),
            thumb="extended", index="extended", middle="extended", ring="extended", pinky="extended",
        )
        return [right], ["Right"]

    else:
        raise ValueError(f"Unknown static word template: '{word}'")


def generate_static_word_samples(
    words: Optional[List[str]] = None,
    samples_per_word: int = 140,
    seed: int = 42,
) -> Dict[str, int]:
    """Generate realistic landmark sample bursts with natural jitter for static ISL words."""
    target_words = words or STATIC_WORDS
    rng = np.random.default_rng(seed)
    created: Dict[str, int] = {}

    burst_size = 28
    num_bursts = max(1, samples_per_word // burst_size)

    for word in target_words:
        base_lms, base_hnds = generate_word_landmarks(word)
        total_added = 0

        for b in range(num_bursts):
            # Burst-level slight position/scale drift
            b_offset = rng.normal(0.0, 0.015, (1, 3)).astype(np.float32)
            b_scale = float(rng.uniform(0.95, 1.05))

            feats_list = []
            for _ in range(burst_size):
                # Frame-level sensor noise
                jittered_lms = []
                for hand_pts in base_lms:
                    noise = rng.normal(0.0, 0.004, hand_pts.shape).astype(np.float32)
                    frame_jitter = rng.normal(0.0, 0.003, (1, 3)).astype(np.float32)
                    jittered = (hand_pts * b_scale) + b_offset + noise + frame_jitter
                    jittered_lms.append(jittered)

                f = extract_features(jittered_lms, base_hnds, dominant=DOMINANT_HAND)
                feats_list.append(f)

            # Store into dataset tagged with session1_template_burst{b}
            n = dataset.append_samples(word, feats_list, source=f"session1_template_b{b}")
            total_added = n

        created[word] = total_added

    return created


def ensure_static_word_samples(force: bool = False) -> Dict[str, int]:
    """Ensure sample files exist for all static ISL words; generates them if missing."""
    if force:
        for w in STATIC_WORDS:
            dataset.reset_label(w)
        return generate_static_word_samples(words=STATIC_WORDS)
    missing = [w for w in STATIC_WORDS if dataset.count_for(w) == 0]
    if not missing:
        return {w: dataset.count_for(w) for w in STATIC_WORDS}
    return generate_static_word_samples(words=missing)
