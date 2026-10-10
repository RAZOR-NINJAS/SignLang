from typing import Any, List, Optional, Tuple, Union
import numpy as np

from .config import (
    DIGIT_CHAINS,
    FEATURE_DIM,
    FINGERTIPS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_TIP,
    SINGLE_HAND_DIM,
    THUMB_TIP,
    WRIST,
)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def normalize_single_hand(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Translate wrist to origin and scale by palm distance (wrist to middle MCP)."""
    p = np.asarray(points, dtype=np.float32).copy()
    p = p - p[WRIST]
    scale = _dist(p[MIDDLE_MCP], p[WRIST])
    if scale < 1e-6:
        scale = 1.0
    return p / scale, scale


def single_hand_features(p: np.ndarray) -> np.ndarray:
    """Compute 74-dimensional geometric shape descriptor for one normalized hand."""
    palm = _dist(p[WRIST], p[MIDDLE_MCP])
    if palm < 1e-6:
        palm = 1.0

    coords = p.reshape(-1)

    extensions = [
        _dist(p[tip], p[mcp]) / palm for mcp, _pip, tip in DIGIT_CHAINS
    ]

    pinch = _dist(p[THUMB_TIP], p[INDEX_TIP]) / palm

    spread_pairs = [
        (INDEX_TIP, MIDDLE_TIP),
        (MIDDLE_TIP, RING_TIP),
        (RING_TIP, PINKY_TIP),
    ]
    spread = sum(_dist(p[a], p[b]) for a, b in spread_pairs) / (3.0 * palm)

    normal = np.cross(p[INDEX_MCP] - p[WRIST], p[PINKY_MCP] - p[WRIST])
    norm_val = np.linalg.norm(normal)
    if norm_val > 1e-6:
        normal = normal / norm_val

    depth = float(p[FINGERTIPS, 2].mean())

    return np.concatenate(
        [
            coords,
            np.asarray(extensions, dtype=np.float32),
            np.asarray([pinch, spread, depth], dtype=np.float32),
            normal.astype(np.float32),
        ]
    ).astype(np.float32)


def canonicalize_hands(
    landmarks_list: Optional[List[Any]],
    handedness_list: Optional[List[str]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Assign hands to canonical [slot_left, slot_right] based on handedness and wrist x position.

    Returns:
        (left_hand, right_hand) where each is either a (21, 3) float32 array or None.
    """
    if not landmarks_list:
        return None, None

    # Convert landmarks to np.ndarray (21, 3)
    parsed = []
    for lm in landmarks_list:
        if lm is None:
            continue
        if isinstance(lm, np.ndarray):
            parsed.append(lm.astype(np.float32).reshape(21, 3))
        elif isinstance(lm, (list, tuple)):
            if len(lm) > 0 and hasattr(lm[0], "x"):
                parsed.append(
                    np.array([[pt.x, pt.y, pt.z] for pt in lm], dtype=np.float32)
                )
            else:
                parsed.append(np.asarray(lm, dtype=np.float32).reshape(21, 3))

    if not parsed:
        return None, None

    h_labels = list(handedness_list or [])
    while len(h_labels) < len(parsed):
        h_labels.append("Unknown")

    left_hand: Optional[np.ndarray] = None
    right_hand: Optional[np.ndarray] = None

    if len(parsed) == 1:
        lbl = h_labels[0]
        if lbl == "Left":
            left_hand = parsed[0]
        elif lbl == "Right":
            right_hand = parsed[0]
        else:
            # Fallback by x coordinate in selfie view (0 is screen-left, 1 is screen-right)
            if parsed[0][WRIST][0] < 0.5:
                left_hand = parsed[0]
            else:
                right_hand = parsed[0]
        return left_hand, right_hand

    # Exactly 2 (or more) hands detected
    # If distinct labels exist:
    has_l = "Left" in h_labels[:2]
    has_r = "Right" in h_labels[:2]
    if has_l and has_r and h_labels[0] != h_labels[1]:
        idx_l = h_labels.index("Left")
        idx_r = h_labels.index("Right")
        left_hand = parsed[idx_l]
        right_hand = parsed[idx_r]
    else:
        # Ambiguous or duplicate labels: sort by wrist x position
        # Left slot = smaller x (viewer left), Right slot = larger x (viewer right)
        hands_sorted = sorted(parsed[:2], key=lambda h: float(h[WRIST][0]))
        left_hand = hands_sorted[0]
        right_hand = hands_sorted[1]

    return left_hand, right_hand


def extract_features(
    landmarks_list: Optional[List[Any]],
    handedness_list: Optional[List[str]],
    dominant: str = "right",
) -> np.ndarray:
    """Extract canonical 164-dimensional feature vector for ISL recognition.

    Vector layout:
      [0..73]:    Left hand shape features (74 floats, zero-imputed if absent)
      [74..147]:  Right hand shape features (74 floats, zero-imputed if absent)
      [148..161]: Inter-hand interaction features (14 floats, zero-imputed if absent)
      [162..163]: Hand presence flags [has_left, has_right] (2 floats)
    """
    left_hand, right_hand = canonicalize_hands(landmarks_list, handedness_list)

    has_left = 1.0 if left_hand is not None else 0.0
    has_right = 1.0 if right_hand is not None else 0.0

    left_feat = np.zeros(SINGLE_HAND_DIM, dtype=np.float32)
    right_feat = np.zeros(SINGLE_HAND_DIM, dtype=np.float32)
    left_norm: Optional[np.ndarray] = None
    right_norm: Optional[np.ndarray] = None
    scale_left, scale_right = 1.0, 1.0

    if left_hand is not None:
        left_norm, scale_left = normalize_single_hand(left_hand)
        left_feat = single_hand_features(left_norm)

    if right_hand is not None:
        right_norm, scale_right = normalize_single_hand(right_hand)
        right_feat = single_hand_features(right_norm)

    # Inter-hand interaction features (14 floats)
    inter_feat = np.zeros(14, dtype=np.float32)

    if left_hand is not None and right_hand is not None:
        avg_scale = max(0.5 * (scale_left + scale_right), 1e-4)

        # 1. Relative wrist offset: (right_wrist - left_wrist) / avg_scale (3 floats)
        wrist_offset = (right_hand[WRIST] - left_hand[WRIST]) / avg_scale

        # 2. Inter-wrist distance (1 float)
        inter_wrist_dist = _dist(right_hand[WRIST], left_hand[WRIST]) / avg_scale

        # Determine dominant vs base hand for asymmetric contact features
        is_right_dom = (dominant.lower() == "right")
        dom_hand = right_hand if is_right_dom else left_hand
        base_hand = left_hand if is_right_dom else right_hand

        # 3. Dominant index tip to base fingertips (5 floats)
        contact_dists = [
            _dist(dom_hand[INDEX_TIP], base_hand[tip]) / avg_scale
            for tip in FINGERTIPS
        ]

        # 4. Dominant index tip to base wrist (1 float)
        dom_idx_to_base_wrist = _dist(dom_hand[INDEX_TIP], base_hand[WRIST]) / avg_scale

        # 5. Dominant thumb tip to base index tip (1 float)
        dom_thumb_to_base_idx = _dist(dom_hand[THUMB_TIP], base_hand[INDEX_TIP]) / avg_scale

        # 6. Minimum fingertip distance between hands (1 float)
        min_tip_dist = min(
            _dist(dom_hand[f1], base_hand[f2])
            for f1 in FINGERTIPS
            for f2 in FINGERTIPS
        ) / avg_scale

        # 7. Palm normal alignment dot product (1 float)
        left_normal = np.cross(left_norm[INDEX_MCP] - left_norm[WRIST], left_norm[PINKY_MCP] - left_norm[WRIST])
        right_normal = np.cross(right_norm[INDEX_MCP] - right_norm[WRIST], right_norm[PINKY_MCP] - right_norm[WRIST])
        nl_val = np.linalg.norm(left_normal)
        nr_val = np.linalg.norm(right_normal)
        if nl_val > 1e-6:
            left_normal /= nl_val
        if nr_val > 1e-6:
            right_normal /= nr_val
        normal_dot = float(np.dot(left_normal, right_normal))

        # 8. Relative height offset: (dom_y - base_y) / avg_scale (1 float)
        rel_height = float(dom_hand[WRIST][1] - base_hand[WRIST][1]) / avg_scale

        inter_feat = np.concatenate(
            [
                wrist_offset,
                np.asarray([inter_wrist_dist], dtype=np.float32),
                np.asarray(contact_dists, dtype=np.float32),
                np.asarray([dom_idx_to_base_wrist, dom_thumb_to_base_idx, min_tip_dist, normal_dot, rel_height], dtype=np.float32),
            ]
        ).astype(np.float32)

    presence = np.array([has_left, has_right], dtype=np.float32)

    return np.concatenate([left_feat, right_feat, inter_feat, presence]).astype(np.float32)


def mirror_feature_vector(feat: np.ndarray) -> np.ndarray:
    """Invert horizontal coordinate features and swap left/right hand slots for left-handed signers.

    Args:
        feat: 164-dimensional feature vector.
    Returns:
        164-dimensional feature vector with mirrored perspective.
    """
    f = feat.copy()
    # Left slot [0..73], Right slot [74..147]
    left_slot = f[0:SINGLE_HAND_DIM].copy()
    right_slot = f[SINGLE_HAND_DIM:SINGLE_HAND_DIM * 2].copy()

    # Invert x coordinates in 21x3 landmarks (indices 0, 3, 6, ... 60)
    for i in range(0, 63, 3):
        left_slot[i] = -left_slot[i]
        right_slot[i] = -right_slot[i]

    # Invert x component of normal vector (last 3 elements of each 74-dim slot)
    left_slot[-3] = -left_slot[-3]
    right_slot[-3] = -right_slot[-3]

    # Swap left and right slots
    f[0:SINGLE_HAND_DIM] = right_slot
    f[SINGLE_HAND_DIM:SINGLE_HAND_DIM * 2] = left_slot

    # Inter-hand offset: invert lateral dx (index 148)
    f[148] = -f[148]

    # Swap presence flags
    has_left = f[162]
    has_right = f[163]
    f[162] = has_right
    f[163] = has_left

    return f


def feature_dim() -> int:
    return FEATURE_DIM
