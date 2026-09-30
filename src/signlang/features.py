import numpy as np

from .config import (
    DIGIT_CHAINS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
    WRIST,
)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def normalize_hand(points, handedness):
    p = np.asarray(points, dtype=np.float32).copy()
    if handedness == "Left":
        p[:, 0] = -p[:, 0]
    p = p - p[WRIST]
    scale = _dist(p[MIDDLE_MCP], p[WRIST])
    if scale < 1e-6:
        scale = 1.0
    return p / scale


def extract_features(points, handedness):
    return landmark_features(normalize_hand(points, handedness))


def landmark_features(p):
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
    depth = float(p[[THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP], 2].mean())

    return np.concatenate(
        [
            coords,
            np.asarray(extensions, dtype=np.float32),
            np.asarray([pinch, spread, depth], dtype=np.float32),
            normal.astype(np.float32),
        ]
    ).astype(np.float32)


def feature_dim():
    dummy = np.zeros((21, 3), dtype=np.float32)
    dummy[9] = [0.0, 0.5, 0.0]
    return int(extract_features(dummy, "Right").shape[0])


LANDMARK_DIM = 63


def landmark_dim():
    return LANDMARK_DIM


def _renorm(p):
    p = p - p[WRIST]
    scale = float(np.linalg.norm(p[MIDDLE_MCP] - p[WRIST]))
    if scale < 1e-6:
        scale = 1.0
    return p / scale


def _rot(p, axis, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == 0:
        m = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    elif axis == 1:
        m = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    else:
        m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return _renorm(p @ m.T)


def augment(
    coords,
    rng,
    roll_deg=11.0,
    pitch_deg=26.0,
    yaw_deg=20.0,
    noise=0.006,
    finger_jitter=0.014,
):
    p = np.asarray(coords, dtype=np.float32).reshape(21, 3).copy()

    p = _rot(p, 2, rng.uniform(-roll_deg, roll_deg))
    if pitch_deg:
        p = _rot(p, 0, rng.uniform(-pitch_deg, pitch_deg))
    if yaw_deg:
        p = _rot(p, 1, rng.uniform(-yaw_deg, yaw_deg))
    p = _renorm(p)

    if finger_jitter > 0:
        anchor = np.array([0, 1, 2, 3, 5, 9, 13, 17], dtype=np.int64)
        free = np.array([i for i in range(21) if i not in set(anchor.tolist())])
        p[free] += rng.normal(0.0, finger_jitter, (len(free), 3)).astype(np.float32)
        p = _renorm(p)

    if noise > 0:
        p += rng.normal(0.0, noise, p.shape).astype(np.float32)
        p = _renorm(p)

    return p.reshape(-1).astype(np.float32)
