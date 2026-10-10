"""Unit tests for ISL two-hand feature engineering."""

import numpy as np
import pytest

from isl.config import FEATURE_DIM, SINGLE_HAND_DIM
from isl.features import (
    canonicalize_hands,
    extract_features,
    feature_dim,
    mirror_feature_vector,
    normalize_single_hand,
    single_hand_features,
)


def _synthetic_hand(seed=0, offset=(0.5, 0.5, 0.0)):
    """Generate reproducible 21x3 hand landmarks."""
    rng = np.random.default_rng(seed)
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = offset
    for i in range(1, 21):
        pts[i] = pts[0] + rng.normal(0.0, 0.05, 3).astype(np.float32)
    # Ensure middle MCP (index 9) is reasonably distant from wrist (index 0)
    pts[9] = pts[0] + np.array([0.0, 0.1, 0.0], dtype=np.float32)
    return pts


def test_feature_dim_is_164():
    assert feature_dim() == 164
    assert FEATURE_DIM == 164


def test_empty_landmarks_returns_zeros():
    feat = extract_features([], [])
    assert feat.shape == (164,)
    assert np.all(feat == 0.0)


def test_single_hand_presence_and_zero_imputation():
    # Only Left hand present
    left = _synthetic_hand(seed=1, offset=(0.3, 0.5, 0.0))
    feat_left_only = extract_features([left], ["Left"])
    assert feat_left_only.shape == (164,)
    assert feat_left_only[162] == 1.0  # has_left
    assert feat_left_only[163] == 0.0  # has_right
    # Right slot [74..147] must be all zeros
    assert np.all(feat_left_only[SINGLE_HAND_DIM:SINGLE_HAND_DIM * 2] == 0.0)
    # Inter-hand features [148..161] must be all zeros
    assert np.all(feat_left_only[SINGLE_HAND_DIM * 2:162] == 0.0)
    # Left slot [0..73] must NOT be all zeros
    assert not np.all(feat_left_only[0:SINGLE_HAND_DIM] == 0.0)

    # Only Right hand present
    right = _synthetic_hand(seed=2, offset=(0.7, 0.5, 0.0))
    feat_right_only = extract_features([right], ["Right"])
    assert feat_right_only.shape == (164,)
    assert feat_right_only[162] == 0.0  # has_left
    assert feat_right_only[163] == 1.0  # has_right
    # Left slot [0..73] must be all zeros
    assert np.all(feat_right_only[0:SINGLE_HAND_DIM] == 0.0)
    # Right slot [74..147] must NOT be all zeros
    assert not np.all(feat_right_only[SINGLE_HAND_DIM:SINGLE_HAND_DIM * 2] == 0.0)


def test_canonical_hand_ordering_distinct_labels():
    h_l = _synthetic_hand(seed=3, offset=(0.2, 0.5, 0.0))
    h_r = _synthetic_hand(seed=4, offset=(0.8, 0.5, 0.0))

    # Passed in correct order [Left, Right]
    l1, r1 = canonicalize_hands([h_l, h_r], ["Left", "Right"])
    assert np.allclose(l1, h_l)
    assert np.allclose(r1, h_r)

    # Passed in inverted order [Right, Left]
    l2, r2 = canonicalize_hands([h_r, h_l], ["Right", "Left"])
    assert np.allclose(l2, h_l)
    assert np.allclose(r2, h_r)


def test_canonical_hand_ordering_tie_break_on_wrist_x():
    h_screen_left = _synthetic_hand(seed=5, offset=(0.25, 0.5, 0.0))
    h_screen_right = _synthetic_hand(seed=6, offset=(0.75, 0.5, 0.0))

    # Identical labels (flicker)
    l1, r1 = canonicalize_hands([h_screen_right, h_screen_left], ["Right", "Right"])
    assert np.allclose(l1, h_screen_left)
    assert np.allclose(r1, h_screen_right)


def test_two_hands_produces_full_features():
    h_l = _synthetic_hand(seed=7, offset=(0.3, 0.5, 0.0))
    h_r = _synthetic_hand(seed=8, offset=(0.7, 0.5, 0.0))

    feat = extract_features([h_l, h_r], ["Left", "Right"])
    assert feat.shape == (164,)
    assert feat[162] == 1.0  # has_left
    assert feat[163] == 1.0  # has_right
    assert not np.all(feat[0:74] == 0.0)
    assert not np.all(feat[74:148] == 0.0)
    assert not np.all(feat[148:162] == 0.0)


def test_mirror_feature_vector_symmetry():
    h_l = _synthetic_hand(seed=9, offset=(0.3, 0.5, 0.0))
    h_r = _synthetic_hand(seed=10, offset=(0.7, 0.5, 0.0))

    feat = extract_features([h_l, h_r], ["Left", "Right"])
    mirrored = mirror_feature_vector(feat)

    # Mirrored presence flags
    assert mirrored[162] == feat[163]
    assert mirrored[163] == feat[162]

    # Inverting mirrored again should restore original vector
    double_mirrored = mirror_feature_vector(mirrored)
    assert np.allclose(feat, double_mirrored, atol=1e-6)
