"""Unit tests for words/normalize.py invariants and extraction."""

import numpy as np
import pytest

from words.config import TOTAL_LANDMARKS
from words.normalize import (
    BODY_LEFT_SHOULDER_IDX,
    BODY_RIGHT_SHOULDER_IDX,
    extract_landmarks,
    flatten_features,
    normalize_landmarks,
    normalize_sequence,
    unflatten_features,
)


def _make_dummy_frame(left_sh=(-0.4, 0.2, 0.1), right_sh=(0.6, 0.3, -0.1), seed=42):
    rng = np.random.default_rng(seed)
    frame = rng.uniform(-1.0, 1.0, size=(TOTAL_LANDMARKS, 3)).astype(np.float32)
    frame[BODY_LEFT_SHOULDER_IDX] = left_sh
    frame[BODY_RIGHT_SHOULDER_IDX] = right_sh
    return frame


def test_shoulder_midpoint_invariant():
    """Origin must be at the shoulder midpoint (0, 0, 0)."""
    frame = _make_dummy_frame(left_sh=(-0.3, 0.4, 0.5), right_sh=(0.7, 0.8, -0.1))
    normed = normalize_landmarks(frame)

    left_norm = normed[BODY_LEFT_SHOULDER_IDX]
    right_norm = normed[BODY_RIGHT_SHOULDER_IDX]
    midpoint = (left_norm + right_norm) / 2.0

    np.testing.assert_allclose(midpoint, np.zeros(3), atol=1e-6)


def test_shoulder_distance_invariant():
    """Distance between shoulders must normalize to exactly 1.0."""
    frame = _make_dummy_frame(left_sh=(10.0, 20.0, 0.0), right_sh=(13.0, 24.0, 0.0))  # dist = 5.0
    normed = normalize_landmarks(frame)

    left_norm = normed[BODY_LEFT_SHOULDER_IDX]
    right_norm = normed[BODY_RIGHT_SHOULDER_IDX]
    dist = np.linalg.norm(left_norm - right_norm)

    assert pytest.approx(dist, abs=1e-6) == 1.0


def test_translation_invariance():
    """Translating landmarks by an arbitrary vector must not change normalized values."""
    frame = _make_dummy_frame()
    normed_original = normalize_landmarks(frame)

    # Apply large translation
    offset = np.array([123.45, -67.89, 42.0], dtype=np.float32)
    translated = frame + offset
    normed_translated = normalize_landmarks(translated)

    np.testing.assert_allclose(normed_translated, normed_original, atol=1e-5)


def test_scale_invariance():
    """Scaling landmarks by an arbitrary positive factor must not change normalized values."""
    frame = _make_dummy_frame()
    normed_original = normalize_landmarks(frame)

    # Apply scaling
    for scale in [0.1, 2.5, 100.0]:
        scaled = frame * scale
        normed_scaled = normalize_landmarks(scaled)
        np.testing.assert_allclose(normed_scaled, normed_original, atol=1e-5)


def test_degenerate_zero_distance_handling():
    """When shoulders are coincident, normalization must not raise or produce NaN/Inf."""
    frame = _make_dummy_frame(left_sh=(0.5, 0.5, 0.5), right_sh=(0.5, 0.5, 0.5))
    normed = normalize_landmarks(frame)

    assert not np.isnan(normed).any()
    assert not np.isinf(normed).any()


def test_sequence_normalization_per_frame():
    """Normalizing a 30-frame sequence (30, 51, 3) maintains invariants on each frame."""
    T = 30
    frames = np.stack([_make_dummy_frame(seed=i) for i in range(T)])
    assert frames.shape == (30, TOTAL_LANDMARKS, 3)

    norm_seq = normalize_sequence(frames, per_frame=True)
    assert norm_seq.shape == (30, TOTAL_LANDMARKS, 3)

    for t in range(T):
        left_norm = norm_seq[t, BODY_LEFT_SHOULDER_IDX]
        right_norm = norm_seq[t, BODY_RIGHT_SHOULDER_IDX]
        midpoint = (left_norm + right_norm) / 2.0
        dist = np.linalg.norm(left_norm - right_norm)

        np.testing.assert_allclose(midpoint, np.zeros(3), atol=1e-6)
        assert pytest.approx(dist, abs=1e-6) == 1.0


def test_sequence_normalization_flattened():
    """Normalizing a flattened 2D sequence (30, 153) returns (30, 153)."""
    frames = np.stack([_make_dummy_frame(seed=i) for i in range(30)])
    flat = flatten_features(frames)
    assert flat.shape == (30, 153)

    norm_flat = normalize_sequence(flat)
    assert norm_flat.shape == (30, 153)

    # Check unflattening matches
    unflat = unflatten_features(norm_flat)
    left_norm = unflat[0, BODY_LEFT_SHOULDER_IDX]
    right_norm = unflat[0, BODY_RIGHT_SHOULDER_IDX]
    dist = np.linalg.norm(left_norm - right_norm)
    assert pytest.approx(dist, abs=1e-6) == 1.0


def test_extract_landmarks_from_dict():
    """extract_landmarks correctly pulls coordinates from mock holistic results."""
    mock_results = {
        "pose_landmarks": [
            [i * 0.1, i * 0.2, i * 0.3] for i in range(33)
        ],
        "left_hand_landmarks": [
            [0.1 * i, 0.2, 0.3] for i in range(21)
        ],
        "right_hand_landmarks": [
            [0.5, 0.1 * i, 0.4] for i in range(21)
        ],
    }

    coords = extract_landmarks(mock_results)
    assert coords.shape == (51, 3)

    # Check nose (pose idx 0)
    np.testing.assert_allclose(coords[0], [0.0, 0.0, 0.0])
    # Check left shoulder (BODY_LEFT_SHOULDER_IDX is pose idx 11)
    np.testing.assert_allclose(coords[BODY_LEFT_SHOULDER_IDX], [1.1, 2.2, 3.3])
    # Check left hand first point (coords idx 9)
    np.testing.assert_allclose(coords[9], [0.0, 0.2, 0.3])
    # Check right hand first point (coords idx 30)
    np.testing.assert_allclose(coords[30], [0.5, 0.0, 0.4])
