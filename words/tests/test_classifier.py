"""Unit tests for DTW calculation and kNN classifier."""

import numpy as np
import pytest

from words.classifier import (
    DTWKNNClassifier,
    dtw_cost_matrix,
    dtw_distance,
    dtw_distance_matrix,
)


def _generate_synthetic_wave(T=30, D=153, freq=1.0, phase=0.0):
    """Generate smooth sinusoidal trajectory sequence."""
    t = np.linspace(0, 2 * np.pi * freq, T) + phase
    base = np.sin(t)[:, None]  # (T, 1)
    # Project to D dimensions
    weights = np.linspace(0.5, 1.5, D)[None, :]
    return (base * weights).astype(np.float32)


def test_dtw_cost_matrix_shape():
    """Cost matrix must have shape (N, M)."""
    s1 = np.zeros((10, 5), dtype=np.float32)
    s2 = np.ones((15, 5), dtype=np.float32)
    cost = dtw_cost_matrix(s1, s2)
    assert cost.shape == (10, 15)
    # Distance between 0 and 1 in 5 dims is sqrt(5)
    np.testing.assert_allclose(cost, np.sqrt(5.0), atol=1e-5)


def test_dtw_distance_identity():
    """DTW distance of sequence to itself must be exactly 0.0."""
    s1 = _generate_synthetic_wave(T=20, D=10, freq=1.0)
    dist = dtw_distance(s1, s1)
    assert pytest.approx(dist, abs=1e-6) == 0.0


def test_dtw_distance_symmetry():
    """DTW distance must be symmetric: d(A, B) == d(B, A)."""
    s1 = _generate_synthetic_wave(T=20, D=10, freq=1.0)
    s2 = _generate_synthetic_wave(T=25, D=10, freq=1.5)

    d12 = dtw_distance(s1, s2)
    d21 = dtw_distance(s2, s1)
    assert pytest.approx(d12, abs=1e-5) == d21


def test_dtw_distance_non_negative():
    """DTW distance must be >= 0."""
    rng = np.random.default_rng(123)
    s1 = rng.standard_normal((30, 20)).astype(np.float32)
    s2 = rng.standard_normal((30, 20)).astype(np.float32)
    assert dtw_distance(s1, s2) >= 0.0


def test_dtw_time_warping_invariance():
    """DTW should have much lower distance under time-stretching than Euclidean."""
    base = _generate_synthetic_wave(T=15, D=10, freq=1.0)
    # Stretched by duplicating each frame
    stretched = np.repeat(base, 2, axis=0)  # T=30

    dtw_dist = dtw_distance(base, stretched)
    # Because every point in stretched matches a point in base exactly, DTW distance is zero!
    assert pytest.approx(dtw_dist, abs=1e-5) == 0.0


def test_dtw_distance_matrix():
    """Batch distance matrix computes all pairwise distances."""
    s1 = _generate_synthetic_wave(T=20, D=10, freq=1.0)
    s2 = _generate_synthetic_wave(T=20, D=10, freq=2.0)
    queries = np.stack([s1, s2])
    targets = np.stack([s1, s2])

    mat = dtw_distance_matrix(queries, targets)
    assert mat.shape == (2, 2)
    assert pytest.approx(mat[0, 0], abs=1e-6) == 0.0
    assert pytest.approx(mat[1, 1], abs=1e-6) == 0.0
    assert mat[0, 1] > 0.0


def test_classifier_fit_and_perfect_prediction():
    """Classifier should recognize training templates with high confidence."""
    s_hello = _generate_synthetic_wave(T=30, D=153, freq=1.0)
    s_thanks = _generate_synthetic_wave(T=30, D=153, freq=3.0)
    s_yes = _generate_synthetic_wave(T=30, D=153, freq=5.0)

    X = np.stack([s_hello, s_thanks, s_yes])
    y = np.array(["HELLO", "THANKYOU", "YES"])

    clf = DTWKNNClassifier(n_neighbors=1, confidence_threshold=0.6)
    clf.fit(X, y)

    # Test exact queries
    label, conf, details = clf.predict_single(s_hello)
    assert label == "HELLO"
    assert conf > 0.8
    assert details["best_dist"] == pytest.approx(0.0, abs=1e-5)

    label, conf, _ = clf.predict_single(s_thanks)
    assert label == "THANKYOU"
    assert conf > 0.8

    label, conf, _ = clf.predict_single(s_yes)
    assert label == "YES"
    assert conf > 0.8


def test_classifier_confidence_threshold_rejection():
    """Dissimilar or noisy sequence should be rejected when below threshold."""
    s_hello = _generate_synthetic_wave(T=30, D=153, freq=1.0)
    s_thanks = _generate_synthetic_wave(T=30, D=153, freq=2.0)

    X = np.stack([s_hello, s_thanks])
    y = np.array(["HELLO", "THANKYOU"])

    clf = DTWKNNClassifier(n_neighbors=1, confidence_threshold=0.7)
    clf.fit(X, y)

    # Very noisy random query that does not match templates
    rng = np.random.default_rng(999)
    noise_query = rng.uniform(-10.0, 10.0, size=(30, 153)).astype(np.float32)

    label, conf, details = clf.predict_single(noise_query)
    # Should be rejected (label is None) and confidence is low
    assert label is None
    assert conf < 0.7


def test_classifier_save_load(tmp_path):
    """Saving and loading classifier preserves predictions."""
    s1 = _generate_synthetic_wave(T=10, D=20, freq=1.0)
    s2 = _generate_synthetic_wave(T=10, D=20, freq=2.0)
    X = np.stack([s1, s2])
    y = np.array(["WORD1", "WORD2"])

    clf = DTWKNNClassifier(n_neighbors=1, confidence_threshold=0.5)
    clf.fit(X, y)

    save_path = tmp_path / "model.npz"
    clf.save(save_path)
    assert save_path.exists()

    loaded = DTWKNNClassifier.load(save_path)
    assert loaded.n_neighbors == 1
    assert loaded.confidence_threshold == 0.5

    lbl1, conf1, _ = loaded.predict_single(s1)
    assert lbl1 == "WORD1"
    assert conf1 > 0.8
