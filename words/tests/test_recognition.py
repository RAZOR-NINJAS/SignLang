"""Unit tests for simulated sequence recognition and end-to-end motion classification."""

import numpy as np
import pytest

from words.classifier import DTWKNNClassifier
from words.config import SEQUENCE_LENGTH, TOTAL_LANDMARKS, WORDS
from words.dataset import _generate_synthetic_sequence_for_word
from words.live import compute_motion_energy, resample_sequence
from words.normalize import extract_landmarks, normalize_sequence


def test_resample_sequence():
    """Resampling properly changes sequence length while preserving start, end, and shape."""
    # Sequence of length 15 with linear ramp
    t15 = np.linspace(0.0, 10.0, 15)[:, None, None]
    seq15 = np.broadcast_to(t15, (15, TOTAL_LANDMARKS, 3)).copy()

    resampled = resample_sequence(seq15, target_length=SEQUENCE_LENGTH)
    assert resampled.shape == (SEQUENCE_LENGTH, TOTAL_LANDMARKS, 3)

    # First and last frames must match
    np.testing.assert_allclose(resampled[0], seq15[0], atol=1e-5)
    np.testing.assert_allclose(resampled[-1], seq15[-1], atol=1e-5)


def test_resample_identity():
    """Resampling a sequence already of target length returns the sequence unchanged."""
    seq30 = np.ones((SEQUENCE_LENGTH, TOTAL_LANDMARKS, 3), dtype=np.float32)
    res = resample_sequence(seq30, target_length=SEQUENCE_LENGTH)
    assert res.shape == (SEQUENCE_LENGTH, TOTAL_LANDMARKS, 3)
    np.testing.assert_allclose(res, seq30)


def test_compute_motion_energy():
    """Energy calculation detects static hands vs moving hands."""
    static1 = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float32)
    static2 = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float32)
    energy_static = compute_motion_energy(static1, static2)
    assert energy_static == pytest.approx(0.0, abs=1e-6)

    # Move right wrist and fingers
    moving = static1.copy()
    moving[6] = [0.2, 0.2, 0.0]  # Right wrist moved 0.28
    moving[30:] += [0.2, 0.2, 0.0]  # Right hand moved
    energy_moving = compute_motion_energy(static1, moving)
    assert energy_moving > 0.1


def test_simulated_sequence_recognition_multiclass():
    """Train on synthetic sequences and test on unseen synthetic sequences."""
    test_words = ["HELLO", "THANKYOU", "PLEASE", "YES", "EAT"]
    rng_train = np.random.default_rng(1001)
    rng_test = np.random.default_rng(2002)

    X_train_list, y_train_list = [], []
    for w in test_words:
        for _ in range(4):
            seq = _generate_synthetic_sequence_for_word(w, rng_train)
            X_train_list.append(seq)
            y_train_list.append(w)

    X_train = np.stack(X_train_list, axis=0)
    y_train = np.array(y_train_list)

    clf = DTWKNNClassifier(n_neighbors=1, confidence_threshold=0.60)
    clf.fit(X_train, y_train)

    # Test on unseen test sequences
    correct = 0
    total = 0
    for w in test_words:
        for _ in range(3):
            test_seq = _generate_synthetic_sequence_for_word(w, rng_test)
            pred, conf, details = clf.predict_single(test_seq)
            assert conf > 0.60, f"Confidence {conf} was below threshold for {w}"
            if pred == w:
                correct += 1
            total += 1

    accuracy = correct / total
    assert accuracy >= 0.90, f"Simulated accuracy {accuracy} was below 90%"


def test_simulated_sequence_noise_rejection():
    """Unstructured random sequence must not be recognized with high confidence."""
    test_words = ["HELLO", "THANKYOU", "PLEASE"]
    rng = np.random.default_rng(42)

    X_train = np.stack([_generate_synthetic_sequence_for_word(w, rng) for w in test_words for _ in range(2)])
    y_train = np.array([w for w in test_words for _ in range(2)])

    clf = DTWKNNClassifier(n_neighbors=1, confidence_threshold=0.70)
    clf.fit(X_train, y_train)

    # Random noise sequence
    random_seq = rng.normal(0.0, 1.0, size=(SEQUENCE_LENGTH, TOTAL_LANDMARKS, 3)).astype(np.float32)
    norm_random = normalize_sequence(random_seq, per_frame=True)

    pred, conf, details = clf.predict_single(norm_random)
    # Must be rejected (pred is None) because confidence is below 0.70
    assert pred is None
    assert conf < 0.70
