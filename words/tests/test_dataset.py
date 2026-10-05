"""Unit tests for dataset storage, shapes, and loading."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from words.config import FEATURE_DIM, SEQUENCE_LENGTH, TOTAL_LANDMARKS, WORDS
from words.dataset import (
    count_samples,
    load_dataset,
    load_word_samples,
    save_word_samples,
)


def test_save_word_samples_various_shapes(tmp_path, monkeypatch):
    """save_word_samples accepts 2D (T, D), 3D (T, N, C), 3D batch (B, T, D), and 4D (B, T, N, C)."""
    monkeypatch.setattr("words.dataset.SAMPLE_DIR", tmp_path)

    # 1. Single 2D sequence (30, 153)
    seq_2d = np.ones((SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    c1 = save_word_samples("TEST_WORD", seq_2d, append=False)
    assert c1 == 1
    loaded = load_word_samples("TEST_WORD")
    assert loaded.shape == (1, SEQUENCE_LENGTH, TOTAL_LANDMARKS, 3)

    # 2. Single 3D sequence (30, 51, 3)
    seq_3d_single = np.ones((SEQUENCE_LENGTH, TOTAL_LANDMARKS, 3), dtype=np.float32) * 2.0
    c2 = save_word_samples("TEST_WORD", seq_3d_single, append=True)
    assert c2 == 2
    loaded = load_word_samples("TEST_WORD")
    assert loaded.shape == (2, SEQUENCE_LENGTH, TOTAL_LANDMARKS, 3)

    # 3. Batch 3D flattened (3, 30, 153)
    batch_3d = np.ones((3, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32) * 3.0
    c3 = save_word_samples("TEST_WORD", batch_3d, append=True)
    assert c3 == 5

    # 4. Batch 4D (2, 30, 51, 3)
    batch_4d = np.ones((2, SEQUENCE_LENGTH, TOTAL_LANDMARKS, 3), dtype=np.float32) * 4.0
    c4 = save_word_samples("TEST_WORD", batch_4d, append=True)
    assert c4 == 7


def test_save_word_samples_invalid_shape_raises(tmp_path, monkeypatch):
    """Invalid shapes should raise ValueError."""
    monkeypatch.setattr("words.dataset.SAMPLE_DIR", tmp_path)
    with pytest.raises(ValueError):
        save_word_samples("TEST_ERR", np.zeros((15, 10), dtype=np.float32))


def test_load_dataset_empty_shape(tmp_path, monkeypatch):
    """load_dataset on an empty directory returns proper shapes for both flatten options."""
    monkeypatch.setattr("words.dataset.SAMPLE_DIR", tmp_path)
    X_flat, y_flat = load_dataset(words=["NONEXISTENT"], flatten=True)
    assert X_flat.shape == (0, SEQUENCE_LENGTH, FEATURE_DIM)
    assert len(y_flat) == 0

    X_unflat, y_unflat = load_dataset(words=["NONEXISTENT"], flatten=False)
    assert X_unflat.shape == (0, SEQUENCE_LENGTH, TOTAL_LANDMARKS, 3)
    assert len(y_unflat) == 0
