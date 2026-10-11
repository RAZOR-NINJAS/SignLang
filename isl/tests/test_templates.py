"""Tests for ISL static word geometric templates and generation."""

import numpy as np
import pytest

from isl.config import EXPECTED_HANDS, FEATURE_DIM, STATIC_WORDS
from isl.features import extract_features
from isl.templates import generate_word_landmarks


def test_static_words_all_have_templates():
    for word in STATIC_WORDS:
        lms, hnds = generate_word_landmarks(word)
        exp_hands = EXPECTED_HANDS[word]
        assert len(lms) == exp_hands, f"{word} expected {exp_hands} hands, got {len(lms)}"
        assert len(hnds) == exp_hands, f"{word} expected {exp_hands} handedness entries, got {len(hnds)}"
        for hand_pts in lms:
            assert hand_pts.shape == (21, 3), f"{word} hand landmark shape is {hand_pts.shape}, expected (21, 3)"
            assert not np.isnan(hand_pts).any(), f"{word} contains NaNs"
            assert not np.isinf(hand_pts).any(), f"{word} contains Infs"


def test_static_words_produce_valid_features():
    for word in STATIC_WORDS:
        lms, hnds = generate_word_landmarks(word)
        feat = extract_features(lms, hnds)
        assert feat.shape == (FEATURE_DIM,), f"{word} feature shape is {feat.shape}, expected ({FEATURE_DIM},)"
        assert not np.isnan(feat).any(), f"{word} feature contains NaNs"
        assert not np.isinf(feat).any(), f"{word} feature contains Infs"


def test_unknown_word_template_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown static word template"):
        generate_word_landmarks("UNKNOWN_WORD_XYZ")
