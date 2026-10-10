"""Unit tests for ISL model evaluation."""

import numpy as np
import pytest

from isl.eval import _selftest, compute_metrics, group_bursts, parse_session


def test_parse_session():
    assert parse_session("session1_cam0") == "session1"
    assert parse_session("session2_net") == "session2"
    assert parse_session("custom") == "custom"


def test_group_bursts():
    y = np.array(["A"] * 20 + ["B"] * 10)
    groups = group_bursts(y, burst_size=14)
    # "A" (20 rows) splits into [0:14], [14:20] -> 2 groups
    # "B" (10 rows) splits into [20:30] -> 1 group
    assert len(groups) == 3
    assert len(groups[0]) == 14
    assert len(groups[1]) == 6
    assert len(groups[2]) == 10


def test_compute_metrics_breakdown():
    # C and SPACE are 1-handed in ISL config; A, B are 2-handed
    labels = ["A", "B", "C", "SPACE"]
    y_true = np.array(["A", "A", "B", "B", "C", "C", "SPACE", "SPACE"])
    # Predictions:
    # A: 2 correct
    # B: 1 correct, 1 wrong (predicted A)
    # C: 2 correct
    # SPACE: 1 correct, 1 wrong (predicted C)
    y_pred = np.array(["A", "A", "B", "A", "C", "C", "SPACE", "C"])

    metrics = compute_metrics(y_true, y_pred, labels)

    assert metrics["overall_accuracy"] == 6 / 8  # 75%
    # 1-handed (C, SPACE): 4 total, 3 correct -> 75%
    assert metrics["one_hand_total"] == 4
    assert metrics["one_hand_accuracy"] == 3 / 4
    # 2-handed (A, B): 4 total, 3 correct -> 75%
    assert metrics["two_hand_total"] == 4
    assert metrics["two_hand_accuracy"] == 3 / 4

    cm = metrics["confusion_matrix"]
    assert cm.shape == (4, 4)
    # B predicted as A: true B (idx 1) -> pred A (idx 0)
    assert cm[1, 0] == 1


def test_eval_selftest():
    assert _selftest() == 0
