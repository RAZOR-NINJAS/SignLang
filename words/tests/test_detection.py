"""Regression tests for the words detection pipeline.

Each test here pins down a specific defect that previously made live recognition
report nothing:

- confidence was computed from a hardcoded distance scale, so genuine webcam
  performances (farther from every reference than synthetic templates are) were
  always scored below threshold;
- motion energy averaged over 44 landmarks, diluting wrist-driven motion ~22x so
  the signing state was never entered;
- a dropped hand arrived as exact zeros, which normalized into a large fixed
  artifact and polluted both motion energy and the distance metric.
"""

import numpy as np
import pytest

from words.classifier import DTWKNNClassifier, dtw_distance
from words.config import (
    CONFIDENCE_THRESHOLD,
    ENERGY_ACTIVE_THRESHOLD,
    MAX_ACCEPT_DISTANCE_RATIO,
    NUM_HAND_LANDMARKS,
    SEQUENCE_LENGTH,
    TOTAL_LANDMARKS,
)
from words.live import compute_motion_energy, hand_centroids
from words.normalize import extract_landmarks_masked, impute_invalid, normalize_sequence


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_class(seed, T=SEQUENCE_LENGTH, D=153):
    """A deterministic stand-in for one sign's landmark sequences."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, size=(T, D)).astype(np.float32)
    variants = np.stack([base + rng.normal(0.0, 0.02, size=base.shape) for _ in range(4)])
    return variants.astype(np.float32)


def _training_set():
    classes = ["HELLO", "THANKYOU", "YES", "NO"]
    xs, ys = [], []
    for i, name in enumerate(classes):
        for seq in _make_class(100 + i):
            xs.append(seq)
            ys.append(name)
    return np.stack(xs), np.array(ys)


def _frame(hand_dx=0.0, hand_present=True):
    """A plausible single frame of 51 landmarks, optionally with a moved hand."""
    f = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float32)
    f[0] = (0.50, 0.20, 0.0)   # nose
    f[1] = (0.45, 0.40, 0.0)   # left shoulder
    f[2] = (0.55, 0.40, 0.0)   # right shoulder
    f[3] = (0.42, 0.55, 0.0)   # left elbow
    f[4] = (0.58, 0.55, 0.0)   # right elbow
    f[5] = (0.40, 0.70, 0.0)   # left wrist
    f[6] = (0.60, 0.70, 0.0)   # right wrist
    f[7] = (0.47, 0.80, 0.0)   # left hip
    f[8] = (0.53, 0.80, 0.0)   # right hip
    for i in range(NUM_HAND_LANDMARKS):
        f[9 + i] = (0.40 + i * 0.001, 0.70 + i * 0.002, 0.0)
        f[30 + i] = (0.60 + i * 0.001, 0.70 + i * 0.002, 0.0)
    if hand_present:
        f[30:, 0] += hand_dx
        f[6, 0] += hand_dx
    return f


# --------------------------------------------------------------------------
# Confidence calibration
# --------------------------------------------------------------------------

def _offset_for_target_distance(clf, base, direction, target_distance):
    """Find a scalar t such that dtw(base + t*direction, refs) ~= target_distance.

    DTW distance is close to linear in the offset for a fixed direction, but
    bisecting keeps the test independent of that assumption.
    """
    direction = direction / max(np.linalg.norm(direction), 1e-9)

    def dist_at(t):
        q = base + np.float32(t) * direction.astype(np.float32)
        return float(clf._compute_query_distances(q).min())

    lo, hi = 0.0, 1.0
    while dist_at(hi) < target_distance and hi < 1e6:
        hi *= 2.0
    if dist_at(hi) < target_distance:
        return None
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if dist_at(mid) < target_distance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _wide_spread_setup():
    """Tight training clusters plus a query drawn with much wider spread.

    This mirrors the shipped dataset: synthetic templates sit ~0.05 from each
    other, while a genuine performance of the same sign sits 1.2-1.9 away and is
    still nearest to the correct class. Reproducing that relationship is the
    point of the regression below.
    """
    rng = np.random.default_rng(2024)
    classes = ["HELLO", "THANKYOU", "YES", "NO"]

    means = {}
    for name in classes:
        means[name] = rng.normal(0.0, 1.0, size=(SEQUENCE_LENGTH, 153)).astype(np.float32)

    xs, ys = [], []
    for name in classes:
        for _ in range(4):
            xs.append(means[name] + rng.normal(0.0, 0.02, size=means[name].shape))
            ys.append(name)
    X = np.stack(xs).astype(np.float32)
    y = np.array(ys)

    # A query for HELLO, displaced far enough to sit beyond the ~1.1 cutoff the
    # old formula imposed, but not so far that its class identity is lost.
    query = means["HELLO"] + rng.normal(0.0, 0.35, size=means["HELLO"].shape)
    return X, y, query.astype(np.float32), "HELLO"


def test_confidence_does_not_depend_on_absolute_distance_scale():
    """A clearly-nearest query must be accepted even when it is far from all references.

    This is the core regression, and it is deliberately run at the threshold the
    project used to ship (0.65) rather than the current default, so it pins the
    original failure mode exactly.

    The previous formula was ``vote_prob * 1/(1 + best_dist/2.0)``. For the query
    built below the nearest reference is 2.22 away, which that formula scores at
    0.47 - below 0.65, so it was refused even though the correct class won by a
    factor of three. That is why live recognition reported nothing: real webcam
    sequences sit well outside the distance the formula tolerated. Separation
    between the winning and runner-up class is scale-free and scores the same
    query at 0.75, so it is accepted.
    """
    X, y, query, expected = _wide_spread_setup()
    legacy_threshold = 0.65
    clf = DTWKNNClassifier(n_neighbors=3, confidence_threshold=legacy_threshold).fit(X, y)

    label, conf, details = clf.predict_single(query)

    # Reproduce the historical formula to document what used to happen.
    legacy_conf = 1.0 / (1.0 + details["best_dist"] / 2.0)
    assert legacy_conf < legacy_threshold, (
        "test no longer reproduces the original bug: the old formula would have "
        f"scored this query {legacy_conf:.2f}, above the threshold"
    )
    assert details["best_dist"] > 1.1, (
        f"query must exceed the old formula's tolerance, got {details['best_dist']:.3f}"
    )

    assert label == expected, "nearest class must win"
    assert conf >= legacy_threshold, (
        f"conf {conf:.3f} below the legacy threshold {legacy_threshold}: "
        "confidence is still being crushed by absolute distance"
    )


def test_default_distance_gate_matches_configured_value():
    """The classifier default must be the calibrated config value, not a stale 1.5.

    A too-tight default silently reinstates the original failure for every caller
    that does not pass the ratio explicitly, which includes live mode.
    """
    clf = DTWKNNClassifier()
    assert clf.max_accept_distance_ratio == MAX_ACCEPT_DISTANCE_RATIO
    assert clf.max_accept_distance_ratio > 1.5, (
        "a ratio near 1.5 rejects real-webcam-scale input, which is the bug being fixed"
    )


def test_input_unlike_anything_is_rejected_by_distance_gate():
    """Wildly distant input is refused outright, with a stated reason."""
    X, y = _training_set()
    clf = DTWKNNClassifier(n_neighbors=3, confidence_threshold=CONFIDENCE_THRESHOLD).fit(X, y)

    rng = np.random.default_rng(5)
    nonsense = rng.normal(0.0, 1.0, size=(SEQUENCE_LENGTH, 153)).astype(np.float32) * np.float32(400.0)

    label, conf, details = clf.predict_single(nonsense)

    assert label is None
    assert "too far" in (details["reject_reason"] or "")
    assert details["best_dist"] > MAX_ACCEPT_DISTANCE_RATIO * clf.inter_scale_


def test_calibration_scales_come_from_training_data():
    """intra/inter scales are measured, not hardcoded."""
    X, y = _training_set()
    clf = DTWKNNClassifier().fit(X, y)

    assert clf.intra_scale_ is not None and clf.intra_scale_ > 0
    assert clf.inter_scale_ is not None and clf.inter_scale_ > 0
    # Classes are well separated here, so different-class distance must exceed
    # same-class distance.
    assert clf.inter_scale_ > clf.intra_scale_

    # Calibration must survive a round trip through disk.
    clf.save("/tmp/_signlang_calib_test.npz")
    loaded = DTWKNNClassifier.load("/tmp/_signlang_calib_test.npz")
    assert loaded.intra_scale_ == pytest.approx(clf.intra_scale_)
    assert loaded.inter_scale_ == pytest.approx(clf.inter_scale_)


def test_random_noise_is_rejected():
    """Unstructured input must not be forced into a class."""
    X, y = _training_set()
    clf = DTWKNNClassifier(n_neighbors=3, confidence_threshold=CONFIDENCE_THRESHOLD).fit(X, y)

    rng = np.random.default_rng(4)
    for _ in range(5):
        noise = rng.normal(0.0, 1.0, size=(SEQUENCE_LENGTH, 153)).astype(np.float32)
        label, conf, details = clf.predict_single(noise)
        assert label is None, f"noise accepted as {label}"
        assert details["separation"] < 0.2, (
            f"noise should not separate any class, got {details['separation']}"
        )


def test_ambiguous_query_is_rejected_with_a_reason():
    """A query between two classes must be refused, and must say why."""
    X, y = _training_set()
    clf = DTWKNNClassifier(n_neighbors=3, confidence_threshold=CONFIDENCE_THRESHOLD).fit(X, y)

    midpoint = 0.5 * (X[0] + X[4])  # midway between HELLO and THANKYOU
    label, conf, details = clf.predict_single(midpoint)

    assert label is None
    assert conf < clf.confidence_threshold
    assert details["reject_reason"] is not None


def test_distances_used_for_confidence_are_class_level():
    """Confidence must compare classes, not just individual samples."""
    X, y = _training_set()
    clf = DTWKNNClassifier(n_neighbors=3, confidence_threshold=CONFIDENCE_THRESHOLD).fit(X, y)

    _, _, details = clf.predict_single(X[0])
    assert details["runner_up_dist"] >= details["best_dist"] - 1e-6
    assert details["separation"] == pytest.approx(
        (details["runner_up_dist"] - details["best_dist"]) / details["runner_up_dist"], abs=1e-5
    )


# --------------------------------------------------------------------------
# Motion energy
# --------------------------------------------------------------------------

def test_wrist_motion_is_not_diluted_below_threshold():
    """Hand movement must be able to trip the signing threshold.

    Averaging displacement over 2 wrists plus 42 hand points scaled wrist-driven
    motion down by roughly 22x, leaving the fixed ENERGY_ACTIVE_THRESHOLD
    unreachable, so the state machine never left REST.
    """
    still = _frame(hand_dx=0.0)
    moved = _frame(hand_dx=0.30)

    energy = compute_motion_energy(still, moved)

    assert energy >= ENERGY_ACTIVE_THRESHOLD, (
        f"energy {energy:.4f} below threshold {ENERGY_ACTIVE_THRESHOLD}: "
        "wrist-driven motion is being diluted"
    )


def test_static_hands_produce_no_energy():
    still = _frame(hand_dx=0.0)
    assert compute_motion_energy(still, still.copy()) == pytest.approx(0.0, abs=1e-6)


def test_one_handed_sign_triggers():
    """A sign driven by a single hand must still be detected."""
    still = _frame(hand_dx=0.0)

    # Move only the right hand, leaving the left hand perfectly still.
    moved = still.copy()
    moved[6] += [0.30, 0.0, 0.0]
    moved[30:, 0] += 0.30

    assert compute_motion_energy(still, moved) >= ENERGY_ACTIVE_THRESHOLD


def test_hand_appearing_produces_no_motion_spike():
    """MediaPipe reports a newly detected hand as a jump from zeros.

    Treating those zeros as real coordinates fabricates a large burst of motion
    and would trigger a spurious sign.
    """
    absent = _frame(hand_present=False)
    present = _frame(hand_present=True)

    energy = compute_motion_energy(absent, present)
    assert energy == pytest.approx(0.0, abs=1e-6), (
        f"hand dropout/appearance produced phantom energy {energy}"
    )


def test_motion_energy_respects_validity_masks():
    """An explicitly invalid hand must be excluded even if its coordinates differ."""
    prev = _frame(hand_dx=0.0)
    curr = _frame(hand_dx=0.30)
    mask = np.ones(TOTAL_LANDMARKS, dtype=bool)
    mask[30:] = False  # right hand untracked in the current frame
    mask[9:30] = False  # left hand untracked too

    assert compute_motion_energy(prev, curr, mask, mask) == pytest.approx(0.0, abs=1e-6)


def test_hand_centroids_reports_absence():
    """A hand with too few tracked landmarks must report None, not a centroid."""
    frame = _frame()
    frame[30:] = 0.0  # right hand fully dropped

    left, right = hand_centroids(frame)
    assert left is not None
    assert right is None


# --------------------------------------------------------------------------
# Dropout masking and imputation
# --------------------------------------------------------------------------

class _LM:
    def __init__(self, x, y, z=0.0, visibility=0.9):
        self.x, self.y, self.z = x, y, z
        self.visibility = visibility


class _LandmarkList:
    def __init__(self, landmarks):
        self.landmark = landmarks


class _Results:
    def __init__(self, pose=None, left=None, right=None):
        self.pose_landmarks = pose
        self.left_hand_landmarks = left
        self.right_hand_landmarks = right


def _pose_landmarks():
    """Full 33-point pose list with the 9 tracked indices populated."""
    pose = [None] * 33
    values = {
        0: (0.50, 0.20), 11: (0.45, 0.40), 12: (0.55, 0.40),
        13: (0.42, 0.55), 14: (0.58, 0.55), 15: (0.40, 0.70),
        16: (0.60, 0.70), 23: (0.47, 0.80), 24: (0.53, 0.80),
    }
    for idx, (x, y) in values.items():
        pose[idx] = _LM(x, y)
    return _LandmarkList(pose)


def test_mask_flags_dropped_hand():
    coords, valid = extract_landmarks_masked(_Results(pose=_pose_landmarks()))
    assert valid[9:30].sum() == 0, "absent left hand must be marked invalid"
    assert valid[:9].all(), "visible body landmarks must be marked valid"


def test_mask_flags_low_visibility_landmark():
    pose = _pose_landmarks().landmark
    pose[11] = _LM(0.45, 0.40, visibility=0.1)  # shoulder seen poorly
    _, valid = extract_landmarks_masked(_Results(pose=_LandmarkList(pose)))
    assert not valid[1], "low-visibility shoulder must be treated as untracked"


def test_imputation_anchors_dropped_hand_to_body_wrist():
    """A dropped hand must collapse onto its wrist, not onto the body centre.

    Left at zero, the hand points normalize to a fixed offset from the shoulder
    midpoint, which looks like a large position error to the distance metric.
    """
    coords, valid = extract_landmarks_masked(_Results(pose=_pose_landmarks()))
    filled = impute_invalid(coords, valid)

    assert np.allclose(filled[9], filled[5]), "dropped hand should sit at the tracked wrist"
    assert not np.any(np.all(np.isclose(filled, 0.0), axis=1)), "no landmark may remain at origin"


def test_dropout_no_longer_creates_normalization_artifact():
    """The imputed frame must not differ from a seen frame by a large offset."""
    coords, valid = extract_landmarks_masked(_Results(pose=_pose_landmarks()))

    with_hand = coords.copy()
    with_hand[9:30] = np.array([0.40, 0.72, 0.0], dtype=np.float32)

    a = normalize_sequence(impute_invalid(coords, valid)[None, ...], per_frame=True)
    b = normalize_sequence(with_hand[None, ...], per_frame=True)

    # Compare only the body landmarks, which are shared.
    np.testing.assert_allclose(a[:, :9], b[:, :9], atol=1e-5)
