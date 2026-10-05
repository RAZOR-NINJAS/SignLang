"""Landmark normalization for ASL Words recognition.

Normalizes holistic landmarks relative to the body's shoulder reference points:
- Origin is set to the midpoint between left shoulder and right shoulder.
- Scale is normalized by the Euclidean distance between left and right shoulder.

This guarantees:
1. Translation invariance: subject moving around the camera frame does not change features.
2. Scale invariance: distance from camera does not affect feature representation.
3. Shoulder distance invariant: normalized shoulder distance is always identically 1.0.
4. Shoulder midpoint invariant: normalized shoulder midpoint is always identically (0, 0, 0).
"""

from typing import Any, Optional, Tuple, Union
import numpy as np

from .config import (
    BODY_REFERENCE_INDICES,
    COORDS_PER_LANDMARK,
    FEATURE_DIM,
    LEFT_SHOULDER,
    NUM_HAND_LANDMARKS,
    RIGHT_SHOULDER,
    TOTAL_LANDMARKS,
)

# In the extracted 9-point body landmarks:
# 0: NOSE, 1: LEFT_SHOULDER, 2: RIGHT_SHOULDER, 3: LEFT_ELBOW, 4: RIGHT_ELBOW,
# 5: LEFT_WRIST, 6: RIGHT_WRIST, 7: LEFT_HIP, 8: RIGHT_HIP
BODY_LEFT_SHOULDER_IDX = 1
BODY_RIGHT_SHOULDER_IDX = 2

EPSILON = 1e-6


def extract_landmarks(results: Any) -> np.ndarray:
    """Extract a 51x3 array of landmark coordinates from MediaPipe Holistic results.

    Layout of the 51 landmarks:
      0..8:   9 pose reference landmarks (nose, shoulders, elbows, wrists, hips)
      9..29:  21 left hand landmarks
      30..50: 21 right hand landmarks

    Args:
        results: MediaPipe Holistic process() result object or dict.

    Returns:
        np.ndarray of shape (51, 3) and dtype float32.
    """
    coords = np.zeros((TOTAL_LANDMARKS, COORDS_PER_LANDMARK), dtype=np.float32)

    # 1. Pose landmarks (9 reference points)
    pose_lms = getattr(results, "pose_landmarks", None)
    if isinstance(results, dict):
        pose_lms = results.get("pose_landmarks", pose_lms)

    if pose_lms is not None:
        lm_list = getattr(pose_lms, "landmark", pose_lms)
        for out_idx, pose_idx in enumerate(BODY_REFERENCE_INDICES):
            if pose_idx < len(lm_list):
                lm = lm_list[pose_idx]
                coords[out_idx, 0] = getattr(lm, "x", lm[0] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
                coords[out_idx, 1] = getattr(lm, "y", lm[1] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
                coords[out_idx, 2] = getattr(lm, "z", lm[2] if isinstance(lm, (list, tuple, np.ndarray)) and len(lm) > 2 else 0.0)

    # 2. Left hand landmarks (21 points)
    lh_lms = getattr(results, "left_hand_landmarks", None)
    if isinstance(results, dict):
        lh_lms = results.get("left_hand_landmarks", lh_lms)

    if lh_lms is not None:
        lm_list = getattr(lh_lms, "landmark", lh_lms)
        for i in range(min(NUM_HAND_LANDMARKS, len(lm_list))):
            lm = lm_list[i]
            coords[9 + i, 0] = getattr(lm, "x", lm[0] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
            coords[9 + i, 1] = getattr(lm, "y", lm[1] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
            coords[9 + i, 2] = getattr(lm, "z", lm[2] if isinstance(lm, (list, tuple, np.ndarray)) and len(lm) > 2 else 0.0)

    # 3. Right hand landmarks (21 points)
    rh_lms = getattr(results, "right_hand_landmarks", None)
    if isinstance(results, dict):
        rh_lms = results.get("right_hand_landmarks", rh_lms)

    if rh_lms is not None:
        lm_list = getattr(rh_lms, "landmark", rh_lms)
        for i in range(min(NUM_HAND_LANDMARKS, len(lm_list))):
            lm = lm_list[i]
            coords[30 + i, 0] = getattr(lm, "x", lm[0] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
            coords[30 + i, 1] = getattr(lm, "y", lm[1] if isinstance(lm, (list, tuple, np.ndarray)) else 0.0)
            coords[30 + i, 2] = getattr(lm, "z", lm[2] if isinstance(lm, (list, tuple, np.ndarray)) and len(lm) > 2 else 0.0)

    return coords


def normalize_landmarks(
    landmarks: np.ndarray,
    left_shoulder_idx: int = BODY_LEFT_SHOULDER_IDX,
    right_shoulder_idx: int = BODY_RIGHT_SHOULDER_IDX,
) -> np.ndarray:
    """Normalize a single frame of landmarks relative to shoulders.

    Sets the origin at the shoulder midpoint and scales by shoulder distance.

    Args:
        landmarks: Array of shape (N, C) where C >= 2 (typically (51, 3)).
        left_shoulder_idx: Index of left shoulder in landmarks.
        right_shoulder_idx: Index of right shoulder in landmarks.

    Returns:
        Normalized array of same shape (N, C), dtype float32.
    """
    pts = np.asarray(landmarks, dtype=np.float32).copy()
    if pts.ndim != 2:
        raise ValueError(f"Expected 2D array (N, C), got shape {pts.shape}")

    left_sh = pts[left_shoulder_idx]
    right_sh = pts[right_shoulder_idx]

    midpoint = (left_sh + right_sh) / 2.0
    dist = float(np.linalg.norm(left_sh - right_sh))

    if dist < EPSILON:
        dist = 1.0

    return (pts - midpoint) / dist


def normalize_sequence(
    sequence: np.ndarray,
    left_shoulder_idx: int = BODY_LEFT_SHOULDER_IDX,
    right_shoulder_idx: int = BODY_RIGHT_SHOULDER_IDX,
    per_frame: bool = True,
) -> np.ndarray:
    """Normalize a sequence of landmark frames.

    Supports:
    - 3D sequence: shape (T, N, C) e.g. (30, 51, 3)
    - 2D sequence: shape (T, D) e.g. (30, 153)

    Args:
        sequence: Array of shape (T, N, C) or (T, D).
        left_shoulder_idx: Index of left shoulder point.
        right_shoulder_idx: Index of right shoulder point.
        per_frame: If True, normalizes each frame by its own shoulders.
            If False, normalizes using the median shoulder position/distance
            across valid frames (smoother under frame-to-frame jitter).

    Returns:
        Normalized sequence with the same shape as input.
    """
    seq = np.asarray(sequence, dtype=np.float32).copy()
    was_flattened = False
    orig_shape = seq.shape

    if seq.ndim == 2:
        # Flattened sequence (T, FEATURE_DIM)
        was_flattened = True
        T, D = seq.shape
        if D % COORDS_PER_LANDMARK != 0:
            raise ValueError(f"Feature dimension {D} not divisible by {COORDS_PER_LANDMARK}")
        N = D // COORDS_PER_LANDMARK
        seq = seq.reshape(T, N, COORDS_PER_LANDMARK)
    elif seq.ndim == 3:
        T, N, C = seq.shape
    else:
        raise ValueError(f"Expected 2D or 3D sequence, got ndim={seq.ndim}")

    if per_frame:
        norm_seq = np.empty_like(seq)
        for t in range(T):
            norm_seq[t] = normalize_landmarks(
                seq[t],
                left_shoulder_idx=left_shoulder_idx,
                right_shoulder_idx=right_shoulder_idx,
            )
    else:
        # Median reference normalization
        left_shs = seq[:, left_shoulder_idx]
        right_shs = seq[:, right_shoulder_idx]
        dists = np.linalg.norm(left_shs - right_shs, axis=-1)
        valid = dists > EPSILON

        if np.any(valid):
            ref_midpoint = np.median((left_shs[valid] + right_shs[valid]) / 2.0, axis=0)
            ref_dist = float(np.median(dists[valid]))
        else:
            ref_midpoint = np.zeros(seq.shape[-1], dtype=np.float32)
            ref_dist = 1.0

        if ref_dist < EPSILON:
            ref_dist = 1.0

        norm_seq = (seq - ref_midpoint) / ref_dist

    if was_flattened:
        return norm_seq.reshape(orig_shape)
    return norm_seq


def flatten_features(landmarks: np.ndarray) -> np.ndarray:
    """Flatten (N, C) or (T, N, C) landmarks into 1D (D,) or 2D (T, D) feature array."""
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.ndim == 2:
        return arr.reshape(-1)
    elif arr.ndim == 3:
        T = arr.shape[0]
        return arr.reshape(T, -1)
    raise ValueError(f"Unexpected shape for flatten_features: {arr.shape}")


def unflatten_features(features: np.ndarray, num_landmarks: int = TOTAL_LANDMARKS) -> np.ndarray:
    """Unflatten (D,) or (T, D) back to (N, C) or (T, N, C)."""
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(num_landmarks, COORDS_PER_LANDMARK)
    elif arr.ndim == 2:
        T = arr.shape[0]
        return arr.reshape(T, num_landmarks, COORDS_PER_LANDMARK)
    raise ValueError(f"Unexpected shape for unflatten_features: {arr.shape}")
