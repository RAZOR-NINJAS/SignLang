"""DTW (Dynamic Time Warping) and k-Nearest Neighbors classifier for ASL words.

Uses pure NumPy vectorized cost matrices and fast dynamic programming for DTW.
Provides scikit-learn compatible classifier with confidence thresholding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .config import CONFIDENCE_THRESHOLD, KNN_NEIGHBORS, MODEL_DIR


def dtw_cost_matrix(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Compute pairwise frame Euclidean distances between two sequences.

    Args:
        s1: Array of shape (N, D).
        s2: Array of shape (M, D).

    Returns:
        np.ndarray of shape (N, M) containing Euclidean distances.
    """
    s1 = np.asarray(s1, dtype=np.float32)
    s2 = np.asarray(s2, dtype=np.float32)

    # Flatten if passed as 3D (T, num_landmarks, coords)
    if s1.ndim == 3:
        s1 = s1.reshape(s1.shape[0], -1)
    if s2.ndim == 3:
        s2 = s2.reshape(s2.shape[0], -1)

    diff = s1[:, None, :] - s2[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def dtw_distance(
    s1: np.ndarray,
    s2: np.ndarray,
    window: Optional[int] = None,
) -> float:
    """Compute normalized Dynamic Time Warping distance between two sequences.

    Args:
        s1: Array of shape (N, D).
        s2: Array of shape (M, D).
        window: Optional Sakoe-Chiba band width. If None, full warping window is used.

    Returns:
        Normalized DTW distance float >= 0.0.
    """
    cost = dtw_cost_matrix(s1, s2)
    N, M = cost.shape

    if N == 0 or M == 0:
        return float("inf")

    prev = np.full(M + 1, np.inf, dtype=np.float32)
    curr = np.full(M + 1, np.inf, dtype=np.float32)
    prev[0] = 0.0

    w = max(window, abs(N - M)) if window is not None else max(N, M)

    for i in range(1, N + 1):
        curr.fill(np.inf)
        j_start = max(1, i - w)
        j_end = min(M + 1, i + w + 1)
        cost_i = cost[i - 1]
        for j in range(j_start, j_end):
            curr[j] = cost_i[j - 1] + min(
                prev[j],      # insertion
                curr[j - 1],  # deletion
                prev[j - 1],  # match
            )
        prev, curr = curr, prev

    return float(prev[M] / (N + M))


def dtw_distance_matrix(
    queries: np.ndarray,
    targets: np.ndarray,
    window: Optional[int] = None,
) -> np.ndarray:
    """Compute pairwise DTW distances between a set of queries and targets.

    Args:
        queries: Array of shape (Q, T_q, D) or (T_q, D) for single query.
        targets: Array of shape (K, T_t, D).
        window: Optional Sakoe-Chiba band width.

    Returns:
        np.ndarray of shape (Q, K) or (K,) for single query.
    """
    is_single = (queries.ndim == 2) or (queries.ndim == 3 and queries.shape[0] == 1)
    if queries.ndim == 2:
        queries = queries[None, ...]

    Q = queries.shape[0]
    K = targets.shape[0]
    dist_mat = np.empty((Q, K), dtype=np.float32)

    for q in range(Q):
        for k in range(K):
            dist_mat[q, k] = dtw_distance(queries[q], targets[k], window=window)

    if is_single:
        return dist_mat[0]
    return dist_mat


class DTWKNNClassifier(BaseEstimator, ClassifierMixin):
    """k-Nearest Neighbors classifier using Dynamic Time Warping distance.

    Features:
    - Pure NumPy DTW distance computation
    - Inverse-distance weighting with match quality penalization
    - Calibrated confidence thresholding: returns None or "UNKNOWN" when below threshold
    - Full scikit-learn estimator interface (fit, predict, predict_proba, score)
    - Persistence via save() and load()
    """

    def __init__(
        self,
        n_neighbors: int = KNN_NEIGHBORS,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        window: Optional[int] = None,
        distance_scale: float = 2.0,
    ):
        self.n_neighbors = n_neighbors
        self.confidence_threshold = confidence_threshold
        self.window = window
        self.distance_scale = distance_scale

        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DTWKNNClassifier":
        """Fit the classifier using training landmark sequences.

        Args:
            X: Array of shape (N_samples, T_frames, D_features) or (N_samples, T, L, C).
            y: Array of shape (N_samples,) containing class labels.
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 4:
            N, T, L, C = X_arr.shape
            X_arr = X_arr.reshape(N, T, L * C)
        elif X_arr.ndim != 3:
            raise ValueError(f"Expected 3D or 4D X array, got shape {X_arr.shape}")

        y_arr = np.asarray(y)
        if len(X_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: len(X)={len(X_arr)}, len(y)={len(y_arr)}")
        if len(X_arr) == 0:
            raise ValueError("Cannot fit classifier with empty dataset.")

        self.X_ = X_arr
        self.y_ = y_arr
        self.classes_ = np.unique(y_arr)
        return self

    def _compute_query_distances(self, query: np.ndarray) -> np.ndarray:
        """Compute DTW distances from query to all stored samples."""
        if self.X_ is None or self.y_ is None:
            raise RuntimeError("Classifier has not been fitted yet.")

        q = np.asarray(query, dtype=np.float32)
        if q.ndim == 3:
            q = q.reshape(q.shape[0], -1)

        N = len(self.X_)
        distances = np.empty(N, dtype=np.float32)
        for i in range(N):
            distances[i] = dtw_distance(q, self.X_[i], window=self.window)
        return distances

    def predict_single(
        self,
        query: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Predict class and confidence for a single sequence.

        Args:
            query: Sequence of shape (T, D) or (T, L, C).
            threshold: Confidence threshold. If None, self.confidence_threshold is used.

        Returns:
            Tuple of:
            - predicted_label (str or None if below threshold)
            - confidence (float in [0, 1])
            - details (dict with voting probabilities, top-k distances, raw label)
        """
        thresh = self.confidence_threshold if threshold is None else threshold
        q_arr = np.asarray(query, dtype=np.float32)
        if len(q_arr) == 0:
            return None, 0.0, {
                "raw_label": None,
                "vote_prob": 0.0,
                "best_dist": float("inf"),
                "quality": 0.0,
                "top_distances": [],
                "top_labels": [],
                "class_probs": {},
            }

        distances = self._compute_query_distances(query)

        k = min(self.n_neighbors, len(distances))
        nn_indices = np.argsort(distances)[:k]
        top_distances = distances[nn_indices]
        top_labels = self.y_[nn_indices]

        # Inverse distance weights
        weights = 1.0 / (top_distances + 1e-4)
        total_weight = float(np.sum(weights))

        if total_weight <= 0.0 or not np.isfinite(total_weight):
            return None, 0.0, {
                "raw_label": None,
                "vote_prob": 0.0,
                "best_dist": float("inf"),
                "quality": 0.0,
                "top_distances": top_distances.tolist(),
                "top_labels": top_labels.tolist(),
                "class_probs": {},
            }

        # Class voting probabilities
        class_probs: Dict[str, float] = {}
        for idx, lbl in enumerate(top_labels):
            class_probs[lbl] = class_probs.get(lbl, 0.0) + float(weights[idx] / total_weight)

        # Winning class
        best_label = max(class_probs, key=class_probs.get)
        vote_prob = class_probs[best_label]

        # Quality factor based on nearest neighbor distance
        best_dist = float(top_distances[0])
        quality = 1.0 / (1.0 + (best_dist / self.distance_scale))

        confidence = float(vote_prob * quality)

        details = {
            "raw_label": best_label,
            "vote_prob": vote_prob,
            "best_dist": best_dist,
            "quality": quality,
            "top_distances": top_distances.tolist(),
            "top_labels": top_labels.tolist(),
            "class_probs": class_probs,
        }

        if confidence >= thresh:
            return best_label, confidence, details
        return None, confidence, details

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probability matrix for batch of sequences.

        Args:
            X: Array of shape (N_samples, T, D) or (N_samples, T, L, C).

        Returns:
            np.ndarray of shape (N_samples, n_classes) with class probabilities.
        """
        if self.classes_ is None:
            raise RuntimeError("Classifier has not been fitted yet.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 4:
            N, T, L, C = X_arr.shape
            X_arr = X_arr.reshape(N, T, L * C)

        n_samples = len(X_arr)
        n_classes = len(self.classes_)
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        probs = np.zeros((n_samples, n_classes), dtype=np.float32)

        for i in range(n_samples):
            _, _, details = self.predict_single(X_arr[i], threshold=0.0)
            for cls, p in details["class_probs"].items():
                if cls in class_to_idx:
                    probs[i, class_to_idx[cls]] = p

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for batch of sequences.

        Labels with confidence below threshold are labeled as "UNKNOWN".
        """
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 4:
            N, T, L, C = X_arr.shape
            X_arr = X_arr.reshape(N, T, L * C)

        preds = []
        for i in range(len(X_arr)):
            lbl, _, _ = self.predict_single(X_arr[i])
            preds.append(lbl if lbl is not None else "UNKNOWN")
        return np.array(preds, dtype=object)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy on test set (ignoring UNKNOWN / threshold rejects as misses)."""
        preds = self.predict(X)
        y_arr = np.asarray(y)
        return float(np.mean(preds == y_arr))

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted classifier model to an NPZ archive."""
        if self.X_ is None or self.y_ is None or self.classes_ is None:
            raise RuntimeError("Cannot save unfitted classifier.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            X=self.X_,
            y=self.y_,
            classes=self.classes_,
            n_neighbors=self.n_neighbors,
            confidence_threshold=self.confidence_threshold,
            distance_scale=self.distance_scale,
            window=self.window if self.window is not None else -1,
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "DTWKNNClassifier":
        """Load fitted classifier from an NPZ archive."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        data = np.load(path, allow_pickle=True)
        window_val = int(data["window"])
        window = window_val if window_val >= 0 else None

        clf = cls(
            n_neighbors=int(data["n_neighbors"]),
            confidence_threshold=float(data["confidence_threshold"]),
            window=window,
            distance_scale=float(data["distance_scale"]),
        )
        clf.X_ = data["X"]
        clf.y_ = data["y"]
        clf.classes_ = data["classes"]
        return clf
