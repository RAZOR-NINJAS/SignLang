"""Evaluation script for ASL Words recognition model.

Performs stratified cross-validation, computes precision, recall, F1-score,
confusion matrix, and classification latency.
"""

import argparse
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from .classifier import DTWKNNClassifier
from .config import CONFIDENCE_THRESHOLD, KNN_NEIGHBORS, MODEL_DIR, WORDS
from .dataset import generate_synthetic_dataset, load_dataset


def evaluate_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_neighbors: int = KNN_NEIGHBORS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run stratified cross-validation on landmark sequences.

    Args:
        X: Sequence array of shape (N, 30, 153) or (N, 30, 51, 3).
        y: Array of labels of shape (N,).
        n_splits: Number of cross-validation folds.
        n_neighbors: k in kNN classifier.
        confidence_threshold: Confidence rejection threshold.
        random_state: Random state for K-fold splitting.

    Returns:
        Dict with metrics, per-fold scores, confusion matrix, and latency.
    """
    labels = np.unique(y)
    min_count = min(np.sum(y == lbl) for lbl in labels)
    k_folds = max(2, min(n_splits, int(min_count)))

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    fold_accuracies: List[float] = []
    latencies: List[float] = []
    all_y_true: List[str] = []
    all_y_pred: List[str] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = DTWKNNClassifier(
            n_neighbors=n_neighbors,
            confidence_threshold=confidence_threshold,
        )
        clf.fit(X_train, y_train)

        fold_preds = []
        for i in range(len(X_val)):
            t0 = time.perf_counter()
            pred, conf, _ = clf.predict_single(X_val[i])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

            pred_lbl = pred if pred is not None else "UNKNOWN"
            fold_preds.append(pred_lbl)

        fold_preds = np.array(fold_preds)
        acc = float(np.mean(fold_preds == y_val))
        fold_accuracies.append(acc)

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(fold_preds.tolist())

    mean_acc = float(np.mean(fold_accuracies))
    std_acc = float(np.std(fold_accuracies))
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    all_labels_for_cm = list(labels)
    if "UNKNOWN" in all_y_pred:
        all_labels_for_cm.append("UNKNOWN")

    cm = confusion_matrix(all_y_true, all_y_pred, labels=all_labels_for_cm)
    report_dict = classification_report(
        all_y_true,
        all_y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        all_y_true,
        all_y_pred,
        labels=labels,
        zero_division=0,
    )

    return {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "fold_accuracies": fold_accuracies,
        "k_folds": k_folds,
        "mean_latency_ms": mean_latency,
        "confusion_matrix": cm,
        "cm_labels": all_labels_for_cm,
        "report_dict": report_dict,
        "report_str": report_str,
    }


def print_confusion_matrix(cm: np.ndarray, labels: List[str]) -> None:
    """Print an ASCII confusion matrix table."""
    col_w = max(7, max(len(str(lbl)) for lbl in labels) + 1)
    header = "True \\ Pred".ljust(col_w) + "".join(lbl[:6].rjust(7) for lbl in labels)
    print(header)
    print("-" * len(header))
    for i, row_lbl in enumerate(labels):
        row_str = row_lbl[: col_w - 1].ljust(col_w)
        for j in range(len(labels)):
            val = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
            row_str += str(val).rjust(7)
        print(row_str)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="words.eval",
        description="Evaluate ASL Words recognition using cross-validation.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=KNN_NEIGHBORS,
        help="k in kNN classifier (default: 3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence rejection threshold (default: 0.65).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Generate synthetic samples if dataset is empty (default: 10).",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Train on all data and save model to words/models/words_classifier.npz.",
    )

    args = parser.parse_args(argv)

    print("Loading ASL Words dataset...")
    X, y = load_dataset()
    if len(X) == 0:
        print(f"No existing samples found. Generating {args.samples} synthetic samples per word...")
        generate_synthetic_dataset(samples_per_word=args.samples)
        X, y = load_dataset()

    print(f"Loaded {len(X)} samples across {len(np.unique(y))} vocabulary words.")

    print(f"\nRunning {args.cv}-fold cross-validation (k={args.k}, threshold={args.threshold:.2f})...")
    results = evaluate_cross_validation(
        X,
        y,
        n_splits=args.cv,
        n_neighbors=args.k,
        confidence_threshold=args.threshold,
    )

    print("\n" + "=" * 60)
    print(f"  CROSS-VALIDATION RESULTS ({results['k_folds']} folds)")
    print(f"  Overall Accuracy: {results['mean_accuracy'] * 100:.2f}% ± {results['std_accuracy'] * 100:.2f}%")
    print(f"  Mean Classification Latency: {results['mean_latency_ms']:.2f} ms / sample")
    print("=" * 60)

    print("\nPer-word Classification Report:")
    print(results["report_str"])

    print("Confusion Matrix:")
    print_confusion_matrix(results["confusion_matrix"], results["cm_labels"])

    if args.save_model:
        model_path = MODEL_DIR / "words_classifier.npz"
        print(f"\nTraining on all {len(X)} samples and saving model to {model_path}...")
        clf = DTWKNNClassifier(
            n_neighbors=args.k,
            confidence_threshold=args.threshold,
        )
        clf.fit(X, y)
        clf.save(model_path)
        print("Model saved successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
