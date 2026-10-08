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
from .dataset import generate_synthetic_dataset, load_dataset, load_sample_sources


def evaluate_holdout_by_source(
    X: np.ndarray,
    y: np.ndarray,
    sources: List[str],
    n_neighbors: int = KNN_NEIGHBORS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    window: Optional[int] = None,
) -> Dict[str, Any]:
    """Train on synthetic data and score only on real (non-synthetic) data.

    Cross-validation over a synthetic-dominated dataset is close to meaningless:
    a synthetic sequence is nearly a duplicate of the template it was generated
    from, so leave-one-out scores sit near 100% while live recognition fails.
    This trains on synthetic sequences only and reports how the real sequences
    fare, which is the number that actually predicts live behaviour.

    A sequence tagged anything other than "synthetic" is treated as real, so
    future webcam sources are picked up without changing this function.

    Returns:
        Dict with train/test counts, accuracy, coverage (fraction accepted),
        calibration scales, and the per-sample predictions.
    """
    is_synthetic = np.array([s == "synthetic" for s in sources])
    is_real = ~is_synthetic

    n_train = int(is_synthetic.sum())
    n_test = int(is_real.sum())

    result: Dict[str, Any] = {
        "n_train": n_train,
        "n_test": n_test,
        "accuracy": None,
        "coverage": None,
        "predictions": [],
        "note": "",
    }

    if n_train == 0 or n_test == 0:
        result["note"] = "need both synthetic and real samples to run this test"
        return result

    clf = DTWKNNClassifier(
        n_neighbors=n_neighbors,
        confidence_threshold=confidence_threshold,
        window=window,
    )
    clf.fit(X[is_synthetic], y[is_synthetic])

    preds: List[str] = []
    n_correct = 0
    n_accepted = 0
    for i in np.where(is_real)[0]:
        pred, _conf, _details = clf.predict_single(X[i])
        if pred is not None:
            n_accepted += 1
        preds.append(f"{y[i]}->{pred if pred is not None else 'UNKNOWN'}")
        if pred == y[i]:
            n_correct += 1

    result["accuracy"] = n_correct / n_test
    result["coverage"] = n_accepted / n_test
    result["predictions"] = preds
    result["intra_scale"] = float(clf.intra_scale_)
    result["inter_scale"] = float(clf.inter_scale_)
    return result


def evaluate_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_neighbors: int = KNN_NEIGHBORS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    window: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run stratified cross-validation on landmark sequences.

    Note that on a synthetic-dominated dataset this measures template recall,
    not real recognition. See ``evaluate_holdout_by_source`` for the figure that
    reflects live behaviour.
    """
    labels = np.unique(y)
    min_count = min(np.sum(y == lbl) for lbl in labels)
    k_folds = max(2, min(n_splits, int(min_count)))

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    fold_accuracies: List[float] = []
    latencies: List[float] = []
    all_y_true: List[str] = []
    all_y_pred: List[str] = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        clf = DTWKNNClassifier(
            n_neighbors=n_neighbors,
            confidence_threshold=confidence_threshold,
            window=window,
        )
        clf.fit(X_train, y_train)

        fold_preds = []
        for i in range(len(X_val)):
            t0 = time.perf_counter()
            pred, _conf, _details = clf.predict_single(X_val[i])
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

            fold_preds.append(pred if pred is not None else "UNKNOWN")

        fold_preds = np.array(fold_preds)
        fold_accuracies.append(float(np.mean(fold_preds == y_val)))

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(fold_preds.tolist())

    all_labels_for_cm = list(labels)
    if "UNKNOWN" in all_y_pred:
        all_labels_for_cm.append("UNKNOWN")

    return {
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "fold_accuracies": fold_accuracies,
        "k_folds": k_folds,
        "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "confusion_matrix": confusion_matrix(all_y_true, all_y_pred, labels=all_labels_for_cm),
        "cm_labels": all_labels_for_cm,
        "report_str": classification_report(
            all_y_true, all_y_pred, labels=labels, output_dict=False, zero_division=0
        ),
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
        help=f"Confidence rejection threshold (default: {CONFIDENCE_THRESHOLD}).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="DTW Sakoe-Chiba band window (default: None for full window).",
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

    sources = load_sample_sources()
    source_counts: Dict[str, int] = {}
    for s in sources:
        source_counts[s] = source_counts.get(s, 0) + 1
    print("\nSample provenance:")
    for tag, count in sorted(source_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {tag}: {count}")

    real_total = sum(c for t, c in source_counts.items() if t != "synthetic")
    if source_counts.get("synthetic", 0) and real_total:
        print(
            "\nWARNING: this dataset is dominated by synthetic sequences. The\n"
            "cross-validation accuracy below is optimistic and does not predict\n"
            "live recognition. Record real samples with:\n"
            "  python main.py --mode words record --camera --samples 20"
        )

    holdout = evaluate_holdout_by_source(
        X, y, sources,
        n_neighbors=args.k,
        confidence_threshold=args.threshold,
        window=args.window,
    )
    print("\n" + "-" * 60)
    print("  REAL-HOLD-OUT TEST (train on synthetic, test on real only)")
    print("-" * 60)
    if holdout["accuracy"] is None:
        print(f"  Skipped: {holdout['note']}")
    else:
        print(f"  Train (synthetic): {holdout['n_train']} samples")
        print(f"  Test  (real):      {holdout['n_test']} samples")
        print(f"  Accuracy:          {holdout['accuracy'] * 100:.1f}%")
        print(f"  Coverage:          {holdout['coverage'] * 100:.1f}% of real inputs accepted")
        print(f"  intra/inter scale: {holdout['intra_scale']:.3f} / {holdout['inter_scale']:.3f}")
        for p in holdout["predictions"]:
            print(f"    {p}")
    print("-" * 60)

    print(f"\nRunning {args.cv}-fold cross-validation (k={args.k}, threshold={args.threshold:.2f}, window={args.window})...")
    results = evaluate_cross_validation(
        X,
        y,
        n_splits=args.cv,
        n_neighbors=args.k,
        confidence_threshold=args.threshold,
        window=args.window,
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
            window=args.window,
        )
        clf.fit(X, y)
        clf.save(model_path)
        print("Model saved successfully.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
