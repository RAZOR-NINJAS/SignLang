"""Evaluation and benchmark script for ISL recognition models.

Supports honest cross-session holdout evaluation (e.g., train on session1, test
on session2), burst-level cross-validation, per-class metrics, 1-handed vs
2-handed breakdown, and confusion matrix reporting.
"""

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from . import dataset as ds
from .config import BURST_FRAMES, EXPECTED_HANDS, FEATURE_DIM, WEIGHTS_PATH
from .features import extract_features
from .model import ISL_MLP, get_torch_model, load_checkpoint


def parse_session(source_tag: str) -> str:
    """Extract session identifier from source tag (e.g. 'session1_cam0' -> 'session1')."""
    return str(source_tag).split("_")[0]


def group_bursts(y_str: np.ndarray, burst_size: int = BURST_FRAMES) -> List[np.ndarray]:
    """Group row indices into bursts without letting a burst cross a label boundary."""
    groups = []
    start = 0
    n = len(y_str)
    while start < n:
        label = y_str[start]
        end = start
        while end < n and y_str[end] == label:
            end += 1
        for i in range(start, end, burst_size):
            groups.append(np.arange(i, min(i + burst_size, end)))
        start = end
    return groups


def print_confusion_matrix(cm: np.ndarray, labels: List[str]) -> None:
    """Print an ASCII confusion matrix."""
    col_w = max(6, max(len(str(lbl)) for lbl in labels) + 1)
    header = "True \\ Pred".ljust(col_w) + "".join(lbl[:5].rjust(6) for lbl in labels)
    print(header)
    print("-" * len(header))
    for i, row_lbl in enumerate(labels):
        row_str = row_lbl[: col_w - 1].ljust(col_w)
        for j in range(len(labels)):
            val = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
            row_str += str(val).rjust(6)
        print(row_str)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
) -> Dict[str, object]:
    """Compute overall, per-class, and 1-hand vs 2-hand accuracy metrics."""
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)

    correct_per_class = Counter()
    total_per_class = Counter()

    for yt, yp in zip(y_true, y_pred):
        total_per_class[yt] += 1
        if yt == yp:
            correct_per_class[yt] += 1
        if yt in label_to_idx and yp in label_to_idx:
            cm[label_to_idx[yt], label_to_idx[yp]] += 1

    overall_acc = (
        sum(correct_per_class.values()) / max(sum(total_per_class.values()), 1)
    )

    one_hand_labels = [l for l in labels if EXPECTED_HANDS.get(l, 2) == 1]
    two_hand_labels = [l for l in labels if EXPECTED_HANDS.get(l, 2) == 2]

    one_hand_corr = sum(correct_per_class[l] for l in one_hand_labels)
    one_hand_tot = sum(total_per_class[l] for l in one_hand_labels)
    one_hand_acc = one_hand_corr / max(one_hand_tot, 1) if one_hand_tot else 0.0

    two_hand_corr = sum(correct_per_class[l] for l in two_hand_labels)
    two_hand_tot = sum(total_per_class[l] for l in two_hand_labels)
    two_hand_acc = two_hand_corr / max(two_hand_tot, 1) if two_hand_tot else 0.0

    return {
        "overall_accuracy": overall_acc,
        "one_hand_accuracy": one_hand_acc,
        "one_hand_total": one_hand_tot,
        "two_hand_accuracy": two_hand_acc,
        "two_hand_total": two_hand_tot,
        "correct_per_class": correct_per_class,
        "total_per_class": total_per_class,
        "confusion_matrix": cm,
        "labels": labels,
    }


def print_evaluation_report(metrics: Dict[str, object]) -> None:
    labels = metrics["labels"]
    corr = metrics["correct_per_class"]
    tot = metrics["total_per_class"]
    cm = metrics["confusion_matrix"]

    print("\n" + "=" * 68)
    print("  ISL MODEL EVALUATION REPORT")
    print("=" * 68)
    print(f"  Overall Accuracy:      {metrics['overall_accuracy'] * 100:5.1f}% ({sum(tot.values())} samples)")
    print(f"  1-Handed Signs Acc:    {metrics['one_hand_accuracy'] * 100:5.1f}% ({metrics['one_hand_total']} samples)")
    print(f"  2-Handed Signs Acc:    {metrics['two_hand_accuracy'] * 100:5.1f}% ({metrics['two_hand_total']} samples)")
    print("=" * 68)

    print("\nPer-sign Accuracy:")
    print("  sign   hands      n      acc     grade")
    print("  --------------------------------------")
    rows = []
    for l in labels:
        n = tot[l]
        acc = corr[l] / n if n else 0.0
        grade = "safe" if acc >= 0.90 else "ok" if acc >= 0.75 else "risky"
        hands = EXPECTED_HANDS.get(l, 2)
        rows.append((acc, l, hands, n, grade))

    for acc, l, hands, n, grade in sorted(rows, key=lambda r: (r[0], r[1])):
        bar = "#" * int(round(acc * 20))
        print(f"  {l:5s}   {hands:2d}h   {n:5d}   {acc*100:5.1f}%   {grade:5s}  {bar}")

    # Top confusions
    conf = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                conf.append((cm[i, j], labels[i], labels[j]))

    if conf:
        print("\nTop Confusions (True -> Predicted):")
        for count, yt, yp in sorted(conf, reverse=True)[:10]:
            print(f"    {yt} -> {yp} : {count} times")
    else:
        print("\nTop Confusions: None (100% clean predictions)")

    print("\nConfusion Matrix:")
    print_confusion_matrix(cm, labels)


def evaluate_existing_checkpoint(
    checkpoint_path: Path,
    test_session: Optional[str] = None,
) -> int:
    """Evaluate an existing trained checkpoint using pure NumPy ISL_MLP."""
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at: {checkpoint_path}")
        print("Train a model first with: signlang isl train")
        return 1

    model, labels, mean, std, meta = load_checkpoint(checkpoint_path)
    X, y_str, kept, sources = ds.load_all(with_sources=True)

    if len(X) == 0:
        print("No evaluation data found in isl/data/samples.")
        return 1

    mask = np.ones(len(y_str), dtype=bool)
    if test_session:
        mask = np.array([parse_session(s) == test_session for s in sources])
        if not np.any(mask):
            print(f"No samples found with session tag '{test_session}'.")
            return 1
        print(f"Filtering to session '{test_session}': {mask.sum()}/{len(y_str)} samples.")

    X_test = X[mask]
    y_test = y_str[mask]

    # Predict via NumPy ISL_MLP
    preds = []
    for row in X_test:
        probs = model.predict_proba(row)
        pred_label = labels[int(np.argmax(probs))]
        preds.append(pred_label)

    metrics = compute_metrics(y_test, np.array(preds), labels)
    print_evaluation_report(metrics)
    return 0


def run_cross_validation(folds: int = 5, epochs: int = 40, seed: int = 42) -> int:
    """Run honest burst-level cross-validation."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    from .train import ISLLandmarkDataset

    X, y_str, labels, sources = ds.load_all(with_sources=True)
    if len(X) == 0:
        print("No evaluation data found.")
        return 1
    if len(labels) < 2:
        print(f"Need at least 2 distinct classes to run cross-validation (have {len(labels)}).")
        return 1

    groups = group_bursts(y_str)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y_str], dtype=np.int64)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    k_folds = min(folds, len(groups))

    print(f"Running {k_folds}-fold burst cross-validation across {len(groups)} bursts ({len(X)} samples)...")
    all_preds = np.empty(len(y_str), dtype=object)

    for fold in range(k_folds):
        val_group_indices = {order[i] for i in range(len(groups)) if i % k_folds == fold}
        tr_indices = np.concatenate([groups[i] for i in range(len(groups)) if i not in val_group_indices])
        va_indices = np.concatenate([groups[i] for i in sorted(val_group_indices)])

        mean = X[tr_indices].mean(axis=0)
        std = X[tr_indices].std(axis=0) + 1e-6

        train_ds = ISLLandmarkDataset(X[tr_indices], y_idx[tr_indices], mean, std, augment=True, seed=seed + fold)
        val_ds = ISLLandmarkDataset(X[va_indices], y_idx[va_indices], mean, std, augment=False, seed=seed + fold)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = get_torch_model(in_dim=FEATURE_DIM, n_classes=len(labels))
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        fold_preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                pred_ids = model(xb).argmax(1).cpu().numpy()
                fold_preds.extend([labels[pid] for pid in pred_ids])

        all_preds[va_indices] = fold_preds
        fold_acc = sum(all_preds[va_indices] == y_str[va_indices]) / len(va_indices)
        print(f"  Fold {fold + 1}/{k_folds}: Acc = {fold_acc * 100:.1f}% ({len(va_indices)} held-out frames)")

    metrics = compute_metrics(y_str, all_preds, labels)
    print_evaluation_report(metrics)
    return 0


def _selftest() -> int:
    """Headless synthetic selftest for ISL eval logic."""
    saved_dir = ds.SAMPLE_DIR
    tmp = tempfile.mkdtemp(prefix="isl-eval-selftest-")
    ds.SAMPLE_DIR = Path(tmp)

    try:
        from .collect import _synthetic_hand

        # Create synthetic samples for 1-hand ('C') and 2-hand ('A', 'B') across session1 & session2
        test_labels = ["A", "B", "C"]
        for sess in ["session1", "session2"]:
            for lbl in test_labels:
                h_count = EXPECTED_HANDS.get(lbl, 2)
                feats = []
                for i in range(10):
                    if h_count == 1:
                        f = extract_features([_synthetic_hand(i)], ["Right"])
                    else:
                        f = extract_features([_synthetic_hand(i), _synthetic_hand(i + 5)], ["Left", "Right"])
                    feats.append(f)
                ds.append_samples(lbl, feats, source=f"{sess}_cam0")

        X, y_str, kept, sources = ds.load_all(with_sources=True)
        assert len(X) == 60
        assert set(kept) == set(test_labels)

        # Mock predictions: 90% correct
        rng = np.random.default_rng(42)
        mock_preds = y_str.copy()
        flip_idx = rng.choice(len(y_str), size=6, replace=False)
        for idx in flip_idx:
            mock_preds[idx] = "B" if mock_preds[idx] == "A" else "A"

        metrics = compute_metrics(y_str, mock_preds, test_labels)
        assert metrics["overall_accuracy"] == 0.90
        assert metrics["one_hand_total"] == 20
        assert metrics["two_hand_total"] == 40
        assert metrics["confusion_matrix"].shape == (3, 3)

        print("ISL EVAL SELFTEST OK: metrics calculation and breakdown verified.")
        return 0
    finally:
        ds.SAMPLE_DIR = saved_dir
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in args_list:
        return _selftest()

    parser = argparse.ArgumentParser(prog="signlang isl eval")
    parser.add_argument("--checkpoint", type=Path, default=WEIGHTS_PATH, help="Path to isl_mlp.pt checkpoint")
    parser.add_argument("--test-session", type=str, default=None, help="Holdout session name to test on")
    parser.add_argument("--cv", action="store_true", help="Run burst cross-validation instead of checkpoint eval")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs per CV fold")
    args = parser.parse_args(args_list)

    if args.cv:
        return run_cross_validation(folds=args.folds, epochs=args.epochs)
    return evaluate_existing_checkpoint(args.checkpoint, test_session=args.test_session)


if __name__ == "__main__":
    raise SystemExit(main())
