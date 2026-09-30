"""Robustness checks that catch failures which look like 'the model is bad'.

Run: .venv/bin/python -m signlang.selftest
"""
import sys

import numpy as np
import torch

from . import dataset as ds
from .features import _renorm, augment, landmark_features
from .model import Predictor


def _rot(p, axis, deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == 0:
        m = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    elif axis == 1:
        m = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    else:
        m = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    return _renorm(p @ m.T)


def _acc_at(predictor, X, y, labels, truth, axis, deg):
    idx = np.flatnonzero(y == truth)[:100]
    if not len(idx):
        return None
    hit = 0
    for j in idx:
        q = landmark_features(_rot(X[j].reshape(21, 3), axis, deg))
        hit += predictor.labels[predictor.probs(q).argmax()] == truth
    return hit / len(idx)


def main():
    try:
        predictor = Predictor()
    except FileNotFoundError as exc:
        print(f"SKIP: {exc}")
        return 0

    X, y, labels, _ = ds.load_all(with_sources=True)
    if len(X) == 0:
        print("SKIP: no recorded data")
        return 0

    failures = []

    base = {}
    for l in labels:
        a = np.mean(
            [
                predictor.labels[predictor.probs(landmark_features(X[j].reshape(21, 3))).argmax()] == l
                for j in np.flatnonzero(y == l)[:100]
            ]
        )
        base[l] = a
    print("clean accuracy: " + "  ".join(f"{l}={base[l]:.2f}" for l in labels))
    for l, a in base.items():
        if a < 0.9:
            failures.append(f"clean accuracy for {l} is only {a:.2f}")

    print("\nview robustness (tilting the hand should NOT change the letter):")
    for l in labels:
        row = []
        for axis, name in ((0, "pitch"), (1, "yaw")):
            worst = 1.0
            for deg in (15, 25, 35):
                a = _acc_at(predictor, X, y, labels, l, axis, deg)
                if a is not None:
                    worst = min(worst, a)
            row.append(f"{name}={worst:.2f}")
            if worst < 0.85:
                failures.append(
                    f"{l} breaks under {name} tilt (worst {worst:.2f} within 35deg) "
                    f"- view-dependent, needs out-of-plane augmentation"
                )
        print(f"  {l:<8s} " + "  ".join(row))

    print("\naugmentation invariants:")
    rng = np.random.default_rng(0)
    bad = 0
    for k in range(300):
        a = augment(X[k % len(X)], rng).reshape(21, 3)
        if not np.allclose(a[0], 0, atol=1e-4) or abs(np.linalg.norm(a[9]) - 1) > 1e-3:
            bad += 1
    print(f"  renormalisation held in {300 - bad}/300 samples")
    if bad:
        failures.append(f"augment() broke normalisation in {bad}/300 samples")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
