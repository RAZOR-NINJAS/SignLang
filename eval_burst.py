#!/usr/bin/env python
"""Honest per-letter accuracy for the bundled model.

train.py splits rows at random, but rows arrive in bursts of BURST_FRAMES
near-identical frames. A random split puts neighbours on both sides, so
validation scores ~1.00 and tells you nothing about a new hand.

This script holds out whole bursts instead: every frame of a burst is either
all-train or all-val. That is the number to quote at an exhibition.

Run from the project root:

    .venv/bin/python eval_burst.py
"""

import argparse
from collections import Counter

import numpy as np
import torch

from signlang import dataset as ds
from signlang.config import BURST_FRAMES
from signlang.features import landmark_features
from signlang.model import SignMLP


def burst_groups(labels_per_row, burst_frames=BURST_FRAMES):
    """Group row indices into bursts, never letting a burst straddle a label."""
    groups = []
    start = 0
    rows = len(labels_per_row)
    while start < rows:
        label = labels_per_row[start]
        end = start
        while end < rows and labels_per_row[end] == label:
            end += 1
        seg = end - start
        for i in range(start, end, burst_frames):
            groups.append(np.arange(i, min(i + burst_frames, end)))
        start = end
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5, help="burst-wise cross-validation folds")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    X, y, kept = ds.load_all()
    if not kept:
        raise SystemExit("no data found; run `signlang collect` first")

    feats = np.stack([landmark_features(c.reshape(21, 3)) for c in X]).astype(np.float32)
    label_names = list(kept)
    yi = np.array([label_names.index(v) for v in y])
    counts = Counter(y)
    groups = burst_groups(y)

    print(f"signs: {len(kept)}   samples: {len(X)}   bursts: {len(groups)}")
    thin = [(k, counts[k]) for k in kept if counts[k] < BURST_FRAMES * 6]
    if thin:
        print(f"  thin signs (<{BURST_FRAMES * 6} samples): " +
              ", ".join(f"{k}={n}" for k, n in thin))
    print()

    # Per-class weighting, same rationale as train.py: G/H/T have far fewer rows.
    freq = np.array([counts[k] for k in kept], dtype=np.float64)
    weight = len(freq) / np.maximum(freq, 1.0)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(groups))

    correct = np.zeros(len(kept), dtype=np.int64)
    total = np.zeros(len(kept), dtype=np.int64)
    confusion = np.zeros((len(kept), len(kept)), dtype=np.int64)

    for fold in range(args.folds):
        val_groups = {order[i] for i in range(len(groups)) if i % args.folds == fold}
        tr_idx = np.concatenate([groups[i] for i in range(len(groups)) if i not in val_groups])
        va_idx = np.concatenate([groups[i] for i in sorted(val_groups)])

        mean = feats[tr_idx].mean(0)
        std = feats[tr_idx].std(0) + 1e-6
        xt = torch.from_numpy((feats[tr_idx] - mean) / std)
        yt = torch.from_numpy(yi[tr_idx])
        xv = torch.from_numpy((feats[va_idx] - mean) / std)

        torch.manual_seed(args.seed)
        model = SignMLP(feats.shape[1], len(kept))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        lossf = torch.nn.CrossEntropyLoss(weight=torch.from_numpy((weight / weight.mean()).astype(np.float32)))

        model.train()
        for _ in range(args.epochs):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 256):
                batch = perm[i:i + 256]
                opt.zero_grad()
                loss = lossf(model(xt[batch]), yt[batch])
                loss.backward()
                opt.step()
            sched.step()

        model.eval()
        with torch.no_grad():
            pred = model(xv).argmax(1).numpy()
        for p, t in zip(pred, yi[va_idx]):
            total[t] += 1
            if p == t:
                correct[t] += 1
            confusion[t, p] += 1
        print(f"  fold {fold + 1}/{args.folds} done ({len(va_idx)} held-out frames)")

    print()
    print("per-letter accuracy on unseen bursts  (this is the honest number)")
    print("  letter   n     acc     grade")
    rows = []
    for i, k in enumerate(kept):
        acc = correct[i] / total[i] if total[i] else 0.0
        grade = "safe" if acc >= 0.90 else "ok" if acc >= 0.75 else "risky"
        rows.append((acc, k, total[i], grade))
    for acc, k, n, grade in sorted(rows):
        bar = "#" * int(round(acc * 20))
        print(f"    {k}     {n:5d}  {acc:5.1%}  {grade:5s} {bar}")

    overall = correct.sum() / max(total.sum(), 1)
    print(f"\n  overall: {overall:.1%} over {total.sum()} held-out frames")

    print("\ntop confusions (true -> predicted)")
    conf = [(confusion[i, j], kept[i], kept[j]) for i in range(len(kept))
            for j in range(len(kept)) if i != j and confusion[i, j]]
    for n, t, p in sorted(conf, reverse=True)[:12]:
        print(f"    {t} -> {p}   {n} times")

    safe = [k for _, k, _, g in rows if g == "safe"]
    ok = [k for _, k, _, g in rows if g == "ok"]
    risky = [k for _, k, _, g in rows if g == "risky"]
    print(f"\nrecommended game pool")
    print(f"  safe  ({len(safe)}): {' '.join(sorted(safe))}")
    print(f"  ok    ({len(ok)}): {' '.join(sorted(ok))}")
    print(f"  risky ({len(risky)}): {' '.join(sorted(risky))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())