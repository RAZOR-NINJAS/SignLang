import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import dataset as ds
from .features import augment, feature_dim, landmark_features
from .model import SignMLP, save_checkpoint


class LandmarkSet(Dataset):
    def __init__(self, coords, targets, mean, std, augment_on, seed):
        self.coords = coords
        self.targets = targets
        self.mean = mean
        self.std = std
        self.augment_on = augment_on
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        c = self.coords[i]
        if self.augment_on:
            c = augment(c, self.rng)
        f = landmark_features(c.reshape(21, 3))
        f = (f - self.mean) / self.std
        return torch.from_numpy(f.astype(np.float32)), int(self.targets[i])


def stratified_split(y, n_classes, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for i in range(n_classes):
        idx = np.flatnonzero(y == i)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        if len(idx) >= 4 and n_val >= 1:
            va.extend(idx[:n_val].tolist())
            tr.extend(idx[n_val:].tolist())
        else:
            tr.extend(idx.tolist())
    return np.array(tr, int), np.array(va, int)


def fit_temperature(logits, targets, lo=0.01, hi=30.0, steps=120):
    best_t, best_nll = 1.0, float("inf")
    for t in np.geomspace(lo, hi, steps):
        nll = nn.functional.cross_entropy(logits / float(t), targets).item()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return max(best_t, 0.05), best_nll


def main(argv=None):
    ap = argparse.ArgumentParser(prog="signlang train")
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    coords, y_str, labels, srcs = ds.load_all(with_sources=True)
    if len(coords) == 0:
        print("No training data found. Run `signlang collect` first.")
        return 1
    if len(labels) < 2:
        print(f"Only {len(labels)} sign(s) recorded ({labels}). Need at least 2.")
        return 1

    print(f"samples={len(coords)}  signs={len(labels)}  landmark_dim={coords.shape[1]}")
    uniq, ucount = np.unique(srcs, return_counts=True)
    print("by source: " + "  ".join(f"{u}={c}" for u, c in zip(uniq, ucount)))

    label_index = {lab: i for i, lab in enumerate(labels)}
    y = np.array([label_index[v] for v in y_str], dtype=np.int64)

    feats_all = np.stack([landmark_features(c.reshape(21, 3)) for c in coords])
    counts = np.bincount(y, minlength=len(labels))
    thin = [labels[i] for i, c in enumerate(counts) if c < 25]
    if thin:
        print(f"WARNING thin signs (<25): {' '.join(thin)}")

    tr, va = stratified_split(y, len(labels), args.val_frac, args.seed)
    if len(tr) == 0 or len(va) == 0:
        print(
            f"not enough data to split: train={len(tr)} val={len(va)}. "
            f"Record more signs with `signlang collect`."
        )
        return 1
    print(f"split: train={len(tr)} val={len(va)}  feature_dim={feature_dim()}")

    mean = feats_all[tr].mean(0)
    std = feats_all[tr].std(0) + 1e-6

    train_set = LandmarkSet(coords[tr], y[tr], mean, std, not args.no_augment, args.seed)
    val_set = LandmarkSet(coords[va], y[va], mean, std, False, args.seed)
    train_ld = DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=False)
    val_ld = DataLoader(val_set, batch_size=128, shuffle=False)

    model = SignMLP(feats_all.shape[1], len(labels))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params} params  {model.in_dim}->{model.n_classes}")

    w = torch.tensor(
        (len(tr) / (len(labels) * np.maximum(counts, 1))), dtype=torch.float32
    )
    w = w / w.mean()
    lossf = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(train_ld), 1),
        pct_start=0.25,
    )

    best_acc, best_state, best_epoch = -1.0, None, -1
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, corr, lsum = 0, 0, 0.0
        for xb, yb in train_ld:
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            lsum += loss.item() * len(yb)
            corr += (out.argmax(1) == yb).sum().item()
            tot += len(yb)
        tr_acc = corr / max(tot, 1)

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for xb, yb in val_ld:
                out = model(xb)
                vc += (out.argmax(1) == yb).sum().item()
                vt += len(yb)
        va_acc = vc / max(vt, 1)
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == 1 or va_acc >= best_acc:
            print(
                f"  ep{ep:3d}  loss={lsum/max(tot,1):.4f}  train={tr_acc:.3f}  "
                f"val={va_acc:.3f}  best={best_acc:.3f}@{best_epoch}  {time.time()-t0:.1f}s"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    temperature = 1.0
    if len(va):
        with torch.no_grad():
            lv = model(
                torch.from_numpy(((feats_all[va] - mean) / std).astype(np.float32))
            )
        temperature, _ = fit_temperature(lv, torch.from_numpy(y[va]))
        pv = lv.argmax(1).numpy()
    else:
        preds = None
        pv = np.zeros(0, dtype=np.int64)
    per = {}
    for i, lab in enumerate(labels):
        sel = y[va] == i
        if sel.sum():
            per[lab] = float((pv[sel] == i).mean())
    if per:
        worst = sorted(per.items(), key=lambda kv: kv[1])[:8]
        print("\nweakest signs (val accuracy):")
        for lab, acc in worst:
            print(f"  {lab:<10s} {acc:.2f}")

        conf = {}
        for t, p in zip(y[va], pv):
            if t != p:
                conf[(labels[t], labels[p])] = conf.get((labels[t], labels[p]), 0) + 1
        if conf:
            print("\ntop confusions (true -> predicted):")
            for (a, b), c in sorted(conf.items(), key=lambda kv: -kv[1])[:8]:
                print(f"  {a:<10s} -> {b:<10s} x{c}")

    meta = {
        "val_acc": best_acc,
        "best_epoch": best_epoch,
        "n_samples": int(len(coords)),
        "epochs": args.epochs,
        "temperature": temperature,
        "per_class_val": per,
    }
    path = save_checkpoint(None, model, labels, mean, std, meta)

    if len(va):
        with torch.no_grad():
            lg = model(
                torch.from_numpy(((feats_all[va] - mean) / std).astype(np.float32))
            )
        before = torch.softmax(lg, 1).max(1).values.mean().item()
        after = torch.softmax(lg / max(temperature, 1e-6), 1).max(1).values.mean().item()
        print(f"\nsaved {path}  (best val acc {best_acc:.3f} @ epoch {best_epoch})")
        print(
            f"temperature {temperature:.3f}  ->  mean val confidence "
            f"{before:.3f} -> {after:.3f}"
        )
    else:
        print(f"\nsaved {path}  (best val acc {best_acc:.3f} @ epoch {best_epoch})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
