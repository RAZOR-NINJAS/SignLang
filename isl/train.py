"""Training pipeline for ISL MLP with multi-session validation and mirror augmentation."""

import argparse
import time
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import dataset as ds
from .config import BURST_FRAMES, FEATURE_DIM, WEIGHTS_PATH
from .features import mirror_feature_vector
from .model import get_torch_model, save_checkpoint


class ISLLandmarkDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, mean: np.ndarray, std: np.ndarray, augment: bool, seed: int = 42):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        f = self.features[idx].copy()
        if self.augment:
            # 50% probability of mirror augmentation (swapping hands for left-handed signers)
            if self.rng.random() < 0.5:
                f = mirror_feature_vector(f)
            # Add small sensor noise
            noise = self.rng.normal(0.0, 0.008, f.shape).astype(np.float32)
            # Only add noise to non-presence flags
            f[:162] += noise[:162]

        norm_f = (f - self.mean) / self.std
        return torch.from_numpy(norm_f.astype(np.float32)), self.targets[idx]


def split_by_session_or_burst(
    y: np.ndarray,
    sources: np.ndarray,
    n_classes: int,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Split dataset by session if multiple sessions exist, otherwise by burst."""
    sessions = set()
    for s in sources:
        tag = str(s).split("_")[0]
        sessions.add(tag)

    if len(sessions) >= 2:
        # Dedicated cross-session split
        sorted_sess = sorted(list(sessions))
        val_sess = sorted_sess[-1]  # Hold out newest session for validation
        tr_idx = np.flatnonzero([not str(s).startswith(val_sess) for s in sources])
        va_idx = np.flatnonzero([str(s).startswith(val_sess) for s in sources])
        if len(tr_idx) > 0 and len(va_idx) > 0:
            return tr_idx, va_idx, f"cross-session (val: {val_sess})"

    # Fallback to stratified burst splitting
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for cls in range(n_classes):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_frac))
        if len(idx) >= 4 and n_val >= 1:
            va.extend(idx[:n_val].tolist())
            tr.extend(idx[n_val:].tolist())
        else:
            tr.extend(idx.tolist())
    return np.array(tr, dtype=int), np.array(va, dtype=int), "stratified burst split"


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, steps: int = 100) -> float:
    best_t, best_nll = 1.0, float("inf")
    for t in np.geomspace(0.05, 10.0, steps):
        nll = nn.functional.cross_entropy(logits / float(t), targets).item()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return max(best_t, 0.05)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="signlang isl train")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    X, y_str, labels, sources = ds.load_all(with_sources=True)

    if len(X) == 0:
        print("No training data found. Run 'signlang isl collect' first to record samples.")
        return 1
    if len(labels) < 2:
        print(f"Only {len(labels)} sign(s) recorded ({labels}). Need at least 2 signs to train.")
        return 1

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y = np.array([label_to_idx[s] for s in y_str], dtype=np.int64)

    tr_idx, va_idx, split_desc = split_by_session_or_burst(y, sources, len(labels), seed=args.seed)
    print(f"ISL Training: {len(X)} samples across {len(labels)} signs | Split: {split_desc}")
    print(f"  Train: {len(tr_idx)} | Validation: {len(va_idx)}")

    mean = X[tr_idx].mean(axis=0)
    std = X[tr_idx].std(axis=0) + 1e-6

    train_set = ISLLandmarkDataset(X[tr_idx], y[tr_idx], mean, std, augment=not args.no_augment, seed=args.seed)
    val_set = ISLLandmarkDataset(X[va_idx], y[va_idx], mean, std, augment=False, seed=args.seed)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)

    model = get_torch_model(in_dim=FEATURE_DIM, n_classes=len(labels))
    class_counts = np.bincount(y[tr_idx], minlength=len(labels))
    weights = torch.tensor(
        len(tr_idx) / (len(labels) * np.maximum(class_counts, 1)),
        dtype=torch.float32,
    )
    weights /= weights.mean()

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * max(len(train_loader), 1), pct_start=0.25
    )

    best_acc, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss, corr, tot = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

            tot_loss += loss.item() * len(yb)
            corr += (out.argmax(1) == yb).sum().item()
            tot += len(yb)

        # Validation
        model.eval()
        v_corr, v_tot = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                out = model(xb)
                v_corr += (out.argmax(1) == yb).sum().item()
                v_tot += len(yb)

        va_acc = v_corr / max(v_tot, 1)
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == 1 or ep == args.epochs:
            print(f"  Epoch {ep:2d}/{args.epochs:2d} | Train Acc: {corr/tot:.3f} | Val Acc: {va_acc:.3f} (best: {best_acc:.3f} @ ep {best_epoch})")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Temperature fitting on validation set
    temp = 1.0
    if len(va_idx):
        with torch.no_grad():
            v_logits = model(torch.from_numpy(((X[va_idx] - mean) / std).astype(np.float32)))
            temp = fit_temperature(v_logits, torch.from_numpy(y[va_idx]))

    meta = {
        "val_acc": best_acc,
        "best_epoch": best_epoch,
        "n_samples": int(len(X)),
        "epochs": args.epochs,
        "temperature": temp,
        "split": split_desc,
    }
    saved_path = save_checkpoint(None, model, labels, mean, std, meta=meta)
    print(f"\nTrained ISL model successfully saved to: {saved_path} (Best Val Acc: {best_acc:.3f}, Temperature: {temp:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
