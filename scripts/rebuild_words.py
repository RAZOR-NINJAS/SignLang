#!/usr/bin/env python
"""Rebuild the words dataset + classifier.

- Backs up current samples to words/data/samples.bak-<ts>
- Regenerates N synthetic 30-frame templates per word (all 25 in WORDS)
- Re-appends any pre-existing REAL (non-synthetic) webcam samples so they aren't lost
- Fits a DTW-kNN classifier and saves words/models/words_classifier.npz
- Prints dataset composition + a synthetic holdout sanity check

Run: PYTHONPATH=. .venv/bin/python scripts/rebuild_words.py [samples_per_word]
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from words.classifier import DTWKNNClassifier
from words.config import CONFIDENCE_THRESHOLD, KNN_NEIGHBORS, MODEL_DIR, WORDS, WORD_GLOSS
from words.dataset import (
    generate_synthetic_dataset,
    load_dataset,
    load_sample_sources,
    load_word_samples,
    path_for_word,
    save_word_samples,
    sources_path_for_word,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "words" / "data" / "samples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", type=int, nargs="?", default=15)
    args = ap.parse_args()
    n = args.samples

    # 1. Back up existing samples
    bak = SAMPLE_DIR.parent / f"samples.bak-{int(time.time())}"
    if SAMPLE_DIR.exists():
        shutil.copytree(SAMPLE_DIR, bak)
        print(f"backed up samples -> {bak}")

    # 2. Preserve real (non-synthetic) webcam samples before regenerating
    real: dict[str, list[np.ndarray]] = {}
    for w in WORDS:
        sam = load_word_samples(w)
        sp = sources_path_for_word(w)
        if sam is None or not sp.exists():
            continue
        tags = np.load(sp, allow_pickle=True)
        real[w] = [s for s, t in zip(sam, tags) if str(t) != "synthetic"]
        if real[w]:
            print(f"  preserving {len(real[w])} real webcam sample(s) for {w}")

    # 3. Regenerate synthetic templates fresh for the full vocabulary
    print(f"generating {n} synthetic templates x {len(WORDS)} words ...")
    generate_synthetic_dataset(samples_per_word=n, seed=42, overwrite=True)

    # 4. Re-append the preserved real samples
    for w, seqs in real.items():
        if seqs:
            save_word_samples(w, np.stack(seqs, axis=0), source="webcam0", append=True)

    # 5. Load and sanity-report composition
    X, y = load_dataset()
    sources = load_sample_sources()
    n_syn = sum(1 for s in sources if s == "synthetic")
    n_real = len(sources) - n_syn
    print(f"dataset: {len(X)} sequences  ({n_syn} synthetic, {n_real} real webcam, {len(set(y))} words)")

    # 6. Fit + save classifier
    clf = DTWKNNClassifier(n_neighbors=KNN_NEIGHBORS, confidence_threshold=CONFIDENCE_THRESHOLD)
    clf.fit(X, y)
    model_path = MODEL_DIR / "words_classifier.npz"
    clf.save(model_path)
    print(f"saved classifier -> {model_path}")
    print(f"  classes ({len(clf.classes_)}): {' '.join(str(c) for c in clf.classes_)}")
    print(f"  calibration: same-sign d~{clf.intra_scale_:.3f}, diff-sign d~{clf.inter_scale_:.3f}")

    # 7. Synthetic holdout sanity check: train on half of synthetic, test on the
    #    other half. This verifies the TEMPLATES are separable (not real-world acc).
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X))
    syn_mask = np.array([s == "synthetic" for s in sources])
    syn_idx = np.where(syn_mask)[0]
    half = syn_idx[: len(syn_idx) // 2]
    te_half = syn_idx[len(syn_idx) // 2:]
    clf2 = DTWKNNClassifier(n_neighbors=3, confidence_threshold=0.0)
    clf2.fit(X[half], y[half])
    acc = clf2.score(X[te_half], y[te_half])
    print(f"  synthetic-template holdout accuracy: {acc:.1%}  (templates separable, NOT real-world accuracy)")


if __name__ == "__main__":
    main()