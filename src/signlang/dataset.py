import json
from pathlib import Path

import numpy as np

from .config import DATA_DIR, SAMPLE_DIR, SOURCE, TRAIN_LABELS


def source_tag():
    s = str(SOURCE)
    return f"cam{s}" if s.isdigit() else "net"


def _path_for(label):
    safe = label.replace(" ", "_")
    return SAMPLE_DIR / f"{safe}.npy"


def _src_path_for(label):
    safe = label.replace(" ", "_")
    return SAMPLE_DIR / f"{safe}.sources.npy"


def append_samples(label, features, source=None):
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(features, dtype=np.float32)
    tag = source or source_tag()
    path = _path_for(label)
    spath = _src_path_for(label)

    if path.exists():
        arr = np.concatenate([np.load(path), arr], axis=0)
    np.save(path, arr)

    tags = np.array([tag] * arr.shape[0], dtype=object)
    if spath.exists():
        prior = np.load(spath, allow_pickle=True)
        n_new = arr.shape[0] - (prior.shape[0] if prior.size else 0)
        tags = np.concatenate(
            [prior, np.array([tag] * max(n_new, 0), dtype=object)]
        )
    np.save(spath, tags)
    return arr.shape[0]


def count_for(label):
    path = _path_for(label)
    if not path.exists():
        return 0
    return int(np.load(path, mmap_mode="r").shape[0])


def sources_for(label):
    path = _path_for(label)
    spath = _src_path_for(label)
    if not path.exists():
        return []
    n = int(np.load(path, mmap_mode="r").shape[0])
    if not spath.exists():
        return ["unknown"] * n
    tags = np.load(spath, allow_pickle=True)
    if tags.shape[0] != n:
        return ["unknown"] * n
    return [str(t) for t in tags]


def reset_label(label):
    for p in (_path_for(label), _src_path_for(label)):
        if p.exists():
            p.unlink()


def counts(labels=None):
    return {lab: count_for(lab) for lab in (labels or TRAIN_LABELS)}


def source_breakdown(labels=None):
    out = {}
    for lab in labels or TRAIN_LABELS:
        for s in sources_for(lab):
            out[s] = out.get(s, 0) + 1
    return out


def _as_rows(arr):
    if arr.ndim == 3 and arr.shape[1:] == (21, 3):
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim == 2:
        return arr
    return None


def load_all(labels=None, with_sources=False):
    labels = labels or TRAIN_LABELS
    xs, ys, ss, kept = [], [], [], []
    for label in labels:
        path = _path_for(label)
        if not path.exists():
            continue
        rows = _as_rows(np.load(path))
        if rows is None or rows.shape[0] == 0:
            continue
        xs.append(rows.astype(np.float32))
        ys.append(np.full(rows.shape[0], label, dtype=object))
        ss.append(np.array(sources_for(label), dtype=object))
        kept.append(label)
    if not xs:
        empty = np.zeros((0, 63), np.float32)
        return (empty, np.zeros((0,), object), [], np.zeros((0,), object)) if with_sources else (empty, np.zeros((0,), object), [])
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    if with_sources:
        return X, y, kept, np.concatenate(ss)
    return X, y, kept


def class_report(labels=None):
    X, y, kept = load_all(labels)
    out = {}
    for label in TRAIN_LABELS:
        out[label] = int((y == label).sum()) if len(y) else 0
    return out, kept
