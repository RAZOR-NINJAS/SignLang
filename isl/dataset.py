"""Dataset management and persistence for ISL landmark features."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .config import DATA_DIR, SAMPLE_DIR, SOURCE, TRAIN_LABELS


def source_tag(session: Optional[str] = None) -> str:
    s = str(SOURCE)
    cam = f"cam{s}" if s.isdigit() else "net"
    sess = session or "session1"
    return f"{sess}_{cam}"


def _path_for(label: str) -> Path:
    safe = label.replace(" ", "_")
    return SAMPLE_DIR / f"{safe}.npy"


def _src_path_for(label: str) -> Path:
    safe = label.replace(" ", "_")
    return SAMPLE_DIR / f"{safe}.sources.npy"


def append_samples(label: str, features: Union[List, np.ndarray], source: Optional[str] = None) -> int:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    tag = source or source_tag()
    path = _path_for(label)
    spath = _src_path_for(label)

    if path.exists():
        prior = np.load(path)
        arr = np.concatenate([prior, arr], axis=0)
    np.save(path, arr)

    tags = np.array([tag] * arr.shape[0], dtype=object)
    if spath.exists():
        prior_tags = np.load(spath, allow_pickle=True)
        n_new = arr.shape[0] - (prior_tags.shape[0] if prior_tags.size else 0)
        tags = np.concatenate(
            [prior_tags, np.array([tag] * max(n_new, 0), dtype=object)]
        )
    np.save(spath, tags)
    return arr.shape[0]


def count_for(label: str) -> int:
    path = _path_for(label)
    if not path.exists():
        return 0
    return int(np.load(path, mmap_mode="r").shape[0])


def sources_for(label: str) -> List[str]:
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


def reset_label(label: str) -> None:
    for p in (_path_for(label), _src_path_for(label)):
        if p.exists():
            p.unlink()


def counts(labels: Optional[List[str]] = None) -> Dict[str, int]:
    return {lab: count_for(lab) for lab in (labels or TRAIN_LABELS)}


def load_all(
    labels: Optional[List[str]] = None,
    with_sources: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, List[str]], Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]]:
    target_labels = labels or TRAIN_LABELS
    xs, ys, ss, kept = [], [], [], []
    for label in target_labels:
        path = _path_for(label)
        if not path.exists():
            continue
        rows = np.load(path)
        if rows.size == 0:
            continue
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        xs.append(rows.astype(np.float32))
        ys.append(np.full(rows.shape[0], label, dtype=object))
        ss.append(np.array(sources_for(label), dtype=object))
        kept.append(label)

    if not xs:
        empty = np.zeros((0, 164), dtype=np.float32)
        if with_sources:
            return empty, np.zeros((0,), dtype=object), [], np.zeros((0,), dtype=object)
        return empty, np.zeros((0,), dtype=object), []

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    sources = np.concatenate(ss, axis=0)
    if with_sources:
        return X, y, kept, sources
    return X, y, kept


def class_report(labels: Optional[List[str]] = None) -> Tuple[Dict[str, int], List[str]]:
    X, y, kept = load_all(labels)
    out = {}
    for label in (labels or TRAIN_LABELS):
        out[label] = int((y == label).sum()) if len(y) else 0
    return out, kept
