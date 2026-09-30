"""Package and unpack training data for moving between machines.

    signlang export                 -> signlang-transfer.tar.gz
    signlang import                 <- signlang-transfer.tar.gz

Samples are normalised landmark coordinates, not images, so a bundle is small
and portable. A bundle carries three things:

  data/samples/*.npy   recordings, including the .sources.npy sidecars
  models/signs_mlp.pt  trained weights (if present)
  models/labels.json   sign names the weights expect (if present)

The MediaPipe landmarker is deliberately NOT included: it is 7.8 MB of a fixed
third-party file that both machines can fetch with scripts/fetch_models.sh.
Import therefore never touches it.

Import merges recordings by appending to whatever is already there, and reports
what changed. Existing recordings are never deleted.
"""

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

from . import dataset as ds
from .config import (
    LANDMARKER_PATH,
    LABELS_PATH,
    PROJECT_ROOT,
    SAMPLE_DIR,
    WEIGHTS_PATH,
)

BUNDLE = "signlang-transfer.tar.gz"
MANIFEST = "signlang-transfer.json"
SKIP = {LANDMARKER_PATH.name}


def _fmt_counts(counts):
    live = {k: v for k, v in sorted(counts.items()) if v}
    if not live:
        return "  (none recorded)"
    return "\n".join(f"  {k}  {v}" for k, v in live.items())


def _payload():
    """Everything that should travel, relative to the project root."""
    out = {}
    if SAMPLE_DIR.exists():
        for p in sorted(SAMPLE_DIR.glob("*.npy")):
            out[str(p.relative_to(PROJECT_ROOT))] = p
    for p in (WEIGHTS_PATH, LABELS_PATH):
        if p.exists():
            out[str(p.relative_to(PROJECT_ROOT))] = p
    return out


def export(dest=None, include_model=True):
    out = Path(dest) if dest else PROJECT_ROOT / BUNDLE
    # --no-model keeps recordings only; by default the weights travel too
    payload = _payload()
    if not include_model:
        payload = {k: v for k, v in payload.items() if not k.startswith("models/")}

    if not payload:
        print("nothing to export: no recordings under data/samples/", file=sys.stderr)
        print("run `signlang collect` first.", file=sys.stderr)
        return 1

    counts = ds.counts()
    manifest = {
        "format": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {k: v for k, v in sorted(counts.items()) if v},
        "has_model": (WEIGHTS_PATH in payload.values()),
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        blob = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(blob)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(blob))
        for rel, path in sorted(payload.items()):
            tar.add(path, arcname=rel)

    size_kb = out.stat().st_size / 1024.0
    print(f"wrote {out}  ({size_kb:.0f} KB)")
    print(f"  signs recorded : {len([v for v in counts.values() if v])}")
    print(f"  samples        : {sum(counts.values())}")
    print(f"  trained model  : {'included' if manifest['has_model'] else 'not included'}")
    print()
    print("copy that file to the other machine, then run:  signlang import")
    return 0


def _safe_dest(name):
    """Resolve a bundle member, refusing paths outside the project.

    A tar file can name entries like ../../etc/passwd, so every path is
    resolved and checked before anything is written.
    """
    dest = (PROJECT_ROOT / name).resolve()
    try:
        dest.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    return dest


def _read_member(tar, name):
    member = tar.getmember(name)
    if not member.isfile():
        return None
    fh = tar.extractfile(member)
    return None if fh is None else fh.read()


def _merge_recording(tar, name, blob):
    """Append an incoming recording to what is already on this machine.

    Recordings are plain arrays of landmark rows, so concatenating is safe and
    keeps samples collected on both laptops. The .sources.npy sidecar must grow
    by the same number of rows or the source breakdown falls out of sync, so it
    is written here rather than as a separate member.
    """
    import numpy as np

    dest = _safe_dest(name)
    if dest is None:
        return "refusing path outside the project"

    rows = ds._as_rows(np.load(io.BytesIO(blob)))
    if rows is None:
        return "unrecognised array shape"
    rows = rows.astype(np.float32)

    prior_rows = 0
    if dest.exists():
        existing = ds._as_rows(np.load(dest))
        if existing is None:
            return "existing file is not a landmark array"
        prior_rows = existing.shape[0]
        rows = np.concatenate([existing, rows], axis=0)

    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, rows)

# keep source tags aligned with the rows they describe
    incoming_n = ds._as_rows(np.load(io.BytesIO(blob))).shape[0]
    tags = np.array(["imported"] * incoming_n, dtype=object)
    # A.npy's sidecar is stored as A.sources.npy, matching dataset.py
    sidecar_name = name[: -len(".npy")] + ".sources.npy"
    sidecar_blob = _read_member(tar, sidecar_name)
    if sidecar_blob:
        try:
            candidate = np.load(io.BytesIO(sidecar_blob), allow_pickle=True)
            if candidate.shape[0] == rows.shape[0] - prior_rows:
                tags = candidate
        except Exception:  # noqa: BLE001 - sidecar is optional
            pass

    sidecar = dest.with_suffix(".sources.npy")
    prior_tags = np.array(["unknown"] * prior_rows, dtype=object)
    if sidecar.exists():
        try:
            candidate = np.load(sidecar, allow_pickle=True)
            if candidate.shape[0] == prior_rows:
                prior_tags = candidate
        except Exception:  # noqa: BLE001
            pass
    np.save(sidecar, np.concatenate([prior_tags, tags]))
    return None


def import_bundle(src=None, dry_run=False):
    path = Path(src) if src else PROJECT_ROOT / BUNDLE
    if not path.exists():
        print(f"no bundle at {path}", file=sys.stderr)
        return 1

    before = ds.counts()

    with tarfile.open(path, "r:gz") as tar:
        names = tar.getnames()

        if any(n in SKIP for n in names):
            print(
                "note: this bundle contains a hand landmarker. Skipping it; "
                "fetch it here instead with scripts/fetch_models.sh",
            )
        wanted = [n for n in names if n != MANIFEST and n not in SKIP]

        if not wanted:
            print("bundle contains no usable files", file=sys.stderr)
            return 1

        print(f"bundle: {path}")
        for n in sorted(wanted):
            print(f"  + {n}")

        if dry_run:
            print("\ndry run, nothing written")
            return 0

        print()
        problems = []
        for name in sorted(wanted):
            if name.endswith(".sources.npy"):
                continue  # written alongside the recording it describes

            member = tar.getmember(name)
            if member.isdir():
                continue  # directory entries are implied by the files
            if not member.isfile():
                problems.append(f"{name}: skipped, not a regular file")
                continue
            blob = _read_member(tar, name)
            if blob is None:
                problems.append(f"{name}: skipped, unreadable")
                continue

            if name.startswith("data/samples/"):
                err = _merge_recording(tar, name, blob)
                if err:
                    problems.append(f"{name}: {err}")
                continue

            dest = _safe_dest(name)
            if dest is None:
                problems.append(f"{name}: refusing path outside the project")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)

        for p in problems:
            print(f"  ! {p}", file=sys.stderr)

    after = ds.counts()
    print("sample counts after import:")
    print(_fmt_counts(after))

    grew = [k for k in after if after[k] != before.get(k, 0)]
    if grew:
        print("\nmerged into existing recordings (nothing overwritten):")
        for k in sorted(grew):
            was = before.get(k, 0)
            print(f"  {k}  {was} -> {after[k]}")

    if WEIGHTS_PATH.exists():
        labels = []
        if LABELS_PATH.exists():
            try:
                labels = json.loads(LABELS_PATH.read_text())
            except (OSError, ValueError):
                labels = []
        print(f"\nmodel ready: {' '.join(labels) if labels else 'labels unreadable'}")
        print("run `signlang live` to use it.")
    else:
        print("\nno trained model in the bundle; run `signlang train`.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="signlang export / import")
    sub = ap.add_subparsers(dest="which", required=True)

    p_ex = sub.add_parser("export", help="package recordings (+ model) to move")
    p_ex.add_argument("--out", help=f"destination path (default ./{BUNDLE})")
    p_ex.add_argument(
        "--no-model",
        action="store_true",
        help="recordings only, leave the trained weights behind",
    )

    p_im = sub.add_parser("import", help="merge a bundle into this project")
    p_im.add_argument("--from", dest="src", help=f"bundle path (default ./{BUNDLE})")
    p_im.add_argument(
        "--dry-run", action="store_true", help="list contents without writing"
    )

    args = ap.parse_args(argv)
    if args.which == "export":
        return export(args.out, include_model=not args.no_model)
    return import_bundle(args.src, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())