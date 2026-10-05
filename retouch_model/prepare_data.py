"""
Build the before/after training pairs from the downloaded archives.

Inputs (in data/downloads/):
  ffhqr_part1_head.tar   first ~4 GB of FFHQR part 1 (retouched faces). It is
                         deliberately truncated; every image that was fully
                         downloaded is used, the cut-off one is skipped.
  ffhq/<folder>.tar      FFHQ originals, one WebDataset shard per 1000-image
                         folder (lossless WebP), from gaunernst/ffhq-1024-wds.

Outputs (in data/pairs/):
  retouched/<id>.png     the professional edit
  original/<id>.webp     the untouched photo
  index.json             per-pair edit statistics, where the edits are (for
                         biased crop sampling), and the train/val split

Usage:
  python prepare_data.py extract-retouched     # step 1, needs only the FFHQR tar
  python prepare_data.py folders               # which FFHQ shards to download
  python prepare_data.py extract-original      # step 2, needs the FFHQ shards
  python prepare_data.py index                 # step 3, verify pairs + stats
"""

from __future__ import annotations
import json
import os
import sys
import tarfile
from collections import Counter
from multiprocessing import Pool

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(HERE, "data", "downloads")
PAIRS = os.path.join(HERE, "data", "pairs")
RET_DIR = os.path.join(PAIRS, "retouched")
ORIG_DIR = os.path.join(PAIRS, "original")
FFHQR_TAR = os.path.join(DL, "ffhqr_part1_head.tar")
FFHQ_DIR = os.path.join(DL, "ffhq")
INDEX = os.path.join(PAIRS, "index.json")

SIZE = 1024
CHANGE_THR = 3        # a pixel counts as edited if any channel moved > 3/255
CELL = 32             # edit locations are recorded on a 32 px grid
MIN_CELL_PIXELS = 8   # cell needs this many edited pixels to count
VAL_EVERY = 10        # every 10th image (id % 10 == 9) is held out


def _folder_of(img_id: int) -> str:
    return f"{(img_id // 1000) * 1000:05d}"


def extract_retouched() -> None:
    """Stream the truncated FFHQR tar and keep every complete PNG."""
    os.makedirs(RET_DIR, exist_ok=True)
    n = 0
    try:
        with tarfile.open(FFHQR_TAR, mode="r|") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".png")):
                    continue
                f = tf.extractfile(m)
                data = f.read() if f else b""
                if len(data) != m.size:   # cut off by the truncated download
                    break
                out = os.path.join(RET_DIR, os.path.basename(m.name))
                with open(out, "wb") as fh:
                    fh.write(data)
                n += 1
                if n % 500 == 0:
                    print(f"  {n} retouched images extracted")
    except (tarfile.ReadError, EOFError, OSError) as e:
        print(f"  reached end of truncated archive ({type(e).__name__})")
    print(f"Extracted {n} retouched images -> {RET_DIR}")
    folders()


def _retouched_ids() -> list[int]:
    if not os.path.isdir(RET_DIR):
        return []
    return sorted(int(f[:-4]) for f in os.listdir(RET_DIR) if f.endswith(".png"))


def folders() -> None:
    """Report which 1000-image folders the retouched set covers."""
    counts = Counter(_folder_of(i) for i in _retouched_ids())
    print("Retouched images per FFHQ folder (shard to download):")
    for k in sorted(counts):
        print(f"  {k}.tar  {counts[k]} images")


def extract_original() -> None:
    """Pull the matching originals out of the FFHQ WebDataset shards."""
    os.makedirs(ORIG_DIR, exist_ok=True)
    want = set(_retouched_ids())
    by_folder = Counter(_folder_of(i) for i in want)
    total = 0
    for folder in sorted(by_folder):
        shard = os.path.join(FFHQ_DIR, f"{folder}.tar")
        if not os.path.exists(shard):
            print(f"  MISSING shard {shard} ({by_folder[folder]} pairs skipped)")
            continue
        got = 0
        with tarfile.open(shard, mode="r") as tf:
            for m in tf.getmembers():
                stem, ext = os.path.splitext(os.path.basename(m.name))
                if ext != ".webp" or not stem.isdigit() or int(stem) not in want:
                    continue
                data = tf.extractfile(m).read()
                with open(os.path.join(ORIG_DIR, f"{int(stem):05d}.webp"), "wb") as fh:
                    fh.write(data)
                got += 1
        print(f"  {folder}.tar: {got} originals")
        total += got
    print(f"Extracted {total} originals -> {ORIG_DIR}")


def _stats(img_id: int):
    o = cv2.imread(os.path.join(ORIG_DIR, f"{img_id:05d}.webp"), cv2.IMREAD_COLOR)
    r = cv2.imread(os.path.join(RET_DIR, f"{img_id:05d}.png"), cv2.IMREAD_COLOR)
    if o is None or r is None:
        return {"id": img_id, "ok": False, "why": "unreadable"}
    if o.shape != (SIZE, SIZE, 3) or r.shape != (SIZE, SIZE, 3):
        return {"id": img_id, "ok": False, "why": f"shape {o.shape} vs {r.shape}"}
    d = r.astype(np.int16) - o.astype(np.int16)
    a = np.abs(d).max(axis=2)
    changed = a > CHANGE_THR
    # Edited-pixel count per 32x32 cell -> list of cell centres with edits.
    grid = changed.reshape(SIZE // CELL, CELL, SIZE // CELL, CELL).sum(axis=(1, 3))
    ys, xs = np.nonzero(grid >= MIN_CELL_PIXELS)
    cells = [[int(y * CELL + CELL // 2), int(x * CELL + CELL // 2)] for y, x in zip(ys, xs)]
    return {
        "id": img_id,
        "ok": True,
        "frac": float(changed.mean()),          # share of pixels edited
        "mean_abs": float(np.abs(d).mean()),     # average change (0..255)
        "shift": [float(v) for v in d.reshape(-1, 3).mean(axis=0)],  # global BGR shift
        "max": int(a.max()),
        "cells": cells,
    }


def index() -> None:
    ids = [i for i in _retouched_ids()
           if os.path.exists(os.path.join(ORIG_DIR, f"{i:05d}.webp"))]
    print(f"Analysing {len(ids)} pairs...")
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as pool:
        rows = pool.map(_stats, ids, chunksize=16)
    bad = [r for r in rows if not r["ok"]]
    good = [r for r in rows if r["ok"]]
    for r in bad[:10]:
        print(f"  bad pair {r['id']:05d}: {r['why']}")

    frac = np.array([r["frac"] for r in good])
    shift = np.array([r["shift"] for r in good])
    ncell = np.array([len(r["cells"]) for r in good])
    print(f"Good pairs: {len(good)}   bad: {len(bad)}")
    print(f"Edited pixels per image: median {np.median(frac)*100:.2f}%  "
          f"p90 {np.percentile(frac, 90)*100:.2f}%  max {frac.max()*100:.2f}%")
    print(f"Images with no edits: {(frac == 0).sum()}")
    print(f"Edit cells per image: median {np.median(ncell):.0f}  p90 {np.percentile(ncell, 90):.0f}")
    print(f"Global colour shift (BGR, 0..255): mean {shift.mean(axis=0).round(3).tolist()}  "
          f"max |shift| {np.abs(shift).max():.3f}")

    val = [r["id"] for r in good if r["id"] % VAL_EVERY == VAL_EVERY - 1]
    train = [r["id"] for r in good if r["id"] % VAL_EVERY != VAL_EVERY - 1]
    with open(INDEX, "w") as fh:
        json.dump({"pairs": good, "train": train, "val": val,
                   "change_thr": CHANGE_THR}, fh)
    print(f"Wrote {INDEX}  (train {len(train)}, val {len(val)})")


if __name__ == "__main__":
    cmds = {"extract-retouched": extract_retouched, "folders": folders,
            "extract-original": extract_original, "index": index}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(1)
    cmds[sys.argv[1]]()
